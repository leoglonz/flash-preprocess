r"""Extract AORC forcing data for a set of NextGen catchments.

Reads the NOAA AORC v1.1 1km zarr from S3 (hourly, 1979-2025), spatially
averages each variable over catchment pixels, and optionally disaggregates
to 15-minute intervals.

Time range is set via --start / --end (ISO 8601 dates or datetimes), or
--year YYYY as a shorthand for a full calendar year. One yearly zarr is
opened per calendar year spanned and sliced to the exact range.

Disaggregation (--timestep 15min):
  APCP_surface (accumulated precip): uniform split — each hourly value
    divided by 4 and repeated, conserving the total accumulation.
  All other variables (instantaneous): linearly interpolated between
    hourly values, with the last value held for trailing sub-hour steps.

Index format (auto-detected):
  Equal-weight, built by index_hf.py
  Area-weighted, built by index_hf_weighted.py (more accurate at boundaries)

Output
------
  <output-dir>/forcing.nc  single NetCDF, shape (catchment, time) per variable
    Row i corresponds to station_ids[i] from the index pkl.

Usage
-----
    # Full year, hourly
    python engine/forcing/aorc/extract.py --year 2022 --index /path/to/index.pkl

    # Arbitrary range, 15-minute output
    python engine/forcing/aorc/extract.py \\
        --start 2019-10-01 --end 2022-09-30T23:00 \\
        --timestep 15min \\
        --index /path/to/index.pkl --output-dir /path/to/output/
"""

import os
import time
import argparse
import pickle
from datetime import datetime, timedelta

import netCDF4
import numpy as np

from flash_preprocess.aorc import (
    VARIABLE_LIST,
    open_aorc,
    spatial_subset_weighted,
    spatial_subset_equal,
    build_weight_matrix,
    weighted_mean,
    groupby_mean_equal,
    disaggregate_to_15min,
)
from flash_preprocess.utils import (
    build_upstream_graph,
    expand_upstream,
    HF_PATH_DEFAULT,
)


def round_to_nearest_15min(dt_str: str) -> np.datetime64:
    """Parse an ISO 8601 datetime string and round to nearest 15min (half-up).

    Parameters
    ----------
    dt_str
        ISO 8601 datetime string, e.g. '2021-10-09T 13:37:30'.

    Returns
    -------
    np.datetime64
        Rounded datetime64[m].
    """
    for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M', '%Y-%m-%d'):
        try:
            dt = datetime.strptime(dt_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Cannot parse datetime: {dt_str!r}")
    total_minutes = dt.hour * 60 + dt.minute + dt.second / 60
    rounded = int(total_minutes / 15 + 0.5) * 15
    dt_r = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        minutes=rounded,
    )
    return np.datetime64(dt_r, 'm')


def parse_window(window_str: str) -> int:
    """Parse window string into total minutes. Accepts '5d', '120h', '7200min'.

    Parameters
    ----------
    window_str
        String specifying a duration in days, hours, or minutes.

    Returns
    -------
    int
        Total duration in minutes.
    """
    if window_str.endswith('d'):
        return int(window_str[:-1]) * 24 * 60
    elif window_str.endswith('h'):
        return int(window_str[:-1]) * 60
    elif window_str.endswith('min'):
        return int(window_str[:-3])
    raise ValueError(
        f"Unrecognised window format {window_str!r}. Use e.g. '5d', '120h', '7200min'.",
    )


def main():
    """Parse CLI args and run the AORC extraction."""
    parser = argparse.ArgumentParser(
        description="Extract AORC forcing data to catchments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Arg parse
    time_grp = parser.add_mutually_exclusive_group(required=True)
    time_grp.add_argument('--year', type=int, help="Full calendar year (e.g. 2022)")
    time_grp.add_argument(
        '--start',
        metavar='DATETIME',
        help="Start of range, ISO 8601. Pair with --end.",
    )
    time_grp.add_argument(
        '--center',
        metavar='DATETIME',
        help="Centre of a symmetric time window (ISO 8601, e.g. "
        "'2021-10-09T13:37:30'). Rounded to nearest 15 min "
        "when --timestep 15min. Pair with --window.",
    )

    parser.add_argument(
        '--end',
        metavar='DATETIME',
        default=None,
        help="End of range, inclusive. Required with --start.",
    )
    parser.add_argument(
        '--window',
        metavar='DURATION',
        default=None,
        help="Window size around --center, e.g. '5d', '120h', "
        "'7200min'. Required with --center.",
    )

    # spatial index
    parser.add_argument(
        '--index',
        required=True,
        help="Index pkl (index_hf22.py or index_hf22_weighted.py)",
    )

    # optional catchment filter
    parser.add_argument(
        '--catchment-ids',
        nargs="+",
        metavar='ID',
        default=None,
        help="Subset of catchment IDs to output. If omitted, all "
        "catchments in the index are used.",
    )
    parser.add_argument(
        '--upstream',
        action='store_true',
        help="Expand --catchment-ids to include all upstream "
        "catchments (reads hydrofabric network).",
    )
    parser.add_argument(
        '--hydrofabric',
        default=HF_PATH_DEFAULT,
        help="Path to conus_nextgen.gpkg, needed for --upstream.",
    )

    # output
    parser.add_argument('--output-dir', default='.', help="Output directory")
    parser.add_argument(
        '--variables',
        nargs="+",
        default=VARIABLE_LIST,
        help="Variables to extract (default: all 8)",
    )
    parser.add_argument(
        '--timestep',
        choices=['1h', '15min'],
        default='1h',
        help="Output timestep: 1h (default) or 15min.",
    )
    args = parser.parse_args()

    if args.start and args.end is None:
        parser.error("--end is required when --start is used")
    if args.center and args.window is None:
        parser.error("--window is required when --center is used")
    if args.upstream and not args.catchment_ids:
        parser.error("--upstream requires --catchment-ids")

    do_15min = args.timestep == "15min"

    # resolve time bounds and optional trim window
    trim_slice: tuple | None = None

    if args.year:
        start = np.datetime64(f"{args.year}-01-01T00:00", 'h')
        end = np.datetime64(f"{args.year}-12-31T23:00", 'h')

    elif args.start:
        start = np.datetime64(args.start, 'h')
        end = np.datetime64(args.end, 'h')

    else:  # --center / --window
        window_min = parse_window(args.window)
        step_min = 15 if do_15min else 60
        n_window = window_min // step_min

        if do_15min:
            center = round_to_nearest_15min(args.center)
            print(f"Center rounded to nearest 15 min: {center}")
        else:
            center = np.datetime64(args.center, 'h')

        half = n_window // 2
        win_start = center - np.timedelta64(int(half * step_min), 'm')
        win_end = win_start + np.timedelta64(int((n_window - 1) * step_min), 'm')
        print(f"Window: {win_start} -> {win_end} ({n_window} * {args.timestep})")

        if do_15min:
            # open one extra hour past the end so interpolation is valid at win_end
            start = win_start.astype('datetime64[h]')
            end = win_end.astype('datetime64[h]') + np.timedelta64(1, 'h')
            i_start = int(
                (win_start.astype('datetime64[m]') - start.astype('datetime64[m]'))
                / np.timedelta64(15, 'm'),
            )
            trim_slice = (i_start, i_start + n_window)
        else:
            start = win_start.astype('datetime64[h]')
            end = win_end.astype('datetime64[h]')

    # load index
    print(f"Loading index: {args.index}")
    with open(args.index, 'rb') as f:
        idx = pickle.load(f)

    weighted = 'weights' in idx
    all_cat_ids = idx['station_ids']

    # resolve catchment selection
    if args.catchment_ids:
        seed_ids = set(args.catchment_ids)
        if args.upstream:
            graph = build_upstream_graph(args.hydrofabric)
            seed_ids = expand_upstream(seed_ids, graph)
            print(f"  Expanded to {len(seed_ids)} catchments (including upstream)")
        cat_mask = np.array([c in seed_ids for c in all_cat_ids])
    else:
        cat_mask = np.ones(len(all_cat_ids), dtype=bool)

    out_cat_ids = all_cat_ids[cat_mask]
    n_basins = int(cat_mask.sum())
    print(f"  Outputting {n_basins} catchments")

    if weighted:
        cell_ids_list = [idx['cell_ids'][i] for i in np.where(cat_mask)[0]]
        weights_list = [idx['weights'][i] for i in np.where(cat_mask)[0]]
        print("  Area-weighted index")
    else:
        row_list_sel = [idx['row_list'][i] for i in np.where(cat_mask)[0]]
        col_list_sel = [idx['col_list'][i] for i in np.where(cat_mask)[0]]
        pixel_counts = [len(r) for r in row_list_sel]
        row_flat = [x for sub in row_list_sel for x in sub]
        col_flat = [x for sub in col_list_sel for x in sub]
        print(f"  Equal-weight index, {sum(pixel_counts)} total pixels")

    # open and slice AORC
    print(f"Opening AORC zarr(s) for {start} -> {end} ...")
    ds = open_aorc(start, end)
    time_vals = ds.time.values
    n_hours = len(time_vals)
    print(f"  {n_hours} hourly timesteps loaded")

    # subset spatially to the pixel bbox — avoids downloading the full CONUS grid
    print("  Subsetting to pixel bounding box...")
    if weighted:
        ds, cell_ids_list = spatial_subset_weighted(ds, cell_ids_list)
    else:
        ds, row_flat, col_flat = spatial_subset_equal(ds, row_flat, col_flat)

    if do_15min:
        print("  Will disaggregate to 15-min after extraction")

    # compute catchment centroids from the bbox grid coords (used in NetCDF output)
    lat_bbox = ds.latitude.values
    lon_bbox = ds.longitude.values

    if weighted:
        n_sub_cols = len(lon_bbox)
        cat_lats, cat_lons = [], []
        for cids, w in zip(cell_ids_list, weights_list):
            wn = w / w.sum()
            cat_lats.append(float(np.dot(wn, lat_bbox[cids // n_sub_cols])))
            cat_lons.append(float(np.dot(wn, lon_bbox[cids % n_sub_cols])))
    else:
        cuts = np.cumsum(pixel_counts)[:-1]
        rows_per = np.split(np.asarray(row_flat, dtype=int), cuts)
        cols_per = np.split(np.asarray(col_flat, dtype=int), cuts)
        cat_lats = [float(lat_bbox[r].mean()) for r in rows_per]
        cat_lons = [float(lon_bbox[c].mean()) for c in cols_per]

    cat_lats = np.array(cat_lats, dtype=np.float32)
    cat_lons = np.array(cat_lons, dtype=np.float32)

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # build the output time coordinate once, before the variable loop
    if do_15min:
        t0_m = time_vals[0].astype('datetime64[m]')
        out_times = t0_m + np.arange(n_hours * 4) * np.timedelta64(15, 'm')
        if trim_slice is not None:
            out_times = out_times[trim_slice[0] : trim_slice[1]]
    else:
        out_times = time_vals.astype('datetime64[h]')

    n_steps = len(out_times)

    nc_path = os.path.join(args.output_dir, 'aorc_extracted.nc')
    nc_out = netCDF4.Dataset(nc_path, 'w', format='NETCDF4')
    nc_out.createDimension('catchment', n_basins)
    nc_out.createDimension('time', n_steps)

    epoch = np.datetime64('1970-01-01T00:00', 'm')
    time_num = (out_times.astype('datetime64[m]') - epoch).astype(np.int64)
    v_time = nc_out.createVariable('time', 'i8', ('time',))
    v_time.units = "minutes since 1970-01-01 00:00:00 UTC"
    v_time.calendar = 'standard'
    v_time[:] = time_num

    v_cat = nc_out.createVariable('divide_id', str, ('catchment',))
    v_cat.long_name = "NextGen catchment ID"
    v_cat[:] = np.array(out_cat_ids, dtype=object)

    v_lat = nc_out.createVariable('latitude', 'f4', ('catchment',))
    v_lat.units = 'degrees_north'
    v_lat.standard_name = 'latitude'
    v_lat[:] = cat_lats

    v_lon = nc_out.createVariable('longitude', 'f4', ('catchment',))
    v_lon.units = 'degrees_east'
    v_lon.standard_name = 'longitude'
    v_lon[:] = cat_lons

    # pre-build sparse weight matrix (weighted path) — built once, reused per variable
    if weighted:
        n_pixels = len(lat_bbox) * len(lon_bbox)
        print("  Building sparse weight matrix...")
        W = build_weight_matrix(cell_ids_list, weights_list, n_basins, n_pixels)

    # fetch all variables in a single dask graph execution (parallel S3 reads)
    print(f"  Fetching all {len(args.variables)} variables from zarr...")
    t_fetch = time.time()
    ds_computed = ds[list(args.variables)].compute()
    print(f"  Fetch complete ({time.time() - t_fetch:.1f}s)")

    print(f"Streaming to {nc_path} ({n_basins} catchments * {n_steps} steps)")

    for var_name in args.variables:
        print(f"  {var_name}...", end=" ", flush=True)
        t_var = time.time()

        raw = ds_computed[var_name].values  # (hours, bbox_lat, bbox_lon)

        if weighted:
            flat = raw.reshape(n_hours, -1)
            result = weighted_mean(flat, W)
        else:
            raw_sel = raw[..., row_flat, col_flat]  # (hours, total_pixels)
            result = groupby_mean_equal(raw_sel.T, pixel_counts)
        del raw

        if do_15min:
            result, _ = disaggregate_to_15min(result, var_name, time_vals)
            if trim_slice is not None:
                result = result[:, trim_slice[0] : trim_slice[1]]

        nc_var = nc_out.createVariable(
            var_name,
            'f4',
            ('catchment', 'time'),
            zlib=True,
            complevel=1,
            chunksizes=(n_basins, min(n_steps, 1000)),
        )
        nc_var.coordinates = "divide_id latitude longitude"
        nc_var[:] = result.astype(np.float32)

        del result
        print(f"done ({time.time() - t_var:.1f}s)")

    del ds_computed
    ds.close()

    nc_out.close()

    print(f"\nComplete in {time.time() - t0:.1f}s  |  shape: ({n_basins}, {n_steps})")


if __name__ == '__main__':
    main()
