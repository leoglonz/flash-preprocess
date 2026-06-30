"""Extract AORC forcing data for a set of NextGen catchments.

Reads the NOAA AORC v1.1 1km zarr from S3 (hourly, 1979-2025), spatially
averages each variable over catchment pixels, and optionally disaggregates
to 15-minute intervals.

Time range: specify --start / --end (ISO 8601 dates or datetimes). The script
opens one yearly zarr per calendar year spanned and slices to the exact range.
Use --year YYYY as a shorthand for a full calendar year.

Timestep: use --timestep 15min to disaggregate from hourly to 15-minute.
  APCP_surface (accumulated precip): uniform split — each hourly value is
    divided by 4 and repeated, conserving the total accumulation.
  All other variables (instantaneous): linearly interpolated between hourly
    values; the last hourly value is held for the trailing sub-hour steps.

Supports two index formats (auto-detected):
  Equal-weight (built by index_hf22.py)
  Area-weighted (built by index_hf22_weighted.py) <- more accurate at boundaries

Usage:
    # Full year, hourly
    python aorc_extract_hourly.py --year 2022 --index /path/to/index.pkl

    # Arbitrary range, 15-minute output
    python aorc_extract_hourly.py \\
        --start 2019-10-01 --end 2022-09-30T23:00 \\
        --timestep 15min \\
        --index /path/to/index.pkl --output-dir /path/to/output/

Output files saved to <output-dir>/:
    APCP_surface.npy
        shape: (basins, timesteps)
    DSWRF_surface.npy
    ...
    time.npy
        UTC timestamps as numpy datetime64, shape: (timesteps,)

Row i corresponds to station_ids[i] from the index pkl.
"""

import os
import time
import argparse
import pickle
from datetime import datetime, timedelta
from multiprocessing.pool import ThreadPool

import dask
import s3fs
import xarray as xr
import numpy as np
import pandas as pd

from flash_preprocess.utils import build_upstream_graph, expand_upstream, HF_PATH_DEFAULT

dask.config.set(pool=ThreadPool(4))


VARIABLE_LIST = [
    "APCP_surface",
    "DSWRF_surface",
    "TMP_2maboveground",
    "DLWRF_surface",
    "PRES_surface",
    "SPFH_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
]

_AORC_NCOLS = 8401  # Full CONUS grid width (constant across all years)

ACCUM_VARS = {"APCP_surface"}  # Variables treated as hourly accumulations.


def groupby_mean_equal(data: np.ndarray, interval: np.ndarray) -> np.ndarray:
    """Equal-weight average of pixel rows within groups.

    Parameters
    ----------
    data
        Shape (total_pixels, hours).
    interval
        Per-catchment pixel counts.

    Returns
    -------
    np.ndarray
        Shape (num_catchments, hours).
    """
    bins = np.insert(np.cumsum(interval), 0, 0)[:-1]
    mask = np.isnan(data)
    data = data.copy()
    data[mask] = 0.0
    g_count = np.add.reduceat(~mask, bins, axis=0)
    g_sum = np.add.reduceat(data, bins, axis=0)
    return g_sum / g_count


def weighted_mean(
    flat_raster: np.ndarray,
    cell_ids_list: list,
    weights_list: list,
    num_catchments: int,
    num_hours: int,
) -> np.ndarray:
    """Area-weighted average using exactextract coverage fractions.

    Parameters
    ----------
    flat_raster
        Shape (hours, n_pixels) for some spatial subset.

    Returns
    -------
    np.ndarray
        Shape (num_catchments, hours).
    """
    out = np.full((num_catchments, num_hours), np.nan, dtype=np.float32)
    for i, (cids, w) in enumerate(zip(cell_ids_list, weights_list)):
        cols = flat_raster[:, cids]
        has_nan = np.isnan(cols).any(axis=1)
        out[i] = np.nansum(cols * w, axis=1) / w.sum()
        out[i, has_nan] = np.nan
    return out


def spatial_subset_weighted(
    ds: xr.Dataset,
    cell_ids: list,
) -> tuple[xr.Dataset, list]:
    """Subset data to pixel bounding box and remap cell_ids to local flat index.

    Parameters
    ----------
    ds
        AORC dataset with dimensions (time, latitude, longitude).
    cell_ids
        List of 1D arrays of full-grid cell_ids for each catchment.
    
    Returns
    -------
    tuple[xr.Dataset, list]
        ds_sub: subsetted dataset with dimensions (time, latitude, longitude)
        local_cell_ids: list of 1D arrays of local flat indices for each cat.
    
    Full-grid cell_id = row * 8401 + col, row 0 = highest latitude.
    After isel, local_cell_id = (row - r0) * n_sub_cols + (col - c0).
    """
    all_cids = np.concatenate(cell_ids)
    rows = all_cids // _AORC_NCOLS
    cols = all_cids % _AORC_NCOLS
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    n_sub = c1 - c0 + 1

    print(f"  Bbox: rows {r0}-{r1}, cols {c0}-{c1} "
          f"> {r1-r0+1} * {n_sub} = {(r1-r0+1)*n_sub:,} pixels "
          f"(was {4201*8401:,})")

    ds_sub = ds.isel(latitude=slice(r0, r1 + 1), longitude=slice(c0, c1 + 1))
    local_cids = [
        (cids // _AORC_NCOLS - r0) * n_sub + (cids % _AORC_NCOLS - c0)
        for cids in cell_ids
    ]
    return ds_sub, local_cids


def spatial_subset_equal(
    ds: xr.Dataset,
    row_flat: np.ndarray,
    col_flat: np.ndarray,
) -> tuple[xr.Dataset, tuple[list, list]]:
    """Subset dataset to the pixel bounding box and return local row/col arrays.
    
    Parameters
    ----------
    ds
        AORC dataset with dimensions (time, latitude, longitude).
    row_flat
        1D array of full-grid row indices for all catchment pixels.
    col_flat
        1D array of full-grid column indices for all catchment pixels.
    
    Returns
    -------
    tuple[xr.Dataset, tuple[list, list]]
        ds_sub: subsetted dataset with dimensions (time, latitude, longitude)
        row_local, col_local: lists of 1D arrays of local row/col indices
        for each catchment.
    """
    rows = np.asarray(row_flat)
    cols = np.asarray(col_flat)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    print(f"  Bbox: rows {r0}-{r1}, cols {c0}-{c1} "
          f"> {r1-r0+1} * {c1-c0+1} = {(r1-r0+1)*(c1-c0+1):,} pixels "
          f"(was {4201*8401:,})")

    ds_sub = ds.isel(latitude=slice(r0, r1 + 1), longitude=slice(c0, c1 + 1))
    return ds_sub, (rows - r0).tolist(), (cols - c0).tolist()


def disaggregate_to_15min(
    data: np.ndarray, var_name: str, time_vals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Disaggregate hourly catchment data to 15-minute intervals.

    Parameters
    ----------
    data
        Shape (n_basins, n_hours).
    var_name
        Used to select accumulation vs. interpolation treatment.
    time_vals
        Hourly datetime64 array, shape (n_hours,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        data_15min (n_basins, n_hours * 4) and time_15min datetime64[m] array.
    """
    n_hours = data.shape[1]
    n_steps = n_hours * 4

    if var_name in ACCUM_VARS:
        # uniform split: divide by 4 and repeat each value across 4 sub-steps
        data_15min = np.repeat(data / 4.0, 4, axis=1)
    else:
        # linear interpolation between hourly values; hold last value at trailing steps
        x_15 = np.arange(n_steps, dtype=np.float64) * 0.25
        i_low = np.clip(np.floor(x_15).astype(int), 0, n_hours - 1)
        i_high = np.clip(np.ceil(x_15).astype(int), 0, n_hours - 1)
        frac = (x_15 - i_low).astype(np.float32)
        data_15min = data[:, i_low] * (1.0 - frac) + data[:, i_high] * frac

    t0 = time_vals[0].astype("datetime64[m]")
    time_15min = t0 + np.arange(n_steps) * np.timedelta64(15, "m")
    return data_15min, time_15min


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
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(dt_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Cannot parse datetime: {dt_str!r}")
    total_minutes = dt.hour * 60 + dt.minute + dt.second / 60
    rounded = int(total_minutes / 15 + 0.5) * 15
    dt_r = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=rounded)
    return np.datetime64(dt_r, "m")


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
    if window_str.endswith("d"):
        return int(window_str[:-1]) * 24 * 60
    elif window_str.endswith("h"):
        return int(window_str[:-1]) * 60
    elif window_str.endswith("min"):
        return int(window_str[:-3])
    raise ValueError(f"Unrecognised window format {window_str!r}. Use e.g. '5d', '120h', '7200min'.")


def open_aorc(start: np.datetime64, end: np.datetime64) -> xr.Dataset:
    """Open AORC zarr stores for every calendar year spanned by [start, end].
    
    Parameters
    ----------
    start
        Start of range, inclusive, as datetime64[h].
    end
        End of range, inclusive, as datetime64[h].
    
    Returns
    -------
    xr.Dataset
        AORC dataset with dimensions (time, latitude, longitude),
        sliced to [start,end].
    """
    y0 = int(str(start)[:4])
    y1 = int(str(end)[:4])
    years = range(y0, y1 + 1)

    _s3 = s3fs.S3FileSystem(anon=True)
    stores = [
        s3fs.S3Map(root=f"s3://noaa-nws-aorc-v1-1-1km/{y}.zarr", s3=_s3, check=False)
        for y in years
    ]
    ds = xr.open_mfdataset(stores, engine="zarr", parallel=True, consolidated=True)
    ds = ds.sortby("latitude", ascending=False)
    ds = ds.sel(time=slice(start, end))
    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Extract AORC forcing data to catchments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Arg parse
    time_grp = parser.add_mutually_exclusive_group(required=True)
    time_grp.add_argument("--year", type=int,
                          help="Full calendar year (e.g. 2022)")
    time_grp.add_argument("--start", metavar="DATETIME",
                          help="Start of range, ISO 8601. Pair with --end.")
    time_grp.add_argument("--center", metavar="DATETIME",
                          help="Centre of a symmetric time window (ISO 8601, e.g. "
                               "'2021-10-09T13:37:30'). Rounded to nearest 15 min "
                               "when --timestep 15min. Pair with --window.")

    parser.add_argument("--end", metavar="DATETIME", default=None,
                        help="End of range, inclusive. Required with --start.")
    parser.add_argument("--window", metavar="DURATION", default=None,
                        help="Window size around --center, e.g. '5d', '120h', "
                             "'7200min'. Required with --center.")

    # spatial index
    parser.add_argument("--index", required=True,
                        help="Index pkl (index_hf22.py or index_hf22_weighted.py)")

    # optional catchment filter
    parser.add_argument("--catchment-ids", nargs="+", metavar="ID", default=None,
                        help="Subset of catchment IDs to output. If omitted, all "
                             "catchments in the index are used.")
    parser.add_argument("--upstream", action="store_true",
                        help="Expand --catchment-ids to include all upstream "
                             "catchments (reads hydrofabric network).")
    parser.add_argument("--hydrofabric", default=HF_PATH_DEFAULT,
                        help="Path to conus_nextgen.gpkg, needed for --upstream.")

    # output
    parser.add_argument("--output-dir", default=".",
                        help="Output directory")
    parser.add_argument("--output-format", choices=["npy", "netcdf"], default="npy",
                        help="npy: one .npy per variable + time.npy (default). "
                             "netcdf: single forcing.nc with all variables, time "
                             "coordinate, and catchment-id coordinate.")
    parser.add_argument("--variables", nargs="+", default=VARIABLE_LIST,
                        help="Variables to extract (default: all 8)")
    parser.add_argument("--timestep", choices=["1h", "15min"], default="1h",
                        help="Output timestep: 1h (default) or 15min.")
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
        start = np.datetime64(f"{args.year}-01-01T00:00", "h")
        end = np.datetime64(f"{args.year}-12-31T23:00", "h")

    elif args.start:
        start = np.datetime64(args.start, "h")
        end = np.datetime64(args.end, "h")

    else:  # --center / --window
        window_min = parse_window(args.window)
        step_min = 15 if do_15min else 60
        n_window = window_min // step_min

        if do_15min:
            center = round_to_nearest_15min(args.center)
            print(f"Center rounded to nearest 15 min: {center}")
        else:
            center = np.datetime64(args.center, "h")

        half = n_window // 2
        win_start = center - np.timedelta64(int(half * step_min), "m")
        win_end = win_start + np.timedelta64(int((n_window - 1) * step_min), "m")
        print(f"Window: {win_start} → {win_end} ({n_window} × {args.timestep})")

        if do_15min:
            # open one extra hour past the end so interpolation is valid at win_end
            start = win_start.astype("datetime64[h]")
            end = win_end.astype("datetime64[h]") + np.timedelta64(1, "h")
            i_start = int(
                (win_start.astype("datetime64[m]") - start.astype("datetime64[m]"))
                / np.timedelta64(15, "m")
            )
            trim_slice = (i_start, i_start + n_window)
        else:
            start = win_start.astype("datetime64[h]")
            end = win_end.astype("datetime64[h]")

    # load index
    print(f"Loading index: {args.index}")
    with open(args.index, "rb") as f:
        idx = pickle.load(f)

    weighted = "weights" in idx
    all_cat_ids = idx["station_ids"]

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
        cell_ids_list = [idx["cell_ids"][i] for i in np.where(cat_mask)[0]]
        weights_list = [idx["weights"][i] for i in np.where(cat_mask)[0]]
        print(f"  Area-weighted index")
    else:
        row_list_sel = [idx["row_list"][i] for i in np.where(cat_mask)[0]]
        col_list_sel = [idx["col_list"][i] for i in np.where(cat_mask)[0]]
        pixel_counts = [len(r) for r in row_list_sel]
        row_flat = [x for sub in row_list_sel for x in sub]
        col_flat = [x for sub in col_list_sel for x in sub]
        print(f"  Equal-weight index, {sum(pixel_counts)} total pixels")

    # open and slice AORC
    print(f"Opening AORC zarr(s) for {start} → {end} ...")
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
        t0_m = time_vals[0].astype("datetime64[m]")
        out_times = t0_m + np.arange(n_hours * 4) * np.timedelta64(15, "m")
        if trim_slice is not None:
            out_times = out_times[trim_slice[0]:trim_slice[1]]
    else:
        out_times = time_vals.astype("datetime64[h]")

    results_store: dict = {}  # only used for netcdf output

    for var_name in args.variables:
        print(f"  {var_name}...", end=" ", flush=True)
        t_var = time.time()

        raw = ds[var_name].compute().values  # (hours, bbox_lat, bbox_lon)

        if weighted:
            flat = raw.reshape(n_hours, -1)
            result = weighted_mean(flat, cell_ids_list, weights_list, n_basins, n_hours)
        else:
            raw_sel = raw[..., row_flat, col_flat]  # (hours, total_pixels)
            result = groupby_mean_equal(raw_sel.T, pixel_counts)
        del raw

        if do_15min:
            result, _ = disaggregate_to_15min(result, var_name, time_vals)
            if trim_slice is not None:
                result = result[:, trim_slice[0]:trim_slice[1]]

        if args.output_format == "npy":
            np.save(os.path.join(args.output_dir, f"{var_name}.npy"), result)
        else:
            results_store[var_name] = result

        del result
        print(f"done ({time.time() - t_var:.1f}s)")

    ds.close()

    # write output
    if args.output_format == "netcdf":
        print("Writing forcing.nc...", end=" ", flush=True)
        ds_out = xr.Dataset(
            {v: xr.DataArray(d.astype(np.float32), dims=["catchment", "time"])
             for v, d in results_store.items()},
            coords={
                "time": out_times.astype("datetime64[ns]"),
                "catchment": out_cat_ids,
                "latitude": xr.DataArray(cat_lats, dims=["catchment"], attrs={
                    "standard_name": "latitude",
                    "long_name": "catchment centroid latitude",
                    "units": "degrees_north",
                }),
                "longitude": xr.DataArray(cat_lons, dims=["catchment"], attrs={
                    "standard_name": "longitude",
                    "long_name": "catchment centroid longitude",
                    "units": "degrees_east",
                }),
            },
        )
        nc_path = os.path.join(args.output_dir, "forcing.nc")
        ds_out.to_netcdf(nc_path)
        print(f"done → {nc_path}")
    else:
        np.save(os.path.join(args.output_dir, "time.npy"), out_times)

    n_steps = len(out_times)
    print(f"\nComplete in {time.time() - t0:.1f}s  |  shape: ({n_basins}, {n_steps})")


if __name__ == "__main__":
    main()
