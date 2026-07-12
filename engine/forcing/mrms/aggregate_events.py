r"""Aggregate per-storm MRMS 15-min NetCDFs into a single padded NetCDF.

Reads all matching NC files from an input directory and stacks them into
one (event, time_step, catchment) dataset padded to the longest event.

Output
------
  P               (event, time_step, catchment)  f32     NaN outside valid range
  storm_id        (event,)                       i32
  n_steps         (event,)                       i32     actual valid timestep count
  ts_start        (event,)                       f64     time of step 0 (source units)
  ts_end          (event,)                       f64     time of last valid step (source units)
  divide_id       (catchment,)                   str

  Downstream: slice P[i, :n_steps[i], :] to unpad an event.

Usage
-----
    python engine/forcing/mrms/aggregate_events.py \\
        --input-dir /path/to/forcing_15min --output /path/to/mrms_15min.nc

    # Filter to storms listed in a manifest CSV (column: storm_index):
    python engine/forcing/mrms/aggregate_events.py \\
        --input-dir /path/to/forcing_15min --manifest /path/to/events_original.csv \\
        --output /path/to/mrms_15min.nc
"""

import argparse
import glob
import os
import re
import sys

import netCDF4
import numpy as np


_FILE_PATTERN = 'storm_*_15min.nc'
_STORM_ID_RE = re.compile(r"storm_(\d+)_")


def _scan_files(input_dir: str, pattern: str, allowed_ids: set | None) -> list[dict]:
    """Return sorted list of {path, storm_id} dicts, optionally filtered."""
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        sys.exit(f"No files matched '{pattern}' in {input_dir}")

    records = []
    for p in paths:
        m = _STORM_ID_RE.search(os.path.basename(p))
        if not m:
            print(
                f"  WARNING: cannot parse storm ID from {os.path.basename(p)}, skipping",
            )
            continue
        sid = int(m.group(1))
        if allowed_ids is not None and sid not in allowed_ids:
            continue
        records.append({'path': p, 'storm_id': sid})

    records.sort(key=lambda r: r['storm_id'])
    return records


def _read_meta(records: list[dict]) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """First pass: read n_steps, time bounds, and the union of catchments.

    Each event file only carries the catchments in that event's own
    upstream basin (varies per gage), so the output catchment axis is the
    union across all events; events missing a given catchment are left
    NaN there (same padding convention as the time axis).

    Returns
    -------
    n_steps_arr   int32 array (n_events,)
    max_steps     int
    ts_starts     float64 array (n_events,)  time[0] in source units
    catchments    sorted object array of all divide_ids seen across events
    """
    n_steps_arr = np.empty(len(records), dtype=np.int32)
    ts_starts = np.empty(len(records), dtype=np.float64)
    all_cats: set[str] = set()

    for i, rec in enumerate(records):
        ds = netCDF4.Dataset(rec['path'], "r")
        n_steps_arr[i] = ds.dimensions['time'].size
        ts_starts[i] = float(ds.variables['time'][0])
        all_cats.update(ds.variables['divide_id'][:].tolist())
        ds.close()

    catchments = np.array(sorted(all_cats), dtype=object)
    return n_steps_arr, int(n_steps_arr.max()), ts_starts, catchments


def main():
    """Parse CLI args and aggregate per-storm MRMS files into a padded NetCDF."""
    parser = argparse.ArgumentParser(
        description="Aggregate per-storm MRMS 15-min NC files into one padded NetCDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing per-storm NC files",
    )
    parser.add_argument("--output", required=True, help="Output NetCDF path")
    parser.add_argument(
        "--pattern",
        default=_FILE_PATTERN,
        help=f"Glob pattern for input files (default: {_FILE_PATTERN})",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional CSV; only storms whose ID appears in --id-col will be included",
    )
    parser.add_argument(
        "--id-col",
        default="storm_index",
        help="Column in --manifest whose values match the storm IDs "
        "embedded in the filenames (default: storm_index)",
    )
    parser.add_argument(
        "--var",
        default="P",
        help="Output variable name (default: P)",
    )
    parser.add_argument(
        "--src-var",
        default="depth_mm_15min",
        help="Variable name in source files to read (default: depth_mm_15min)",
    )
    parser.add_argument(
        "--complevel",
        type=int,
        default=4,
        help="zlib compression level 1-9 (default: 4)",
    )
    args = parser.parse_args()

    # resolve manifest filter
    allowed_ids = None
    if args.manifest:
        import pandas as pd

        mdf = pd.read_csv(args.manifest)
        if args.id_col not in mdf.columns:
            sys.exit(
                f"Column '{args.id_col}' not found in {args.manifest}.\n"
                f"Available columns: {mdf.columns.tolist()}\n"
                f"Use --id-col to specify the right column.",
            )
        allowed_ids = set(mdf[args.id_col].dropna().astype(int).tolist())
        print(f"Manifest: {len(allowed_ids)} storms (col: {args.id_col})")

    # scan input files
    records = _scan_files(args.input_dir, args.pattern, allowed_ids)
    if not records:
        msg = "No matching storm files found after filtering."
        if allowed_ids:
            import glob as _glob
            import os as _os

            file_ids = sorted(
                int(m.group(1))
                for p in _glob.glob(_os.path.join(args.input_dir, args.pattern))
                if (m := _STORM_ID_RE.search(_os.path.basename(p)))
            )
            sample_manifest = sorted(allowed_ids)[:5]
            sample_files = file_ids[:5]
            msg += (
                f"\n  Manifest IDs (sample): {sample_manifest}"
                f"\n  File IDs     (sample): {sample_files}"
                f"\n  Check --id-col; current value is '{args.id_col}'."
            )
        sys.exit(msg)
    print(f"Found {len(records)} events in {args.input_dir}")

    # first pass: collect metadata
    n_steps_arr, max_steps, ts_starts, catchments = _read_meta(records)
    n_events = len(records)
    n_catchments = len(catchments)
    print(
        f"Events: {n_events}  |  max timesteps: {max_steps}  |  catchments: {n_catchments}",
    )

    # read time units from first file
    with netCDF4.Dataset(records[0]['path'], "r") as ds0:
        time_units = getattr(ds0.variables['time'], 'units', 'minutes since 2000-01-01')

    # create output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f"Writing {args.output} ...")
    nc_out = netCDF4.Dataset(args.output, 'w', format='NETCDF4')

    nc_out.createDimension('event', n_events)
    nc_out.createDimension('time_step', max_steps)
    nc_out.createDimension('catchment', n_catchments)

    # coordinates
    v_sid = nc_out.createVariable('storm_id', 'i4', ('event',))
    v_sid.long_name = 'storm index'
    v_sid[:] = np.array([r['storm_id'] for r in records], dtype=np.int32)

    v_ns = nc_out.createVariable('n_steps', 'i4', ('event',))
    v_ns.long_name = "number of valid 15-min timesteps in this event"
    v_ns[:] = n_steps_arr

    v_ts = nc_out.createVariable('ts_start', 'f8', ('event',))
    v_ts.units = time_units
    v_ts.long_name = "start time of the 5-day timeseries window (time step 0)"
    v_ts[:] = ts_starts

    v_te = nc_out.createVariable('ts_end', 'f8', ('event',))
    v_te.units = time_units
    v_te.long_name = "end time of the 5-day timeseries window (time step n_steps-1)"
    v_te[:] = ts_starts + (n_steps_arr - 1) * 15.0

    v_cat = nc_out.createVariable('divide_id', str, ('catchment',))
    v_cat.long_name = "NextGen catchment ID"
    v_cat[:] = catchments.astype(object)

    # main data variable — pre-filled with NaN for padding
    chunk_e = min(n_events, 16)
    chunk_t = min(max_steps, 481)
    chunk_c = min(n_catchments, 64)
    v_data = nc_out.createVariable(
        args.var,
        'f4',
        ('event', 'time_step', 'catchment'),
        fill_value=np.nan,
        zlib=True,
        complevel=args.complevel,
        chunksizes=(chunk_e, chunk_t, chunk_c),
    )
    v_data.units = "mm [15 min]-1"
    v_data.long_name = "MRMS precipitation depth"
    v_data.coordinates = "storm_id n_steps ts_start ts_end divide_id"

    catchment_idx = {c: j for j, c in enumerate(catchments.tolist())}

    # second pass: stream each event into the output, scattering its own
    # catchments into the shared global catchment axis
    for i, rec in enumerate(records):
        ds = netCDF4.Dataset(rec['path'], 'r')
        data = ds.variables[args.src_var][:]  # (time, catchment) — event-local catchments
        cats = ds.variables['divide_id'][:]
        ds.close()

        n = data.shape[0]
        cols = np.array([catchment_idx[c] for c in cats])
        v_data[i, :n, cols] = data  # (n, len(cols)) matches (time, catchment)

        if (i + 1) % 20 == 0 or i == n_events - 1:
            print(f"  {i + 1}/{n_events} events written")

    nc_out.close()
    print(f"Done -> {args.output}")
    print(
        f"  Shape: ({n_events} events, {max_steps} max steps, {n_catchments} catchments)",
    )


if __name__ == '__main__':
    main()
