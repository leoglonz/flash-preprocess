r"""Merge AORC and MRMS 15-min NetCDFs into a single combined forcing file.

MRMS's `storm_id` is the same ID as AORC's `event_id` (aggregate_events.py
names its files storm_<event_id>_15min.nc). This script:
  1. Joins AORC and MRMS on event_id (MRMS storm_id cast to str).
  2. Aligns catchments by ID (inner join on divide_id / catchment strings).
  3. Writes a combined (event, time_step, catchment) NetCDF with variables
     from both sources.

Output
------
  Coordinates:
    event_id        (event,)     str     shared AORC/MRMS event ID
    n_steps         (event,)     i32     valid 15-min steps (minimum of both sources)
    ts_start        (event,)     f64     start of the 5-day window (minutes since 1970-01-01)
    ts_end          (event,)     f64     end of the 5-day window (minutes since 1970-01-01)
    event_gage_id   (event,)     str     zero-padded 8-digit USGS gauge downstream of event
    event_divide_id (event,)     str     NextGen catchment ID downstream of event
    catchment       (catchment,) str     NextGen divide ID (intersection of both files)
    latitude        (catchment,) f32
    longitude       (catchment,) f32

  Variables:
    P     (event, time_step, catchment)  f32  from MRMS
    T               (event, time_step, catchment)  f32  from AORC
    PET                (event, time_step, catchment)  f32  from AORC

Usage
-----
    python engine/forcing/merge_15min.py \\
        --aorc     /path/to/aorc_15min.nc \\
        --aorc-hr  /path/to/aorc_hr.nc \\
        --mrms     /path/to/mrms_15min.nc \\
        --output   /path/to/forcing_15min.nc
"""

import argparse
import sys
from pathlib import Path

import netCDF4
import numpy as np
from tqdm.auto import tqdm


def _load_str_var(ds: netCDF4.Dataset, name: str) -> np.ndarray:
    """Read a netCDF4 string variable as a plain numpy object array of strings, shape (n,)."""
    raw = ds.variables[name][:]
    if isinstance(raw, np.ma.MaskedArray):
        raw = raw.filled()
    out = np.empty(len(raw), dtype=object)
    for i, v in enumerate(raw):
        out[i] = str(v) if not isinstance(v, str) else v
    return out


def main() -> None:
    """Parse CLI arguments and run the AORC + MRMS merge pipeline."""
    parser = argparse.ArgumentParser(
        description="Merge AORC and MRMS 15-min forcing NetCDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--aorc",
        required=True,
        help="AORC 15-min NC (output of to_events.py, events_15min.nc)",
    )
    parser.add_argument(
        "--aorc-hr",
        required=True,
        help="AORC hourly antecedent NC (aorc_hr.nc), used only to check that "
        "its warmup window ends exactly where the MRMS event window begins",
    )
    parser.add_argument(
        "--mrms",
        required=True,
        help="MRMS 15-min NC (output of aggregate_events.py)",
    )
    parser.add_argument("--output", required=True, help="Output NetCDF path")
    parser.add_argument(
        "--complevel",
        type=int,
        default=4,
        help="zlib compression level 1-9 (default: 4)",
    )
    args = parser.parse_args()

    # open source files
    nc_aorc = netCDF4.Dataset(args.aorc, "r")
    nc_mrms = netCDF4.Dataset(args.mrms, "r")

    aorc_event_ids = _load_str_var(nc_aorc, "event_id")
    mrms_event_ids = np.array(
        [str(int(sid)) for sid in nc_mrms.variables["storm_id"][:]],
        dtype=object,
    )
    event_to_mrms_idx = {eid: i for i, eid in enumerate(mrms_event_ids)}

    # build aligned event list: iterate AORC order, keep events present in MRMS
    aorc_indices, mrms_indices, event_ids = [], [], []
    skipped = 0
    for i, eid in enumerate(aorc_event_ids):
        j = event_to_mrms_idx.get(eid)
        if j is None:
            skipped += 1
            continue
        aorc_indices.append(i)
        mrms_indices.append(j)
        event_ids.append(eid)

    n_events = len(aorc_indices)
    if n_events == 0:
        sys.exit("No events matched between AORC and MRMS files on event_id.")
    print(
        f"Matched {n_events} events ({skipped} AORC events skipped, "
        f"not present in MRMS file)",
    )

    # Both files' event windows come from the same build_manifest() logic,
    # but only if both were built from run_pipeline.py runs with matching
    # WINDOW_DAYS/CENTROID -- a stale file from an older config would merge
    # silently, splicing MRMS precip for one time window onto AORC
    # temperature/PET for a different one. Catch that here instead.
    aorc_ts_start_chk = np.array(nc_aorc.variables["ts_start"][:], dtype=np.float64)
    mrms_ts_start_chk = np.array(nc_mrms.variables["ts_start"][:], dtype=np.float64)
    offset_min = aorc_ts_start_chk[aorc_indices] - mrms_ts_start_chk[mrms_indices]
    bad = np.abs(offset_min) > 1.0  # allow <1 min for float rounding
    if bad.any():
        sys.exit(
            f"{bad.sum()}/{n_events} matched events have AORC/MRMS window start times that "
            f"disagree by more than a minute (median offset {np.median(offset_min[bad]):.0f} "
            f"min, e.g. event {event_ids[np.argmax(bad)]}). The two files were likely built "
            f"with different WINDOW_DAYS/CENTROID settings (or one is stale) -- regenerate "
            f"both from the same run_pipeline.py config before merging.",
        )

    # The AORC hourly antecedent file (30-day warmup) must end exactly where
    # the MRMS event window begins, with no gap or overlap -- its stored
    # ts_end is the timestamp of the *last hourly step itself* (an hourly
    # value at hour H covers [H, H+1)), so the true end of the warmup period
    # is ts_end + 60 min, which should equal MRMS's ts_start exactly.
    nc_aorc_hr = netCDF4.Dataset(args.aorc_hr, "r")
    aorc_hr_event_ids = _load_str_var(nc_aorc_hr, "event_id")
    aorc_hr_idx = {eid: i for i, eid in enumerate(aorc_hr_event_ids)}
    aorc_hr_ts_end = np.array(nc_aorc_hr.variables["ts_end"][:], dtype=np.float64)
    nc_aorc_hr.close()

    missing_hr = [eid for eid in event_ids if eid not in aorc_hr_idx]
    if missing_hr:
        sys.exit(
            f"{len(missing_hr)}/{n_events} matched events are missing from --aorc-hr "
            f"(e.g. event {missing_hr[0]}) -- aorc_hr.nc and the AORC/MRMS 15-min files "
            f"appear to come from different runs.",
        )
    hr_end_for_event = np.array([aorc_hr_ts_end[aorc_hr_idx[eid]] for eid in event_ids])
    gap_min = mrms_ts_start_chk[mrms_indices] - (hr_end_for_event + 60.0)
    bad_gap = np.abs(gap_min) > 1.0
    if bad_gap.any():
        sys.exit(
            f"{bad_gap.sum()}/{n_events} matched events have a gap/overlap between the AORC "
            f"hourly warmup and the MRMS event window (median {np.median(gap_min[bad_gap]):.0f} "
            f"min, e.g. event {event_ids[np.argmax(bad_gap)]}). aorc_hr.ts_end + 60min should "
            f"equal mrms.ts_start exactly -- the two pipelines likely used different "
            f"WINDOW_DAYS/ANTECEDENT_DAYS/CENTROID, or one file is stale.",
        )

    # align catchments: inner join, report drops from each side
    aorc_cats = _load_str_var(nc_aorc, "divide_id")
    mrms_cats = _load_str_var(nc_mrms, "divide_id")
    aorc_set = set(aorc_cats)
    mrms_set = set(mrms_cats)
    common = sorted(aorc_set & mrms_set)
    if not common:
        sys.exit("No common catchments between AORC and MRMS files.")
    only_aorc = sorted(aorc_set - mrms_set)
    only_mrms = sorted(mrms_set - aorc_set)
    if only_aorc:
        print(
            f"  {len(only_aorc)} catchments in AORC but not MRMS (dropped): "
            f"{only_aorc[:5]}{'...' if len(only_aorc) > 5 else ''}",
        )
    if only_mrms:
        print(
            f"  {len(only_mrms)} catchments in MRMS but not AORC (dropped): "
            f"{only_mrms[:5]}{'...' if len(only_mrms) > 5 else ''}",
        )
    aorc_cat_idx = {c: i for i, c in enumerate(aorc_cats)}
    mrms_cat_idx = {c: i for i, c in enumerate(mrms_cats)}
    aorc_ci = np.array([aorc_cat_idx[c] for c in common])
    mrms_ci = np.array([mrms_cat_idx[c] for c in common])
    n_catchments = len(common)
    print(
        f"Catchments: {n_catchments} common "
        f"(AORC: {len(aorc_cats)}, MRMS: {len(mrms_cats)})",
    )

    # figure out max_steps (both sources are nominally 480, take the max)
    aorc_max = nc_aorc.dimensions["time_step"].size
    mrms_max = nc_mrms.dimensions["time_step"].size
    max_steps = max(aorc_max, mrms_max)
    print(f"time_step dim: AORC {aorc_max}, MRMS {mrms_max} -> output {max_steps}")

    # read n_steps and event_start from AORC; n_steps from MRMS for crosscheck
    aorc_n_steps = np.array(nc_aorc.variables["n_steps"][:], dtype=np.int32)
    mrms_n_steps = np.array(nc_mrms.variables["n_steps"][:], dtype=np.int32)
    aorc_ts_starts = aorc_ts_start_chk
    aorc_ts_ends = np.array(nc_aorc.variables["ts_end"][:], dtype=np.float64)
    aorc_lats = np.array(nc_aorc.variables["latitude"][:], dtype=np.float32)
    aorc_lons = np.array(nc_aorc.variables["longitude"][:], dtype=np.float32)
    aorc_gage_ids = _load_str_var(nc_aorc, "event_gage_id")
    aorc_divide_ids = _load_str_var(nc_aorc, "event_divide_id")

    out_n_steps = np.minimum(
        aorc_n_steps[aorc_indices],
        mrms_n_steps[mrms_indices],
    )

    # create output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    nc_out = netCDF4.Dataset(args.output, "w", format="NETCDF4")
    nc_out.createDimension("event", n_events)
    nc_out.createDimension("time_step", max_steps)
    nc_out.createDimension("catchment", n_catchments)

    v = nc_out.createVariable("event_id", str, ("event",))
    v.long_name = "shared AORC/MRMS event ID"
    v[:] = np.array(event_ids, dtype=object)

    v = nc_out.createVariable("n_steps", "i4", ("event",))
    v.long_name = "valid 15-min timesteps (min of AORC and MRMS coverage)"
    v[:] = out_n_steps

    v = nc_out.createVariable("ts_start", "f8", ("event",))
    v.units = "minutes since 1970-01-01 00:00:00 UTC"
    v.long_name = "start of the 5-day timeseries window (from AORC, time step 0)"
    v[:] = aorc_ts_starts[aorc_indices]

    v = nc_out.createVariable("ts_end", "f8", ("event",))
    v.units = "minutes since 1970-01-01 00:00:00 UTC"
    v.long_name = "end of the 5-day timeseries window (from AORC, time step n_steps-1)"
    v[:] = aorc_ts_ends[aorc_indices]

    v = nc_out.createVariable("event_gage_id", str, ("event",))
    v.long_name = "zero-padded 8-digit USGS gauge ID downstream of this event"
    v[:] = aorc_gage_ids[aorc_indices]

    v = nc_out.createVariable("event_divide_id", str, ("event",))
    v.long_name = "NextGen catchment ID downstream of this event"
    v[:] = aorc_divide_ids[aorc_indices]

    v = nc_out.createVariable("divide_id", str, ("catchment",))
    v.long_name = "NextGen catchment ID"
    v[:] = np.array(common, dtype=object)

    v = nc_out.createVariable("latitude", "f4", ("catchment",))
    v.units = "degrees_north"
    v.standard_name = "latitude"
    v[:] = aorc_lats[aorc_ci]

    v = nc_out.createVariable("longitude", "f4", ("catchment",))
    v.units = "degrees_east"
    v.standard_name = "longitude"
    v[:] = aorc_lons[aorc_ci]

    chunk_e = min(n_events, 16)
    chunk_c = min(n_catchments, 64)

    def _make_var(name: str, units: str, long_name: str) -> netCDF4.Variable:
        """Create a compressed (event, time_step, catchment) float32 variable, pre-filled with NaN."""
        nv = nc_out.createVariable(
            name,
            "f4",
            ("event", "time_step", "catchment"),
            fill_value=np.nan,
            zlib=True,
            complevel=args.complevel,
            chunksizes=(chunk_e, max_steps, chunk_c),
        )
        nv.units = units
        nv.long_name = long_name
        nv.coordinates = "event_id n_steps ts_start ts_end event_gage_id event_divide_id divide_id latitude longitude"
        return nv

    # Read every event into plain in-memory arrays, then write each variable
    # to disk in one bulk call below -- a per-event write straight into a
    # compressed netCDF chunk forces a decompress/recompress of that chunk
    # on every call, which dominates runtime once there are thousands of
    # events (reads from the source files don't have this problem, only
    # writes into the new compressed output do).
    P = np.full((n_events, max_steps, n_catchments), np.nan, dtype=np.float32)
    T = np.full((n_events, max_steps, n_catchments), np.nan, dtype=np.float32)
    PET = np.full((n_events, max_steps, n_catchments), np.nan, dtype=np.float32)

    print(f"Reading {n_events} events ...")
    for out_i, (ai, mi) in tqdm(
        enumerate(zip(aorc_indices, mrms_indices)),
        total=n_events,
        desc="Reading events",
    ):
        ns = int(out_n_steps[out_i])
        P[out_i, :ns, :] = nc_mrms.variables["P"][mi, :ns, :][:, mrms_ci]
        T[out_i, :ns, :] = nc_aorc.variables["T"][ai, :ns, :][:, aorc_ci]
        PET[out_i, :ns, :] = nc_aorc.variables["PET"][ai, :ns, :][:, aorc_ci]

    nc_aorc.close()
    nc_mrms.close()

    print("Writing output ...")
    _make_var("P", "mm [15 min]-1", "MRMS precipitation depth")[:] = P
    _make_var("T", "degC", "Air temperature at 2 m (interpolated)")[:] = T
    _make_var("PET", "mm [15 min]-1", "Penman-Monteith ET0 (15-min, uniform split)")[
        :
    ] = PET

    nc_out.close()
    print(f"Done -> {args.output}")
    print(
        f"  Shape: ({n_events} events, {max_steps} time_steps, {n_catchments} catchments)",
    )


if __name__ == "__main__":
    main()
