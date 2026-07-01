"""Merge AORC and MRMS 15-min NetCDFs into a single combined forcing file.

The two input files use different event identifiers (episode_id vs storm_id)
and may cover different catchment sets. This script:
  1. Joins events via a manifest CSV that links the two ID spaces.
  2. Aligns catchments by ID (inner join on divide_id / catchment strings).
  3. Writes a combined (event, time_step, catchment) NetCDF with variables
     from both sources.

Output coordinates
------------------
  episode_id  (event,)   str     AORC/NWS episode ID
  storm_id    (event,)   int32   MRMS internal storm index
  n_steps     (event,)   int32   valid 15-min steps (minimum of both sources)
  event_start (event,)   float64 step-0 time from AORC (minutes since 1970-01-01)
  catchment   (catchment,) str   NextGen divide ID (intersection of both files)
  latitude    (catchment,) f32
  longitude   (catchment,) f32

Output variables
----------------
  TMP_2maboveground  (event, time_step, catchment)  f32  from AORC
  PET                (event, time_step, catchment)  f32  from AORC
  depth_mm_15min     (event, time_step, catchment)  f32  from MRMS

Usage
-----
    python engine/forcing/merge_15min.py \\
        --aorc     /path/to/events_15min.nc \\
        --mrms     /path/to/mrms_15min.nc \\
        --manifest /path/to/events.csv \\
        --output   /path/to/forcing_15min.nc

    # Customise the manifest join columns (defaults shown):
    python engine/forcing/merge_15min.py ... \\
        --episode-col episode_id --storm-col storm_index
"""

import argparse
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd


_EPOCH = np.datetime64("1970-01-01T00:00", "m")


def _load_str_var(ds: netCDF4.Dataset, name: str) -> np.ndarray:
    """Read a netCDF4 string variable as a plain numpy object array.

    Parameters
    ----------
    ds
        Open netCDF4 Dataset.
    name
        Variable name to read.

    Returns
    -------
    np.ndarray
        Object array of Python strings, shape (n,).
    """
    raw = ds.variables[name][:]
    if hasattr(raw, "data"):
        raw = raw.data
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
    parser.add_argument("--aorc", required=True,
                        help="AORC 15-min NC (output of to_events.py, events_15min.nc)")
    parser.add_argument("--mrms", required=True,
                        help="MRMS 15-min NC (output of aggregate_events.py)")
    parser.add_argument("--manifest", required=True,
                        help="CSV that links episode_id to storm_index")
    parser.add_argument("--output", required=True,
                        help="Output NetCDF path")
    parser.add_argument("--episode-col", default="episode_id",
                        help="Manifest column with AORC episode IDs (default: episode_id)")
    parser.add_argument("--storm-col", default="storm_index",
                        help="Manifest column with MRMS storm IDs (default: storm_index)")
    parser.add_argument("--complevel", type=int, default=4,
                        help="zlib compression level 1-9 (default: 4)")
    args = parser.parse_args()

    # load manifest
    mdf = pd.read_csv(args.manifest)
    for col in (args.episode_col, args.storm_col):
        if col not in mdf.columns:
            sys.exit(f"Column '{col}' not found in {args.manifest}.\n"
                     f"Available: {mdf.columns.tolist()}")
    mdf = mdf.dropna(subset=[args.episode_col, args.storm_col])
    mdf["_episode"] = mdf[args.episode_col].astype(int).astype(str)
    mdf["_storm"] = mdf[args.storm_col].astype(int)
    ep_to_storm = dict(zip(mdf["_episode"], mdf["_storm"]))
    print(f"Manifest: {len(mdf)} pairs loaded "
          f"({args.episode_col} <-> {args.storm_col})")

    # open source files
    nc_aorc = netCDF4.Dataset(args.aorc, "r")
    nc_mrms = netCDF4.Dataset(args.mrms, "r")

    aorc_episode_ids = _load_str_var(nc_aorc, "event_id")
    mrms_storm_ids = np.array(nc_mrms.variables["storm_id"][:], dtype=np.int32)
    storm_to_mrms_idx = {int(sid): i for i, sid in enumerate(mrms_storm_ids)}

    # build aligned event list: iterate AORC order, keep events present in MRMS
    aorc_indices, mrms_indices, episode_ids, storm_ids = [], [], [], []
    skipped = 0
    for i, eid in enumerate(aorc_episode_ids):
        sid = ep_to_storm.get(eid)
        if sid is None:
            skipped += 1
            continue
        j = storm_to_mrms_idx.get(sid)
        if j is None:
            skipped += 1
            continue
        aorc_indices.append(i)
        mrms_indices.append(j)
        episode_ids.append(eid)
        storm_ids.append(sid)

    n_events = len(aorc_indices)
    if n_events == 0:
        sys.exit("No events matched between AORC and MRMS files via the manifest.")
    print(f"Matched {n_events} events ({skipped} AORC events skipped — "
          f"not in manifest or MRMS file)")

    # align catchments: inner join, report drops from each side
    aorc_cats = _load_str_var(nc_aorc, "catchment")
    mrms_cats = _load_str_var(nc_mrms, "divide_id")
    aorc_set = set(aorc_cats)
    mrms_set = set(mrms_cats)
    common = sorted(aorc_set & mrms_set)
    if not common:
        sys.exit("No common catchments between AORC and MRMS files.")
    only_aorc = sorted(aorc_set - mrms_set)
    only_mrms = sorted(mrms_set - aorc_set)
    if only_aorc:
        print(f"  {len(only_aorc)} catchments in AORC but not MRMS (dropped): "
              f"{only_aorc[:5]}{'...' if len(only_aorc) > 5 else ''}")
    if only_mrms:
        print(f"  {len(only_mrms)} catchments in MRMS but not AORC (dropped): "
              f"{only_mrms[:5]}{'...' if len(only_mrms) > 5 else ''}")
    aorc_cat_idx = {c: i for i, c in enumerate(aorc_cats)}
    mrms_cat_idx = {c: i for i, c in enumerate(mrms_cats)}
    aorc_ci = np.array([aorc_cat_idx[c] for c in common])
    mrms_ci = np.array([mrms_cat_idx[c] for c in common])
    n_catchments = len(common)
    print(f"Catchments: {n_catchments} common "
          f"(AORC: {len(aorc_cats)}, MRMS: {len(mrms_cats)})")

    # figure out max_steps (both sources are nominally 480, take the max)
    aorc_max = nc_aorc.dimensions["time_step"].size
    mrms_max = nc_mrms.dimensions["time_step"].size
    max_steps = max(aorc_max, mrms_max)
    print(f"time_step dim: AORC {aorc_max}, MRMS {mrms_max} -> output {max_steps}")

    # read n_steps and event_start from AORC; n_steps from MRMS for crosscheck
    aorc_n_steps = np.array(nc_aorc.variables["n_steps"][:], dtype=np.int32)
    mrms_n_steps = np.array(nc_mrms.variables["n_steps"][:], dtype=np.int32)
    aorc_starts = np.array(nc_aorc.variables["event_start"][:], dtype=np.float64)
    aorc_lats = np.array(nc_aorc.variables["latitude"][:], dtype=np.float32)
    aorc_lons = np.array(nc_aorc.variables["longitude"][:], dtype=np.float32)

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

    v = nc_out.createVariable("episode_id", str, ("event",))
    v.long_name = "NWS episode ID"
    v[:] = np.array(episode_ids, dtype=object)

    v = nc_out.createVariable("storm_id", "i4", ("event",))
    v.long_name = "MRMS internal storm index"
    v[:] = np.array(storm_ids, dtype=np.int32)

    v = nc_out.createVariable("n_steps", "i4", ("event",))
    v.long_name = "valid 15-min timesteps (min of AORC and MRMS coverage)"
    v[:] = out_n_steps

    v = nc_out.createVariable("event_start", "f8", ("event",))
    v.units = "minutes since 1970-01-01 00:00:00 UTC"
    v.long_name = "time of step 0 for each event (from AORC window)"
    v[:] = aorc_starts[aorc_indices]

    v = nc_out.createVariable("catchment", str, ("catchment",))
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
        """Create a compressed (event, time_step, catchment) float32 variable.

        Parameters
        ----------
        name
            NetCDF variable name.
        units
            CF units string written as a variable attribute.
        long_name
            Human-readable description written as a variable attribute.

        Returns
        -------
        netCDF4.Variable
            Newly created variable, pre-filled with NaN.
        """
        nv = nc_out.createVariable(
            name, "f4", ("event", "time_step", "catchment"),
            fill_value=np.nan, zlib=True, complevel=args.complevel,
            chunksizes=(chunk_e, max_steps, chunk_c),
        )
        nv.units = units
        nv.long_name = long_name
        return nv

    v_tmp = _make_var("TMP_2maboveground", "K", "Air temperature at 2 m (interpolated)")
    v_pet = _make_var("PET", "mm 15min-1", "Penman-Monteith ET0 (15-min, uniform split)")
    v_dep = _make_var("depth_mm_15min", "mm per 15 min", "MRMS precipitation depth")

    print(f"Writing {n_events} events ...")
    for out_i, (ai, mi) in enumerate(zip(aorc_indices, mrms_indices)):
        ns = int(out_n_steps[out_i])

        tmp_row = nc_aorc.variables["TMP_2maboveground"][ai, :ns, :][:, aorc_ci]
        pet_row = nc_aorc.variables["PET"][ai, :ns, :][:, aorc_ci]
        dep_row = nc_mrms.variables["depth_mm_15min"][mi, :ns, :][:, mrms_ci]

        v_tmp[out_i, :ns, :] = tmp_row
        v_pet[out_i, :ns, :] = pet_row
        v_dep[out_i, :ns, :] = dep_row

        if (out_i + 1) % 20 == 0 or out_i == n_events - 1:
            print(f"  {out_i + 1}/{n_events}")

    nc_aorc.close()
    nc_mrms.close()
    nc_out.close()
    print(f"Done -> {args.output}")
    print(f"  Shape: ({n_events} events, {max_steps} time_steps, {n_catchments} catchments)")


if __name__ == "__main__":
    main()