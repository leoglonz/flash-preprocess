import argparse
import shutil
from pathlib import Path

import pandas as pd

from flash_preprocess.mrms import (
    load_hydrofabric,
    build_crosswalk,
    build_manifest,
    build_fractional_crosswalk,
    build_store,
    extract_all,
    merge_parts,
)


# CONFIG ------------------------------------------
# Every value below is a default, overridable per-invocation via CLI flags
# (see build_args()/parse_args()) -- this is what makes running many parallel
# sharded instances possible without hand-editing this file per process; see
# split_events_shards.py + run_sharded.sh.
EVENTS_CSV = Path("/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/events.csv")
EVENT_IDS = None  # None -> all events in EVENTS_CSV; else e.g. [1266, 4703]

# VPUs to process in this runtime. None -> every VPU present in EVENTS_CSV, all
# in this one run, auto-merged into OUT_NC at the end. A list, e.g. ["03N"],
# restricts this run to just those VPUs and writes a part-file per VPU under
# CACHE_DIR/vpu_runs/<vpu><TAG_SUFFIX>/mrms_15min_part.nc -- run separate
# invocations with disjoint VPU_SUBSETs (e.g. on different machines) to split
# the download, then combine every part with merge.py once they're all done.
VPU_SUBSET = None

# Appended to every per-VPU cache/output path (manifest, event_catchment_windows,
# vpu_runs/<vpu><TAG_SUFFIX>/...) so multiple concurrent instances that both
# touch the same VPU -- e.g. sharded runs, see run_sharded.sh -- never collide
# on the same files. "" (default) reproduces the original single-instance
# per-VPU naming. Non-empty also disables the end-of-run auto-merge (a shard
# only ever has a fraction of any given VPU's events, so merging just its own
# parts would silently produce an incomplete result mislabeled as final) --
# merge all shards' parts together explicitly once every instance is done.
TAG_SUFFIX = ""

CACHE_DIR = Path("/projects/mhpi/leoglonz/sub_hourly/data/_mrms_preprocess/neuse")
OUT_NC = Path("/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/mrms_15min.nc")

BBOX_MARGIN_DEG = 0.1
MAX_WORKERS = 100

# Total width of each event's forcing window, in days (e.g. 5 or 6) -- the
# window is centered on CENTROID and extends WINDOW_DAYS/2 on each side.
WINDOW_DAYS = 6.0

# 'midpoint' -- window centered on the mean of BEGIN_DATE_TIME/END_DATE_TIME.
# 'peak' -- window centered on the event's peak_time column instead.
CENTROID = "peak"

# True -> ignore cached per-VPU windows and downloaded MRMS timesteps, and
# rebuild both from scratch for every VPU in this run. Does NOT touch the
# hydrofabric or crosswalk caches (static geometry, independent of windowing
# or event selection, expensive to rebuild). Needed after any change to
# WINDOW_DAYS/CENTROID -- build_manifest() only caches by file existence, so
# without this a rerun would silently keep reusing the old window boundaries.
FRESH_START = False
# ---------------------------------------------------- #


def parse_args():
    """Parse command-line overrides for the CONFIG block above."""
    p = argparse.ArgumentParser(description="MRMS download + extraction pipeline")
    p.add_argument("--events-csv", type=Path, default=EVENTS_CSV)
    p.add_argument(
        "--vpu-subset",
        default=None,
        help="Comma-separated VPU codes, e.g. '03N,02'. Unset -> every VPU "
        "present in --events-csv (the VPU_SUBSET default).",
    )
    p.add_argument(
        "--tag-suffix",
        default=TAG_SUFFIX,
        help="Appended to per-VPU cache/output paths; non-empty disables "
        "auto-merge at the end of this run (see TAG_SUFFIX above).",
    )
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--out-nc", type=Path, default=OUT_NC)
    p.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    p.add_argument("--window-days", type=float, default=WINDOW_DAYS)
    p.add_argument("--centroid", choices=["midpoint", "peak"], default=CENTROID)
    p.add_argument("--fresh-start", action="store_true", default=FRESH_START)
    return p.parse_args()


def main():
    """Run the MRMS download and extraction pipeline."""
    args = parse_args()
    events_csv = args.events_csv
    vpu_subset = args.vpu_subset.split(",") if args.vpu_subset else None
    tag_suffix = args.tag_suffix
    cache_dir = args.cache_dir
    out_nc = args.out_nc
    max_workers = args.max_workers
    window_days = args.window_days
    centroid = args.centroid
    fresh_start = args.fresh_start

    catchments_master, network, flowpaths, nexus = load_hydrofabric(cache_dir)
    print(f"hydrofabric: {len(catchments_master):,} catchments")

    crosswalk = build_crosswalk(catchments_master, cache_dir)
    print(f"crosswalk: {len(crosswalk):,} MRMS cells")

    events = pd.read_csv(events_csv, dtype={"STAID": str})
    if EVENT_IDS is not None:
        events = events[events["event_id"].isin(EVENT_IDS)]

    cat_vpu = catchments_master.set_index("divide_id")["vpuid"]
    events = events.assign(vpuid=events["gage_cat-id"].map(cat_vpu))
    vpus = sorted(events["vpuid"].dropna().unique())
    if vpu_subset is not None:
        vpus = [v for v in vpus if v in vpu_subset]
    print(
        f"events: {len(events):,} across {len(vpus)} VPU(s): {vpus}"
        + (f"  [tag suffix: {tag_suffix!r}]" if tag_suffix else ""),
    )
    print(f"window: {window_days} day(s) centered on '{centroid}'")

    # 15-min steps in a window_days-wide window, + a small buffer for the
    # outward 15-min-grid rounding in build_manifest possibly adding a step.
    max_steps = int(round(window_days * 24 * 60 / 15)) + 1

    part_ncs = []
    for vpu in vpus:
        vpu_tag = f"{vpu}{tag_suffix}"
        print(f"\n=== VPU {vpu}  (tag: {vpu_tag}) ===")
        vpu_events = events[events["vpuid"] == vpu]
        vpu_dir = cache_dir / "vpu_runs" / vpu_tag

        if fresh_start:
            f_manifest = cache_dir / f"manifest_out_{vpu_tag}.parquet"
            f_windows = cache_dir / f"event_catchment_windows_{vpu_tag}.parquet"
            for f in (f_manifest, f_windows):
                f.unlink(missing_ok=True)
            if vpu_dir.exists():
                shutil.rmtree(vpu_dir)
            print(f"  FRESH_START: cleared manifest cache and {vpu_dir}")

        manifest, event_catchment_windows = build_manifest(
            vpu_events,
            cache_dir,
            tag=vpu_tag,
            window_days=window_days,
            centroid=centroid,
        )
        print(f"  manifest: {len(manifest):,} events resolved to upstream catchments")

        divide_ids = event_catchment_windows["divide_id"].unique()
        cm4326 = catchments_master[
            catchments_master["divide_id"].isin(divide_ids)
        ].to_crs(4326)
        minx, miny, maxx, maxy = cm4326.total_bounds
        bbox = (
            minx - BBOX_MARGIN_DEG,
            miny - BBOX_MARGIN_DEG,
            maxx + BBOX_MARGIN_DEG,
            maxy + BBOX_MARGIN_DEG,
        )

        manifest = manifest.assign(
            grid_start=manifest["win_start"].dt.floor("2min"),
            grid_end=manifest["win_end"].dt.ceil("2min"),
        )
        times = pd.DatetimeIndex(
            sorted(
                set().union(
                    *[
                        set(pd.date_range(r.grid_start, r.grid_end, freq="2min"))
                        for r in manifest.itertuples()
                    ],
                ),
            ),
        )
        print(
            f"  bbox: {bbox}  |  {len(divide_ids):,} catchments  |  {len(times):,} timestamps",
        )

        build_store(
            times,
            bbox,
            str(vpu_dir),
            max_workers=max_workers,
            mask_negative=True,
            use_aws=True,
            verbose=True,
        )

        frac_cw = build_fractional_crosswalk(
            divide_ids,
            catchments_master,
            crosswalk,
            cache_dir,
        )
        print(f"  fractional crosswalk: {len(frac_cw):,} cell/catchment pairs")

        part_nc = vpu_dir / "mrms_15min_part.nc"
        extract_all(
            manifest,
            event_catchment_windows,
            frac_cw,
            vpu_dir / "shards",
            part_nc,
            max_steps=max_steps,
        )
        part_ncs.append(part_nc)

    if vpu_subset is None and not tag_suffix:
        merge_parts(part_ncs, out_nc)
    else:
        print(f"\nVPU subset {vpu_subset} / tag {tag_suffix!r} done -> {part_ncs}")
        print(
            "Run other shards/VPU subsets separately, then merge.py all part files together.",
        )


if __name__ == "__main__":
    main()
