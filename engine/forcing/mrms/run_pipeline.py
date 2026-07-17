import shutil
from pathlib import Path

import pandas as pd

from flash_preprocess.mrms import (
    load_hydrofabric, build_crosswalk, build_manifest, build_fractional_crosswalk,
    build_store, extract_all, merge_parts,
)


# CONFIG ------------------------------------------
EVENTS_CSV = Path("/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/events.csv")
EVENT_IDS = None  # None -> all events in EVENTS_CSV; else e.g. [1266, 4703]

# VPUs to process in this runtime. None -> every VPU present in EVENTS_CSV, all
# in this one run, auto-merged into OUT_NC at the end. A list, e.g. ["03N"],
# restricts this run to just those VPUs and writes a part-file per VPU under
# CACHE_DIR/vpu_runs/<vpu>/mrms_15min_part.nc -- run separate invocations with
# disjoint VPU_SUBSETs (e.g. on different machines) to split the download, then
# combine every part with merge.py once they're all done.
VPU_SUBSET = None

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


def main():
    catchments_master, network, flowpaths, nexus = load_hydrofabric(CACHE_DIR)
    print(f"hydrofabric: {len(catchments_master):,} catchments")

    crosswalk = build_crosswalk(catchments_master, CACHE_DIR)
    print(f"crosswalk: {len(crosswalk):,} MRMS cells")

    events = pd.read_csv(EVENTS_CSV, dtype={"STAID": str})
    if EVENT_IDS is not None:
        events = events[events["event_id"].isin(EVENT_IDS)]

    cat_vpu = catchments_master.set_index("divide_id")["vpuid"]
    events = events.assign(vpuid=events["gage_cat-id"].map(cat_vpu))
    vpus = sorted(events["vpuid"].dropna().unique())
    if VPU_SUBSET is not None:
        vpus = [v for v in vpus if v in VPU_SUBSET]
    print(f"events: {len(events):,} across {len(vpus)} VPU(s): {vpus}")
    print(f"window: {WINDOW_DAYS} day(s) centered on '{CENTROID}'")

    # 15-min steps in a WINDOW_DAYS-wide window, + a small buffer for the
    # outward 15-min-grid rounding in build_manifest possibly adding a step.
    max_steps = int(round(WINDOW_DAYS * 24 * 60 / 15)) + 1

    part_ncs = []
    for vpu in vpus:
        print(f"\n=== VPU {vpu} ===")
        vpu_events = events[events["vpuid"] == vpu]
        vpu_dir = CACHE_DIR / "vpu_runs" / vpu

        if FRESH_START:
            f_manifest = CACHE_DIR / f"manifest_out_{vpu}.parquet"
            f_windows = CACHE_DIR / f"event_catchment_windows_{vpu}.parquet"
            for f in (f_manifest, f_windows):
                f.unlink(missing_ok=True)
            if vpu_dir.exists():
                shutil.rmtree(vpu_dir)
            print(f"  FRESH_START: cleared manifest cache and {vpu_dir}")

        manifest, event_catchment_windows = build_manifest(
            vpu_events, CACHE_DIR, tag=vpu, window_days=WINDOW_DAYS, centroid=CENTROID)
        print(f"  manifest: {len(manifest):,} events resolved to upstream catchments")

        divide_ids = event_catchment_windows["divide_id"].unique()
        cm4326 = catchments_master[catchments_master["divide_id"].isin(divide_ids)].to_crs(4326)
        minx, miny, maxx, maxy = cm4326.total_bounds
        bbox = (minx - BBOX_MARGIN_DEG, miny - BBOX_MARGIN_DEG,
                maxx + BBOX_MARGIN_DEG, maxy + BBOX_MARGIN_DEG)

        manifest = manifest.assign(grid_start=manifest["win_start"].dt.floor("2min"),
                                    grid_end=manifest["win_end"].dt.ceil("2min"))
        times = pd.DatetimeIndex(sorted(set().union(*[
            set(pd.date_range(r.grid_start, r.grid_end, freq="2min")) for r in manifest.itertuples()
        ])))
        print(f"  bbox: {bbox}  |  {len(divide_ids):,} catchments  |  {len(times):,} timestamps")

        build_store(times, bbox, str(vpu_dir), max_workers=MAX_WORKERS, mask_negative=True,
                    use_aws=True, verbose=True)

        frac_cw = build_fractional_crosswalk(divide_ids, catchments_master, crosswalk, CACHE_DIR)
        print(f"  fractional crosswalk: {len(frac_cw):,} cell/catchment pairs")

        part_nc = vpu_dir / "mrms_15min_part.nc"
        extract_all(manifest, event_catchment_windows, frac_cw, vpu_dir / "shards", part_nc,
                     max_steps=max_steps)
        part_ncs.append(part_nc)

    if VPU_SUBSET is None:
        merge_parts(part_ncs, OUT_NC)
    else:
        print(f"\nVPU subset {VPU_SUBSET} done -> {part_ncs}")
        print("Run other VPU subsets separately, then merge.py all part files together.")


if __name__ == "__main__":
    main()
