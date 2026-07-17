import shutil
from pathlib import Path

import pandas as pd

from flash_preprocess.mrms import load_hydrofabric, build_manifest
from flash_preprocess.aorc import (
    build_weighted_crosswalk, build_shards, extract_all, merge_hr_parts, merge_15min_parts,
)


# CONFIG ------------------------------------------
EVENTS_CSV = Path("/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/events.csv")
EVENT_IDS = None  # None -> all events in EVENTS_CSV; else e.g. [1266, 4703]

# VPUs to process in this runtime. None -> every VPU present in EVENTS_CSV, all
# in this one run, auto-merged into OUT_*_NC at the end. A list, e.g. ["03N"],
# restricts this run to just those VPUs and writes part-files per VPU under
# CACHE_DIR/aorc_runs/<vpu>/ -- run separate invocations with disjoint
# VPU_SUBSETs (e.g. on different machines) to split the S3 fetch, then merge
# every part with merge_hr_parts/merge_15min_parts once they're all done.
VPU_SUBSET = None

# Same CACHE_DIR as engine/forcing/mrms/run_pipeline.py -- both pipelines
# call the identical build_manifest(), so keeping this in sync means the
# second pipeline you run reuses the first one's cached per-VPU windows
# instead of redoing the upstream-catchment BFS.
CACHE_DIR = Path("/projects/mhpi/leoglonz/sub_hourly/data/_mrms_preprocess/neuse")
OUT_HR_NC = Path("/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/aorc_hr.nc")
OUT_15MIN_NC = Path("/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/aorc_15min.nc")

MAX_WORKERS = 16  # parallel workers for exactextract area-weight computation

# Total width of each event's forcing window, in days -- must match the
# WINDOW_DAYS used for the MRMS run this is meant to align with (both feed
# build_manifest(), and merge_15min.py joins the two outputs by event_id).
WINDOW_DAYS = 6.0

# 'midpoint' -- window centered on the mean of BEGIN_DATE_TIME/END_DATE_TIME.
# 'peak' -- window centered on the event's peak_time column instead.
CENTROID = "peak"

# Hourly warmup window preceding each event's WINDOW_DAYS window.
ANTECEDENT_DAYS = 30.0

# True -> ignore cached per-VPU windows/weights/shards and rebuild from
# scratch for every VPU in this run. Does NOT touch the hydrofabric cache.
# Needed after any change to WINDOW_DAYS/CENTROID/ANTECEDENT_DAYS.
FRESH_START = False
# ---------------------------------------------------- #


def main():
    catchments_master, *_ = load_hydrofabric(CACHE_DIR)
    print(f"hydrofabric: {len(catchments_master):,} catchments")

    events = pd.read_csv(EVENTS_CSV, dtype={"STAID": str})
    if EVENT_IDS is not None:
        events = events[events["event_id"].isin(EVENT_IDS)]

    cat_vpu = catchments_master.set_index("divide_id")["vpuid"]
    events = events.assign(vpuid=events["gage_cat-id"].map(cat_vpu))
    vpus = sorted(events["vpuid"].dropna().unique())
    if VPU_SUBSET is not None:
        vpus = [v for v in vpus if v in VPU_SUBSET]
    print(f"events: {len(events):,} across {len(vpus)} VPU(s): {vpus}")
    print(f"window: {WINDOW_DAYS} day(s) centered on '{CENTROID}', {ANTECEDENT_DAYS}d antecedent")

    # 15-min steps in a WINDOW_DAYS-wide window, + a small buffer for the
    # outward hour-grid rounding in build_manifest possibly adding a step.
    max_15min_steps = int(round(WINDOW_DAYS * 24 * 60 / 15)) + 1

    hr_parts, min15_parts = [], []
    for vpu in vpus:
        print(f"\n=== VPU {vpu} ===")
        vpu_events = events[events["vpuid"] == vpu]
        vpu_dir = CACHE_DIR / "aorc_runs" / vpu

        if FRESH_START:
            f_weights = CACHE_DIR / f"aorc_weights_{vpu}.pkl"
            f_weights.unlink(missing_ok=True)
            if vpu_dir.exists():
                shutil.rmtree(vpu_dir)
            print(f"  FRESH_START: cleared weight cache and {vpu_dir}")

        manifest, event_catchment_windows = build_manifest(
            vpu_events, CACHE_DIR, tag=vpu, window_days=WINDOW_DAYS, centroid=CENTROID)
        print(f"  manifest: {len(manifest):,} events resolved to upstream catchments")

        divide_id_of = dict(zip(vpu_events["event_id"].astype(str), vpu_events["gage_cat-id"]))

        divide_ids = event_catchment_windows["divide_id"].unique()
        weight_idx = build_weighted_crosswalk(divide_ids, catchments_master, CACHE_DIR, tag=vpu,
                                               max_workers=MAX_WORKERS)
        print(f"  weighted crosswalk: {len(weight_idx['station_ids']):,} catchments")

        build_shards(manifest, weight_idx, vpu_dir, antecedent_days=ANTECEDENT_DAYS)

        hr_part = vpu_dir / "aorc_hr_part.nc"
        min15_part = vpu_dir / "aorc_15min_part.nc"
        extract_all(manifest, weight_idx, vpu_dir / "shards", hr_part, min15_part, divide_id_of,
                    antecedent_days=ANTECEDENT_DAYS, max_15min_steps=max_15min_steps)
        hr_parts.append(hr_part)
        min15_parts.append(min15_part)

    if VPU_SUBSET is None:
        merge_hr_parts(hr_parts, OUT_HR_NC)
        merge_15min_parts(min15_parts, OUT_15MIN_NC)
    else:
        print(f"\nVPU subset {VPU_SUBSET} done -> {hr_parts + min15_parts}")
        print("Run other VPU subsets separately, then merge remaining parts together.")


if __name__ == "__main__":
    main()
