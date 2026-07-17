from pathlib import Path

import pandas as pd

from flash_preprocess.mrms import load_hydrofabric, build_manifest
from flash_preprocess.aorc import (
    build_weighted_crosswalk, build_shards, extract_all, merge_hr_parts, merge_15min_parts,
)

# CONFIG
EVENTS_CSV = Path("/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/events.csv")
EVENT_IDS = None  # None -> all events in EVENTS_CSV; else e.g. [1266, 4703]

# VPUs to process in this runtime. None -> every VPU present in EVENTS_CSV, all
# in this one run, auto-merged into OUT_*_NC at the end. A list, e.g. ["03N"],
# restricts this run to just those VPUs and writes part-files per VPU under
# CACHE_DIR/aorc_runs/<vpu>/ -- run separate invocations with disjoint
# VPU_SUBSETs (e.g. on different machines) to split the S3 fetch, then merge
# every part with merge_hr_parts/merge_15min_parts once they're all done.
VPU_SUBSET = None

CACHE_DIR = Path("/projects/mhpi/leoglonz/sub_hourly/data/_mrms_preprocess")
OUT_HR_NC = Path("/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/aorc_hr.nc")
OUT_15MIN_NC = Path("/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/aorc_15min.nc")


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

    hr_parts, min15_parts = [], []
    for vpu in vpus:
        print(f"\n=== VPU {vpu} ===")
        vpu_events = events[events["vpuid"] == vpu]
        # same manifest/window logic MRMS uses (5-day window, upstream catchments)
        # -- this file adds the 30-day antecedent lookback on top of it.
        manifest, event_catchment_windows = build_manifest(vpu_events, CACHE_DIR, tag=vpu)
        print(f"  manifest: {len(manifest):,} events resolved to upstream catchments")

        divide_id_of = dict(zip(vpu_events["event_id"].astype(str), vpu_events["gage_cat-id"]))

        divide_ids = event_catchment_windows["divide_id"].unique()
        weight_idx = build_weighted_crosswalk(divide_ids, catchments_master, CACHE_DIR, tag=vpu)
        print(f"  weighted crosswalk: {len(weight_idx['station_ids']):,} catchments")

        vpu_dir = CACHE_DIR / "aorc_runs" / vpu
        build_shards(manifest, weight_idx, vpu_dir)

        hr_part = vpu_dir / "aorc_hr_part.nc"
        min15_part = vpu_dir / "aorc_15min_part.nc"
        extract_all(manifest, weight_idx, vpu_dir / "shards", hr_part, min15_part, divide_id_of)
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
