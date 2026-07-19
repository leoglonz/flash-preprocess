"""Extract hourly AORC (Analysis Of Record for Calibration) forcing for a set
of flash flood events.

Outputs:
    - 15-min resolution forcing NetCDF for all events (window centered on
      each event's centroid or peak flow time)
    - 1-hr resolution forcing NetCDF for all events (antecedent days
      preceding the 15-min window)

Edit the CONFIG block at the top of this file to set all options, or
override per-invocation via CLI flags (see below).
"""

import shutil

import pandas as pd

from flash_preprocess.mrms import load_hydrofabric, build_manifest
from flash_preprocess.aorc import (
    build_weighted_crosswalk,
    build_shards,
    extract_all,
    merge_hr_parts,
    merge_15min_parts,
)
from flash_preprocess.paths import CACHE_DIR as _CACHE_DIR
from flash_preprocess.paths import EVENTS_CSV as _EVENTS_CSV


# CONFIG -------------------------- #
# Flash flood event registry.
#   None = all events in EVENTS_CSV; else e.g. [1266, 4703]
EVENT_PATH = _EVENTS_CSV
EVENT_IDS = None

# VPUs to process in this runtime.
#   None = every VPU in EVENTS_CSV; else e.g. [ "01", "03N"],
#   Merge with forcing/mrms/merge.py
VPU_SUBSET = None

# Where to cache per-VPU AORC windows, weights, and NetCDF shards.
CACHE_DIR = _CACHE_DIR

# Output NetCDF paths for 1) merged hourly and 2) 15-min AORC forcing.
OUT_HR_NC = _CACHE_DIR / 'aorc_hr.nc'
OUT_15MIN_NC = _CACHE_DIR / 'aorc_15min.nc'

# More workers == faster. Make sure you have enough CPUs (=workers) and RAM.
MAX_WORKERS = 16

# Total width of each event's forcing window (days), centered on event.
#    Must match WINDOW_DAYS used for MRMS run.
WINDOW_DAYS = 6.0

# Event window centroid method.
#   'midpoint' -- center between begin and end times.
#   'peak' (Recommended) -- window centered on the event's reported peak time.
CENTROID = 'peak'

# Hourly warmup window (days) preceding each event's WINDOW_DAYS window.
ANTECEDENT_DAYS = 30.0

# Caching
#   True -- ignore cached per-VPU windows/weights/shards and rebuild.
#   Needed after any change to WINDOW_DAYS/CENTROID/ANTECEDENT_DAYS.
FRESH_START = False
# -------------------------- #


def main():
    """Run the AORC forcing extraction pipeline."""
    catchments_master, *_ = load_hydrofabric(CACHE_DIR)
    print(f"hydrofabric: {len(catchments_master):,} catchments")

    events = pd.read_csv(EVENT_PATH, dtype={'STAID': str})
    if EVENT_IDS is not None:
        events = events[events['event_id'].isin(EVENT_IDS)]

    cat_vpu = catchments_master.set_index('divide_id')['vpuid']
    events = events.assign(vpuid=events['gage_cat-id'].map(cat_vpu))
    vpus = sorted(events['vpuid'].dropna().unique())
    if VPU_SUBSET is not None:
        vpus = [v for v in vpus if v in VPU_SUBSET]
    print(f"events: {len(events):,} across {len(vpus)} VPU(s): {vpus}")
    print(
        f"window: {WINDOW_DAYS} day(s) centered on '{CENTROID}', {ANTECEDENT_DAYS}d antecedent",
    )

    # 15-min steps in a WINDOW_DAYS-wide window, + a small buffer for the
    # outward hour-grid rounding in build_manifest possibly adding a step.
    max_15min_steps = int(round(WINDOW_DAYS * 24 * 60 / 15)) + 1

    hr_parts, min15_parts = [], []
    for vpu in vpus:
        print(f"\n=== VPU {vpu} ===")
        vpu_events = events[events['vpuid'] == vpu]
        vpu_dir = CACHE_DIR / 'aorc_runs' / vpu

        if FRESH_START:
            f_weights = CACHE_DIR / f'aorc_weights_{vpu}.pkl'
            f_weights.unlink(missing_ok=True)
            if vpu_dir.exists():
                shutil.rmtree(vpu_dir)
            print(f"  FRESH_START: cleared weight cache and {vpu_dir}")

        manifest, event_catchment_windows = build_manifest(
            vpu_events,
            CACHE_DIR,
            tag=vpu,
            window_days=WINDOW_DAYS,
            centroid=CENTROID,
        )
        print(f"  manifest: {len(manifest):,} events resolved to upstream catchments")

        divide_id_of = dict(
            zip(vpu_events['event_id'].astype(str), vpu_events['gage_cat-id']),
        )

        divide_ids = event_catchment_windows['divide_id'].unique()
        weight_idx = build_weighted_crosswalk(
            divide_ids,
            catchments_master,
            CACHE_DIR,
            tag=vpu,
            max_workers=MAX_WORKERS,
        )
        print(f"  weighted crosswalk: {len(weight_idx['station_ids']):,} catchments")

        build_shards(manifest, weight_idx, vpu_dir, antecedent_days=ANTECEDENT_DAYS)

        hr_part = vpu_dir / 'aorc_hr_part.nc'
        min15_part = vpu_dir / 'aorc_15min_part.nc'
        extract_all(
            manifest,
            weight_idx,
            vpu_dir / 'shards',
            hr_part,
            min15_part,
            divide_id_of,
            antecedent_days=ANTECEDENT_DAYS,
            max_15min_steps=max_15min_steps,
        )
        hr_parts.append(hr_part)
        min15_parts.append(min15_part)

    if VPU_SUBSET is None:
        merge_hr_parts(hr_parts, OUT_HR_NC)
        merge_15min_parts(min15_parts, OUT_15MIN_NC)
    else:
        print(f"\nVPU subset {VPU_SUBSET} done -> {hr_parts + min15_parts}")
        print("Run other VPU subsets separately, then merge remaining parts together.")


if __name__ == '__main__':
    main()
