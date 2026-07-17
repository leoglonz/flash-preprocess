"""
Standalone multi-HUC8 MRMS download runner.

Reads a pickle mapping HUC8 name -> (bbox, unique_mrms_times), computed once
per HUC8 in the notebook (same as the single-HUC8 nb3_download_state.pkl, but
one entry per HUC8 instead of one region total). Calls build_multi_store(),
which fetches and decodes each full-CONUS timestamp file ONCE regardless of
how many HUC8s need it, then crops it to each HUC8's own bbox and writes each
HUC8's shards to its own subfolder under OUT_ROOT.

Building the input pickle (run once per HUC8 in the notebook, accumulating
into one dict, then pickle it):

    multi_state = {}
    for huc8_name, (bbox, times) in per_huc8_results.items():
        multi_state[huc8_name] = {"bbox": bbox, "unique_mrms_times": times}
    import pickle
    with open("multi_huc8_download_state.pkl", "wb") as f:
        pickle.dump(multi_state, f)

Usage (from a terminal, NOT a notebook cell):

    MALLOC_ARENA_MAX=2 nohup python -u download_mrms_multi_huc8.py \
        > mrms_download_multi.log 2>&1 &
    disown
"""

import pickle
import sys
import time
from pathlib import Path

from flash_preprocess.engine.forcing.mrms.mrms_bbox_downloader import build_multi_store

CACHE = Path("multi_huc8_download_state.pkl")
OUT_ROOT = Path("storm_precip_MRMS_multi").resolve()
DONE_MARKER = OUT_ROOT / "DOWNLOAD_COMPLETE"
MAX_WORKERS = 12


def main():
    if DONE_MARKER.exists():
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"{DONE_MARKER} already exists -- nothing to do. "
              "Delete it if you want to force a re-run.")
        return

    if not CACHE.exists():
        sys.exit(
            f"\u274c {CACHE.resolve()} not found. Build it in the notebook "
            "first -- one bbox + unique_mrms_times entry per HUC8, pickled "
            "as a dict keyed by HUC8 name."
        )

    with open(CACHE, "rb") as f:
        state = pickle.load(f)

    region_times = {name: v["unique_mrms_times"] for name, v in state.items()}
    region_bboxes = {name: v["bbox"] for name, v in state.items()}

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting multi-HUC8 download")
    print(f"  regions        : {list(region_times.keys())}")
    for name in region_times:
        print(f"    [{name}] {len(region_times[name]):,} timestamps, "
              f"bbox={region_bboxes[name]}")
    print(f"  output root    : {OUT_ROOT}")
    print(f"  max_workers    : {MAX_WORKERS}", flush=True)

    summaries = build_multi_store(
        region_times,
        region_bboxes,
        str(OUT_ROOT),
        max_workers=MAX_WORKERS,
        mask_negative=True,
        use_aws=True,
        verbose=True,
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    DONE_MARKER.write_text(
        f"Completed {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{len(region_times)} HUC8 regions.\n"
    )
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
          f"All regions done -- wrote {DONE_MARKER}")
    print(summaries)


if __name__ == "__main__":
    main()
