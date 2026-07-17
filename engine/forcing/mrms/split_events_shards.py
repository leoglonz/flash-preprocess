r"""Split an events CSV into N roughly-equal shards for parallel MRMS
downloading via run_sharded.sh / run_pipeline.py --tag-suffix.

Splits by TIME, sorted then cut into N contiguous chunks -- NOT round-robin.
An earlier round-robin version (assign by row-index modulo N) was measured to
inflate total download volume by ~565% on huc8_top15/events.csv: each VPU's
events span nearly the entire ~5-year catalog, so round-robin spreads that
same near-full date range across every shard, meaning every shard ends up
needing almost the same ~1800 days independently, summing to ~6.6x the work
of a single deduplicated run. Sorting by time first means events close in
time land in the same shard, so each shard's date range is mostly disjoint
from the others' -- measured overhead with this approach: ~2.3%, i.e.
essentially the redundant-download problem goes away. Shard balance (event
count) is unaffected: with several VPUs' events roughly independently spread
across the same time range, any contiguous time slice still gets a
proportional mix of all of them, same as round-robin gave.

Run with:
    python engine/forcing/mrms/split_events_shards.py \
        --events-csv /path/to/events.csv --n-shards 10 --out-dir /path/to/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events-csv", type=Path, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="Defaults to --events-csv's own directory.")
    args = ap.parse_args()

    out_dir = args.out_dir or args.events_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events_csv, dtype={"STAID": str})
    print(f"{len(events):,} events -> {args.n_shards} shard(s)")

    sort_key = pd.to_datetime(events["BEGIN_DATE_TIME"], utc=True)
    events = events.assign(_sort_key=sort_key).sort_values("_sort_key").reset_index(drop=True)
    events["shard"] = np.arange(len(events)) * args.n_shards // len(events)

    stem = args.events_csv.stem
    for i in range(args.n_shards):
        shard = events[events["shard"] == i].drop(columns=["_sort_key", "shard"])
        out_path = out_dir / f"{stem}_shard{i}.csv"
        shard.to_csv(out_path, index=False)
        span = f"{shard['BEGIN_DATE_TIME'].min()} -> {shard['BEGIN_DATE_TIME'].max()}"
        print(f"  shard {i}: {len(shard):,} events ({span}) -> {out_path}")


if __name__ == "__main__":
    main()
