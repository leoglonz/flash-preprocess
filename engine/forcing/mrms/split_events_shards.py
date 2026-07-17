r"""Split an events CSV into N roughly-equal shards for parallel MRMS
downloading via run_sharded.sh / run_pipeline.py --tag-suffix.

Splits by row-index modulo N, NOT by contiguous chunks -- events.csv is
naturally grouped by HUC8 (and HUC8s map very unevenly onto VPUs, e.g. in
huc8_top15/events.csv VPU 02 alone holds 65% of all events), so a contiguous
split would just reproduce that imbalance inside each chunk. Round-robin
assignment spreads every VPU's events evenly across every shard instead,
giving each shard roughly equal total work regardless of VPU boundaries.

Run with:
    python engine/forcing/mrms/split_events_shards.py \
        --events-csv /path/to/events.csv --n-shards 10 --out-dir /path/to/
"""

import argparse
from pathlib import Path

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

    stem = args.events_csv.stem
    for i in range(args.n_shards):
        shard = events.iloc[i::args.n_shards]
        out_path = out_dir / f"{stem}_shard{i}.csv"
        shard.to_csv(out_path, index=False)
        print(f"  shard {i}: {len(shard):,} events -> {out_path}")


if __name__ == "__main__":
    main()
