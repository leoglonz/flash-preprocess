"""Split an events CSV into N ~equal shards for parallel MRMS downloading.

Run:
    python engine/forcing/mrms/split_events_shards.py \
        --events-csv /path/to/events.csv --n-shards 10 --out-dir /path/to/
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger('MRMS-Shard')


def main() -> None:
    """Split an events CSV into N time-contiguous shards."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--events-csv', type=Path, required=True)
    ap.add_argument('--n-shards', type=int, required=True)
    ap.add_argument(
        '--out-dir',
        type=Path,
        default=None,
        help="Defaults to --events-csv's own directory.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or args.events_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events_csv, dtype={'STAID': str})
    log.info('%d events -> %d shard(s)', len(events), args.n_shards)

    sort_key = pd.to_datetime(events['BEGIN_DATE_TIME'], utc=True)
    events = (
        events.assign(_sort_key=sort_key)
        .sort_values('_sort_key')
        .reset_index(drop=True)
    )
    events['shard'] = np.arange(len(events)) * args.n_shards // len(events)

    stem = args.events_csv.stem
    for i in range(args.n_shards):
        shard = events[events['shard'] == i].drop(columns=['_sort_key', 'shard'])
        out_path = out_dir / f'{stem}_shard{i}.csv'
        shard.to_csv(out_path, index=False)
        span = f"{shard['BEGIN_DATE_TIME'].min()} -> {shard['BEGIN_DATE_TIME'].max()}"
        log.info('shard %d: %d events (%s) -> %s', i, len(shard), span, out_path)


if __name__ == '__main__':
    main()
