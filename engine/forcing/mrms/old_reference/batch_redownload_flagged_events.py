r"""Batch re-download raw MRMS PrecipRate for every event flagged as
near-zero-precip in precip_gap_events.csv (see
dmg-flash/example/flash_flood/check_precip_gaps.py), to check/repair the
existing mrms_15min.nc extraction.

Efficiency: builds ONE combined bbox (union of all flagged gauges' catchments)
and ONE deduplicated list of 2-min-grid-aligned timestamps (union across all
flagged events' windows), then makes a single build_store() call. Every
unique (day, 2-min-timestamp) file is downloaded + decoded exactly once,
regardless of how many events/gauges share it -- this is what gets the
~64% dedup savings over downloading per-event.

Resumable: build_store() skips files already present in the day shards and
remembers confirmed-missing timestamps (known_missing.csv), so a killed/
restarted run picks up where it left off.

Edit the CONFIG block at the top of this file to set all options. Run with:
    python engine/forcing/mrms/batch_redownload_flagged_events.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from mrms_bbox_downloader import build_store


# CONFIG
FLAGGED_CSV = Path('/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/precip_gap_events.csv')
MRMS_15MIN_NC = Path('/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/mrms_15min.nc')
CATCHMENTS_MASTER = Path('/projects/mhpi/leoglonz/sub_hourly/data/_mrms_preprocess/catchments_master.parquet')
OUT_DIR = Path('/projects/mhpi/leoglonz/sub_hourly/data/_mrms_preprocess/flagged_events_redownload')

BBOX_MARGIN_DEG = 0.1
MAX_WORKERS = 40


def main() -> None:
    df = pd.read_csv(FLAGGED_CSV, dtype={'event_id': str, 'gauge_id': str},
                      parse_dates=['ts_start', 'ts_end'])
    print(f'{len(df)} flagged events across {df["gauge_id"].nunique()} gauges')

    ds_mrms = xr.open_dataset(MRMS_15MIN_NC)
    storm_ids = ds_mrms['storm_id'].values.astype(str)
    storm_idx = {s: i for i, s in enumerate(storm_ids)}
    divide_ids_all = ds_mrms['divide_id'].values

    # per-gauge union of divide_ids across ALL that gauge's events in the
    # existing file (not just flagged ones) -- maximizes true catchment
    # coverage even where an individual flagged event's own extraction was
    # itself corrupted/incomplete.
    gauge_to_divides: dict[str, set[str]] = {}
    for eid, gid in zip(df['event_id'], df['gauge_id']):
        gid = gid.zfill(8)
        i = storm_idx.get(eid)
        if i is None:
            continue
        P = ds_mrms['P'].isel(event=i).values
        has_data = ~np.isnan(P).all(axis=0)
        gauge_to_divides.setdefault(gid, set()).update(divide_ids_all[has_data].tolist())

    all_divides = set().union(*gauge_to_divides.values())
    print(f'{len(all_divides)} unique catchments across all flagged gauges')

    cm = gpd.read_parquet(CATCHMENTS_MASTER)
    sub = cm[cm['divide_id'].isin(all_divides)].to_crs(4326)
    print(f'matched {len(sub)} / {len(all_divides)} catchments in catchments_master')
    minx, miny, maxx, maxy = sub.total_bounds
    bbox = (minx - BBOX_MARGIN_DEG, miny - BBOX_MARGIN_DEG,
            maxx + BBOX_MARGIN_DEG, maxy + BBOX_MARGIN_DEG)
    print(f'combined bbox (lonmin,latmin,lonmax,latmax): {bbox}')

    # deduplicated, grid-aligned timestamp union across all flagged events
    df['grid_start'] = df['ts_start'].dt.floor('2min')
    df['grid_end'] = df['ts_end'].dt.ceil('2min')
    all_times: set = set()
    for _, r in df.iterrows():
        all_times.update(pd.date_range(r['grid_start'], r['grid_end'], freq='2min'))
    times = pd.DatetimeIndex(sorted(all_times))
    print(f'{len(times):,} unique timestamps to fetch across '
          f'{times.normalize().nunique()} UTC days')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'catchments_used.txt').write_text('\n'.join(sorted(all_divides)))
    (OUT_DIR / 'bbox.txt').write_text(repr(bbox))

    summary = build_store(
        times, bbox, str(OUT_DIR),
        max_workers=MAX_WORKERS, mask_negative=True, use_aws=True, verbose=True,
    )
    print('\nbuild_store summary:', summary)
    print(f'\nDay shards in: {OUT_DIR / "shards"}')
    print('Next: run rebuild_flagged_events_precip.py to re-extract per-event '
          '15-min catchment depth from these shards.')


if __name__ == '__main__':
    main()
