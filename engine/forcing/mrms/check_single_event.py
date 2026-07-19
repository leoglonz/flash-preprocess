r"""Re-download raw MRMS PrecipRate directly from source for one event and
recompute 15-min catchment depth exactly as the pipeline does (Notebook 4's
storm_catchment_rate_2min + to_depth_15), to check whether a near-zero P in
forcing_15min.nc / mrms_15min.nc reflects a genuine MRMS gap or a pipeline bug.

Bypasses the full multi-HUC8 notebook chain: builds the bbox and catchment
centroids directly from the cached hydrofabric (catchments_master.parquet)
and mrms_15min.nc, then calls mrms_bbox_downloader.build_store() for just
this one event's window.

Edit the CONFIG block at the top of this file to set all options. Run with:
    python engine/forcing/mrms/check_single_event.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from flash_preprocess.mrms import build_store


# CONFIG
EVENT_ID = '1891'  # storm_id / event_id to re-check, from mrms_15min.nc
MRMS_15MIN_NC = Path(
    '/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/mrms_15min.nc',
)
CATCHMENTS_MASTER = Path(
    '/projects/mhpi/leoglonz/sub_hourly/data/_mrms_preprocess/catchments_master.parquet',
)
OUT_DIR = (
    Path('/projects/mhpi/leoglonz/sub_hourly/data/_mrms_preprocess/event_checks')
    / f'event_{EVENT_ID}'
)

BBOX_MARGIN_DEG = 0.1
MAX_WORKERS = 8


def main() -> None:
    """Re-download and recompute 15-min catchment depth for one event."""
    ds_mrms = xr.open_dataset(MRMS_15MIN_NC)
    matches = np.where(ds_mrms['storm_id'].values.astype(str) == EVENT_ID)[0]
    if len(matches) == 0:
        sys.exit(f'event_id {EVENT_ID!r} not found in {MRMS_15MIN_NC}')
    mi = matches[0]

    ts_start = pd.Timestamp(ds_mrms['ts_start'].values[mi])
    ts_end = pd.Timestamp(ds_mrms['ts_end'].values[mi])
    print(f'Event {EVENT_ID}: {ts_start} -> {ts_end}')

    P_existing = ds_mrms['P'].isel(event=mi).values
    has_data = ~np.isnan(P_existing).all(axis=0)
    divide_ids = ds_mrms['divide_id'].values[has_data]
    print(
        f'{len(divide_ids)} catchments with data in existing file '
        f'(existing max P = {np.nanmax(P_existing):.4f} mm/15min)',
    )

    cm = gpd.read_parquet(CATCHMENTS_MASTER)
    sub = cm[cm['divide_id'].isin(divide_ids)].to_crs(4326)
    minx, miny, maxx, maxy = sub.total_bounds
    bbox = (
        minx - BBOX_MARGIN_DEG,
        miny - BBOX_MARGIN_DEG,
        maxx + BBOX_MARGIN_DEG,
        maxy + BBOX_MARGIN_DEG,
    )
    print('bbox (lonmin,latmin,lonmax,latmax):', bbox)

    # catchment centroids for nearest-cell precip extraction (lon in 0-360 to
    # match MRMS grid convention, same as notebook 4's storm_catchment_rate_2min)
    sub_proj = sub.to_crs(5070)
    centroids = sub_proj.geometry.centroid.to_crs(4326)
    cat_lat = centroids.y.values
    cat_lon = centroids.x.values % 360.0
    cat_ids = sub['divide_id'].values

    # MRMS files are only published on even 2-min marks (:00, :02, ...); floor/
    # ceil the event window to that grid before building the range, same as
    # Notebook 3 does (sub["win_start"].dt.floor(MRMS_FREQ) / .dt.ceil(...)) --
    # otherwise an odd-minute ts_start (e.g. :01, :15) makes every requested
    # timestamp miss a real file and look like a confirmed archive gap.
    grid_start = ts_start.floor('2min')
    grid_end = ts_end.ceil('2min')
    times = pd.date_range(grid_start, grid_end, freq='2min')
    print(
        f'{len(times)} timestamps to fetch (native 2-min MRMS cadence, '
        f'grid-aligned {grid_start} -> {grid_end})',
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_store(
        times,
        bbox,
        str(OUT_DIR),
        max_workers=MAX_WORKERS,
        mask_negative=True,
        use_aws=True,
        verbose=True,
    )
    print('\nbuild_store summary:', summary)

    # Load the freshly downloaded shards, extract nearest-cell precip rate
    # per catchment centroid, resample to 15-min depth, same as the pipeline
    # (Notebook 4: storm_catchment_rate_2min + to_depth_15).
    shard_dir = OUT_DIR / 'shards'
    files = sorted(shard_dir.glob('pr_*.nc'))
    print(f'\n{len(files)} day shard(s) downloaded: {[f.name for f in files]}')
    if not files:
        sys.exit('No shard files -- nothing downloaded, cannot proceed.')

    ds = xr.open_mfdataset([str(f) for f in files], combine='by_coords')
    da = ds['precip_rate']

    pts = da.sel(
        latitude=xr.DataArray(cat_lat, dims='cell'),
        longitude=xr.DataArray(cat_lon, dims='cell'),
        method='nearest',
    )
    rate2min = pd.DataFrame(
        pts.values,
        index=pd.DatetimeIndex(da['time'].values),
        columns=cat_ids,
    )
    rate2min = rate2min.sort_index()
    rate2min = rate2min[(rate2min.index >= ts_start) & (rate2min.index <= ts_end)]

    rate15 = rate2min.resample('15min', label='left', closed='left').mean()
    depth15 = rate15 * 0.25  # mm/15min

    print(
        f'\nRe-extracted depth: {depth15.shape[0]} timesteps x {depth15.shape[1]} catchments',
    )
    print('max single-catchment 15-min depth (mm):', np.nanmax(depth15.values))
    print(
        'basin-mean total depth over event (mm):',
        np.nanmean(np.nansum(depth15.values, axis=0)),
    )
    print(
        'n timesteps with any nonzero rain (any catchment):',
        int((np.nansum(depth15.values, axis=1) > 0.01).sum()),
        '/',
        depth15.shape[0],
    )

    out_csv = OUT_DIR / 'depth15_reextracted.csv'
    depth15.to_csv(out_csv)
    print(f'\nSaved -> {out_csv}')


if __name__ == '__main__':
    main()
