r"""Re-extract 15-min catchment precipitation depth for every flagged event,
from the raw MRMS shards fetched by batch_redownload_flagged_events.py.

Same method as Notebook 4's storm_catchment_rate_2min + to_depth_15
(nearest-cell 2-min PrecipRate -> 15-min mean rate -> depth = rate * 0.25h),
except using nearest-cell centroid lookup rather than the full
fraction_inside area-weighted crosswalk (good enough to quantify/repair the
precip gap; swap in the real crosswalk later if bit-exact parity with the
official product matters).

Output: one combined NetCDF, same schema as mrms_15min.nc (storm_id, n_steps,
ts_start, ts_end, divide_id, P), covering only the flagged events -- ready to
compare against or merge back into the existing mrms_15min.nc.

Run with:
    python engine/forcing/mrms/rebuild_flagged_events_precip.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import netCDF4
from tqdm.auto import tqdm


# CONFIG
FLAGGED_CSV = Path('/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/precip_gap_events.csv')
MRMS_15MIN_NC = Path('/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/mrms_15min.nc')
CATCHMENTS_MASTER = Path('/projects/mhpi/leoglonz/sub_hourly/data/_mrms_preprocess/catchments_master.parquet')
REDOWNLOAD_DIR = Path('/projects/mhpi/leoglonz/sub_hourly/data/_mrms_preprocess/flagged_events_redownload')
OUT_NC = Path('/projects/mhpi/leoglonz/sub_hourly/data/upper_neuse_usgs/mrms_15min_reextracted.nc')


def main() -> None:
    df = pd.read_csv(FLAGGED_CSV, dtype={'event_id': str, 'gauge_id': str},
                      parse_dates=['ts_start', 'ts_end'])
    print(f'{len(df)} flagged events to re-extract')

    ds_mrms = xr.open_dataset(MRMS_15MIN_NC)
    storm_ids = ds_mrms['storm_id'].values.astype(str)
    storm_idx = {s: i for i, s in enumerate(storm_ids)}
    divide_ids_all = ds_mrms['divide_id'].values

    gauge_to_divides: dict[str, list[str]] = {}
    for eid, gid in zip(df['event_id'], df['gauge_id']):
        gid = gid.zfill(8)
        i = storm_idx.get(eid)
        if i is None:
            continue
        P = ds_mrms['P'].isel(event=i).values
        has_data = ~np.isnan(P).all(axis=0)
        gauge_to_divides.setdefault(gid, set()).update(divide_ids_all[has_data].tolist())
    gauge_to_divides = {g: sorted(d) for g, d in gauge_to_divides.items()}

    cm = gpd.read_parquet(CATCHMENTS_MASTER).to_crs(5070)
    cm_lookup = cm.set_index('divide_id')
    centroids_4326 = cm_lookup.geometry.centroid.to_crs(4326)
    cat_lat_lookup = centroids_4326.y
    cat_lon_lookup = (centroids_4326.x % 360.0)

    shard_dir = REDOWNLOAD_DIR / 'shards'
    files = sorted(shard_dir.glob('pr_*.nc'))
    if not files:
        raise SystemExit(f'No shard files in {shard_dir} -- run '
                          'batch_redownload_flagged_events.py first.')
    print(f'Loading {len(files)} day shard(s) from {shard_dir} ...')
    ds_shards = xr.open_mfdataset([str(f) for f in files], combine='by_coords')
    da = ds_shards['precip_rate']

    max_steps = 481  # matches mrms_15min.nc convention (n_steps <= 481)
    n_events = len(df)

    out_records = []  # (storm_id, gauge, n_steps, ts_start, ts_end, depth_df)
    for _, row in tqdm(df.iterrows(), total=n_events, desc='events'):
        eid = row['event_id']
        gid = row['gauge_id'].zfill(8)
        ts_start, ts_end = row['ts_start'], row['ts_end']
        divide_ids = gauge_to_divides.get(gid, [])
        if not divide_ids:
            continue

        cat_lat = cat_lat_lookup.reindex(divide_ids).values
        cat_lon = cat_lon_lookup.reindex(divide_ids).values

        pts = da.sel(
            latitude=xr.DataArray(cat_lat, dims='cell'),
            longitude=xr.DataArray(cat_lon, dims='cell'),
            method='nearest',
        )
        window = pts.sel(time=slice(ts_start.floor('2min'), ts_end.ceil('2min')))
        if window.sizes.get('time', 0) == 0:
            continue

        rate2min = pd.DataFrame(
            window.values,
            index=pd.DatetimeIndex(window['time'].values),
            columns=divide_ids,
        ).sort_index()
        rate2min = rate2min[(rate2min.index >= ts_start) & (rate2min.index <= ts_end)]
        if rate2min.empty:
            continue

        rate15 = rate2min.resample('15min', label='left', closed='left').mean()
        depth15 = (rate15 * 0.25).iloc[:max_steps]

        out_records.append({
            'storm_id': eid,
            'divide_ids': divide_ids,
            'n_steps': len(depth15),
            'ts_start': depth15.index[0],
            'ts_end': depth15.index[-1],
            'depth': depth15.values.astype('float32'),
        })

    ds_shards.close()

    if not out_records:
        raise SystemExit('No events re-extracted -- nothing to write.')

    all_cats = sorted(set().union(*[set(r['divide_ids']) for r in out_records]))
    cat_index = {c: j for j, c in enumerate(all_cats)}
    n_out = len(out_records)
    n_cat = len(all_cats)

    print(f'\nWriting {OUT_NC} ({n_out} events x {max_steps} steps x {n_cat} catchments) ...')
    nc_out = netCDF4.Dataset(OUT_NC, 'w', format='NETCDF4')
    nc_out.createDimension('event', n_out)
    nc_out.createDimension('time_step', max_steps)
    nc_out.createDimension('catchment', n_cat)

    v_sid = nc_out.createVariable('storm_id', str, ('event',))
    v_sid[:] = np.array([r['storm_id'] for r in out_records], dtype=object)

    v_ns = nc_out.createVariable('n_steps', 'i4', ('event',))
    v_ns[:] = np.array([r['n_steps'] for r in out_records], dtype=np.int32)

    epoch = np.datetime64('1970-01-01T00:00', 'm')
    v_ts = nc_out.createVariable('ts_start', 'f8', ('event',))
    v_ts.units = 'minutes since 1970-01-01 00:00:00 UTC'
    v_ts[:] = [(np.datetime64(r['ts_start']) - epoch) / np.timedelta64(1, 'm') for r in out_records]

    v_te = nc_out.createVariable('ts_end', 'f8', ('event',))
    v_te.units = 'minutes since 1970-01-01 00:00:00 UTC'
    v_te[:] = [(np.datetime64(r['ts_end']) - epoch) / np.timedelta64(1, 'm') for r in out_records]

    v_cat = nc_out.createVariable('divide_id', str, ('catchment',))
    v_cat[:] = np.array(all_cats, dtype=object)

    v_data = nc_out.createVariable(
        'P', 'f4', ('event', 'time_step', 'catchment'),
        fill_value=np.nan, zlib=True, complevel=4,
    )
    v_data.units = 'mm [15 min]-1'
    v_data.long_name = 'MRMS precipitation depth (re-extracted, nearest-cell)'

    data = np.full((n_out, max_steps, n_cat), np.nan, dtype='float32')
    for i, r in enumerate(out_records):
        cols = [cat_index[c] for c in r['divide_ids']]
        n = r['n_steps']
        data[i, :n, cols] = r['depth'].T
    v_data[:] = data

    nc_out.close()
    print(f'Done -> {OUT_NC}')


if __name__ == '__main__':
    main()
