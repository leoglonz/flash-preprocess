"""AORC + Hydrofabric pipeline.

- area-weighted catchment crosswalk
- checkpointed per-year S3 fetch
- per-event antecedent/event-window extraction to aorc_hr.nc / aorc_15min.nc.

@drworm
"""

import atexit
import multiprocessing
import os
import pickle
from multiprocessing.pool import ThreadPool
from pathlib import Path

import dask
import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from dask.diagnostics import ProgressBar
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from flash_preprocess.pet import penman_monteith_pet
from flash_preprocess.utils import get_cell_weights


_pool = ThreadPool(int(os.environ.get('AORC_S3_THREADS', 64)))
dask.config.set(pool=_pool)
atexit.register(_pool.terminate)


### DEFAULTS --------------- #
VARIABLE_LIST = [
    "APCP_surface",
    "DSWRF_surface",
    "TMP_2maboveground",
    "DLWRF_surface",
    "PRES_surface",
    "SPFH_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
]

# Accumulated (uniform split to 15-min) vs. instantaneous (linear interp)
ACCUM_VARS = {"APCP_surface"}

# Hourly warmup window preceding each event's sub-hourly window (default: 6d).
ANTECEDENT_DAYS = 30.0

_EPOCH = np.datetime64("1970-01-01T00:00", "m")
_AORC_NCOLS = 8401  # CONUS grid width
_AORC_NROWS = 4201
_AORC_LAT0 = 20.0  # southernmost latitude
_AORC_LON0 = -130.0  # westernmost longitude
_AORC_DLAT = 1.0 / 120.0  # ~0.008333 deg (~1 km)

# Below this, one exact_extract call beats worker overhead
_WEIGHT_PARALLEL_THRESHOLD = 500
# -------------------------- #


def open_aorc(start: np.datetime64, end: np.datetime64) -> xr.Dataset:
    """Open AORC v1.1 zarr stores for every calendar year spanned by [start, end].

    Parameters
    ----------
    start
        Start of range, inclusive, datetime64[h].
    end
        End of range, inclusive, datetime64[h].

    Returns
    -------
    xr.Dataset
        Lazily opened dataset with dims (time, latitude, longitude),
        latitude sorted descending, sliced to [start, end].
    """
    y0 = int(str(start)[:4])
    y1 = int(str(end)[:4])
    _s3 = s3fs.S3FileSystem(anon=True)
    stores = [
        s3fs.S3Map(root=f"s3://noaa-nws-aorc-v1-1-1km/{y}.zarr", s3=_s3, check=False)
        for y in range(y0, y1 + 1)
    ]
    ds = xr.open_mfdataset(stores, engine="zarr", parallel=True, consolidated=True)
    ds = ds.sortby("latitude", ascending=False)
    return ds.sel(time=slice(start, end))


def spatial_subset_points(
    ds: xr.Dataset,
    flat_cell_ids: np.ndarray,
) -> xr.Dataset:
    """Subset to exactly the unique pixels needed, via vectorized indexing.

    A single bounding box is wasteful when catchments are geographically
    scattered (e.g. several disjoint HUC8s across CONUS): the enclosing
    rectangle can be >100x larger than the pixels actually used. Vectorized
    (point-wise) isel selects only the requested (row, col) pairs, so
    dask/zarr fetch only the chunks that intersect them.

    Parameters
    ----------
    ds
        AORC dataset, dims (time, latitude, longitude).
    flat_cell_ids
        1D array of *unique* full-grid flat cell indices (row * _AORC_NCOLS + col)
        actually needed, in the desired output order.

    Returns
    -------
    ds_sub
        Dataset with latitude/longitude collapsed into a single 'pixel'
        dimension of length len(flat_cell_ids), aligned to that order.
    """
    rows = flat_cell_ids // _AORC_NCOLS
    cols = flat_cell_ids % _AORC_NCOLS
    print(
        f"  Point selection: {len(flat_cell_ids):,} unique pixels "
        f"(vs {_AORC_NROWS * _AORC_NCOLS:,} full grid)",
    )
    row_idx = xr.DataArray(rows, dims="pixel")
    col_idx = xr.DataArray(cols, dims="pixel")
    return ds.isel(latitude=row_idx, longitude=col_idx)


def build_weight_matrix(
    cell_ids_list: list,
    weights_list: list,
    n_basins: int,
    n_pixels: int,
) -> csr_matrix:
    """Build a (n_basins, n_pixels) CSR weight matrix (rows sum to 1).

    Parameters
    ----------
    cell_ids_list
        Per-catchment local flat pixel indices into the bbox grid.
    weights_list
        Per-catchment area weights (unnormalised).
    n_basins
        Number of catchments.
    n_pixels
        Total pixels in the bbox (bbox_lat * bbox_lon).

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape (n_basins, n_pixels), float32.
    """
    rows = np.concatenate(
        [np.full(len(c), i, dtype=np.int32) for i, c in enumerate(cell_ids_list)],
    )
    cols = np.concatenate(cell_ids_list).astype(np.int32)
    vals = np.concatenate([w / w.sum() for w in weights_list]).astype(np.float32)
    return csr_matrix(
        (vals, (rows, cols)),
        shape=(n_basins, n_pixels),
        dtype=np.float32,
    )


def weighted_mean(flat_raster: np.ndarray, W: csr_matrix) -> np.ndarray:
    """Area-weighted catchment average via sparse matmul.

    Parameters
    ----------
    flat_raster
        Shape (num_hours, n_pixels_in_bbox).
    W
        Pre-built weight matrix from build_weight_matrix, shape (n_basins, n_pixels).

    Returns
    -------
    np.ndarray
        Shape (num_catchments, num_hours), float32. NaN propagates if any pixel is NaN.
    """
    return np.asarray(W @ flat_raster.T, dtype=np.float32)  # (n_basins, n_hours)


def groupby_mean_equal(data: np.ndarray, interval: np.ndarray) -> np.ndarray:
    """Equal-weight catchment average (point-in-polygon index).

    Parameters
    ----------
    data
        Shape (total_pixels, hours).
    interval
        Per-catchment pixel counts.

    Returns
    -------
    np.ndarray
        Shape (num_catchments, hours).
    """
    bins = np.insert(np.cumsum(interval), 0, 0)[:-1]
    mask = np.isnan(data)
    data = data.copy()
    data[mask] = 0.0
    g_count = np.add.reduceat(~mask, bins, axis=0)
    g_sum = np.add.reduceat(data, bins, axis=0)
    return g_sum / g_count


def disaggregate_to_15min(
    data: np.ndarray,
    var_name: str,
    time_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Disaggregate hourly catchment data to 15-minute intervals.

    Accumulated variables (APCP_surface): uniform split / 4.
    Instantaneous variables: linear interpolation between hourly values.

    Parameters
    ----------
    data
        Shape (n_basins, n_hours).
    var_name
        Variable name, used to select accumulation vs. interpolation.
    time_vals
        Hourly datetime64 array, shape (n_hours,).

    Returns
    -------
    data_15min
        Shape (n_basins, n_hours * 4).
    time_15min
        datetime64[m] array, shape (n_hours * 4,).
    """
    n_hours = data.shape[1]
    n_steps = n_hours * 4

    if var_name in ACCUM_VARS:
        data_15min = np.repeat(data / 4.0, 4, axis=1)
    else:
        x_15 = np.arange(n_steps, dtype=np.float64) * 0.25
        i_low = np.clip(np.floor(x_15).astype(int), 0, n_hours - 1)
        i_high = np.clip(np.ceil(x_15).astype(int), 0, n_hours - 1)
        frac = (x_15 - i_low).astype(np.float32)
        data_15min = data[:, i_low] * (1.0 - frac) + data[:, i_high] * frac

    t0 = time_vals[0].astype("datetime64[m]")
    time_15min = t0 + np.arange(n_steps) * np.timedelta64(15, "m")
    return data_15min, time_15min


def _atomic_write(write_fn, path):
    """write_fn(tmp_path) then atomic rename -- avoids a torn file if a
    run is interrupted or a concurrent VPU job races on the same cache entry.
    """
    tmp = f"{path}.tmp{os.getpid()}"
    write_fn(tmp)
    os.replace(tmp, path)


def _mins(t) -> float:
    return float((np.datetime64(t) - _EPOCH) / np.timedelta64(1, "m"))


#### AORC 1km grid geometry + area-weighted catchment crosswalk


def build_aorc_grid() -> xr.Dataset:
    """AORC 1km grid geometry, identical every year (1979-2025). Latitude
    descending (row 0 = ~55N) to match the flat cell_id convention used
    throughout this module (cell_id = row * _AORC_NCOLS + col).
    """
    lat = np.linspace(
        _AORC_LAT0 + (_AORC_NROWS - 1) * _AORC_DLAT,
        _AORC_LAT0,
        _AORC_NROWS,
    )
    lon = np.linspace(
        _AORC_LON0,
        _AORC_LON0 + (_AORC_NCOLS - 1) * _AORC_DLAT,
        _AORC_NCOLS,
    )
    dummy = np.zeros((_AORC_NROWS, _AORC_NCOLS), dtype=np.float32)
    return xr.Dataset({"dummy": (["y", "x"], dummy)}, coords={"y": lat, "x": lon})


def _compute_weights(
    gdf_4326: gpd.GeoDataFrame,
    grid_ds: xr.Dataset,
    max_workers: int | None = None,
) -> pd.DataFrame:
    wkt = gdf_4326.crs.to_wkt()
    if len(gdf_4326) < _WEIGHT_PARALLEL_THRESHOLD:
        return get_cell_weights(grid_ds, gdf_4326, wkt)
    if max_workers is None:
        max_workers = multiprocessing.cpu_count() - 1
    n_workers = max(1, min(max_workers, len(gdf_4326) // 50, 16))
    idx_splits = np.array_split(np.arange(len(gdf_4326)), n_workers)
    chunks = [gdf_4326.iloc[idx] for idx in idx_splits if len(idx) > 0]
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.starmap(get_cell_weights, [(grid_ds, c, wkt) for c in chunks])
    return pd.concat(results)


def build_weighted_crosswalk(
    divide_ids,
    catchments_master: gpd.GeoDataFrame,
    cache_dir: Path,
    tag: str = "",
    max_workers: int | None = None,
) -> dict:
    """Area-weighted AORC-pixel <-> catchment crosswalk for `divide_ids`, cached
    per tag (VPU). Returns dict(station_ids, cell_ids_list, weights_list).
    """
    cache_dir = Path(cache_dir)
    suffix = f"_{tag}" if tag else ""
    f = cache_dir / f"aorc_weights{suffix}.pkl"
    divide_ids = set(divide_ids)
    if f.exists():
        idx = pickle.loads(f.read_bytes())
        if divide_ids <= set(idx["station_ids"]):
            return idx

    gdf = catchments_master[catchments_master["divide_id"].isin(divide_ids)][
        ["divide_id", "geometry"]
    ].to_crs(4326)
    weights_df = _compute_weights(gdf, build_aorc_grid(), max_workers)

    station_ids, cell_ids_list, weights_list = [], [], []
    for cat_id in sorted(divide_ids):
        if cat_id not in weights_df.index:
            continue
        row = weights_df.loc[cat_id]
        station_ids.append(cat_id)
        cell_ids_list.append(np.asarray(row["cell_id"], dtype=np.int64))
        w = np.asarray(row["coverage"], dtype=np.float32)
        weights_list.append(w / w.sum())

    idx = {
        "station_ids": np.array(station_ids),
        "cell_ids_list": cell_ids_list,
        "weights_list": weights_list,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(lambda p: Path(p).write_bytes(pickle.dumps(idx)), f)
    return idx


#### checkpointed per-year S3 fetch


def _year_bounds(manifest: pd.DataFrame, antecedent_days: float) -> dict:
    """Per calendar year spanned by any event's [antecedent_start, win_end],
    the tightest [lo, hi] bound covering every event's need that year.
    """
    bounds = {}
    for row in manifest.itertuples():
        ant_start = row.win_start - pd.Timedelta(days=antecedent_days)
        for year in range(ant_start.year, row.win_end.year + 1):
            y_lo = max(ant_start, pd.Timestamp(year, 1, 1))
            y_hi = min(row.win_end, pd.Timestamp(year, 12, 31, 23))
            if y_lo > y_hi:
                continue
            lo, hi = bounds.get(year, (y_lo, y_hi))
            bounds[year] = (min(lo, y_lo), max(hi, y_hi))
    return bounds


def build_shards(
    manifest: pd.DataFrame,
    weight_idx: dict,
    vpu_dir: Path,
    variables=VARIABLE_LIST,
    antecedent_days: float = ANTECEDENT_DAYS,
    verbose: bool = True,
) -> list:
    """Fetch + cache AORC data for every year spanned by `manifest`'s
    antecedent + event windows, one NetCDF shard per calendar year under
    <vpu_dir>/shards/. Resumable: an interrupted run leaves finished years'
    shard files in place and a rerun only re-fetches missing years.
    """
    shard_dir = Path(vpu_dir) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    all_cell_ids = np.unique(np.concatenate(weight_idx["cell_ids_list"]))
    bounds = _year_bounds(manifest, antecedent_days)

    for year, (lo, hi) in sorted(bounds.items()):
        f = shard_dir / f"aorc_{year}.nc"
        if f.exists():
            continue
        if verbose:
            print(
                f"  fetching AORC {year}: {lo} -> {hi} ({len(all_cell_ids):,} pixels)",
            )
        ds = open_aorc(np.datetime64(lo, "h"), np.datetime64(hi, "h"))
        ds = spatial_subset_points(ds, all_cell_ids)[list(variables)]
        with ProgressBar():
            ds_computed = ds.compute()

        def _write(tmp, ds_computed=ds_computed):
            nc = netCDF4.Dataset(tmp, "w", format="NETCDF4")
            nc.createDimension("time", ds_computed.sizes["time"])
            nc.createDimension("pixel", len(all_cell_ids))
            v_t = nc.createVariable("time", "i8", ("time",))
            v_t.units = "minutes since 1970-01-01 00:00:00 UTC"
            v_t[:] = (
                ds_computed["time"].values.astype("datetime64[m]") - _EPOCH
            ).astype(np.int64)
            nc.createVariable("pixel_id", "i8", ("pixel",))[:] = all_cell_ids
            nc.createVariable("latitude", "f4", ("pixel",))[:] = ds_computed[
                "latitude"
            ].values.astype(np.float32)
            nc.createVariable("longitude", "f4", ("pixel",))[:] = ds_computed[
                "longitude"
            ].values.astype(np.float32)
            for var in variables:
                nv = nc.createVariable(
                    var,
                    "f4",
                    ("time", "pixel"),
                    zlib=True,
                    complevel=1,
                )
                nv[:] = ds_computed[var].values.astype(np.float32)
            nc.close()

        _atomic_write(_write, f)
    return sorted(shard_dir.glob("aorc_*.nc"))


def open_shards(shard_dir: Path) -> xr.Dataset:
    """Open all AORC shards in shard_dir as one lazily-concatenated dataset."""
    files = sorted(Path(shard_dir).glob("aorc_*.nc"))
    if not files:
        raise FileNotFoundError(f"no AORC shards in {shard_dir}")
    return xr.open_mfdataset(
        [str(f) for f in files],
        combine="nested",
        concat_dim="time",
        engine="netcdf4",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    ).sortby("time")


#### per-event extraction: antecedent hourly window + 5-day 15-min window


def _create_output_nc(
    path: Path,
    n_events: int,
    max_steps: int,
    meta: dict,
    data_vars: dict,
) -> netCDF4.Dataset:
    n_basins = meta["n_basins"]
    nc = netCDF4.Dataset(path, "w", format="NETCDF4")
    nc.createDimension("event", n_events)
    nc.createDimension("time_step", max_steps)
    nc.createDimension("catchment", n_basins)

    nc.createVariable("event_id", str, ("event",))
    nc.createVariable("n_steps", "i4", ("event",))
    v = nc.createVariable("ts_start", "f8", ("event",))
    v.units = "minutes since 1970-01-01 00:00:00 UTC"
    v = nc.createVariable("ts_end", "f8", ("event",))
    v.units = "minutes since 1970-01-01 00:00:00 UTC"
    nc.createVariable("event_gage_id", str, ("event",))
    nc.createVariable("event_divide_id", str, ("event",))

    v = nc.createVariable("divide_id", str, ("catchment",))
    v[:] = np.array(meta["station_ids"], dtype=object)
    v = nc.createVariable("latitude", "f4", ("catchment",))
    v[:] = meta["cat_lats"]
    v = nc.createVariable("longitude", "f4", ("catchment",))
    v[:] = meta["cat_lons"]

    chunk_e, chunk_c = min(n_events, 16), min(n_basins, 64)
    for vname, (units, long_name) in data_vars.items():
        nv = nc.createVariable(
            vname,
            "f4",
            ("event", "time_step", "catchment"),
            fill_value=np.nan,
            zlib=True,
            complevel=4,
            chunksizes=(chunk_e, max_steps, chunk_c),
        )
        nv.units, nv.long_name = units, long_name
    return nc


def _catchment_metadata(ds: xr.Dataset, weight_idx: dict):
    """Localize global pixel_ids to ds's pixel-dim ordering, build the sparse
    weight matrix, and compute area-weighted catchment centroids.
    """
    pixel_id = ds["pixel_id"].values
    id_to_local = {v: i for i, v in enumerate(pixel_id)}
    local_cids_list = [
        np.array([id_to_local[c] for c in cids], dtype=np.int64)
        for cids in weight_idx["cell_ids_list"]
    ]
    n_basins, n_pixels = len(weight_idx["station_ids"]), ds.sizes["pixel"]
    W = build_weight_matrix(
        local_cids_list,
        weight_idx["weights_list"],
        n_basins,
        n_pixels,
    )

    lat_px = ds["latitude"].values.astype(np.float32)
    lon_px = ds["longitude"].values.astype(np.float32)
    cat_lats, cat_lons = [], []
    for cids, w in zip(local_cids_list, weight_idx["weights_list"]):
        wn = w / w.sum()
        cat_lats.append(float(np.dot(wn, lat_px[cids])))
        cat_lons.append(float(np.dot(wn, lon_px[cids])))
    return W, np.array(cat_lats, dtype=np.float32), np.array(cat_lons, dtype=np.float32)


def _catchment_pet(vd: dict) -> np.ndarray:
    return penman_monteith_pet(
        temp=vd["TMP_2maboveground"],
        spfh=vd["SPFH_2maboveground"],
        dlwrf=vd["DLWRF_surface"],
        dswrf=vd["DSWRF_surface"],
        pres=vd["PRES_surface"],
        ugrd_10m=vd["UGRD_10maboveground"],
        vgrd_10m=vd["VGRD_10maboveground"],
    )


def extract_all(
    manifest: pd.DataFrame,
    weight_idx: dict,
    shard_dir: Path,
    out_hr_nc: Path,
    out_15min_nc: Path,
    divide_id_of: dict,
    antecedent_days: float = ANTECEDENT_DAYS,
    max_15min_steps: int = 481,
):
    """Extract per-event hourly and 15-min AORC forcing to out_hr_nc/out_15min_nc."""
    # Eager-load once: shards are netCDF4-backed dask arrays, so repeated
    # per-event/per-variable .isel().values calls below would otherwise
    # re-hit disk for every slice instead of just indexing an in-memory array.
    ds = open_shards(shard_dir).load()
    time_dt = pd.DatetimeIndex(ds["time"].values)
    W, cat_lats, cat_lons = _catchment_metadata(ds, weight_idx)
    n_basins = len(weight_idx["station_ids"])
    n_hours_ant = int(antecedent_days * 24)
    meta = {
        "n_basins": n_basins,
        "station_ids": weight_idx["station_ids"],
        "cat_lats": cat_lats,
        "cat_lons": cat_lons,
    }

    n_events = len(manifest)

    # Accumulate into plain numpy arrays and write each variable to disk in a
    # single bulk call at the end -- a per-event netCDF4 write into a
    # compressed chunk forces a decompress/recompress of that chunk on every
    # call, which dominates runtime at thousands of events; one bulk write
    # lets the library compress each chunk exactly once.
    P_ant = np.full((n_events, n_hours_ant, n_basins), np.nan, dtype=np.float32)
    T_ant = np.full((n_events, n_hours_ant, n_basins), np.nan, dtype=np.float32)
    PET_ant = np.full((n_events, n_hours_ant, n_basins), np.nan, dtype=np.float32)
    T_15 = np.full((n_events, max_15min_steps, n_basins), np.nan, dtype=np.float32)
    PET_15 = np.full((n_events, max_15min_steps, n_basins), np.nan, dtype=np.float32)

    event_ids = np.empty(n_events, dtype=object)
    gage_ids = np.empty(n_events, dtype=object)
    divide_ids_out = np.empty(n_events, dtype=object)
    n_steps_hr = np.zeros(n_events, dtype=np.int32)
    ts_start_hr = np.zeros(n_events, dtype=np.float64)
    ts_end_hr = np.zeros(n_events, dtype=np.float64)
    n_steps_15 = np.zeros(n_events, dtype=np.int32)
    ts_start_15 = np.zeros(n_events, dtype=np.float64)
    ts_end_15 = np.zeros(n_events, dtype=np.float64)

    skipped = []
    for i, row in enumerate(
        tqdm(manifest.itertuples(), total=n_events, desc="extract"),
    ):
        sid = str(row.storm_index)
        event_ids[i] = sid
        gage_ids[i] = row.recording_gage_STAID
        divide_ids_out[i] = divide_id_of.get(sid, "")

        ant_start = row.win_start - pd.Timedelta(days=antecedent_days)
        ant_mask = (time_dt >= ant_start) & (time_dt < row.win_start)
        evt_mask = (time_dt >= row.win_start) & (time_dt <= row.win_end)
        if ant_mask.sum() < n_hours_ant or not evt_mask.any():
            skipped.append(
                {
                    "storm_id": sid,
                    "gauge_id": row.recording_gage_STAID,
                    "reason": f"incomplete shard coverage (ant={int(ant_mask.sum())}/{n_hours_ant}, "
                    f"evt={int(evt_mask.sum())})",
                },
            )
            continue

        raw_ant = {v: ds[v].isel(time=ant_mask).values for v in VARIABLE_LIST}
        cat_ant = {v: weighted_mean(raw_ant[v], W) for v in raw_ant}
        temp_ant_c = cat_ant["TMP_2maboveground"] - 273.15
        pet_ant = _catchment_pet({**cat_ant, "TMP_2maboveground": temp_ant_c})
        time_ant = time_dt[ant_mask]

        n_steps_hr[i] = n_hours_ant
        ts_start_hr[i] = _mins(time_ant[0])
        ts_end_hr[i] = _mins(time_ant[-1])
        P_ant[i] = cat_ant["APCP_surface"].T
        T_ant[i] = temp_ant_c.T
        PET_ant[i] = pet_ant.T

        raw_evt = {v: ds[v].isel(time=evt_mask).values for v in VARIABLE_LIST}
        cat_evt = {v: weighted_mean(raw_evt[v], W) for v in raw_evt}
        temp_evt_c = cat_evt["TMP_2maboveground"] - 273.15
        pet_evt = _catchment_pet({**cat_evt, "TMP_2maboveground": temp_evt_c})
        time_evt = time_dt[evt_mask]
        tmp_15min, _ = disaggregate_to_15min(temp_evt_c, "TMP", time_evt.values)
        pet_15min = np.repeat(pet_evt / 4.0, 4, axis=1)
        n15 = min(tmp_15min.shape[1], max_15min_steps)

        n_steps_15[i] = n15
        ts_start_15[i] = _mins(time_evt[0])
        ts_end_15[i] = _mins(time_evt[0]) + (n15 - 1) * 15
        T_15[i, :n15, :] = tmp_15min[:, :n15].T
        PET_15[i, :n15, :] = pet_15min[:, :n15].T

    ds.close()

    nc_hr = _create_output_nc(
        out_hr_nc,
        n_events,
        n_hours_ant,
        meta,
        {
            "P": ("kg m-2", "Precipitation"),
            "T": ("degC", "Air temperature at 2 m"),
            "PET": ("mm h-1", "Penman-Monteith ET0 (FAO-56 hourly)"),
        },
    )
    nc_hr.variables["event_id"][:] = event_ids
    nc_hr.variables["n_steps"][:] = n_steps_hr
    nc_hr.variables["ts_start"][:] = ts_start_hr
    nc_hr.variables["ts_end"][:] = ts_end_hr
    nc_hr.variables["event_gage_id"][:] = gage_ids
    nc_hr.variables["event_divide_id"][:] = divide_ids_out
    nc_hr.variables["P"][:] = P_ant
    nc_hr.variables["T"][:] = T_ant
    nc_hr.variables["PET"][:] = PET_ant
    nc_hr.close()

    nc_15 = _create_output_nc(
        out_15min_nc,
        n_events,
        max_15min_steps,
        meta,
        {
            "T": ("degC", "Air temperature at 2 m (interpolated)"),
            "PET": ("mm 15min-1", "Penman-Monteith ET0 (15-min, uniform split)"),
        },
    )
    nc_15.variables["event_id"][:] = event_ids
    nc_15.variables["n_steps"][:] = n_steps_15
    nc_15.variables["ts_start"][:] = ts_start_15
    nc_15.variables["ts_end"][:] = ts_end_15
    nc_15.variables["event_gage_id"][:] = gage_ids
    nc_15.variables["event_divide_id"][:] = divide_ids_out
    nc_15.variables["T"][:] = T_15
    nc_15.variables["PET"][:] = PET_15
    nc_15.close()

    if skipped:
        pd.DataFrame(skipped).to_csv(
            Path(out_hr_nc).with_name("extraction_skipped_events.csv"),
            index=False,
        )
        print(
            f"WARNING: {len(skipped)} / {n_events} events skipped (incomplete shard coverage) -- "
            f"logged to extraction_skipped_events.csv",
        )
    print(
        f"wrote {out_hr_nc.name} and {out_15min_nc.name}: {n_events - len(skipped)} events x {n_basins} catchments",
    )


#### merge per-VPU part files into one combined NetCDF


def _merge_event_parts(part_paths, out_nc, data_vars: dict):
    parts = [xr.open_dataset(p, decode_times=False) for p in part_paths]
    all_cats = sorted(
        set().union(*[set(p["divide_id"].values.tolist()) for p in parts]),
    )
    cat_idx = {c: j for j, c in enumerate(all_cats)}
    max_steps = max(p.sizes["time_step"] for p in parts)
    n_ev, n_cat = sum(p.sizes["event"] for p in parts), len(all_cats)

    # Assemble full in-memory arrays and write each variable in one bulk
    # call
    lat_out = np.full(n_cat, np.nan, dtype=np.float32)
    lon_out = np.full(n_cat, np.nan, dtype=np.float32)
    event_ids = np.empty(n_ev, dtype=object)
    gage_ids = np.empty(n_ev, dtype=object)
    divide_ids_out = np.empty(n_ev, dtype=object)
    n_steps = np.zeros(n_ev, dtype=np.int32)
    ts_start = np.zeros(n_ev, dtype=np.float64)
    ts_end = np.zeros(n_ev, dtype=np.float64)
    arrays = {
        name: np.full((n_ev, max_steps, n_cat), np.nan, dtype=np.float32)
        for name in data_vars
    }

    i = 0
    for p in parts:
        cols = np.array([cat_idx[c] for c in p["divide_id"].values.tolist()])
        lat_out[cols] = p["latitude"].values
        lon_out[cols] = p["longitude"].values
        n, ns = p.sizes["event"], p.sizes["time_step"]
        event_ids[i : i + n] = p["event_id"].values
        gage_ids[i : i + n] = p["event_gage_id"].values
        divide_ids_out[i : i + n] = p["event_divide_id"].values
        n_steps[i : i + n] = p["n_steps"].values
        ts_start[i : i + n] = p["ts_start"].values
        ts_end[i : i + n] = p["ts_end"].values
        for name in data_vars:
            arrays[name][i : i + n, :ns, cols] = p[name].values
        i += n
        p.close()

    Path(out_nc).parent.mkdir(parents=True, exist_ok=True)
    nc = netCDF4.Dataset(out_nc, "w", format="NETCDF4")
    nc.createDimension("event", n_ev)
    nc.createDimension("time_step", max_steps)
    nc.createDimension("catchment", n_cat)

    nc.createVariable("event_id", str, ("event",))[:] = event_ids
    nc.createVariable("n_steps", "i4", ("event",))[:] = n_steps
    v_ts = nc.createVariable("ts_start", "f8", ("event",))
    v_ts.units = "minutes since 1970-01-01 00:00:00 UTC"
    v_ts[:] = ts_start
    v_te = nc.createVariable("ts_end", "f8", ("event",))
    v_te.units = "minutes since 1970-01-01 00:00:00 UTC"
    v_te[:] = ts_end
    nc.createVariable("event_gage_id", str, ("event",))[:] = gage_ids
    nc.createVariable("event_divide_id", str, ("event",))[:] = divide_ids_out
    nc.createVariable("divide_id", str, ("catchment",))[:] = np.array(
        all_cats,
        dtype=object,
    )
    nc.createVariable("latitude", "f4", ("catchment",))[:] = lat_out
    nc.createVariable("longitude", "f4", ("catchment",))[:] = lon_out

    for name, (units, long_name) in data_vars.items():
        nv = nc.createVariable(
            name,
            "f4",
            ("event", "time_step", "catchment"),
            fill_value=np.nan,
            zlib=True,
            complevel=4,
            chunksizes=(min(n_ev, 16), max_steps, min(n_cat, 64)),
        )
        nv.units, nv.long_name = units, long_name
        nv[:] = arrays[name]

    nc.close()
    print(
        f"merged {len(part_paths)} part(s) -> {out_nc}: {n_ev} events x {n_cat} catchments",
    )


def merge_hr_parts(part_paths, out_nc):
    """Merge per-VPU hourly AORC part files into out_nc."""
    _merge_event_parts(
        part_paths,
        out_nc,
        {
            "P": ("kg m-2", "Precipitation"),
            "T": ("degC", "Air temperature at 2 m"),
            "PET": ("mm h-1", "Penman-Monteith ET0 (FAO-56 hourly)"),
        },
    )


def merge_15min_parts(part_paths, out_nc):
    """Merge per-VPU 15-min AORC part files into out_nc."""
    _merge_event_parts(
        part_paths,
        out_nc,
        {
            "T": ("degC", "Air temperature at 2 m (interpolated)"),
            "PET": ("mm 15min-1", "Penman-Monteith ET0 (15-min, uniform split)"),
        },
    )
