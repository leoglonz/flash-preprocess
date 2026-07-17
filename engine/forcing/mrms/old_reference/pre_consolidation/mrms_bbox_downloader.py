"""
mrms_bbox_downloader.py  (v2)
=============================
Streaming MRMS PrecipRate downloader + bounding-box subsetter for the
MRMS -> HydroFabric -> dHBV pipeline.

WHAT CHANGED FROM v1
--------------------
v1 parallelised over DAYS and walked each day's ~720 files sequentially, so the
progress bar only moved once per finished day (it looked frozen at "0/6" for
many minutes) and the worker pool was never fully saturated.

v2 parallelises over FILES. Every 2-min file is its own task across the whole
worker pool, so:
  * the progress bar is per FILE and moves continuously (real rate + ETA),
  * all workers stay busy regardless of how many days the window spans,
  * output is still ONE NetCDF per UTC day (resumable, dedup'd) -- unchanged.

Days are processed one at a time so only one day of data is held in RAM
(~60 MB for a typical HUC8 bbox) and each day shard is written atomically.

WHAT DID NOT CHANGE
-------------------
The full-CONUS grid is downloaded over the wire but NEVER written to disk: it is
decoded in memory, sliced to your bbox, and discarded. Only the small bbox slice
is stored. Sources: AWS noaa-mrms-pds (anon S3) with Iowa State HTTPS fallback.

DECODER
-------
Decoding uses the low-level `eccodes` API directly (codes_new_from_message +
codes_get_values), not cfgrib's xr.open_dataset(engine="cfgrib"). cfgrib is
correct but does much more work than a single-variable PrecipRate message
needs (temp-file write, full coordinate/index inference), which is the
difference between ~2 file/s and ~12 file/s. If the fast path ever raises
(unexpected grid/key), _decode_precip() transparently falls back to the old
cfgrib path for that one file.

REQUIREMENTS
------------
    pip install numpy pandas xarray cfgrib eccodes s3fs fsspec geopandas tqdm

NOTEBOOK USAGE
--------------
    from mrms_bbox_downloader import bbox_from_huc8, build_store, consolidate
    bbox = bbox_from_huc8(selected_huc8, margin_deg=0.1)
    build_store(unique_mrms_times, bbox, "mrms_precip_bbox", max_workers=12)

CLI (headless / nohup)
----------------------
    python mrms_bbox_downloader.py --times unique_mrms_times.parquet \
        --huc8-bounds selected_huc8.gpkg --out mrms_precip_bbox --workers 12
"""

from __future__ import annotations

import os
import gzip
import tempfile
import argparse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

try:
    import s3fs
except Exception:  # pragma: no cover
    s3fs = None


# ----------------------------------------------------------------------
# Bounding box
# ----------------------------------------------------------------------
def bbox_from_huc8(huc8, margin_deg: float = 0.1):
    """
    Build a lon/lat bbox (lon_min, lat_min, lon_max, lat_max) in EPSG:4326 from a
    HUC8 polygon, padded by `margin_deg` on each side. Accepts a GeoDataFrame,
    GeoSeries, a single GeoDataFrame row, or a bare shapely geom (assumed 5070).
    """
    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry

    if isinstance(huc8, (gpd.GeoDataFrame, gpd.GeoSeries)):
        geom = huc8.to_crs(4326)
    elif hasattr(huc8, "geometry") and not isinstance(huc8, BaseGeometry):
        crs = getattr(huc8, "crs", None) or 5070
        geom = gpd.GeoSeries([huc8.geometry], crs=crs).to_crs(4326)
    else:
        geom = gpd.GeoSeries([huc8], crs=5070).to_crs(4326)

    minx, miny, maxx, maxy = geom.total_bounds
    return (float(minx) - margin_deg, float(miny) - margin_deg,
            float(maxx) + margin_deg, float(maxy) + margin_deg)


# ----------------------------------------------------------------------
# URL / key construction
# ----------------------------------------------------------------------
def _paths(ts: pd.Timestamp):
    """Return (aws_s3_key, iowa_state_https_url) for one timestamp."""
    ts = pd.Timestamp(ts)
    d = ts.strftime("%Y%m%d")
    hms = ts.strftime("%H%M%S")
    aws = (f"noaa-mrms-pds/CONUS/PrecipRate_00.00/{d}/"
           f"MRMS_PrecipRate_00.00_{d}-{hms}.grib2.gz")
    isu = (f"https://mtarchive.geol.iastate.edu/{ts:%Y/%m/%d}/mrms/ncep/"
           f"PrecipRate/PrecipRate_00.00_{d}-{hms}.grib2.gz")
    return aws, isu


_CONFIRMED_MISSING = object()  # absent in BOTH archives (real gap), not a hiccup


def _fetch_bytes(ts, fs):
    """
    bytes              -> the raw .grib2.gz
    _CONFIRMED_MISSING -> 404 in both archives
    None               -> transient error (retry next run)
    Pass fs=None to skip S3 and use Iowa State only.
    """
    aws_key, isu_url = _paths(ts)
    aws_missing = isu_missing = False

    if fs is not None:
        try:
            return fs.cat_file(aws_key)
        except FileNotFoundError:
            aws_missing = True
        except Exception:
            pass  # transient -> fall through to Iowa State

    try:
        req = urllib.request.Request(isu_url, headers={"User-Agent": "mrms-bbox-dl"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            isu_missing = True
    except Exception:
        pass

    if aws_missing and isu_missing:
        return _CONFIRMED_MISSING
    return None


# ----------------------------------------------------------------------
# Decode + subset (in memory; CONUS never hits disk)
# ----------------------------------------------------------------------
try:
    import eccodes
    _HAVE_ECCODES = True
except Exception:  # pragma: no cover
    _HAVE_ECCODES = False


def _decode_precip_eccodes(raw_gz: bytes) -> xr.DataArray:
    """
    Fast path: decompress in memory and decode with the low-level eccodes API
    directly (codes_new_from_message -> codes_get_values), skipping cfgrib's
    dataset-building machinery entirely and never touching disk. This is the
    ~12 files/s decoder; cfgrib's xr.open_dataset() does a lot of extra work
    (temp file write, coordinate/index inference across the whole message)
    that isn't needed for a single-variable, single-message PrecipRate file.
    """
    grib = gzip.decompress(raw_gz)
    gid = eccodes.codes_new_from_message(grib)
    try:
        ni = eccodes.codes_get(gid, "Ni")
        nj = eccodes.codes_get(gid, "Nj")
        lat1 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
        lon1 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
        di = eccodes.codes_get(gid, "iDirectionIncrementInDegrees")
        dj = eccodes.codes_get(gid, "jDirectionIncrementInDegrees")
        i_neg = bool(eccodes.codes_get(gid, "iScansNegatively"))
        j_pos = bool(eccodes.codes_get(gid, "jScansPositively"))
        missing_value = eccodes.codes_get(gid, "missingValue")
        values = eccodes.codes_get_values(gid).astype("float32")
    finally:
        eccodes.codes_release(gid)

    values = values.reshape(nj, ni)
    values[values == missing_value] = np.nan

    lon = lon1 + np.arange(ni, dtype="float64") * di * (-1.0 if i_neg else 1.0)
    lat = lat1 + np.arange(nj, dtype="float64") * dj * (1.0 if j_pos else -1.0)

    return xr.DataArray(
        values, dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        name="precip_rate",
    )


def _decode_precip_cfgrib(raw_gz: bytes) -> xr.DataArray:
    """Slow-but-robust fallback: same as v1/v2 -- writes a temp file, decodes
    via xr.open_dataset(engine="cfgrib"). Used only if the fast eccodes path
    raises (unexpected grid type, missing keys, etc.)."""
    grib = gzip.decompress(raw_gz)
    fd, tmp = tempfile.mkstemp(suffix=".grib2")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(grib)
        with xr.open_dataset(tmp, engine="cfgrib",
                             backend_kwargs={"indexpath": ""}) as ds:
            ds = ds.load()
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    var = list(ds.data_vars)[0]
    da = ds[var].rename("precip_rate")
    drop = [c for c in da.coords if c not in ("latitude", "longitude")]
    return da.drop_vars(drop, errors="ignore")


def _decode_precip(raw_gz: bytes) -> xr.DataArray:
    """Decompress a .grib2.gz in memory, return PrecipRate as a 2-D DataArray.
    Tries the fast eccodes path first, falls back to cfgrib on any failure."""
    if _HAVE_ECCODES:
        try:
            return _decode_precip_eccodes(raw_gz)
        except Exception:
            pass  # fall through to cfgrib
    return _decode_precip_cfgrib(raw_gz)


def _subset_bbox(da: xr.DataArray, bbox) -> xr.DataArray:
    """Cut to bbox=(lon_min,lat_min,lon_max,lat_max); handles 0..360 lon + ordering."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lon = da["longitude"]
    lat = da["latitude"]
    if float(lon.max()) > 180.0:
        lon_min = lon_min % 360.0
        lon_max = lon_max % 360.0
    lat_asc = float(lat[0]) < float(lat[-1])
    lon_asc = float(lon[0]) < float(lon[-1])
    lat_slice = slice(lat_min, lat_max) if lat_asc else slice(lat_max, lat_min)
    lon_slice = slice(lon_min, lon_max) if lon_asc else slice(lon_max, lon_min)
    return da.sel(latitude=lat_slice, longitude=lon_slice)


# ----------------------------------------------------------------------
# Per-FILE worker  (download -> decode -> subset, no disk write)
# ----------------------------------------------------------------------
def _process_one(ts, bbox, fs, mask_negative=True):
    """
    ("ok", ts, da)        -> small bbox DataArray with a time dim
    ("missing", ts, None) -> confirmed archive gap
    ("error", ts, None)   -> transient/decoding failure (retry next run)
    """
    raw = _fetch_bytes(ts, fs)
    if raw is _CONFIRMED_MISSING:
        return ("missing", ts, None)
    if not isinstance(raw, (bytes, bytearray)):
        return ("error", ts, None)
    try:
        da = _decode_precip(raw)
        da = _subset_bbox(da, bbox)
        if mask_negative:
            da = da.where(da >= 0)            # -3 = "no radar coverage" -> NaN
        da = da.expand_dims(time=[np.datetime64(pd.Timestamp(ts))]).astype("float32")
        return ("ok", ts, da)
    except Exception:
        return ("error", ts, None)


# ----------------------------------------------------------------------
# Day shard helpers
# ----------------------------------------------------------------------
def _read_existing(path):
    """Return (set_of_present_times, existing_precip_DataArray_or_None)."""
    if not os.path.exists(path):
        return set(), None
    try:
        with xr.open_dataset(path) as _ds:
            existing = _ds.load()
        return set(pd.to_datetime(existing["time"].values)), existing["precip_rate"]
    except Exception:
        return set(), None


# def _write_day(path, existing_da, new_slices):
#     """Merge existing + new slices, sort, dedup, write atomically. Returns n stored."""
#     parts = []
#     if existing_da is not None:
#         parts.append(existing_da)
#     if new_slices:
#         parts.append(xr.concat(new_slices, dim="time"))
#     if not parts:
#         return 0
#     da_all = xr.concat(parts, dim="time").sortby("time").drop_duplicates("time")
#     enc = {"precip_rate": {"zlib": True, "complevel": 4, "dtype": "float32"}}
#     tmp = path + ".tmp"
#     da_all.to_dataset(name="precip_rate").to_netcdf(tmp, encoding=enc)
#     os.replace(tmp, path)
#     return int(da_all["time"].size)

def _write_day(path, existing_da, new_slices):
    """Merge existing + new slices, sort, dedup, write atomically. Returns n stored."""
    parts = []
    if existing_da is not None:
        parts.append(existing_da)
    if new_slices:
        parts.append(xr.concat(new_slices, dim="time"))
    if not parts:
        return 0
    da_all = xr.concat(parts, dim="time").sortby("time").drop_duplicates("time")

    # force a clean pandas datetime axis so NetCDF time-encoding works
    da_all = da_all.assign_coords(
        time=pd.DatetimeIndex(pd.to_datetime(da_all["time"].values)))

    ds_out = da_all.to_dataset(name="precip_rate")
    enc = {
        "precip_rate": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "time": {"units": "minutes since 2000-01-01 00:00:00", "dtype": "float64"},
    }
    tmp = path + ".tmp"
    ds_out.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, path)
    return int(da_all["time"].size)


# ----------------------------------------------------------------------
# Orchestrator  (file-level parallelism + file-level progress)
# ----------------------------------------------------------------------
def _run_pass(days, todo_by_day, bbox, fs, mask_negative, pool, bar, shard_dir):
    """Fetch todo_by_day[d] for each day, write/merge into that day's shard.
    Returns (results_by_day, new_missing_timestamps, transient_fail_by_day)."""
    results, new_missing, fail_by_day = {}, [], {}
    for d in days:
        todo = todo_by_day.get(d) or []
        if not todo:
            continue
        slices, n_missing, n_err, failed_ts = [], 0, 0, []
        futures = {pool.submit(_process_one, t, bbox, fs, mask_negative): t for t in todo}
        for fut in as_completed(futures):
            status, ts, da = fut.result()
            if status == "ok":
                slices.append(da)
            elif status == "missing":
                n_missing += 1
                new_missing.append(pd.Timestamp(ts))
            else:
                n_err += 1
                failed_ts.append(pd.Timestamp(ts))
            if bar is not None:
                bar.update(1)
                bar.set_postfix(day=f"{d:%Y-%m-%d}", miss=n_missing, err=n_err)
        path = os.path.join(shard_dir, f"pr_{d:%Y%m%d}.nc")
        _, existing_da = _read_existing(path)
        n_stored = _write_day(path, existing_da, slices)
        results[d] = dict(date=f"{d:%Y-%m-%d}", stored=n_stored, new=len(slices),
                           missing=n_missing, errors=n_err)
        if failed_ts:
            fail_by_day[d] = failed_ts
    return results, new_missing, fail_by_day


def build_store(unique_times, bbox, out_dir, max_workers=12,
                mask_negative=True, use_aws=True, verbose=True, max_retries=3):
    """
    Download + bbox-subset every timestamp, writing one NetCDF per UTC day under
    <out_dir>/shards/. File-level progress bar. Resumable: rerun to fill gaps;
    confirmed-missing timestamps (404 in both archives) are remembered and
    skipped next time. Transient failures (timeouts, decode errors -- NOT a
    confirmed absence) are retried in-process up to max_retries times; any
    still failing after that are logged to transient_errors.csv, not treated
    as missing, and will be retried again on the next build_store() call.
    """
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    missing_csv = os.path.join(out_dir, "known_missing.csv")
    manifest_csv = os.path.join(out_dir, "build_manifest.csv")
    transient_csv = os.path.join(out_dir, "transient_errors.csv")

    times = pd.DatetimeIndex(pd.to_datetime(list(unique_times))).sort_values()

    if os.path.exists(missing_csv):
        known = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(missing_csv)["time"]))
        n0 = len(times)
        times = times.difference(known)
        if verbose and n0 != len(times):
            print(f"Skipping {n0 - len(times)} timestamps already known missing.")

    by_day = {}
    for t in times:
        by_day.setdefault(t.normalize(), []).append(t)
    days = sorted(by_day)

    # Pre-scan existing shards so the progress total = files actually needing work.
    todo_by_day, existing_by_day, already = {}, {}, 0
    for d in days:
        path = os.path.join(shard_dir, f"pr_{d:%Y%m%d}.nc")
        have, existing_da = _read_existing(path)
        todo = [t for t in by_day[d] if pd.Timestamp(t) not in have]
        todo_by_day[d] = todo
        existing_by_day[d] = existing_da
        already += len(have & set(by_day[d]))
    total_todo = sum(len(v) for v in todo_by_day.values())

    if verbose:
        print(f"{len(times)} timestamps across {len(days)} UTC days | "
              f"{already} already stored, {total_todo} to fetch -> {shard_dir}")
    if total_todo == 0:
        if verbose:
            print("Nothing to do -- all timestamps already present.")
        return dict(days=len(days), to_fetch=0, stored_new=0,
                    confirmed_missing=0, transient_errors=0, shard_dir=shard_dir)

    fs = None
    if use_aws and s3fs is not None:
        try:
            fs = s3fs.S3FileSystem(anon=True)
        except Exception as e:
            if verbose:
                print(f"AWS unavailable ({e}); Iowa State only.")

    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=total_todo, desc="files", unit="file", disable=not verbose)
    except Exception:
        bar = None

    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        results, new_missing, fail_by_day = _run_pass(
            days, todo_by_day, bbox, fs, mask_negative, pool, bar, shard_dir)

        retry_round = 0
        while fail_by_day and retry_round < max_retries:
            retry_round += 1
            n_retry = sum(len(v) for v in fail_by_day.values())
            if verbose:
                print(f"\nRetry {retry_round}/{max_retries}: {n_retry} transient failure(s)...")
            retry_days = sorted(fail_by_day)
            retry_bar = None
            if bar is not None:
                try:
                    from tqdm.auto import tqdm
                    retry_bar = tqdm(total=n_retry, desc=f"retry {retry_round}", unit="file", disable=not verbose)
                except Exception:
                    pass
            r_results, r_missing, fail_by_day = _run_pass(
                retry_days, fail_by_day, bbox, fs, mask_negative, pool, retry_bar, shard_dir)
            if retry_bar is not None:
                retry_bar.close()
            new_missing.extend(r_missing)
            for d, r in r_results.items():
                results[d]["stored"] = r["stored"]
                results[d]["new"] += r["new"]
                results[d]["missing"] += r["missing"]
                results[d]["errors"] = r["errors"]
    finally:
        pool.shutdown(wait=True)
        if bar is not None:
            bar.close()

    if new_missing:
        prev = []
        if os.path.exists(missing_csv):
            prev = list(pd.to_datetime(pd.read_csv(missing_csv)["time"]))
        allm = pd.DatetimeIndex(pd.to_datetime(prev + new_missing)).unique().sort_values()
        pd.DataFrame({"time": allm}).to_csv(missing_csv, index=False)

    persistent_fails = sorted(ts for tss in fail_by_day.values() for ts in tss)
    if persistent_fails:
        pd.DataFrame({"time": persistent_fails}).to_csv(transient_csv, index=False)
    elif os.path.exists(transient_csv):
        os.remove(transient_csv)

    pd.DataFrame(list(results.values())).to_csv(manifest_csv, index=False)

    summary = dict(
        days=len(days),
        to_fetch=int(total_todo),
        stored_new=int(sum(r["new"] for r in results.values())),
        confirmed_missing=int(len(new_missing)),
        transient_errors=int(len(persistent_fails)),
        shard_dir=shard_dir,
        manifest=manifest_csv,
    )
    if verbose:
        print("Done:", summary)
        if summary["transient_errors"]:
            print(f"  -> {summary['transient_errors']} still failing after {max_retries} retries, "
                  f"logged to {transient_csv} (will retry again next build_store() call).")
    return summary


# ----------------------------------------------------------------------
# Consolidation (optional)
# ----------------------------------------------------------------------
def consolidate(shard_dir, out_path, fmt="netcdf"):
    """Stack all pr_*.nc day shards into one store along time (streams via dask)."""
    import glob
    files = sorted(glob.glob(os.path.join(shard_dir, "pr_*.nc")))
    if not files:
        raise FileNotFoundError(f"No day shards found in {shard_dir}")
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="time",
                           engine="netcdf4", chunks={"time": 720}).sortby("time")
    if fmt == "zarr":
        ds.to_zarr(out_path, mode="w")
    elif fmt == "netcdf":
        enc = {"precip_rate": {"zlib": True, "complevel": 4, "dtype": "float32"}}
        ds.to_netcdf(out_path, encoding=enc)
    else:
        raise ValueError("fmt must be 'zarr' or 'netcdf'")
    ds.close()
    print(f"Consolidated {len(files)} day files -> {out_path}")
    return out_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _load_times(path):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if df.shape[1] == 1:
        col = df.columns[0]
    else:
        cand = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        col = cand[0] if cand else df.columns[0]
    return pd.DatetimeIndex(pd.to_datetime(df[col])).unique()


def _main():
    p = argparse.ArgumentParser(description="Stream MRMS PrecipRate, subset to a bbox.")
    p.add_argument("--times", required=True, help="parquet/csv of unique UTC timestamps")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--bbox", nargs=4, type=float,
                   metavar=("LONMIN", "LATMIN", "LONMAX", "LATMAX"))
    p.add_argument("--huc8-bounds", help="vector file whose total bounds define the bbox")
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--no-aws", action="store_true")
    p.add_argument("--consolidate", choices=["zarr", "netcdf"])
    args = p.parse_args()

    if args.bbox:
        bbox = tuple(args.bbox)
    elif args.huc8_bounds:
        import geopandas as gpd
        bbox = bbox_from_huc8(gpd.read_file(args.huc8_bounds), margin_deg=args.margin)
    else:
        p.error("provide either --bbox or --huc8-bounds")

    print("bbox (lonmin,latmin,lonmax,latmax):", bbox)
    times = _load_times(args.times)
    build_store(times, bbox, args.out, max_workers=args.workers, use_aws=not args.no_aws)
    if args.consolidate:
        ext = "zarr" if args.consolidate == "zarr" else "nc"
        consolidate(os.path.join(args.out, "shards"),
                    os.path.join(args.out, f"precip_rate.{ext}"), fmt=args.consolidate)


if __name__ == "__main__":
    _main()