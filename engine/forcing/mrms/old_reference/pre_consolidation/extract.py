from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm


def open_shards(shard_dir: Path) -> xr.Dataset:
    files = sorted(Path(shard_dir).glob("pr_*.nc"))
    if not files:
        raise FileNotFoundError(f"no day shards in {shard_dir}")
    return xr.open_mfdataset([str(f) for f in files], combine="nested", concat_dim="time",
                              engine="netcdf4", data_vars="minimal", coords="minimal",
                              compat="override").sortby("time")


def storm_catchment_rate_2min(ds: xr.Dataset, win_start, win_end, divide_ids, frac_cw: pd.DataFrame):
    window = ds["precip_rate"].sel(time=slice(win_start, win_end))
    if window.sizes.get("time", 0) == 0:
        return None
    cw = frac_cw[frac_cw["divide_id"].isin(divide_ids)]
    if len(cw) == 0:
        return None

    pts = window.sel(latitude=xr.DataArray(cw["lat"].values, dims="cell"),
                      longitude=xr.DataArray(cw["lon_360"].values, dims="cell"), method="nearest")
    values = pts.values  # (time, cell)
    times = pd.DatetimeIndex(window["time"].values)

    # area-weighted catchment mean via sparse matmul, renormalized per-timestep
    # over non-NaN cells only (same math as the old per-timestep groupby, but
    # ~2 matmuls instead of a pandas groupby over time*cell rows).
    cats = sorted(set(cw["divide_id"]))
    cat_row = {c: i for i, c in enumerate(cats)}
    W = csr_matrix((cw["fraction_inside"].values,
                     (cw["divide_id"].map(cat_row).values, np.arange(len(cw)))),
                    shape=(len(cats), len(cw)))

    valid = ~np.isnan(values)
    num = W @ np.nan_to_num(values, nan=0.0).T   # (n_cat, n_time)
    den = W @ valid.astype(np.float32).T          # (n_cat, n_time)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = (num / den).T
    rate[den.T == 0] = np.nan

    return pd.DataFrame(rate, index=times, columns=cats)


def to_depth_15(rate2min: pd.DataFrame) -> pd.DataFrame:
    rate15 = rate2min.resample("15min", label="left", closed="left").mean()
    return rate15 * 0.25


def extract_all(manifest: pd.DataFrame, event_catchment_windows: pd.DataFrame,
                 frac_cw: pd.DataFrame, shard_dir: Path, out_nc: Path, max_steps: int = 481,
                 zero_precip_threshold_mm: float = 1.0):
    ds = open_shards(shard_dir)
    cats_by_event = event_catchment_windows.groupby("storm_index")["divide_id"].apply(list)

    records, flagged, failed = [], [], []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="extract"):
        sid = row["storm_index"]
        divide_ids = cats_by_event.get(sid)
        if not divide_ids:
            failed.append({"storm_id": sid, "gauge_id": row["recording_gage_STAID"],
                            "reason": "no catchments in event_catchment_windows"})
            continue
        depth15 = storm_catchment_rate_2min(ds, row["win_start"], row["win_end"], divide_ids, frac_cw)
        if depth15 is None or depth15.empty:
            reason = "no fractional_crosswalk match" if frac_cw[frac_cw["divide_id"].isin(divide_ids)].empty \
                else "no shard data for window"
            failed.append({"storm_id": sid, "gauge_id": row["recording_gage_STAID"], "reason": reason})
            continue
        depth15 = to_depth_15(depth15).iloc[:max_steps]
        vals = depth15.values.astype("float32")
        records.append({"storm_id": int(sid), "divide_ids": list(depth15.columns),
                         "n_steps": len(depth15), "ts_start": depth15.index[0],
                         "ts_end": depth15.index[-1], "depth": vals})

        has_data = ~np.isnan(vals).all(axis=0)
        basin_mean_total = float(np.nanmean(np.nansum(vals[:, has_data], axis=0))) if has_data.any() else np.nan
        if np.isnan(basin_mean_total) or basin_mean_total < zero_precip_threshold_mm:
            flagged.append({"storm_id": int(sid), "gauge_id": row["recording_gage_STAID"],
                             "ts_start": depth15.index[0], "ts_end": depth15.index[-1],
                             "n_catchments": len(divide_ids), "n_catchments_with_data": int(has_data.sum()),
                             "basin_mean_total_mm": basin_mean_total})
    ds.close()
    if not records:
        raise RuntimeError("no events extracted")

    if flagged:
        flagged_csv = Path(out_nc).with_name("zero_precip_events.csv")
        pd.DataFrame(flagged).sort_values("ts_start").to_csv(flagged_csv, index=False)
        print(f"WARNING: {len(flagged)} / {len(records)} events have basin-mean total precip "
              f"< {zero_precip_threshold_mm} mm over their full window -- logged to {flagged_csv}")
    if failed:
        failed_csv = Path(out_nc).with_name("extraction_failed_events.csv")
        pd.DataFrame(failed).to_csv(failed_csv, index=False)
        print(f"WARNING: {len(failed)} events could not be extracted at all -- logged to {failed_csv}")

    all_cats = sorted(set().union(*[set(r["divide_ids"]) for r in records]))
    cat_idx = {c: j for j, c in enumerate(all_cats)}
    n_ev, n_cat = len(records), len(all_cats)

    Path(out_nc).parent.mkdir(parents=True, exist_ok=True)
    nc = netCDF4.Dataset(out_nc, "w", format="NETCDF4")
    nc.createDimension("event", n_ev)
    nc.createDimension("time_step", max_steps)
    nc.createDimension("catchment", n_cat)

    nc.createVariable("storm_id", "i4", ("event",))[:] = np.array([r["storm_id"] for r in records], dtype=np.int32)
    nc.createVariable("n_steps", "i4", ("event",))[:] = np.array([r["n_steps"] for r in records], dtype=np.int32)

    epoch = np.datetime64("1970-01-01T00:00", "m")
    v_ts = nc.createVariable("ts_start", "f8", ("event",))
    v_ts.units = "minutes since 1970-01-01 00:00:00 UTC"
    v_ts[:] = [(np.datetime64(r["ts_start"]) - epoch) / np.timedelta64(1, "m") for r in records]
    v_te = nc.createVariable("ts_end", "f8", ("event",))
    v_te.units = "minutes since 1970-01-01 00:00:00 UTC"
    v_te[:] = [(np.datetime64(r["ts_end"]) - epoch) / np.timedelta64(1, "m") for r in records]

    v_cat = nc.createVariable("divide_id", str, ("catchment",))
    v_cat[:] = np.array(all_cats, dtype=object)

    v_p = nc.createVariable("P", "f4", ("event", "time_step", "catchment"), fill_value=np.nan,
                             zlib=True, complevel=4, chunksizes=(min(n_ev, 16), max_steps, min(n_cat, 64)))
    v_p.units = "mm [15 min]-1"
    v_p.long_name = "MRMS precipitation depth"

    for i, r in enumerate(records):
        cols = [cat_idx[c] for c in r["divide_ids"]]
        n = r["n_steps"]
        v_p[i, :n, cols] = r["depth"]

    nc.close()
    print(f"wrote {out_nc}: {n_ev} events x {max_steps} steps x {n_cat} catchments")
