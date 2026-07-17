import os
import pickle
from pathlib import Path

import pandas as pd

from flash_preprocess.utils import build_upstream_graph, expand_upstream
from hydrofabric import GPKG_PATH

WINDOW_DAYS = 5.0
HALF = pd.to_timedelta(WINDOW_DAYS / 2, unit="D")


def _atomic_write_bytes(path: Path, data: bytes):
    tmp = f"{path}.tmp{os.getpid()}"
    Path(tmp).write_bytes(data)
    os.replace(tmp, path)  # shared cache; multiple concurrent VPU runs may race to build this


def _upstream_graph(cache_dir: Path) -> dict:
    f = Path(cache_dir) / "upstream_graph.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    graph = build_upstream_graph(str(GPKG_PATH))
    _atomic_write_bytes(f, pickle.dumps(graph))
    return graph


def build_manifest(events: pd.DataFrame, cache_dir: Path, tag: str = ""):
    cache_dir = Path(cache_dir)
    suffix = f"_{tag}" if tag else ""
    f_manifest = cache_dir / f"manifest_out{suffix}.parquet"
    f_windows = cache_dir / f"event_catchment_windows{suffix}.parquet"
    if f_manifest.exists() and f_windows.exists():
        return pd.read_parquet(f_manifest), pd.read_parquet(f_windows)

    graph = _upstream_graph(cache_dir)
    upstream_cache: dict[str, set] = {}

    def upstream_cats_of(cat_id):
        if cat_id not in upstream_cache:
            upstream_cache[cat_id] = expand_upstream({cat_id}, graph)
        return upstream_cache[cat_id]

    ev = events.copy()
    ev["begin_time"] = pd.to_datetime(ev["BEGIN_DATE_TIME"], errors="coerce", utc=True).dt.tz_localize(None)
    ev["end_time"] = pd.to_datetime(ev["END_DATE_TIME"], errors="coerce", utc=True).dt.tz_localize(None)
    ev["midpoint"] = ev["begin_time"] + (ev["end_time"] - ev["begin_time"]) / 2
    ev["win_start"] = ev["midpoint"] - HALF
    ev["win_end"] = ev["midpoint"] + HALF

    rows = []
    for _, s in ev.iterrows():
        cats = upstream_cats_of(s["gage_cat-id"])
        if not cats:
            continue
        rows.append({
            "storm_index": str(s["event_id"]), "recording_gage_STAID": str(s["STAID"]).zfill(8),
            "n_catchments": len(cats), "win_start": s["win_start"], "win_end": s["win_end"],
            "_divide_ids": sorted(cats),
        })
    manifest = pd.DataFrame(rows)
    event_catchment_windows = (
        manifest.explode("_divide_ids").rename(columns={"_divide_ids": "divide_id"})
        .dropna(subset=["divide_id"])
        [["storm_index", "recording_gage_STAID", "divide_id", "win_start", "win_end"]]
        .reset_index(drop=True)
    )
    manifest_out = manifest.drop(columns="_divide_ids")

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_out.to_parquet(f_manifest, index=False)
    event_catchment_windows.to_parquet(f_windows, index=False)
    return manifest_out, event_catchment_windows
