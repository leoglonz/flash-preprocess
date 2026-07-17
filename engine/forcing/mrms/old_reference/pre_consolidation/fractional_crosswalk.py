import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

MRMS_LON_MIN_EDGE = -130.0
MRMS_LAT_MAX_EDGE = 55.0
MRMS_RES = 0.01
MRMS_NLON = 7000
MRMS_NLAT = 3500
HALF = MRMS_RES / 2.0
AREA_CRS = 5070
PAD_CELLS = 2


def _lon_from_col(col):
    return MRMS_LON_MIN_EDGE + HALF + col * MRMS_RES


def _lat_from_row(row):
    return MRMS_LAT_MAX_EDGE - HALF - row * MRMS_RES


def _cell_id(row, col):
    return row.astype(np.int64) * MRMS_NLON + col.astype(np.int64)


def _mrms_cell_polys(rows, cols):
    rows, cols = np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)
    lon, lat = _lon_from_col(cols), _lat_from_row(rows)
    gdf = gpd.GeoDataFrame(
        {"cell_id": _cell_id(rows, cols), "row": rows, "col": cols, "lon": lon, "lat": lat,
         "geometry": [box(x - HALF, y - HALF, x + HALF, y + HALF) for x, y in zip(lon, lat)]},
        geometry="geometry", crs="EPSG:4326",
    ).to_crs(AREA_CRS)
    gdf["cell_area_m2"] = gdf.geometry.area
    return gdf


def _rows_cols_for_bbox(west, south, east, north, pad=2):
    i0 = max(0, int(np.floor((west - MRMS_LON_MIN_EDGE - HALF) / MRMS_RES)) - pad)
    i1 = min(MRMS_NLON - 1, int(np.ceil((east - MRMS_LON_MIN_EDGE - HALF) / MRMS_RES)) + pad)
    j0 = max(0, int(np.floor((MRMS_LAT_MAX_EDGE - HALF - north) / MRMS_RES)) - pad)
    j1 = min(MRMS_NLAT - 1, int(np.ceil((MRMS_LAT_MAX_EDGE - HALF - south) / MRMS_RES)) + pad)
    if i1 < i0 or j1 < j0:
        return pd.DataFrame(columns=["row", "col"])
    COL, ROW = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
    return pd.DataFrame({"row": ROW.ravel().astype(np.int32), "col": COL.ravel().astype(np.int32)})


def _add_neighbors(df, pad=2):
    base = df[["row", "col"]].drop_duplicates()
    pieces = []
    for dr in range(-pad, pad + 1):
        for dc in range(-pad, pad + 1):
            t = base.copy()
            t["row"] += dr
            t["col"] += dc
            pieces.append(t)
    out = pd.concat(pieces, ignore_index=True)
    out = out[out["row"].between(0, MRMS_NLAT - 1) & out["col"].between(0, MRMS_NLON - 1)]
    return out.drop_duplicates().reset_index(drop=True)


def _intersection_area(candidates, chunk=150_000):
    areas = []
    for s in range(0, len(candidates), chunk):
        c = candidates.iloc[s:s + chunk]
        cat_geoms = gpd.GeoSeries(c["catchment_geometry"], crs=AREA_CRS, index=c.index)
        areas.append(c.geometry.intersection(cat_geoms).area.to_numpy())
    return np.concatenate(areas)


def build_fractional_crosswalk(divide_ids, catchments_master: gpd.GeoDataFrame,
                                crosswalk: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    # No top-level "combined" shortcut cache: this function is called once per
    # VPU group (see run_pipeline.py), so the per-VPU sub-cache below (keyed by
    # VPU code, collision-free across concurrent VPU runs) already covers it --
    # a shared combined-file cache would just be a race target with no benefit.
    cache_dir = Path(cache_dir)
    sub = catchments_master[catchments_master["divide_id"].isin(divide_ids)].to_crs(AREA_CRS).copy()
    sub["geometry"] = sub.geometry.buffer(0)
    sel_cw = crosswalk[crosswalk["divide_id"].isin(divide_ids)]

    vpu_cache = cache_dir / "fractional_vpu_cache"
    vpu_cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for vpu in sorted(sub["vpuid"].dropna().astype(str).unique()):
        vsub = sub[sub["vpuid"].astype(str) == vpu]
        divide_ids_vpu = vsub["divide_id"].dropna().unique()

        out_file = vpu_cache / f"vpu_{vpu}.parquet"
        if out_file.exists():
            cached_vpu = pd.read_parquet(out_file)
            if set(divide_ids_vpu) <= set(cached_vpu["divide_id"]):
                parts.append(cached_vpu[cached_vpu["divide_id"].isin(divide_ids_vpu)])
                continue

        cw_vpu = sel_cw[sel_cw["divide_id"].isin(divide_ids_vpu)][["divide_id", "cell_id", "row", "col"]]
        candidate_cells = _add_neighbors(cw_vpu[["row", "col"]], pad=PAD_CELLS)

        has_center = set(cw_vpu["divide_id"].dropna().unique())
        missing = vsub[~vsub["divide_id"].isin(has_center)]
        if len(missing):
            extra = [_rows_cols_for_bbox(*geom.bounds, pad=PAD_CELLS) for geom in missing.to_crs(4326).geometry]
            candidate_cells = pd.concat([candidate_cells] + extra, ignore_index=True)
        candidate_cells = candidate_cells.drop_duplicates(["row", "col"]).reset_index(drop=True)
        if len(candidate_cells) == 0:
            continue

        mrms_cells = _mrms_cell_polys(candidate_cells["row"].values, candidate_cells["col"].values)
        join_cols = [c for c in ["divide_id", "vpuid", "nexus_id", "nexus_type", "is_terminal",
                                  "terminal_nexus_id", "geometry"] if c in vsub.columns]
        candidates = gpd.sjoin(mrms_cells, vsub[join_cols], how="inner", predicate="intersects")
        if len(candidates) == 0:
            continue
        candidates = candidates.join(vsub[join_cols].geometry.rename("catchment_geometry"), on="index_right")
        candidates["intersection_area_m2"] = _intersection_area(candidates)
        candidates = candidates[candidates["intersection_area_m2"] > 0.01].copy()
        candidates["fraction_inside"] = (candidates["intersection_area_m2"] / candidates["cell_area_m2"]).clip(0, 1)

        keep = ["cell_id", "row", "col", "lon", "lat", "divide_id", "vpuid", "cell_area_m2",
                "intersection_area_m2", "fraction_inside"]
        df = (pd.DataFrame(candidates[keep]).drop_duplicates(["divide_id", "cell_id"])
              .sort_values(["divide_id", "cell_id"]).reset_index(drop=True))
        df["weight"] = df["fraction_inside"] / df.groupby("divide_id")["fraction_inside"].transform("sum")
        tmp = f"{out_file}.tmp{os.getpid()}"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out_file)
        parts.append(df)

    result = pd.concat(parts, ignore_index=True).drop_duplicates(["divide_id", "cell_id"]).reset_index(drop=True)
    result["weight"] = result["fraction_inside"] / result.groupby("divide_id")["fraction_inside"].transform("sum")
    result["lon_360"] = result["lon"] % 360.0
    return result
