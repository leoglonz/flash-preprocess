import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

MRMS_LON_MIN_EDGE = -130.0
MRMS_LAT_MAX_EDGE = 55.0
MRMS_RES = 0.01
MRMS_NLON = 7000
MRMS_NLAT = 3500
HALF = MRMS_RES / 2.0

CARRY = ["divide_id", "vpuid", "nexus_id", "nexus_type", "is_terminal", "geometry"]


def mrms_centers_for_bbox(west, east, south, north, pad=1):
    i0 = max(0, int(np.floor((west - MRMS_LON_MIN_EDGE - HALF) / MRMS_RES)) - pad)
    i1 = min(MRMS_NLON - 1, int(np.ceil((east - MRMS_LON_MIN_EDGE - HALF) / MRMS_RES)) + pad)
    j0 = max(0, int(np.floor((MRMS_LAT_MAX_EDGE - HALF - north) / MRMS_RES)) - pad)
    j1 = min(MRMS_NLAT - 1, int(np.ceil((MRMS_LAT_MAX_EDGE - HALF - south) / MRMS_RES)) + pad)
    if i1 < i0 or j1 < j0:
        return None
    cols, rows = np.arange(i0, i1 + 1), np.arange(j0, j1 + 1)
    LON, LAT = np.meshgrid(MRMS_LON_MIN_EDGE + HALF + cols * MRMS_RES,
                            MRMS_LAT_MAX_EDGE - HALF - rows * MRMS_RES)
    COL, ROW = np.meshgrid(cols, rows)
    return ROW.ravel(), COL.ravel(), LON.ravel(), LAT.ravel()


def build_crosswalk(catchments_master: gpd.GeoDataFrame, cache_dir: Path) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    combined = cache_dir / "mrms_hf_crosswalk_conus.parquet"
    if combined.exists():
        return pd.read_parquet(combined)

    cache_dir.mkdir(parents=True, exist_ok=True)
    vpus = sorted(catchments_master["vpuid"].dropna().unique())
    parts = []
    t0 = time.time()
    for vpu in vpus:
        out = cache_dir / f"crosswalk_vpu_{vpu}.parquet"
        if out.exists():
            parts.append(pd.read_parquet(out))
            continue
        sub = catchments_master[catchments_master["vpuid"] == vpu]
        west, south, east, north = sub.to_crs(4326).total_bounds
        res = mrms_centers_for_bbox(west, east, south, north)
        if res is None:
            continue
        row, col, lon, lat = res
        pts = gpd.GeoDataFrame(
            {"cell_id": row.astype(np.int64) * MRMS_NLON + col.astype(np.int64),
             "row": row.astype(np.int32), "col": col.astype(np.int32), "lon": lon, "lat": lat},
            geometry=gpd.points_from_xy(lon, lat), crs=4326,
        ).to_crs(5070)
        joined = gpd.sjoin(pts, sub[CARRY], how="inner", predicate="within")
        df = pd.DataFrame(joined.drop(columns=["geometry", "index_right"])).drop_duplicates("cell_id")
        tmp = f"{out}.tmp{os.getpid()}"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        parts.append(df)
    crosswalk = pd.concat(parts, ignore_index=True).drop_duplicates("cell_id")
    tmp = f"{combined}.tmp{os.getpid()}"
    crosswalk.to_parquet(tmp, index=False)
    os.replace(tmp, combined)
    print(f"crosswalk built in {time.time() - t0:.0f}s: {len(crosswalk):,} cells")
    return crosswalk
