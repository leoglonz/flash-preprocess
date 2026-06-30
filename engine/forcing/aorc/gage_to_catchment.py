"""Verify every USGS gage in a CSV is associated with a hydrofabric catchment.

For each gage (identified by LAT_GAGE / LNG_GAGE):
  1. Point-in-polygon spatial join against hydrofabric divides → found_divide_id
  2. If the CSV already has a gage_divide_id, compare to the spatial result
  3. Report: matched, missing (gage outside all divides), and mismatched

Usage:
    python gage_to_catchment.py
    python gage_to_catchment.py --events /path/to/events.csv --out fixed_events.csv
"""

import argparse
import sqlite3

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


# CSV with gage events and coordinates (LAT_GAGE, LNG_GAGE) for each STAID.
EVENTS_CSV = "/Users/leoglonz/Desktop/noaa/flash-preprocess/data/huc8_03020201_events_and_gages.csv"


# Default path where hydrofabric v2.2 geopackage is stored.
HF_PATH_DEFAULT = "/Users/leoglonz/.ngiab/hydrofabric/v2.2/conus_nextgen.gpkg"
HF_CRS = "EPSG:5070"  # NAD83 / Conus Albers — native CRS of the hydrofabric


def load_divides_bbox(hydrofabric_path: str, gages_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Load only the divides that overlap the gage bounding box."""
    bounds = tuple(gages_gdf.to_crs(HF_CRS).total_bounds)
    pad = 5_000  # metres padding in Albers
    bbox = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)
    print(f"  Loading divides for bbox ({bbox[0]:.0f}, {bbox[1]:.0f}) → "
          f"({bbox[2]:.0f}, {bbox[3]:.0f}) [EPSG:5070] ...")
    divides = gpd.read_file(hydrofabric_path, layer="divides", bbox=bbox)
    print(f"  {len(divides)} divides loaded")
    return divides[["divide_id", "geometry"]]


def main():
    parser = argparse.ArgumentParser(description="Verify gage → catchment associations.")
    parser.add_argument("--events", default=EVENTS_CSV, help="Events+gages CSV")
    parser.add_argument("--hydrofabric", default=HF_PATH_DEFAULT, help="conus_nextgen.gpkg")
    parser.add_argument("--out", default=None,
                        help="If provided, write a corrected CSV with found_divide_id column")
    args = parser.parse_args()

    # load gages — one row per unique STAID with valid coordinates
    events = pd.read_csv(args.events)
    gages = (
        events
        .dropna(subset=["LAT_GAGE", "LNG_GAGE"])
        .drop_duplicates("STAID")
        .reset_index(drop=True)
    )
    print(f"Events CSV: {len(events)} rows, {events['STAID'].nunique()} unique STAIDs")
    print(f"Gages with coordinates: {len(gages)}")

    gages_gdf = gpd.GeoDataFrame(
        gages,
        geometry=gpd.points_from_xy(gages["LNG_GAGE"], gages["LAT_GAGE"]),
        crs="EPSG:4326",
    )

    # spatial join: point-in-polygon against hydrofabric divides
    divides = load_divides_bbox(args.hydrofabric, gages_gdf)
    gages_proj = gages_gdf.to_crs(HF_CRS)

    joined = gpd.sjoin(gages_proj, divides, how="left", predicate="within")
    joined = joined.drop(columns=["geometry"])
    joined = joined.rename(columns={"divide_id": "found_divide_id"})

    # report
    matched = joined["found_divide_id"].notna()
    missing = ~matched

    print(f"\n{'='*60}")
    print(f"  Matched (gage inside a divide):  {matched.sum():>3}")
    print(f"  Missing (no containing divide):  {missing.sum():>3}")

    if missing.any():
        print("\nMissing gages (no catchment found):")
        cols = ["STAID", "LAT_GAGE", "LNG_GAGE"]
        if "gage_divide_id" in joined.columns:
            cols.append("gage_divide_id")
        print(joined.loc[missing, cols].to_string(index=False))

    if "gage_divide_id" in joined.columns:
        has_existing = joined["gage_divide_id"].notna() & matched
        mismatch = has_existing & (joined["gage_divide_id"] != joined["found_divide_id"])
        if mismatch.any():
            print(f"\nMismatches (CSV divide_id ≠ spatial result):  {mismatch.sum()}")
            print(joined.loc[mismatch, ["STAID", "gage_divide_id", "found_divide_id",
                                        "LAT_GAGE", "LNG_GAGE"]].to_string(index=False))
        else:
            print(f"\nNo mismatches with existing gage_divide_id values.")

    print(f"\nMatched gages:")
    disp_cols = ["STAID", "found_divide_id", "LAT_GAGE", "LNG_GAGE"]
    if "gage_snap_dist_m" in joined.columns:
        disp_cols.append("gage_snap_dist_m")
    print(joined.loc[matched, disp_cols].to_string(index=False))

    # optional: write corrected CSV
    if args.out:
        mapping = joined[["STAID", "found_divide_id"]].dropna(subset=["STAID"])
        out_df = events.merge(mapping, on="STAID", how="left")
        out_df.to_csv(args.out, index=False)
        print(f"\nCorrected CSV written → {args.out}")


if __name__ == "__main__":
    main()
