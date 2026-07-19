r"""Assign hydrofabric catchment (divide) IDs to USGS gages by lat/lon.

Snaps each gage point to the nearest hydrofabric flowpath and maps that
flowpath to its containing divide, adding a `gage_cat-id` column
(cat-XXXXX) to the input CSV. This is the same snap-to-flowpath logic
used in event_pipeline.py (read_and_snap_gages), reused here for CSVs
that only carry STAID + lat/lon and no longer carry gage_cat-id directly.

The output CSV can be fed straight into extract_hf.py via
`--csv-column gage_cat-id`.

Usage
-----
    python engine/geo/assign_gage_catchment.py \\
        --csv /path/to/events.csv \\
        --gpkg /path/to/conus_nextgen.gpkg \\
        --staid-col STAID --lat-col gage_lat --lon-col gage_lon \\
        --output /path/to/events_with_cat_id.csv
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from flash_preprocess.paths import HYDROFABRIC_GPKG

log = logging.getLogger('AssignGageCatchment')


def read_hydrofabric(hydrofabric_gpkg: str) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Read only the hydrofabric layers/columns needed for gage snapping."""
    gpkg = Path(hydrofabric_gpkg)
    if not gpkg.exists():
        raise FileNotFoundError(f'Hydrofabric GeoPackage not found: {gpkg}')

    log.info('Reading hydrofabric: %s', gpkg)
    network = gpd.read_file(
        gpkg,
        layer='network',
        columns=['id', 'toid', 'divide_id', 'vpuid'],
        read_geometry=False,
    )
    flowpaths = gpd.read_file(
        gpkg,
        layer='flowpaths',
        columns=['id', 'geometry'],
    )
    network['id'] = network['id'].astype(str)
    network['divide_id'] = network['divide_id'].astype(str)
    flowpaths['id'] = flowpaths['id'].astype(str)
    return network, flowpaths


def assign_gage_catchments(
    gages: pd.DataFrame,
    network: pd.DataFrame,
    flowpaths: gpd.GeoDataFrame,
    staid_col: str,
    lat_col: str,
    lon_col: str,
) -> pd.DataFrame:
    """Snap gages to the nearest flowpath and map to `gage_cat-id`."""
    gages = gages.copy()
    gages[staid_col] = gages[staid_col].astype(str).str.strip().str.zfill(8)

    gages_gdf = gpd.GeoDataFrame(
        gages,
        geometry=gpd.points_from_xy(gages[lon_col], gages[lat_col]),
        crs='EPSG:4326',
    ).to_crs(flowpaths.crs)

    snapped = gpd.sjoin_nearest(
        gages_gdf,
        flowpaths[['id', 'geometry']],
        how='left',
        distance_col='gage_snap_dist_m',
    ).rename(columns={'id': 'gage_flowpath-id'})

    snapped['gage_flowpath-id'] = snapped['gage_flowpath-id'].astype(str)

    flowpath_to_divide = (
        network[['id', 'divide_id']]
        .dropna()
        .drop_duplicates(subset='id')
        .set_index('id')['divide_id']
    )
    snapped['gage_cat-id'] = snapped['gage_flowpath-id'].map(flowpath_to_divide)

    matched = snapped['gage_cat-id'].notna().sum()
    log.info('Gages matched to catchments: %d / %d', matched, len(snapped))

    return pd.DataFrame(snapped.drop(columns=['geometry'], errors='ignore'))


def main():
    """Parse CLI args and run gage-to-catchment assignment."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--csv',
        type=Path,
        required=True,
        help='CSV with gage STAID + lat/lon',
    )
    parser.add_argument(
        '--gpkg',
        type=Path,
        default=HYDROFABRIC_GPKG,
        help='Path to conus_nextgen.gpkg (default: config.yaml hydrofabric_gpkg)',
    )
    parser.add_argument(
        '--staid-col',
        default='STAID',
        help='STAID column name (default: %(default)s)',
    )
    parser.add_argument(
        '--lat-col',
        default='gage_lat',
        help='Latitude column name (default: %(default)s)',
    )
    parser.add_argument(
        '--lon-col',
        default='gage_lon',
        help='Longitude column name (default: %(default)s)',
    )
    parser.add_argument('--output', type=Path, required=True, help='Output CSV path')
    args = parser.parse_args()

    gages = pd.read_csv(args.csv, dtype={args.staid_col: str}, low_memory=False)
    network, flowpaths = read_hydrofabric(str(args.gpkg))

    out = assign_gage_catchments(
        gages,
        network,
        flowpaths,
        args.staid_col,
        args.lat_col,
        args.lon_col,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    log.info('Wrote %s', args.output)


if __name__ == '__main__':
    main()
