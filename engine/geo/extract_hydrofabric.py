"""Extract and subset the CONUS NextGen HydroFabric for flash-flood event sims.

Reads divide IDs from 15-min MRMS NetCDF files (or an explicit list), then
subsets HydroFabric to those catchments plus their full upstream
network, derives the river topology DAG, locates USGS gages within the region,
and writes everything needed for dMG runtime
(see https://github.com/mhpi/generic_deltamodel).

Outputs (written to --output-dir):
    divides.gpkg
        Catchment divide geometries + key attributes
    topology.json
        River network DAG:
            {
                "nodes": [int, ...],
                "edges": [[upstream_int, downstream_int], ...],
                "gage_hf": {"USGS_gage_id": outlet_int_divide_id, ...}
            }
    gauges.csv
        USGS gage metadata = STAID, DRAIN_SQKM, divide_id

Usage
-----
    # From 15-min NC files (divide ID auto-detected)
    python extract_hydrofabric.py \\
        --nc-dir ../data/forcing_15min_huc8_all_upstream \\
        --gpkg ~/.ngiab/hydrofabric/v2.2/conus_nextgen.gpkg \\
        --output-dir ../data/hydrofabric_subset

    # From an explicit list of divide IDs
    python extract_hydrofabric.py \\
        --divide-ids cat-251968 cat-251969 cat-251970 \\
        --gpkg ~/.ngiab/hydrofabric/v2.2/conus_nextgen.gpkg \\
        --output-dir ../data/hydrofabric_subset
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger('ExtractHF')


def _cat_to_int(cat_id: str) -> int:
    """Convert 'cat-XXXXX' to integer XXXXX."""
    return int(cat_id.split('-', 1)[1])


def _int_to_cat(n: int) -> str:
    """Convert integer XXXXX to 'cat-XXXXX'."""
    return f'cat-{n}'


def collect_divide_ids_from_nc(nc_dir: Path) -> list[str]:
    """Return sorted unique divide_ids found across all NC files in nc_dir.
    
    Parameters
    ----------
    nc_dir
        Directory containing 15-min MRMS NetCDF files with 'divide_id' coordinate.
    
    Returns
    -------
    list[str]
        List of unique divide IDs (cat-XXXXX strings) found in the NC files.
    """
    ids: set[str] = set()
    files = list(nc_dir.glob('*.nc'))
    if not files:
        raise FileNotFoundError(f'No .nc files found in {nc_dir}')
    for f in files:
        ds = xr.open_dataset(f)
        if 'divide_id' in ds.coords:
            ids.update(ds['divide_id'].values.tolist())
        elif 'divide_id' in ds.data_vars:
            ids.update(ds['divide_id'].values.tolist())
        ds.close()
    log.info('Found %d unique divide IDs across %d NC files', len(ids), len(files))
    return sorted(ids)


def build_full_upstream_network(
    conn: sqlite3.Connection,
    divide_ids: list[str],
) -> tuple[set[int], list[tuple[int, int]]]:
    """Walk upstream from divides and return all reachable nodes and edges.

    Parameters
    ----------
    conn
        Open SQLite connection to the hydrofabric GeoPackage.
    divide_ids
        List of divide IDs (cat-XXXXX strings) to start from.

    Returns
    -------
    set[int]
        Set of all integer divide IDs in the subgraph (seeds + all upstream).
    list[tuple[int, int]]
        List of (upstream_int, downstream_int) tuples representing flow direction.
    """
    # Build the full DAG from the network table in one shot, then do BFS.
    # Each row: wb-A -> nex-X -> wb-B  == cat-A flows to cat-B.
    log.info('Loading network topology from GeoPackage ...')
    # some sql nastiness
    rows = conn.execute(
        """
        SELECT DISTINCT
            CAST(SUBSTR(w_up.id, 4) AS INTEGER)  AS up_id,
            CAST(SUBSTR(w_dn.id, 4) AS INTEGER)  AS dn_id
        FROM   network w_up
        JOIN   network nex ON w_up.toid = nex.id
        JOIN   network w_dn ON nex.toid = w_dn.id
        WHERE  w_up.id  LIKE 'wb-%'
          AND  nex.id   LIKE 'nex-%'
          AND  w_dn.id  LIKE 'wb-%'
        """
    ).fetchall()

    # adjacency: downstream[node] = set of direct downstream nodes
    downstream: dict[int, set[int]] = {}
    # reverse adjacency for BFS upstream
    upstream: dict[int, set[int]] = {}
    all_edge_set: set[tuple[int, int]] = set()
    for up_id, dn_id in rows:
        downstream.setdefault(up_id, set()).add(dn_id)
        upstream.setdefault(dn_id, set()).add(up_id)
        all_edge_set.add((up_id, dn_id))

    # BFS: start from seeds, walk *upstream* to find the full contributing area.
    seeds = {_cat_to_int(d) for d in divide_ids}
    visited: set[int] = set()
    queue = list(seeds)
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        for parent in upstream.get(node, []):
            if parent not in visited:
                queue.append(parent)

    # Keep only edges within the subgraph.
    subgraph_edges = [(u, d) for u, d in all_edge_set if u in visited and d in visited]
    log.info(
        'Subgraph: %d nodes, %d edges (seeds: %d)',
        len(visited),
        len(subgraph_edges),
        len(seeds),
    )
    return visited, subgraph_edges


def find_gages_in_region(
    conn: sqlite3.Connection,
    node_ids: set[int],
) -> pd.DataFrame:
    """Return USGS gages whose outlet waterbody falls within node_ids.

    Parameters
    ----------
    conn
        Open SQLite connection to the hydrofabric GeoPackage.
    node_ids
        Integer divide IDs in the study subgraph.

    Returns
    -------
    DataFrame
        columns: STAID, divide_id (int), DRAIN_SQKM.
    """
    # hydrolocations.id = 'wb-XXXXX' for the waterbody the gage is at.
    # Matches cat-XXXXX nodes in our set when the integer matches.
    rows = conn.execute(
        """
        SELECT h.hl_link                                  AS STAID,
               CAST(SUBSTR(h.id, 4) AS INTEGER)          AS divide_id,
               d.tot_drainage_areasqkm                   AS DRAIN_SQKM
        FROM   hydrolocations h
        JOIN   divides d ON d.divide_id = h.nex_id OR
                            'wb' || SUBSTR(d.divide_id, 4) = h.id
        WHERE  h.hl_reference IN ('gages', 'usgs-gage')
          AND  h.id LIKE 'wb-%'
        """
    ).fetchall()

    df = pd.DataFrame(rows, columns=['STAID', 'divide_id', 'DRAIN_SQKM'])
    df = df[df['divide_id'].isin(node_ids)].drop_duplicates('STAID').reset_index(drop=True)
    log.info('Found %d USGS gages in study region', len(df))
    return df


def extract_divides_geodataframe(
    conn: sqlite3.Connection,
    node_ids: set[int],
) -> gpd.GeoDataFrame:
    """Read divide geometries and attributes for the given integer node IDs.
    
    Parameters
    ----------
    conn
        Open SQLite connection to the hydrofabric GeoPackage.
    node_ids
        Integer divide IDs in the study subgraph.

    Returns
    -------
    GeoDataFrame
        Geometries and attributes for the given divide IDs.
    """
    cat_ids = [_int_to_cat(n) for n in sorted(node_ids)]

    try:
        gdf = gpd.read_file(
            conn.execute('PRAGMA database_list').fetchone()[2],  # gpkg path
            layer='divides',
            where=f"divide_id IN ({','.join(repr(c) for c in cat_ids)})",
        )
    except Exception:
        # Fallback: read all then filter (slower but safe)
        gdf = gpd.read_file(
            conn.execute('PRAGMA database_list').fetchone()[2],
            layer='divides',
        )
        gdf = gdf[gdf['divide_id'].isin(set(cat_ids))].reset_index(drop=True)

    return gdf


def build_topology_json(
    nodes: set[int],
    edges: list[tuple[int, int]],
    gages_df: pd.DataFrame,
) -> dict:
    """Assemble the topology dict expected by FlashHydroLoader / MtsHydroLoader.
    
    Parameters
    ----------
    nodes
        Set of integer divide IDs in the subgraph.
    edges
        List of (upstream_int, downstream_int) tuples representing flow
        direction.
    gages_df
        DataFrame of USGS gages in the region, with columns STAID and divide_id
    
    Returns
    -------
    dict
        dict with keys 'nodes', 'edges', and 'gage_hf' for JSON serialization.
    """
    gage_hf = {
        str(row['STAID']).zfill(8): int(row['divide_id'])
        for _, row in gages_df.iterrows()
    }
    return {
        'nodes': sorted(nodes),
        'edges': [[int(u), int(d)] for u, d in edges],
        'gage_hf': gage_hf,
    }


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--nc-dir', type=Path, help='Directory of 15-min MRMS NC files (divide_id coord auto-read)')
    src.add_argument('--divide-ids', nargs='+', metavar='cat-XXXXX', help='Explicit list of seed divide IDs')
    parser.add_argument('--gpkg', type=Path, required=True, help='Path to conus_nextgen.gpkg')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--upstream', action='store_true', default=True, help='Expand to full upstream network (default: on)')
    parser.add_argument('--no-upstream', dest='upstream', action='store_false', help='Only keep seed divide IDs, no upstream expansion')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # collect seed divide ids
    if args.nc_dir:
        seed_ids = collect_divide_ids_from_nc(args.nc_dir)
    else:
        seed_ids = args.divide_ids

    log.info('Seed divide IDs: %d', len(seed_ids))

    conn = sqlite3.connect(args.gpkg)

    # build network subgraph
    if args.upstream:
        nodes, edges = build_full_upstream_network(conn, seed_ids)
    else:
        nodes = {_cat_to_int(d) for d in seed_ids}
        edges = []

    # find gages
    gages_df = find_gages_in_region(conn, nodes)

    # extract divide geometries
    log.info('Extracting divide geometries ...')
    gpkg_path = str(args.gpkg)
    cat_ids = [_int_to_cat(n) for n in sorted(nodes)]
    try:
        gdf = gpd.read_file(
            gpkg_path,
            layer='divides',
            where=f"divide_id IN ({','.join(repr(c) for c in cat_ids)})",
        )
    except Exception:
        log.warning('WHERE clause failed, falling back to full read + filter')
        gdf = gpd.read_file(gpkg_path, layer='divides')
        gdf = gdf[gdf['divide_id'].isin(set(cat_ids))].reset_index(drop=True)

    conn.close()

    # write outputs
    divides_path = args.output_dir / 'divides.gpkg'
    gdf.to_file(divides_path, driver='GPKG')
    log.info('Wrote %s (%d features)', divides_path, len(gdf))

    topo_path = args.output_dir / 'topology.json'
    topo = build_topology_json(nodes, edges, gages_df)
    with open(topo_path, 'w') as f:
        json.dump(topo, f)
    log.info('Wrote %s (%d nodes, %d edges, %d gages)', topo_path, len(topo['nodes']), len(topo['edges']), len(topo['gage_hf']))

    gauges_path = args.output_dir / 'gauges.csv'
    gages_df.to_csv(gauges_path, index=False)
    log.info('Wrote %s', gauges_path)

    log.info('Done.')


if __name__ == '__main__':
    main()
