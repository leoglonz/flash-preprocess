import sqlite3
from collections import deque

import pandas as pd


HF_PATH_DEFAULT = '/Users/leoglonz/.ngiab/hydrofabric/v2.2/conus_nextgen.gpkg'


def build_upstream_graph(hf_path: str) -> dict[str, list[str]]:
    """Build a dict mapping each catchment to its immediate upstream catchments.

    Parameters
    ----------
    hf_path
        Path to conus_nextgen.gpkg (SQLite geopackage).

    Returns
    -------
    dict
        Mapping of divide_id -> list of upstream divide_ids.
    """
    print("  Building upstream graph from hydrofabric...")
    conn = sqlite3.connect(hf_path)
    divides = pd.read_sql("SELECT divide_id, toid FROM divides", conn)
    nexus = pd.read_sql("SELECT id, toid FROM nexus", conn)
    flowpaths = pd.read_sql("SELECT id, divide_id FROM flowpaths", conn)
    conn.close()

    nex_to_wb = nexus.set_index("id")["toid"].to_dict()
    wb_to_cat = flowpaths.set_index("id")["divide_id"].to_dict()

    upstream = {}
    for row in divides.itertuples(index=False):
        wb_ds = nex_to_wb.get(row.toid)
        cat_ds = wb_to_cat.get(wb_ds) if wb_ds else None
        if cat_ds:
            upstream.setdefault(cat_ds, []).append(row.divide_id)
    return upstream


def expand_upstream(
    seed_cats: list[str] | set[str], upstream_graph: dict[str, list[str]]
) -> set[str]:
    """BFS from seed catchments to collect all upstream catchment IDs.

    Parameters
    ----------
    seed_cats
        Seed catchment IDs.
    upstream_graph
        Mapping of catchment IDs to their immediate upstreams.

    Returns
    -------
    set
        Set of all catchment IDs in the upstream network.
    """
    visited = set(seed_cats)
    queue = deque(seed_cats)
    while queue:
        cat = queue.popleft()
        for up in upstream_graph.get(cat, []):
            if up not in visited:
                visited.add(up)
                queue.append(up)
    return visited
