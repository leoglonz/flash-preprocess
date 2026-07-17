import os
import sys
import warnings
from pathlib import Path

proj_data = Path(sys.prefix) / "share" / "proj"
if proj_data.exists():
    os.environ["PROJ_DATA"] = str(proj_data)
    os.environ["PROJ_LIB"] = str(proj_data)

import geopandas as gpd
import pandas as pd
import pyproj

pyproj.network.set_network_enabled(False)
warnings.filterwarnings("ignore")

GPKG_PATH = Path("/projects/mhpi/leoglonz/sub_hourly/data/conus_nextgen.gpkg")


def load_hydrofabric(cache_dir: Path):
    cache_dir = Path(cache_dir)
    f_cm, f_net, f_fp, f_nx = (
        cache_dir / "catchments_master.parquet",
        cache_dir / "network.parquet",
        cache_dir / "flowpaths.parquet",
        cache_dir / "nexus.parquet",
    )
    if f_cm.exists() and f_net.exists() and f_fp.exists() and f_nx.exists():
        return (gpd.read_parquet(f_cm), pd.read_parquet(f_net),
                gpd.read_parquet(f_fp), gpd.read_parquet(f_nx))

    network = gpd.read_file(GPKG_PATH, layer="network",
                             columns=["id", "toid", "divide_id", "vpuid"], read_geometry=False)
    divides = gpd.read_file(GPKG_PATH, layer="divides",
                             columns=["divide_id", "id", "toid", "areasqkm"])
    flowpaths = gpd.read_file(GPKG_PATH, layer="flowpaths", columns=["id", "toid"])
    nexus = gpd.read_file(GPKG_PATH, layer="nexus", columns=["id", "toid"])

    net_lookup = network.dropna(subset=["divide_id"]).drop_duplicates("divide_id")[["divide_id", "vpuid"]]
    catchments_master = divides.rename(columns={"id": "flowpath_id", "toid": "nexus_id"}).merge(
        net_lookup, on="divide_id", how="left")
    catchments_master["nexus_type"] = catchments_master["nexus_id"].astype(str).str.extract(r"^([a-z]+)-")[0]
    catchments_master["is_terminal"] = catchments_master["nexus_type"].isin(["tnx", "cnx", "inx"])

    nexus_next = dict(zip(nexus["id"].astype(str), nexus["toid"].astype(str)))
    net_wb = network[network["id"].astype(str).str.startswith("wb-")]
    wb_next = dict(zip(net_wb["id"].astype(str), net_wb["toid"].astype(str)))
    TERMINAL_WB = "wb-0"
    terminal_cache = {}

    def _down_nexus(n):
        wb = nexus_next.get(n)
        if wb is None or wb == TERMINAL_WB:
            return None
        return wb_next.get(wb)

    def terminal_of(start):
        path, cur, seen = [], start, set()
        while cur is not None and cur not in terminal_cache:
            if cur in seen:
                for p in path:
                    terminal_cache[p] = cur
                return cur
            seen.add(cur)
            path.append(cur)
            nxt = _down_nexus(cur)
            if nxt is None:
                terminal_cache[cur] = cur
                break
            cur = nxt
        result = terminal_cache.get(cur, cur)
        for p in path:
            terminal_cache[p] = result
        return result

    for n in set(catchments_master["nexus_id"].astype(str)):
        terminal_of(n)
    catchments_master["terminal_nexus_id"] = catchments_master["nexus_id"].astype(str).map(terminal_cache)
    catchments_master["terminal_is_clean"] = ~catchments_master["terminal_nexus_id"].astype(str).str.startswith("nex-")

    cache_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_write(df, path):
        tmp = f"{path}.tmp{os.getpid()}"
        df.to_parquet(tmp)
        os.replace(tmp, path)  # shared cache; concurrent VPU runs may race to build this

    _atomic_write(catchments_master, f_cm)
    _atomic_write(network, f_net)
    _atomic_write(flowpaths, f_fp)
    _atomic_write(nexus, f_nx)
    return catchments_master, network, flowpaths, nexus
