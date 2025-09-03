
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict
import pandas as pd

from .file_config import load_config

def _get(d: Dict, key: str, default: str) -> str:
    v = d.get(key, default)
    return str(v) if v is not None else default


def load_lime_input_data(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lime_config = cfg['data']['lime']
    input_file = lime_config.get('datasource', 'data/lime_input.csv')
    feature_cols = lime_config.get('feature_cols', [])
    label_names = lime_config.get('label_names', [])
    df = pd.read_csv(input_file, usecols=feature_cols).fillna(0)
    return df

def load_nodes_edges(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    dnodes = cfg["data"]["nodes"]
    dedges = cfg["data"]["edges"]

    # Nodes config
    n_path = Path(dnodes.get("path", "data/nodes.csv"))
    n_delim = dnodes.get("delimiter", ",")
    id_col = _get(dnodes, "id_col", "id")
    lat_col = _get(dnodes, "lat_col", "lat")
    lon_col = _get(dnodes, "lon_col", "lon")
    name_col = _get(dnodes, "location_name_col", "location_name")
    prov_col = _get(dnodes, "province_col", "province")
    tier_col = _get(dnodes, "tier_col", "tier")
    pop_col = _get(dnodes, "population_col", "population")
    type_col = _get(dnodes, "type_col", "location_type")

    nodes = pd.read_csv(n_path, delimiter=n_delim)
    rename_map = {
        id_col: "id",
        lat_col: "lat",
        lon_col: "lon",
        name_col: "location_name",
        prov_col: "province",
        tier_col: "tier",
        pop_col: "population",
        type_col: "location_type",
    }
    nodes = nodes.rename(columns=rename_map)

    keep_cols = [c for c in ["id","lat","lon","location_name","province","tier","population","location_type"] if c in nodes.columns]
    nodes = nodes[keep_cols].copy()

    nodes["id"] = nodes["id"].astype(str)
    if "lat" in nodes: nodes["lat"] = pd.to_numeric(nodes["lat"], errors="coerce")
    if "lon" in nodes: nodes["lon"] = pd.to_numeric(nodes["lon"], errors="coerce")

    # Edges config
    e_path = Path(dedges.get("path", "data/edges.csv"))
    e_delim = dedges.get("delimiter", ",")
    src_col = _get(dedges, "source_col", "source")
    tgt_col = _get(dedges, "target_col", "target")
    etype_col = _get(dedges, "type_col", "type")

    edges = pd.read_csv(e_path, delimiter=e_delim)
    edges = edges.rename(columns={src_col: "source", tgt_col: "target", etype_col: "type"})
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    if "type" not in edges.columns:
        edges["type"] = "default"

    ids = set(nodes["id"])
    edges = edges[edges["source"].isin(ids) & edges["target"].isin(ids)].copy()

    return nodes, edges
