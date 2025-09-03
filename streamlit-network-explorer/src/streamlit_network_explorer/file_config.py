from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import yaml

DEFAULT_PATH = Path(os.getenv("DATA_CONFIG", "config/data_config.yaml")).resolve()

@dataclass(frozen=True)
class DataConfig:
    nodes_csv: str | None
    edges_csv: str
    node_id_col: str = "location_id"
    edge_src_col: str = "source"
    edge_dst_col: str = "target"

def load_config(path: str | Path | None = None) -> DataConfig:
    p = Path(path).resolve() if path else DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(f"Data config not found at {p}. Set DATA_CONFIG or create config/data_config.yaml")
    with open(p, "r") as f:
        raw = yaml.safe_load(f) or {}
    data = (raw.get("data") or {})
    nodes_csv = data.get("nodes_csv")
    edges_csv = data.get("edges_csv")
    if not edges_csv:
        raise ValueError("Config must include data.edges_csv (path to edges CSV).")
    node_id_col = data.get("node_id_col", "location_id")
    edge_src_col = data.get("edge_src_col", "source")
    edge_dst_col = data.get("edge_dst_col", "target")
    return DataConfig(nodes_csv=nodes_csv, edges_csv=edges_csv,
                      node_id_col=node_id_col, edge_src_col=edge_src_col, edge_dst_col=edge_dst_col)
