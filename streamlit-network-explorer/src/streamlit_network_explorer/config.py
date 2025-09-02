# src/streamlit_network_explorer/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Defaults:
    canvas_height: int = 700
    physics: bool = True
    hierarchical: bool = False
    highlight_hops: int = 1
    avg_degree: int = 4
    demo_nodes: int = 1700
    node_id_col: str = "id"
    edge_src_col: str = "source"
    edge_dst_col: str = "target"
