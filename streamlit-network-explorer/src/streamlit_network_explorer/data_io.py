# src/streamlit_network_explorer/data_io.py
from typing import Dict, Optional
import networkx as nx
import pandas as pd
import streamlit as st

from .config import Defaults

@st.cache_data(show_spinner=True)
def generate_demo_graph(n_nodes: int, avg_degree: int, seed: int = 42) -> nx.Graph:
    rng = nx.utils.create_random_state(seed)
    m = max(1, avg_degree // 2)
    G = nx.barabasi_albert_graph(n=n_nodes, m=m, seed=rng)
    for n in G.nodes:
        G.nodes[n]["label"] = f"Node {n}"
        G.nodes[n]["type"] = "demo"
    for u, v in G.edges:
        G.edges[u, v]["weight"] = 1
    return G

def _load_csv(nodes, edges, node_id_col: str, src: str, dst: str) -> nx.Graph:
    G = nx.Graph()
    if nodes:
        node_df = pd.read_csv(nodes)
        assert node_id_col in node_df.columns, f"Nodes CSV must have '{node_id_col}'"
        for _, row in node_df.iterrows():
            nid = row[node_id_col]
            G.add_node(nid, **row.to_dict())
    if edges:
        edge_df = pd.read_csv(edges)
        assert src in edge_df.columns and dst in edge_df.columns, f"Edges CSV must have '{src}' and '{dst}'"
        for _, row in edge_df.iterrows():
            u, v = row[src], row[dst]
            attrs = row.to_dict()
            attrs.pop(src, None); attrs.pop(dst, None)
            G.add_edge(u, v, **attrs)
    return G

def get_graph_from_config(cfg) -> nx.Graph:
    return _load_csv(cfg.nodes_csv, cfg.edges_csv, cfg.node_id_col, cfg.edge_src_col, cfg.edge_dst_col)



# def _load_parquet(nodes, edges, node_id_col: str, src: str, dst: str) -> nx.Graph:
#     G = nx.Graph()
#     if nodes:
#         node_df = pd.read_parquet(nodes)
#         assert node_id_col in node_df.columns, f"Nodes Parquet must have '{node_id_col}'"
#         for _, row in node_df.iterrows():
#             nid = row[node_id_col]
#             G.add_node(nid, **row.to_dict())
#     if edges:
#         edge_df = pd.read_parquet(edges)
#         assert src in edge_df.columns and dst in edge_df.columns, f"Edges Parquet must have '{src}' and '{dst}'"
#         for _, row in edge_df.iterrows():
#             u, v = row[src], row[dst]
#             attrs = row.to_dict()
#             attrs.pop(src, None); attrs.pop(dst, None)
#             G.add_edge(u, v, **attrs)
#     return G

# def get_graph(source_type: str, data_files: Dict, opts):
#     """Return a NetworkX graph based on source selection and uploaded files."""
#     if source_type == "Demo":
#         return generate_demo_graph(opts.demo_nodes, opts.avg_degree)

#     node_id_col = opts.node_id_col or Defaults.node_id_col
#     src = opts.edge_src_col or Defaults.edge_src_col
#     dst = opts.edge_dst_col or Defaults.edge_dst_col

#     if source_type == "CSV":
#         return _load_csv(data_files.get("nodes"), data_files.get("edges"), node_id_col, src, dst)
#     elif source_type == "Parquet":
#         return _load_parquet(data_files.get("nodes"), data_files.get("edges"), node_id_col, src, dst)
#     else:
#         # Fallback to demo if nothing matches
#         return generate_demo_graph(Defaults.demo_nodes, Defaults.avg_degree)
