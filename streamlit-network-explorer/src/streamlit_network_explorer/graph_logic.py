# src/streamlit_network_explorer/graph_logic.py
from typing import List, Tuple, Optional, Dict, Set
import json
import networkx as nx
from streamlit_agraph import Node, Edge

def k_hop_nodes(G: nx.Graph, source, k: int) -> Set:
    """Return nodes within k hops (including the source)."""
    if source not in G:
        return set()
    visited = {source}
    frontier = {source}
    for _ in range(max(0, k)):
        nxt = set()
        for u in frontier:
            nxt.update(G.neighbors(u))
        visited.update(nxt)
        frontier = nxt
    return visited

def to_agraph(
    G: nx.Graph,
    selected: Optional[str],
    hops: int,
    size_map_attr: Optional[str],
    palette: Dict[str, str],
    max_node_size: int = 12,
) -> Tuple[List[Node], List[Edge]]:
    """Convert graph to streamlit-agraph nodes/edges with neighbor highlighting."""
    hi = k_hop_nodes(G, selected, hops) if selected is not None else set()

    nodes: List[Node] = []
    for n, attrs in G.nodes(data=True):
        label = str(attrs.get("label", n))
        title = json.dumps(attrs, indent=2, default=str)
        color = (
            palette["selected"] if str(n) == str(selected)
            else palette["neighbor"] if (selected is not None and n in hi and str(n) != str(selected))
            else palette["dim"] if selected is not None
            else None
        )
        size = 6
        if size_map_attr and size_map_attr in attrs:
            try:
                val = float(attrs[size_map_attr])
                size = int(max(4, min(max_node_size, 4 + (max_node_size - 4) * (val))))
            except Exception:
                pass
        nodes.append(Node(id=str(n), label=label, size=size, color=color, title=title))

    edges: List[Edge] = []
    for u, v, eattrs in G.edges(data=True):
        color = palette["edge_highlight"] if (selected is not None and (u in hi or v in hi)) else palette["edge"]
        label = str(eattrs["weight"]) if "weight" in eattrs else None
        edges.append(Edge(source=str(u), target=str(v), color=color, label=label))

    return nodes, edges
