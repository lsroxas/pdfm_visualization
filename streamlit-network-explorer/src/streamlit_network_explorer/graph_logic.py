# src/streamlit_network_explorer/graph_logic.py
from typing import List, Tuple, Optional, Dict, Set
import json
import networkx as nx
from streamlit_agraph import Node, Edge

# Type styles: base radius (meters) and color [R,G,B,A]
TYPE_STYLE_GRAPH = {
    "province": {"size": 10, "color": "#DC4437"},  # red-ish
    "municipality": {"size": 5, "color": "#B4B4B4"},   # grey
    "default":  {"size": 5, "color": "#FFAA00"},  # grey
}

# Highlight overrides (HEX)
SELECTED_COLOR_HEX = "#FF7F0E"   # orange
NEIGHBOR_COLOR_HEX = "#1F77B4"   # blue
# If you want to dim non-neighbors when a node is selected, use this:
DIM_COLOR_HEX = "#C7C7C7"
SELECTED_BOOST = 6                      # +meters for selected
NEIGHBOR_BOOST = 3    

# Edge styles by type (GRAPH view)
EDGE_STYLE_GRAPH = {
    "proximity": "#EA4E4E",    # gray
    "ownership": "#292929",     # blue
    "similarity": "#9467BD",    # purple
    "default": "#FFAA00", # fallback
}

EDGE_HIGHLIGHT_WIDTH = 2.5

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
    size_map_attr: Optional[str],
    palette: Dict[str, str],
    hops: int = 1,
    max_node_size: int = 12,
) -> Tuple[List[Node], List[Edge]]:
    """Convert graph to streamlit-agraph nodes/edges with neighbor highlighting."""
    hi = k_hop_nodes(G, selected, hops) if selected is not None else set()

    nodes: List[Node] = []
    for n, attrs in G.nodes(data=True):
        label = str(attrs.get("label", n))
        node_type = str(attrs.get("type", "default")).lower()
        base = TYPE_STYLE_GRAPH.get(node_type, TYPE_STYLE_GRAPH["default"])
        color = base["color"]
        size  = base["size"]

        # Make a Tooltip        
        import json
        title = json.dumps(attrs, indent=2, default=str)
        
        # optional attribute-based scaling (kept if you use size_map_attr)
        # if size_map_attr and size_map_attr in attrs:
        #     try:
        #         val = float(attrs[size_map_attr])
        #         size = int(max(4, min(max_node_size, size + (max_node_size - 4) * (val))))
        #     except Exception:
        #         pass

        is_sel = (str(n) == str(selected))
        is_hi  = (selected is not None and n in hi and not is_sel)
        if is_sel:
            color = SELECTED_COLOR
        elif is_hi:
            color = NEIGHBOR_COLOR
        elif selected is not None:
            # dim non-neighbors when something is selected
            color = "#C7C7C7"

        nodes.append(Node(id=str(n), label=label, size=size, color=color, title=title))

    edges: List[Edge] = []
    for u, v, eattrs in G.edges(data=True):
        etype = str(eattrs.get("type", "default")).lower()
        base_color = EDGE_STYLE_GRAPH.get(etype, EDGE_STYLE_GRAPH["default"])

        # Highlight override if a node is selected
        if selected is not None and (u in hi or v in hi):
            color = palette["edge_highlight"]
        else:
            color = base_color

        label = str(eattrs["weight"]) if "weight" in eattrs else None
        edges.append(Edge(source=str(u), target=str(v), color=color, label=label))

    return nodes, edges
