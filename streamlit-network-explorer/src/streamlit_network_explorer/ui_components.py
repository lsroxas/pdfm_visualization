# src/streamlit_network_explorer/ui_components.py
from dataclasses import dataclass
import streamlit as st
from streamlit_agraph import agraph, Config

@dataclass
class Options:
    canvas_height: int
    physics: bool
    hierarchical: bool
    hops: int
    size_map_attr: str | None
    lock_layout: bool
    layout_algo: str

def header(title: str, subtitle: str | None = None):
    st.title(title)
    if subtitle:
        st.caption(subtitle)

# def source_selector():
#     return st.selectbox("Data source", ["Demo", "CSV", "Parquet"], index=0)

# def data_uploads(source_type: str):
#     data = {}
#     if source_type == "CSV":
#         data["nodes"] = st.file_uploader("Nodes CSV (optional)", type=["csv"])
#         data["edges"] = st.file_uploader("Edges CSV (required)", type=["csv"])
#     elif source_type == "Parquet":
#         data["nodes"] = st.file_uploader("Nodes Parquet (optional)", type=["parquet"])
#         data["edges"] = st.file_uploader("Edges Parquet (required)", type=["parquet"])
#     return data

def display_options() -> Options:
    st.subheader("Display options")
    canvas_height = st.slider("Canvas height (px)", 400, 1200, 700, 50)
    physics = st.checkbox("Enable physics", False)
    hierarchical = st.checkbox("Hierarchical layout", False)
    hops = st.select_slider("Highlight hops", [0, 1, 2], value=1)
    size_map_attr = st.text_input("Size by node attribute (optional)") or None
    lock_layout = st.checkbox("Lock layout (no dragging)", True)
    layout_algo = st.selectbox("Layout algorithm (when locked)",
                               ["spring", "kamada_kawai", "fruchterman_reingold"], index=0)
    return Options(canvas_height, physics, hierarchical, hops, size_map_attr, lock_layout, layout_algo)

def graph_area(nodes, edges, opts: Options):
    cfg = Config(
        width="100%",
        height=opts.canvas_height,
        directed=False,
        physics=opts.physics,
        hierarchical=opts.hierarchical,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
    )
    return agraph(nodes=nodes, edges=edges, config=cfg)

def node_details_panel(G, sel, opts: Options):
    st.subheader("Node Details")
    if not sel:
        st.info("Click a node to see attributes and neighbor summary.")
        return
    attrs = G.nodes.get(sel, {})
    st.json(attrs)
    nbrs = list(G.neighbors(sel))
    st.markdown(f"**Degree:** {len(nbrs)}")
    import pandas as pd
    top = sorted(((n, G.degree(n)) for n in nbrs), key=lambda x: x[1], reverse=True)[:20]
    df_top = pd.DataFrame({"neighbor": [str(n) for n, _ in top], "degree": [d for _, d in top]})
    st.dataframe(df_top, use_container_width=True)

def tips_footer():
    with st.expander("💡 Tips & Notes"):
        st.markdown(
            "- Data is loaded from a static YAML config (config/data_config.yaml)."
            "- Keep physics off + lock layout to prevent any node movement."
            "- Choose a layout algorithm (spring/kamada-kawai/FR) for deterministic positions."
            "- Precompute and save positions if you need cross-session consistency."
        )
