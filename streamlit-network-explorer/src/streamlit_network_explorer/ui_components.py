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

# src/streamlit_network_explorer/ui_components.py

def _swatch_html(color_hex: str) -> str:
    return (
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'border-radius:2px;background:{color_hex};'
        f'margin-right:6px;border:1px solid #999"></span>'
    )

def graph_legend(
    node_style: dict,
    edge_style: dict,
    palette: dict,
    selected_node_color_hex: str,
    neighbor_node_color_hex: str,
):
    """Render a collapsible legend for the Graph (agraph) view.
    Accepts node_style entries like {"type": {"color": "#HEX"}} and
    edge_style entries as either {"type": {"color": "#HEX"}} OR {"type": "#HEX"}.
    """
    with st.expander("Legend", expanded=False):
        st.markdown("### Node types", unsafe_allow_html=True)
        for ntype, sty in node_style.items():
            if ntype == "default":
                continue
            color_hex = sty["color"] if isinstance(sty, dict) else str(sty)
            st.markdown(_swatch_html(color_hex) + ntype, unsafe_allow_html=True)

        st.markdown("### Edge types", unsafe_allow_html=True)
        for etype, sty in edge_style.items():
            if etype == "default":
                continue
            color_hex = sty["color"] if isinstance(sty, dict) else str(sty)
            st.markdown(_swatch_html(color_hex) + etype, unsafe_allow_html=True)

        # st.markdown("### Highlights", unsafe_allow_html=True)
        # st.markdown(_swatch_html(selected_node_color_hex) + "Selected node", unsafe_allow_html=True)
        # st.markdown(_swatch_html(neighbor_node_color_hex) + "Neighbor nodes", unsafe_allow_html=True)
        # st.markdown(_swatch_html(palette.get("edge_highlight", "#ff7f0e")) + "Highlighted edges", unsafe_allow_html=True)
        # st.markdown(_swatch_html(palette.get("edge", "#bbbbbb")) + "Default edges", unsafe_allow_html=True)