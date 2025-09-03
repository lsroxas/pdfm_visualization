# src/streamlit_network_explorer/ui_components.py
from dataclasses import dataclass
import streamlit as st
from streamlit_agraph import agraph, Config
import streamlit as st
import pandas as pd
import networkx as nx

HIDE_KEYS = {
    "lat", "lon", "long", "latitude", "longitude",  # geo already visualized
    "x", "y",                                        # alternates
    "id", "label", "title",                          # shown elsewhere
    "radius", "fill", "sel", "hi",                   # map-only computed
}


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

# def node_details_panel(G, sel, opts: Options):
#     st.subheader("Node Details")
#     if not sel:
#         st.info("Click a node to see attributes and neighbor summary.")
#         return
#     attrs = G.nodes.get(sel, {})
#     st.json(attrs)
#     nbrs = list(G.neighbors(sel))
#     st.markdown(f"**Degree:** {len(nbrs)}")
#     import pandas as pd
#     top = sorted(((n, G.degree(n)) for n in nbrs), key=lambda x: x[1], reverse=True)[:20]
#     df_top = pd.DataFrame({"neighbor": [str(n) for n, _ in top], "degree": [d for _, d in top]})
#     st.dataframe(df_top, use_container_width=True)

def node_details_panel(G: nx.Graph, selected_node: str | None, opts=None, extra_hide: set[str] | None = None):
    with st.container():
        st.subheader("Details")
        if not selected_node:
            st.caption("Click or pick a node to see details.")
            return

        sel_str = str(selected_node)
        key = sel_str if sel_str in G else next((k for k in G.nodes if str(k) == sel_str), None)
        if key is None:
            st.warning(f"Node '{sel_str}' is not in the current graph.")
            return

        attrs = dict(G.nodes[key])

        # Title / display name (keeps your province ALL CAPS rule)
        display_name = attrs.get("location_name") or attrs.get("label") or str(key)
        if str(attrs.get("location_type", "")).lower() == "province":
            display_name = str(display_name).upper()
        st.markdown(f"### {display_name}")

        # 🔑 Show Location ID prominently
        loc_id = _get_location_id(attrs, key)
        st.caption(f"**Location ID:** `{loc_id}`")

        # Hide fields you don’t want in the table
        hide = set(HIDE_KEYS) | {"location_id", "Location_ID", "loc_id"}
        if extra_hide:
            hide |= set(extra_hide)

        rows = []
        for k, v in attrs.items():
            if k in hide or v in (None, ""):
                continue
            rows.append({"Field": _pretty_key(k), "Value": _format_val(v)})

        if rows:
            rows.sort(key=lambda r: r["Field"].lower())
            df = pd.DataFrame(rows, columns=["Field", "Value"])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No additional attributes to display for this node.")
            
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

def _pretty_key(k: str) -> str:
    # "location_type" -> "Location Type"
    return " ".join(part.capitalize() for part in str(k).replace("_", " ").split())

def _format_val(v):
    # Gentle formatting: numbers, short lists, everything else as-is
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".")
    if isinstance(v, (list, tuple, set)):
        if len(v) <= 8:
            return ", ".join(map(str, v))
        return f"{len(v)} items"
    return v

def node_details_panel(G: nx.Graph, selected_node: str | None, opts=None, extra_hide: set[str] | None = None):
    """Right-hand panel with node attributes from the nodes CSV."""
    with st.container():
        st.subheader("Details")
        if not selected_node:
            st.caption("Click a node to see details.")
            return

        if selected_node not in G:
            st.warning(f"Node '{selected_node}' is not in the current graph.")
            return

        attrs = dict(G.nodes[selected_node])
        # Build a presentable name (match your label logic)
        display_name = attrs.get("location_name") or attrs.get("label") or str(selected_node)
        if str(attrs.get("location_type", "")).lower() == "province":
            display_name = str(display_name).upper()

        st.markdown(f"### {display_name}")
        st.caption(f"ID: `{selected_node}`")

        # Filter attributes you don't want to show
        hide = set(HIDE_KEYS)
        if extra_hide:
            hide |= set(extra_hide)

        rows = []
        for k, v in attrs.items():
            if k in hide:
                continue
            if v is None or v == "":
                continue
            rows.append({"Field": _pretty_key(k), "Value": _format_val(v)})

        if not rows:
            st.info("No additional attributes to display for this node.")
            return

        # Stable sort by field name
        rows.sort(key=lambda r: r["Field"].lower())
        df = pd.DataFrame(rows, columns=["Field", "Value"])
        st.dataframe(df, use_container_width=True, hide_index=True)

def node_picker(G: nx.Graph, selected_node: str | None = None) -> str | None:
    names = []
    for n, a in G.nodes(data=True):
        muni = a.get("municipality_name") or a.get("location_name") or a.get("city_name") or a.get("municipality")
        prov = a.get("province_name") or a.get("province") or a.get("prov_name")
        locid = a.get("location_id") or a.get("Location_ID") or a.get("loc_id") or str(n)
        label = ", ".join([x for x in [muni, prov] if x]) or str(n)
        names.append((f"{label} ({locid})", str(n)))

    if not names:
        st.caption("No nodes available.")
        return None

    names.sort(key=lambda t: t[0].lower())
    values = [v for _, v in names]
    labels = {v: d for d, v in names}

    # store selected value in session_state so callback can read it
    if "node_picker_value" not in st.session_state:
        st.session_state.node_picker_value = selected_node if selected_node in values else (values[0])

    def _apply_pick():
        from streamlit_network_explorer import state as app_state
        app_state.set_selected_node(st.session_state.node_picker_value)

    st.selectbox(
        "🔎 Find a node",
        options=values,
        key="node_picker_value",
        index=values.index(selected_node) if selected_node in values else 0,
        format_func=lambda v: labels.get(v, v),
        on_change=_apply_pick,           # <- ensures state is updated before rest renders
    )
    return st.session_state.node_picker_value

def _get_location_id(attrs: dict, key_fallback: str) -> str:
    # try multiple common spellings/casings
    return (
        str(attrs.get("location_id"))
        or str(attrs.get("Location_ID"))
        or str(attrs.get("loc_id"))
        or str(key_fallback)
    )