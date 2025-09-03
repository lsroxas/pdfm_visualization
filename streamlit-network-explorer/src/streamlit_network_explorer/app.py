# src/streamlit_network_explorer/app.py
import streamlit as st
from streamlit_network_explorer.logging_setup import setup_logging
from streamlit_network_explorer import styles, state
from streamlit_network_explorer import data_io, graph_logic, ui_components
from streamlit_network_explorer.file_config import load_config

def main():
    st.set_page_config(page_title="Network Explorer", layout="wide")
    setup_logging()
    styles.inject_base_css()

    ui_components.header(
        title="DKSH Market Nodes",
        subtitle="Overlay graph on the Philippines map. Data loaded from config/data_config.yaml"
    )

    try:
        cfg = load_config()
    except Exception as e:
        st.error(f"Failed to load data config: {e}")
        return

    # Load graph
    G = data_io.get_graph_from_config(cfg)

    # Selection state
    selected_node = state.get_selected_node()

    # Layout
    left, right = st.columns([2, 1], gap="large")

    # Sidebar options
    with st.sidebar:
        opts = ui_components.display_options()

    tabs = st.tabs(["Graph", "Map"])

    with tabs[0]:
        col_graph, col_legend = st.columns([4, 1], gap="large")

        with col_graph:
            nodes, edges = graph_logic.to_agraph(
                G,
                selected=selected_node,
                hops=opts.hops,
                size_map_attr=opts.size_map_attr,
                # size_map_attr=None
                palette=styles.palette(),
                # lock_layout=opts.lock_layout,
                # layout_algo=opts.layout_algo,
            )
            selection = ui_components.graph_area(nodes, edges, opts)
            if isinstance(selection, dict) and selection.get("type") == "node":
                state.set_selected_node(selection.get("id"))
        
        with col_legend:
            # Legend for Graph View
            from streamlit_network_explorer.graph_logic import (
                TYPE_STYLE_GRAPH,
                EDGE_STYLE_GRAPH,
                SELECTED_COLOR_HEX,
                NEIGHBOR_COLOR_HEX,
            )
            ui_components.graph_legend(
                node_style=TYPE_STYLE_GRAPH,
                edge_style=EDGE_STYLE_GRAPH,
                palette=styles.palette(),
                selected_node_color_hex=SELECTED_COLOR_HEX,
                neighbor_node_color_hex=NEIGHBOR_COLOR_HEX,
            )

    with tabs[1]:
        from streamlit_network_explorer.map_view import render_map, render_legend
        col_map, col_legend = st.columns([4, 1], gap="large")
        
        with col_map:
            render_map(
                G,
                selected=state.get_selected_node(),
                hops=opts.hops,
                height=opts.canvas_height,
                initial_zoom=5.0,
                philippines_center=(12.8797, 121.7740),
            )
        
        with col_legend:
            render_legend()

    ui_components.node_details_panel(G, state.get_selected_node(), opts)
    ui_components.tips_footer()

if __name__ == "__main__":
    main()
