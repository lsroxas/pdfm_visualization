
from __future__ import annotations
import streamlit as st

from streamlit_network_explorer.file_config import load_config
from streamlit_network_explorer.data_io import load_nodes_edges
from streamlit_network_explorer import state
from streamlit_network_explorer import ui_components
from streamlit_network_explorer.map_view import render_map

def main():
    st.set_page_config(page_title="PDFM - Philippine Network Map", layout="wide")

    ui_components.header(
        title="PDFM Philippine Network Map",
        subtitle="Map-only viewer loaded via config/data_config.yaml",
    )

    try:
        cfg = load_config()
    except Exception as e:
        st.error(f"Failed to load data config: {e}")
        return

    try:
        nodes_df, edges_df = load_nodes_edges(cfg)
    except Exception as e:
        st.error(f"Failed to load CSVs from config: {e}")
        return

    left, right = st.columns([1, 3], gap="large")

    with left:
        # NEW: filters
        node_types, edge_types = ui_components.filters_panel(nodes_df, edges_df)

        picked = ui_components.node_picker(nodes_df, state.get_selected_node())
        if picked and picked != state.get_selected_node():
            state.set_selected_node(picked)

        ## Add details pane
        ui_components.legend_box()
        ui_components.node_details_panel(nodes_df, state.get_selected_node()) 
        
        st.caption(f"Loaded nodes: {len(nodes_df):,} | edges: {len(edges_df):,}")

    with right:
        render_map(
            nodes_df=nodes_df,
            edges_df=edges_df,
            selected_id=state.get_selected_node(),
            height=720,
            initial_zoom=5.0,
            philippines_center=(12.8797, 121.7740),
            node_types_filter=node_types,   # <-- pass filters
            edge_types_filter=edge_types,   
        )

if __name__ == "__main__":
    main()
