# src/streamlit_network_explorer/app.py
import streamlit as st

from streamlit_network_explorer.logging_setup import setup_logging
from streamlit_network_explorer import styles, state
from streamlit_network_explorer import data_io, graph_logic, ui_components

def main():
    st.set_page_config(page_title="Network Explorer", layout="wide")
    setup_logging()
    styles.inject_base_css()

    ui_components.header(
        title="Network Explorer 🚀",
        subtitle="Visualize ~1.7k-node graphs. Click a node to inspect and highlight its neighbors."
    )

    with st.sidebar:
        source_type = ui_components.source_selector()
        opts = ui_components.display_options()
        data_files = ui_components.data_uploads(source_type=source_type)

    # Load graph
    G = data_io.get_graph(source_type, data_files, opts)

    # Selection state
    selected_node = state.get_selected_node()

    # Layout
    left, right = st.columns([2, 1], gap="large")

    with left:
        nodes, edges = graph_logic.to_agraph(
            G,
            selected=selected_node,
            hops=opts.hops,
            size_map_attr=opts.size_map_attr,
            palette=styles.palette(),
        )
        selection = ui_components.graph_area(nodes, edges, opts)
        if isinstance(selection, dict) and selection.get("type") == "node":
            state.set_selected_node(selection.get("id"))

    with right:
        ui_components.node_details_panel(G, state.get_selected_node(), opts)

    ui_components.tips_footer()

if __name__ == "__main__":
    main()
