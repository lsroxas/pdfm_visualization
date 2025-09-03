
from __future__ import annotations
import streamlit as st
import pandas as pd

# For details pane
from typing import Any, Dict, Iterable, Tuple, List

def header(title: str, subtitle: str | None = None):
    st.title(title)
    if subtitle:
        st.caption(subtitle)

def node_picker(nodes_df: pd.DataFrame, selected_id: str | None) -> str | None:
    if nodes_df.empty:
        st.info("No nodes loaded.")
        return None

    disp = nodes_df.copy()
    # Robust label: "Location — Province" (falls back to id if missing)
    name_col = "location_name" if "location_name" in disp.columns else "id"
    prov_col = "province" if "province" in disp.columns else None

    if prov_col:
        disp["label"] = disp[name_col].astype(str) + " — " + disp[prov_col].astype(str)
    else:
        disp["label"] = disp[name_col].astype(str)

    if "id" not in disp.columns:
        st.error("nodes_df is missing 'id'. Check your data_config.yaml mappings.")
        return None

    options = disp[["id", "label"]].drop_duplicates().sort_values("label").to_records(index=False)
    labels = [lbl for _, lbl in options]
    id_by_label = {lbl: _id for _id, lbl in options}

    # No default selection on first load
    picked_label = st.selectbox(
        "Pick a location",
        labels,
        index=None,                      # <- key bit: no preselection
        placeholder="— Select a location —",
    )

    return id_by_label.get(picked_label) if picked_label else None

def legend_box():
    st.markdown("#### Legend")
    st.markdown(
        '''
        **Nodes**  
        <span style="display:inline-block;width:12px;height:12px;background:#EA4E4E;border:1px solid #999;margin-right:6px;"></span> Province  
        <span style="display:inline-block;width:12px;height:12px;background:#B4B4B4;border:1px solid #999;margin-right:6px;"></span> Municipality  
        
        **Edges**  
        <span style="display:inline-block;width:12px;height:12px;background:#EA4E4E;border:1px solid #999;margin-right:6px;"></span> Proximity  
        <span style="display:inline-block;width:12px;height:12px;background:#6E6E6E;border:1px solid #999;margin-right:6px;"></span> Ownership  
        <span style="display:inline-block;width:12px;height:12px;background:#945FBD;border:1px solid #999;margin-right:6px;"></span> Similarity  
        ''',
        unsafe_allow_html=True,
    )


# For details pane

def _get_first_present(row: pd.Series, keys: list[str], default: str = "—") -> Any:
    for k in keys:
        if k in row.index and pd.notna(row[k]) and row[k] != "":
            return row[k]
    return default

def node_details_panel(nodes_df: pd.DataFrame, selected_id: str | None) -> None:
    """Render a compact details pane for the selected node."""
    st.markdown("#### Details")

    if not selected_id:
        st.info("Select a node from the picker to see details.")
        return

    row = nodes_df.loc[nodes_df["id"].astype(str) == str(selected_id)]
    if row.empty:
        st.warning("Selected node not found in data.")
        return

    r = row.iloc[0]

    # Pull common fields with graceful fallbacks
    name        = _get_first_present(r, ["location_name", "name", "location"])
    province    = _get_first_present(r, ["province"])
    ntype       = _get_first_present(r, ["location_type", "type"])
    tier        = _get_first_present(r, ["tier"])
    population  = _get_first_present(r, ["population"])
    lat         = _get_first_present(r, ["lat", "latitude"])
    lon         = _get_first_present(r, ["lon", "longitude"])

    # Format population nicely if numeric
    try:
        if population not in ("—", ""):
            population = f"{int(float(population)):,}"
    except Exception:
        pass

    st.markdown(f"""
        **{name}**  
        - **Province:** {province}  
        - **Type:** {ntype}  
        - **Tier:** {tier}  
        - **Population:** {population}  
        - **Coordinates:** {lat}, {lon}  
        - **Location ID:** {r.get('id','—')}
        """)

    # Optionally show a tiny raw record expander for debugging
    with st.expander("Raw record", expanded=False):
        st.write(r.to_dict())

def filters_panel(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    key_prefix: str = "filters",
) -> Tuple[List[str], List[str]]:
    """
    Render multiselects for node types and edge types. Returns (node_types, edge_types).
    Defaults select all available types on first render.
    """
    st.markdown("#### Filters")

    # Detect available types
    node_types = sorted(
        nodes_df["location_type"].dropna().astype(str).str.lower().unique().tolist()
    ) if "location_type" in nodes_df.columns else []

    edge_types = sorted(
        edges_df["type"].dropna().astype(str).str.lower().unique().tolist()
    ) if "type" in edges_df.columns else []

    # Session defaults: select all on first run
    n_key = f"{key_prefix}_node_types"
    e_key = f"{key_prefix}_edge_types"
    if n_key not in st.session_state:
        st.session_state[n_key] = node_types[:]  # all
    if e_key not in st.session_state:
        st.session_state[e_key] = edge_types[:]  # all

    picked_nodes = st.multiselect(
        "Node types",
        options=node_types,
        default=st.session_state[n_key],
        help="Choose which node types to show",
        key=n_key,  # persist selection
    )

    picked_edges = st.multiselect(
        "Edge types",
        options=edge_types,
        default=st.session_state[e_key],
        help="Choose which edge types to show",
        key=e_key,  # persist selection
    )

    return picked_nodes, picked_edges