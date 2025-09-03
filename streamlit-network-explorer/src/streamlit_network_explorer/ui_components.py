
from __future__ import annotations
import streamlit as st
import pandas as pd

# For details pane
from typing import Any, Dict, Iterable, Tuple, List

#For additional details
from typing import Optional


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

def full_node_record_panel(nodes_df: pd.DataFrame, selected_id: Optional[str]) -> None:
    """Show every column/value for the selected node as a neat table."""
    st.markdown("#### Node attributes")

    if not selected_id:
        st.info("Pick a node to view all attributes.")
        return

    row = nodes_df.loc[nodes_df["id"].astype(str) == str(selected_id)]
    if row.empty:
        st.warning("Selected node not found in current data/filters.")
        return

    r = row.iloc[0]

    # Put commonly useful fields first (if they exist), then the rest
    preferred = ["id", "location_name", "province", "location_type", "tier", "population", "lat", "lon"]
    cols_present = list(r.index)
    ordered = [c for c in preferred if c in cols_present] + [c for c in cols_present if c not in preferred]

    # Build a (Field, Value) table
    df = pd.DataFrame({"Field": ordered, "Value": [r[c] for c in ordered]})
    st.dataframe(df, hide_index=True, use_container_width=True)


@st.cache_data(show_spinner=False)
def _load_nodes_raw_csv(nodes_cfg: Dict) -> pd.DataFrame:
    """Load the raw nodes CSV as-is, without renaming columns."""
    path = nodes_cfg.get("path", "data/nodes.csv")
    delim = nodes_cfg.get("delimiter", ",")
    df = pd.read_csv(path, delimiter=delim)
    return df

def full_node_record_from_csv(cfg: Dict, selected_id: Optional[str]) -> None:
    """
    Show every column/value for the selected node by looking it up in the *raw*
    nodes CSV using the configured id column (e.g., 'location_id').
    """
    st.markdown("#### Node attributes per modality")

    if not selected_id:
        st.info("Pick a node to view all attributes from the CSV.")
        return

    nodes_cfg = (cfg or {}).get("data", {}).get("nodes", {})
    if not nodes_cfg:
        st.error("Invalid config: missing data.nodes in data_config.yaml")
        return

    id_col = nodes_cfg.get("id_col", "location_id")  # default to 'location_id'
    df = _load_nodes_raw_csv(nodes_cfg)

    if id_col not in df.columns:
        st.error(f"The configured id_col '{id_col}' was not found in the nodes CSV.")
        return

    # Lookup by id (string-compare to be safe)
    m = df[id_col].astype(str) == str(selected_id)
    row = df.loc[m]
    if row.empty:
        st.warning(f"No row found in nodes CSV where {id_col} == {selected_id!r}.")
        return

    r = row.iloc[0]

    # Build a tidy Field/Value table (preferred fields first if present)
    preferred = [
        id_col, "location_name", "province", "location_type", "tier", "population",
        "latitude", "longitude", "lat", "lon",
    ]
    cols_present = list(r.index)
    ordered = [c for c in preferred if c in cols_present] + [c for c in cols_present if c not in preferred]

    table = pd.DataFrame({"Field": ordered, "Value": [r[c] for c in ordered]})
    st.dataframe(table, hide_index=True, use_container_width=True)


@st.cache_data(show_spinner=False)
def _load_nodes_raw_csv(nodes_cfg: Dict) -> pd.DataFrame:
    """Load the raw nodes CSV as-is, without renaming columns."""
    path = nodes_cfg.get("path", "data/nodes.csv")
    delim = nodes_cfg.get("delimiter", ",")
    return pd.read_csv(path, delimiter=delim)

def _field_value_df(row: pd.Series, columns: List[str]) -> pd.DataFrame:
    """Build a (Field, Value) DataFrame for the subset of columns that exist."""
    cols = [c for c in columns if c in row.index]
    if not cols:
        return pd.DataFrame({"Field": [], "Value": []})
    return pd.DataFrame({"Field": cols, "Value": [row[c] for c in cols]})

def full_node_record_grouped_from_csv(cfg: Dict, selected_id: Optional[str]) -> None:
    """
    Show all columns for the selected node, grouped by 'attribute_groups' from data_config.yaml.
    Each group is collapsible. Any remaining columns appear under 'Other'.
    """
    st.markdown("#### Node attributes")

    if not selected_id:
        st.info("Pick a node to view grouped attributes.")
        return

    nodes_cfg = (cfg or {}).get("data", {}).get("nodes", {})
    if not nodes_cfg:
        st.error("Invalid config: missing data.nodes in data_config.yaml")
        return

    id_col = nodes_cfg.get("id_col", "location_id")
    groups_cfg = nodes_cfg.get("attribute_groups", [])

    # Load raw CSV and find the row
    df = _load_nodes_raw_csv(nodes_cfg)
    if id_col not in df.columns:
        st.error(f"The configured id_col '{id_col}' was not found in the nodes CSV.")
        return

    row = df.loc[df[id_col].astype(str) == str(selected_id)]
    if row.empty:
        st.warning(f"No row found in nodes CSV where {id_col} == {selected_id!r}.")
        return
    r = row.iloc[0]

    # Track which columns we’ve displayed
    displayed: set[str] = set()

    # Render defined groups in order
    if isinstance(groups_cfg, dict):
        # also support dict form: {GroupName: [col1, col2, ...]}
        groups_iter = [{"name": k, "columns": v} for k, v in groups_cfg.items()]
    else:
        groups_iter = groups_cfg  # assume list of {name, columns}

    for g in groups_iter:
        name = str(g.get("name", "Group"))
        cols = list(g.get("columns", []))
        tbl = _field_value_df(r, cols)
        displayed.update([c for c in cols if c in r.index])

        with st.expander(name, expanded=False):
            if tbl.empty:
                st.caption("No fields from this group are present in the CSV.")
            else:
                st.dataframe(tbl, hide_index=True, use_container_width=True)

    # Any columns not covered by groups appear under "Other"
    remaining_cols = [c for c in r.index if c not in displayed]
    # Optional: hide pandas index-ish columns if present
    remaining_cols = [c for c in remaining_cols if c not in ("Unnamed: 0",)]

    if remaining_cols:
        with st.expander("Other", expanded=False):
            st.dataframe(_field_value_df(r, remaining_cols), hide_index=True, use_container_width=True)