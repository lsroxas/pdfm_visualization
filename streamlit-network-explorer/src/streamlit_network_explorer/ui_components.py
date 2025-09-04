
from __future__ import annotations
import streamlit as st
import pandas as pd

# For details pane
from typing import Any, Dict, Iterable, Tuple, List

#For additional details
from typing import Optional, Sequence

#For counterfactuals
import json
import pathlib, pickle, hashlib
import numpy as np
import xgboost

# ---- model + lime utils -----------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_pickle_bundle(model_path: str):
    """Load your pickle bundle: expect a dict with at least a 'model' key."""
    p = pathlib.Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    with p.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    if isinstance(obj, dict):
        raise ValueError(f"Pickle contains a dict but no 'model' key. Keys: {list(obj.keys())}")
    # Fallback: user pickled the model directly
    return {"model": obj, "feature_names": getattr(obj, "feature_names_in_", None), "label_map": None}

def _detect_mode(model) -> str:
    return "classification" if hasattr(model, "predict_proba") else "regression"

def _hash_background(df: pd.DataFrame) -> str:
    h = hashlib.md5()
    h.update(str(df.shape).encode())
    h.update(",".join(map(str, df.columns)).encode())
    if not df.empty:
        h.update(pd.util.hash_pandas_object(df.head(50), index=False).values.tobytes())
    return h.hexdigest()

def _safe_predict_fn(model, feature_cols: Sequence[str]):
    """Ensure X arrives with correct shape & columns for model.predict(_proba)."""
    n_feat = len(feature_cols)
    def _to_frame(X):
        if isinstance(X, pd.DataFrame):
            return X.loc[:, feature_cols]
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != n_feat:
            raise ValueError(f"predict_fn received {X.shape[1]} features, expected {n_feat}.")
        return pd.DataFrame(X, columns=feature_cols)
    def _predict(X):
        Xf = _to_frame(X)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(Xf)
        y = model.predict(Xf)
        return y if y.ndim == 1 else y.reshape(-1)
    return _predict

@st.cache_resource(show_spinner=False)
def _build_lime_explainer(
    background_df: pd.DataFrame,
    feature_cols: Sequence[str],
    class_names: Optional[Sequence[str]],
    categorical_features: Optional[Sequence[str]],
    mode: str,
    _bg_key: str,
):
    from lime.lime_tabular import LimeTabularExplainer
    X_bg = background_df[feature_cols].to_numpy()

    cat_idx = None
    if categorical_features:
        name_to_idx = {n: i for i, n in enumerate(feature_cols)}
        cat_idx = [name_to_idx[n] for n in categorical_features if n in name_to_idx]

    return LimeTabularExplainer(
        training_data=X_bg,
        feature_names=list(feature_cols),
        class_names=list(class_names) if class_names is not None else None,
        categorical_features=cat_idx,
        mode=mode,
        discretize_continuous=True,
        random_state=42,
        verbose=False,
    )

def _lime_explain_row(model, explainer, row: pd.Series, feature_cols: Sequence[str], num_features: int = 10):
    label_to_explain = None
    if hasattr(model, "predict_proba"):
        probs = _predict_fn_for(model)(row[feature_cols].to_frame().T)
        label_to_explain = int(np.argmax(probs, axis=1)[0])

    x = row[feature_cols].to_numpy()
    exp = explainer.explain_instance(
        data_row=x,
        predict_fn=lambda X: _predict_fn_for(model)(pd.DataFrame(X, columns=feature_cols)),
        num_features=num_features,
        top_labels=1,
        labels=[label_to_explain] if label_to_explain is not None else None,
    )
    return exp, label_to_explain

# ---- counterfactuals helpers you already have (trimmed to essentials) -------

def _normalize_tier(value) -> str | None:
    if value is None: return None
    import re
    m = re.search(r"[1-4]", str(value))
    return f"Tier {int(m.group(0))}" if m else None

def _desired_tier(current: str | None) -> str | None:
    cur = _normalize_tier(current)
    if not cur: return None
    n = max(1, int(cur.split()[-1]) - 1)
    return f"Tier {n}"

def _parse_top_changes(val) -> list[dict]:
    if val is None or (isinstance(val, float) and pd.isna(val)): return []
    try:
        parsed = json.loads(val) if isinstance(val, str) else val
        if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
            return parsed
    except Exception:
        pass
    return []

def _render_counterfactuals(changes: list[dict]) -> str:
    valid = [d for d in changes if isinstance(d.get("cf"), (int, float)) and d["cf"] < 10]
    valid.sort(key=lambda d: d["cf"])
    if not valid:
        return "No valid counterfactual"
    return "<ul style='margin:0;padding-left:18px;'>" + "".join(
        f"<li>{d.get('feature','?')}: {d['cf']}</li>" for d in valid
    ) + "</ul>"

@st.cache_data(show_spinner=False)
def _load_counterfactuals_csv(cfg: Dict) -> pd.DataFrame:
    path = cfg.get("path", "data/counterfactuals.csv")
    delim = cfg.get("delimiter", ",")
    return pd.read_csv(path, delimiter=delim)



#----- UI Components
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


@st.cache_data(show_spinner=True)
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

####### Counterfactuals pane

# --- Tier helpers -------------------------------------------------------------
def _normalize_tier(value) -> str | None:
    """
    Accepts 'Tier 3', 'tier3', '3', 3, etc. Returns normalized 'Tier X' or None.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Extract the first digit in 1..4
    import re
    m = re.search(r"[1-4]", s)
    if not m:
        return None
    n = int(m.group(0))
    return f"Tier {n}"

def _desired_tier(current: str | None) -> str | None:
    """
    Given a normalized 'Tier N', return the next tier up (lower number).
    Tier 4 -> Tier 3 -> Tier 2 -> Tier 1 -> Tier 1.
    """
    cur = _normalize_tier(current)
    if cur is None:
        return None
    try:
        n = int(cur.split()[-1])
    except Exception:
        return None
    n = max(1, n - 1)  # move up a tier; clamp at Tier 1
    return f"Tier {n}"


@st.cache_data(show_spinner=False)
def _load_counterfactuals_csv(cfg: Dict) -> pd.DataFrame:
    path = cfg.get("path", "data/counterfactuals.csv")
    delim = cfg.get("delimiter", ",")
    return pd.read_csv(path, delimiter=delim)

def _coerce_changes(val: Any) -> List[str]:
    """Turn a cell (JSON/text/NaN) into a list[str] of changes."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    # Try JSON list
    try:
        parsed = json.loads(s)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
    except Exception:
        pass
    # Fallback: split on common separators if not JSON
    if ";" in s:
        return [x.strip() for x in s.split(";") if x.strip()]
    if "|" in s:
        return [x.strip() for x in s.split("|") if x.strip()]
    # Single item
    return [s] if s else []

def _render_bullets(items: List[str]) -> str:
    if not items:
        return "—"
    return "<ul style='margin:0;padding-left:18px;'>" + "".join(
        f"<li>{st._utils.escape_markdown(i, unsafe_allow_html=True) if hasattr(st, '_utils') else i}</li>"
        for i in items
    ) + "</ul>"


def _parse_top_changes(val) -> list[dict]:
    """
    Parse a top_changes cell into a list of {feature, cf} dicts.
    Returns [] if parsing fails.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    try:
        parsed = json.loads(val) if isinstance(val, str) else val
        if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
            return parsed
    except Exception:
        pass
    return []


def _render_counterfactuals(changes: list[dict]) -> str:
    """
    Render a bullet list of feature: cf, sorted ascending by cf, cf<10.
    If none valid, return 'No valid counterfactual'.
    """
    # filter and sort
    valid = [
        d for d in changes
        if "cf" in d and isinstance(d["cf"], (int, float)) and d["cf"] < 10
    ]
    valid.sort(key=lambda d: d["cf"])

    if not valid:
        return "No valid counterfactual"

    return "<ul style='margin:0;padding-left:18px;'>" + "".join(
        f"<li>{d.get('feature', '?')}: {d['cf']}</li>" for d in valid
    ) + "</ul>"




def counterfactuals_panel(cfg: Dict, selected_id: Optional[str], nodes_df: pd.DataFrame, lime_df: pd.DataFrame) -> None:
    """
    Show Counterfactuals:
    - Tier comes from nodes_df (lookup by selected_id).
    - Only read/display top_changes from data.counterfactuals CSV.
    """
    st.markdown("#### Counterfactuals")

    if not selected_id:
        st.info("Pick a node to view counterfactuals.")
        return

    # --- Tier from nodes_df
    tier_val = "—"
    if not nodes_df.empty and "id" in nodes_df.columns:
        row = nodes_df.loc[nodes_df["id"].astype(str) == str(selected_id)]
        if not row.empty:
            # Be defensive about the tier column name just in case
            if "tier" in row.columns:
                tier_val = row.iloc[0]["tier"]
            else:
                # Try a couple of alternates
                for c in ("Tier", "TIER"):
                    if c in row.columns:
                        tier_val = row.iloc[0][c]
                        break
    
    desired_tier = _desired_tier(tier_val) or "—"

    cf_cfg = (cfg or {}).get("data", {}).get("counterfactuals", {})
    if not cf_cfg:
        st.caption("No 'data.counterfactuals' section configured in data_config.yaml.")
        # Still show tier so the pane isn’t empty
        st.markdown(f"**Tier:** {tier_val}")
        return

    id_col  = cf_cfg.get("id_col", "location_id")
    chg_col = cf_cfg.get("top_changes_col", "top_changes")

    df = _load_counterfactuals_csv(cf_cfg)
    # Keep only columns we care about (if present)
    needed = [c for c in (id_col, chg_col) if c in df.columns]
    if not needed or id_col not in df.columns or chg_col not in df.columns:
        st.error(f"Counterfactuals CSV must contain '{id_col}' and '{chg_col}'.")
        st.markdown(f"**Tier:** {tier_val}")
        return
    df = df[needed].copy()

    matches = df.loc[df[id_col].astype(str) == str(selected_id)]
    # # Merge all top changes (if multiple rows, union)
    # all_changes: List[str] = []
    # for _, r in matches.iterrows():
    #     all_changes.extend(_coerce_changes(r.get(chg_col)))
    # # Deduplicate preserving order
    # seen = set()
    # merged_changes = [x for x in all_changes if not (x in seen or seen.add(x))]

    # Collect all top_changes dicts across rows
    all_changes: list[dict] = []
    for _, r in matches.iterrows():
        all_changes.extend(_parse_top_changes(r.get(chg_col)))

    # Render
    c1, c2 = st.columns([1, 2])
    with c1:
        st.caption("DiCE")
        st.markdown(f"**Current:** { _normalize_tier(tier_val) or '—' }")
        st.markdown(f"**Desired:** { desired_tier }")
        st.markdown("**Top changes:**")
        st.markdown(_render_counterfactuals(all_changes), unsafe_allow_html=True)
    with c2:
        # st.markdown("**Top changes:**")
        # st.markdown(_render_counterfactuals(all_changes), unsafe_allow_html=True)
        st.caption("**LIME explanation:**")

        # --- LIME explanation (model from pickle; features from nodes CSV) ---
        lime_cfg   = (cfg or {}).get("data", {}).get("lime", {}) or {}
        model_path = lime_cfg.get("path")

        if not model_path:
            st.caption("Configure `data.lime.path` in data_config.yaml to enable LIME.")
            return

        # 1) Load bundle and extract model + canonical feature names
        try:
            bundle = _load_pickle_bundle(model_path)  # {'model','feature_names','label_map',...}
        except Exception as e:
            st.error(f"Model load failed: {e}")
            return

        model = bundle["model"]

        # Priority for feature list (ORDER MATTERS)
        if bundle.get("feature_names"):
            feature_cols = list(map(str, bundle["feature_names"]))
        elif hasattr(model, "feature_names_in_"):
            feature_cols = list(map(str, model.feature_names_in_))
        else:
            fc = lime_cfg.get("lime_cols", [])
            feature_cols = [c.strip() for c in fc.split(",")] if isinstance(fc, str) else list(map(str, fc))

        if not feature_cols:
            st.error("No feature list found. Provide bundle['feature_names'], model.feature_names_in_, or data.lime.feature_cols.")
            return

        # 2) Align your lime_df to EXACT columns and order (drop extras, add missings)
        df_cols = set(lime_df.columns)
        needed  = list(feature_cols)  # keep order
        missing = [c for c in needed if c not in df_cols]
        extra   = [c for c in lime_df.columns if c not in needed]

        # Create missing columns with neutral values (choose what's appropriate for your model)
        for c in missing:
            # For numeric features, 0 is often okay; adjust if your model expects something else
            lime_df[c] = 0

        # Reindex to EXACT order; extras will be ignored by reindex
        X_all = lime_df.reindex(columns=needed)

        # Optional: basic impute/typing to avoid NaNs and dtype issues
        num_cols = X_all.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in needed if c not in num_cols]
        if num_cols:
            X_all[num_cols] = X_all[num_cols].fillna(X_all[num_cols].median())
        if cat_cols:
            X_all[cat_cols] = X_all[cat_cols].astype("string").fillna("missing")

        # 3) Build background from the ALIGNED matrix
        if X_all.empty:
            st.error("Aligned LIME matrix is empty after reindexing.")
            return
        bg = X_all.sample(min(3000, len(X_all)), random_state=42)
        bg_key = _hash_background(bg)

        # 4) Find the selected row using an ID column PRESENT in lime_df
        id_candidates = [
            (cfg or {}).get("data", {}).get("nodes", {}).get("id_col"),
            (cfg or {}).get("data", {}).get("counterfactuals", {}).get("id_col"),
            "location_id", "Location_ID", "id", "node_id", "Node_ID",
        ]
        id_col_lime = next((c for c in id_candidates if c and c in lime_df.columns), None)
        if not id_col_lime:
            st.error("Could not find an ID column in lime_df to match the selected node.")
            return

        sid = str(selected_id).strip().casefold()
        ids_norm = lime_df[id_col_lime].astype(str).str.strip().str.casefold()
        row_df = X_all.loc[ids_norm == sid]

        if row_df.empty:
            st.error(f"No row matched selected_id={selected_id!r} via {id_col_lime!r}.")
            st.caption("Tip: ensure your node picker emits the same ID used in the lime_df ID column.")
            return
        if len(row_df) > 1:
            st.warning(f"Multiple rows matched {selected_id!r}; using the first.")
            row_df = row_df.iloc[[0]]

        row_series = row_df.iloc[0]  # <- EXACT same columns & order as feature_cols

        # 5) Build explainer and a shape-safe predict_fn
        mode = _detect_mode(model)
        class_names = lime_cfg.get("label_names", None)
        if class_names is None and isinstance(bundle.get("label_map"), dict):
            try:
                inv = {v: k for k, v in bundle["label_map"].items()}
                class_names = [inv[i] for i in sorted(inv)]
            except Exception:
                pass

        categorical_features = lime_cfg.get("categorical_features", None)
        try:
            explainer = _build_lime_explainer(
                background_df=bg,
                feature_cols=feature_cols,
                class_names=class_names,
                categorical_features=categorical_features,
                mode=mode,
                _bg_key=bg_key,
            )
        except ModuleNotFoundError:
            st.error("LIME is not installed. Run: `uv pip install lime`")
            return

        predict_fn = _safe_predict_fn(model, feature_cols)

        # 6) Run LIME on the ALIGNED one-row instance
        try:
            label_idx = None
            if hasattr(model, "predict_proba"):
                proba = predict_fn(row_series.to_numpy())
                label_idx = int(np.argmax(proba, axis=1)[0])

            exp = explainer.explain_instance(
                data_row=row_series.to_numpy(),   # 1D array, EXACT order
                predict_fn=predict_fn,
                num_features=10,
                top_labels=1,
                labels=[label_idx] if label_idx is not None else None,
            )

            pairs = exp.as_list(label=label_idx) if label_idx is not None else exp.as_list()
            wdf = pd.DataFrame(pairs, columns=["Feature (range/condition)", "Weight"])
            st.dataframe(wdf, hide_index=True, use_container_width=True)

            with st.expander("HTML view", expanded=False):
                st.components.v1.html(
                    exp.as_html(label=label_idx) if label_idx is not None else exp.as_html(),
                    height=600, scrolling=True
                )
        except Exception as e:
            # st.error(f"LIME failed: {e}")
            st.error("LIME failed: Unable to load model.")
                    
    if matches.empty:
        st.caption("No matching counterfactual rows found for this node.")


@st.cache_resource(show_spinner=False)
def _load_pickle_model(model_path: str):
    p = pathlib.Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    with p.open("rb") as f:
        obj = pickle.load(f)

    # If pickle contains a dict, extract the model
    if isinstance(obj, dict):
        # adjust the key to match how you saved it
        if "model" in obj:
            return obj["model"]
        else:
            raise ValueError(f"Pickle contains a dict but no 'model' key. Keys: {list(obj.keys())}")

    return obj

# def _load_pickle_model(model_path: str):
#     p = pathlib.Path(model_path)
#     if not p.exists():
#         raise FileNotFoundError(f"Model file not found: {p}")
#     with p.open("rb") as f:
#         return pickle.load(f)

def _detect_mode(model) -> str:
    return "classification" if hasattr(model, "predict_proba") else "regression"

def _predict_fn_for(model):
    # function mapping ndarray/DataFrame -> ndarray for LIME
    if hasattr(model, "predict_proba"):
        return lambda X: model.predict_proba(X)
    return lambda X: np.atleast_2d(model.predict(X)).reshape(-1, 1)

def _hash_background(df: pd.DataFrame) -> str:
    # small stable cache key for background sample
    h = hashlib.md5()
    h.update(str(df.shape).encode())
    h.update(",".join(df.columns).encode())
    # include head few rows to invalidate if data changed a lot
    h.update(pd.util.hash_pandas_object(df.head(50), index=False).values.tobytes())
    return h.hexdigest()

@st.cache_resource(show_spinner=False)
def _build_lime_explainer(
    background_df: pd.DataFrame,
    feature_cols: Sequence[str],
    class_names: Optional[Sequence[str]],
    categorical_features: Optional[Sequence[str]],
    mode: str,
    _bg_key: str,   # cache key derived from background to avoid huge pickles
):
    from lime.lime_tabular import LimeTabularExplainer  # import here to fail gracefully
    X_bg = background_df[feature_cols].to_numpy()

    cat_idx = None
    if categorical_features:
        name_to_idx = {n: i for i, n in enumerate(feature_cols)}
        cat_idx = [name_to_idx[n] for n in categorical_features if n in name_to_idx]

    explainer = LimeTabularExplainer(
        training_data=X_bg,
        feature_names=list(feature_cols),
        class_names=list(class_names) if class_names is not None else None,
        categorical_features=cat_idx,
        mode=mode,
        discretize_continuous=True,
        random_state=42,
        verbose=False,
    )
    return explainer

def _lime_explain_row(
    model,
    explainer,
    row: pd.Series,
    feature_cols: Sequence[str],
    num_features: int = 10,
    top_labels: int = 1,
):
    # Choose label to explain (for classifiers)
    label_to_explain = None
    if hasattr(model, "predict_proba"):
        probs = _predict_fn_for(model)(row[feature_cols].to_frame().T)
        label_to_explain = int(np.argmax(probs, axis=1)[0])

    x = row[feature_cols].to_numpy()
    exp = explainer.explain_instance(
        data_row=x,
        predict_fn=lambda X: _predict_fn_for(model)(pd.DataFrame(X, columns=feature_cols)),
        num_features=num_features,
        top_labels=top_labels,
        labels=[label_to_explain] if label_to_explain is not None else None,
    )
    return exp, label_to_explain

