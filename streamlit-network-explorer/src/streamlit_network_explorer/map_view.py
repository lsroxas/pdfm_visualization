# src/streamlit_network_explorer/map_view.py
from __future__ import annotations
import os
from typing import Optional, Set, Tuple, Dict, Iterable

import pandas as pd
import pydeck as pdk
import streamlit as st
import networkx as nx

from streamlit_network_explorer.graph_logic import k_hop_nodes
from streamlit_deckgl import st_deckgl

# --- Helpers -----------------------------------------------------------------

_LAT_KEYS = ("lat", "latitude", "y")
_LON_KEYS = ("lon", "long", "lng", "longitude", "x")

# Type styles: base radius (meters) and color [R,G,B,A]
TYPE_STYLE = {
    "province": {"radius": 10, "color": [220, 68, 55, 220]},      # red-ish
    "municipality": {"radius": 5, "color": [180, 180, 180, 180]},   # grey
    "default": {"radius": 5, "color": [255, 127, 14, 240]},   # orange
}

# Optional highlight overrides (selected/neighbor)
SELECTED_COLOR = [255, 127, 14, 240]   # orange
NEIGHBOR_COLOR = [31, 119, 180, 220]   # blue
SELECTED_BOOST  = 6                    # +meters for selected
NEIGHBOR_BOOST  = 3

# Edge styles by type (MAP view)
EDGE_STYLE = {
    "proximity": {"color": [136, 136, 136, 160], "width": 1.0},  # gray
    "ownership": {"color": [41, 41, 41, 160], "width": 1.0},     # blue
    "similarity": {"color": [148, 103, 189, 160], "width": 1.0}, # purple
    "default": {"color": [255, 170, 0, 160], "width": 1.0},      # fallback
}

EDGE_HIGHLIGHT_WIDTH = 2.5

def _get_first(attrs: Dict, keys: Iterable[str]):
    for k in keys:
        if k in attrs and attrs[k] not in (None, ""):
            return attrs[k]
    return None

def _extract_lon_lat(attrs: Dict):
    """Return (lon, lat) as floats if possible."""
    lat = _get_first(attrs, _LAT_KEYS)
    lon = _get_first(attrs, _LON_KEYS)
    if lat is None or lon is None:
        return None, None
    try:
        return float(lon), float(lat)
    except Exception:
        return None, None

def _has_geo(G: nx.Graph) -> bool:
    for _, a in G.nodes(data=True):
        lon, lat = _extract_lon_lat(a)
        if lon is not None and lat is not None:
            return True
    return False

def _graph_to_frames(G: nx.Graph, selected: Optional[str], hops: int):
    hi: Set = k_hop_nodes(G, selected, hops) if selected is not None else set()

    nodes = []
    for n, a in G.nodes(data=True):
        lon, lat = _extract_lon_lat(a)
        if lon is None or lat is None:
            continue

        node_type = str(a.get("type", "default")).lower()
        base = TYPE_STYLE.get(node_type, TYPE_STYLE["default"])
        radius = float(base["radius"])
        fill   = list(base["color"])

        is_sel = (str(n) == str(selected))
        is_hi  = (selected is not None and n in hi and not is_sel)

        # Highlight/selection overrides
        if is_sel:
            fill = SELECTED_COLOR
            radius += SELECTED_BOOST
        elif is_hi:
            fill = NEIGHBOR_COLOR
            radius += NEIGHBOR_BOOST

        nodes.append({
            "id": str(n),
            "label": a.get("label", str(n)),
            "type": node_type,
            "lon": lon,
            "lat": lat,
            "deg": int(G.degree(n)),
            "sel": is_sel,
            "hi": is_hi,
            "radius": radius,   # <-- computed here
            "fill": fill,       # <-- computed here
            **{k: v for k, v in a.items() if k not in ("lat","lon","long")},
        })

    ndf = pd.DataFrame(nodes)

    edges = []
    for u, v, e in G.edges(data=True):
        au, av = G.nodes[u], G.nodes[v]
        lon1, lat1 = _extract_lon_lat(au)
        lon2, lat2 = _extract_lon_lat(av)
        if None in (lon1, lat1, lon2, lat2):
            continue

        etype = str(e.get("type", "default")).lower()
        estyle = EDGE_STYLE.get(etype, EDGE_STYLE["default"])

        edges.append({
            "source": str(u), "target": str(v),
            "lon1": lon1, "lat1": lat1,
            "lon2": lon2, "lat2": lat2,
            "weight": e.get("weight", 1),
            "type": etype,
            "base_color": estyle["color"],
            "base_width": estyle["width"],
            "hi": (selected is not None and (u in hi or v in hi)),
        })
    edf = pd.DataFrame(edges)
    return ndf, edf

def _autocenter(nodes_df: pd.DataFrame, fallback_center: Tuple[float, float], fallback_zoom: float):
    if nodes_df.empty:
        return fallback_center[0], fallback_center[1], fallback_zoom
    lat_med = float(nodes_df["lat"].median())
    lon_med = float(nodes_df["lon"].median())
    lat_ptp = float(nodes_df["lat"].max() - nodes_df["lat"].min())
    lon_ptp = float(nodes_df["lon"].max() - nodes_df["lon"].min())
    spread = max(lat_ptp, lon_ptp)
    zoom = 5.0
    if spread < 0.2:   zoom = 8.0
    elif spread < 0.6: zoom = 6.5
    elif spread < 1.5: zoom = 6.0
    return lat_med, lon_med, zoom

# --- Renderer ----------------------------------------------------------------

def render_map(
    G: nx.Graph,
    selected: Optional[str] = None,
    hops: int = 1,
    height: int = 700,
    initial_zoom: float = 5.0,
    philippines_center: Tuple[float, float] = (12.8797, 121.7740),  # lat, lon
    center_on_node: Optional[str] = None,      # <-- add this
    zoom_on_center: float = 8.5,               # <-- and this (used below)
):
    """Render a Deck.gl map with node and edge layers over the Philippines."""
    if not _has_geo(G):
        st.warning(
            "No geographic coordinates found on nodes. "
            "Ensure your nodes CSV has columns like 'lat' and 'lon' (or 'latitude'/'longitude')."
        )
        return

    nodes_df, edges_df = _graph_to_frames(G, selected, hops)

    # Debug panel to verify what we got
    # with st.expander("🧪 Map debug", expanded=False):
    #     st.write(f"Nodes w/ coords: {len(nodes_df)}")
    #     st.write(f"Edges w/ coords: {len(edges_df)}")
    #     if not nodes_df.empty:
    #         st.dataframe(nodes_df.head(5), use_container_width=True)

    # Compute center from data if present; else fallback to Philippines
    lat0, lon0, z0 = _autocenter(nodes_df, philippines_center, initial_zoom)

    # If a specific node is requested, center on it (if it has coords)
    if center_on_node and not nodes_df.empty:
        sel = nodes_df.loc[nodes_df["id"] == str(center_on_node)]
        if not sel.empty:
            lat0 = float(sel["lat"].iloc[0])
            lon0 = float(sel["lon"].iloc[0])
            z0 = float(zoom_on_center)

    # Layers
    edge_layer = pdk.Layer(
        "LineLayer",
        data=edges_df,
        get_source_position='[lon1, lat1]',
        get_target_position='[lon2, lat2]',
        get_width="hi ? highlight_width : base_width",
        get_color="hi ? highlight_color : base_color",
        parameters={
            "highlight_width": EDGE_HIGHLIGHT_WIDTH,
            # "highlight_color": EDGE_HIGHLIGHT_COLOR,
        },
        pickable=False,
        auto_highlight=True,
    )

    node_layer = pdk.Layer(
        "ScatterplotLayer",
        id="nodes",                    # important for selection extraction
        data=nodes_df,
        get_position='[lon, lat]',
        get_radius="radius",
        radius_units="meters",
        get_fill_color="fill",
        get_line_color="[255,255,255,200]",
        line_width_min_pixels=1,
        pickable=True,                 # must be True to receive click events
        auto_highlight=True,
    )

    # Map style: Mapbox if available, else OpenStreetMap TileLayer fallback
    token = st.secrets.get("MAPBOX_API_KEY", os.getenv("MAPBOX_API_KEY"))
    layers = [edge_layer, node_layer]
    view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=z0, bearing=0, pitch=0)

    if token:
        pdk.settings.mapbox_api_key = token
        r = pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v11",
            initial_view_state=view_state,
            layers=layers,
            tooltip={"html": "<b>{label}</b><br/>Degree: {deg}<br/>ID: {id}", "style": {"color": "white"}},
        )
    else:
        # OSM background via TileLayer so you see a basemap even w/o Mapbox token
        tile = pdk.Layer(
            "TileLayer",
            data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            min_zoom=0, max_zoom=19, tile_size=256,
        )
        r = pdk.Deck(
            initial_view_state=view_state,
            layers=layers,
            tooltip={"html": "<b>{label}</b><br/>Degree: {deg}<br/>ID: {id}", "style": {"color": "white"}},
        )

    # --- Render with true click events via streamlit-deckgl ---
    value = st_deckgl(
        r,                      # <-- must be a pydeck.Deck, not a dict
        height=height,
        key="deck_map",
        events=["click"],       # capture clicks on pickable layers
    )

    # Extract clicked object -> update global selected node -> rerun
    info = (value or {}).get("info") or {}
    obj = info.get("object") if isinstance(info, dict) else None
    if obj:
        # nodes_df rows come back as the object; sometimes wrapped in 'properties'
        nid = str(
            obj.get("id")
            or (obj.get("properties") or {}).get("id")
        )
        if nid:
            from streamlit_network_explorer import state as app_state
            if app_state.get_selected_node() != nid:
                app_state.set_selected_node(nid)
                st.rerun()

def render_legend():
    """Legend for Map view: node types + edge types."""
    with st.expander("Legend", expanded=True):
        st.markdown("### Node types")
        for ntype, sty in TYPE_STYLE.items():
            if ntype == "default":
                continue
            rgba = sty["color"]
            swatch = f"background: rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]/255:.2f});"
            st.markdown(
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'border-radius:2px;margin-right:8px;border:1px solid #999;{swatch}"></span>'
                f'{ntype}',
                unsafe_allow_html=True,
            )

        st.markdown("### Edge types")
        for etype, sty in EDGE_STYLE.items():
            if etype == "default":
                continue
            rgba = sty["color"]
            swatch = f"background: rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]/255:.2f});"
            st.markdown(
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'border-radius:2px;margin-right:8px;border:1px solid #999;{swatch}"></span>'
                f'{etype}',
                unsafe_allow_html=True,
            )