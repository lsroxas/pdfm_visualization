from __future__ import annotations
import os
from typing import Optional, Set, Tuple, Dict, Iterable

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import networkx as nx

from streamlit_network_explorer.graph_logic import k_hop_nodes

# --- Helpers -----------------------------------------------------------------

_LAT_KEYS = ("lat", "latitude", "y")
_LON_KEYS = ("lon", "long", "lng", "longitude", "x")

def _get_first(attrs: Dict, keys: Iterable[str]):
    for k in keys:
        if k in attrs and attrs[k] is not None and attrs[k] != "":
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
        nodes.append({
            "id": str(n),
            "label": a.get("label", str(n)),
            "lon": lon,
            "lat": lat,
            "deg": int(G.degree(n)),
            "sel": (str(n) == str(selected)),
            "hi": (selected is not None and n in hi),
            **{k: v for k, v in a.items() if k.lower() not in set(_LAT_KEYS + _LON_KEYS)},
        })
    ndf = pd.DataFrame(nodes)

    edges = []
    for u, v, e in G.edges(data=True):
        au, av = G.nodes[u], G.nodes[v]
        lon1, lat1 = _extract_lon_lat(au)
        lon2, lat2 = _extract_lon_lat(av)
        if None in (lon1, lat1, lon2, lat2):
            continue
        edges.append({
            "source": str(u), "target": str(v),
            "lon1": lon1, "lat1": lat1,
            "lon2": lon2, "lat2": lat2,
            "weight": e.get("weight", 1),
            "hi": (selected is not None and (u in hi or v in hi)),
        })
    edf = pd.DataFrame(edges)
    return ndf, edf

def _autocenter(nodes_df: pd.DataFrame, fallback_center: Tuple[float, float], fallback_zoom: float):
    if nodes_df.empty:
        return fallback_center[0], fallback_center[1], fallback_zoom
    # robust center using median; set zoom from spread
    lat_med = float(nodes_df["lat"].median())
    lon_med = float(nodes_df["lon"].median())
    lat_ptp = float(nodes_df["lat"].max() - nodes_df["lat"].min())
    lon_ptp = float(nodes_df["lon"].max() - nodes_df["lon"].min())
    # crude zoom heuristic: smaller spread -> higher zoom
    spread = max(lat_ptp, lon_ptp)
    zoom = 5.0
    if spread < 0.2: zoom = 8.0
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

    # Layers
    edge_layer = pdk.Layer(
        "LineLayer",
        data=edges_df,
        get_source_position='[lon1, lat1]',
        get_target_position='[lon2, lat2]',
        get_width="hi ? 2.5 : 0.5",
        get_color="hi ? [255,127,14,180] : [180,180,180,100]",
        pickable=False,
    )

    node_layer = pdk.Layer(
        "ScatterplotLayer",
        data=nodes_df,
        get_position='[lon, lat]',
        get_radius="sel ? 100 : hi ? 80 : 55",
        radius_units="meters",
        get_fill_color="sel ? [255,127,14,220] : hi ? [31,119,180,220] : [180,180,180,160]",
        get_line_color="[255,255,255,240]",
        line_width_min_pixels=1,
        pickable=True,
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
        layers = [tile] + layers
        r = pdk.Deck(
            initial_view_state=view_state,
            layers=layers,
            tooltip={"html": "<b>{label}</b><br/>Degree: {deg}<br/>ID: {id}", "style": {"color": "white"}},
        )

    st.pydeck_chart(r, use_container_width=True, height=height)