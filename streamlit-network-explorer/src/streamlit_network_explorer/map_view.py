
from __future__ import annotations
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

NODE_COLORS = {
    "province": "#EA4E4E",
    "municipality": "#0E00A6",
    "default": "#FF7F0E",
}

EDGE_COLORS = {
    "proximity": "#EA4E4E",
    "ownership": "#6E6E6E",
    "similarity": "#9467BD",
    "default": "#FFAA00",
}

NODE_RADIUS = {
    "province": 20,
    "municipality": 10,
    "default": 50,
}

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

def render_map(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    selected_id: Optional[str] = None,
    height: int = 700,
    initial_zoom: float = 5.0,
    philippines_center: Tuple[float, float] = (12.8797, 121.7740),
    node_types_filter: Optional[list[str]] = None,   # for node filtering
    edge_types_filter: Optional[list[str]] = None,   # for edge filtering
):
    if nodes_df.empty:
        st.warning("No nodes to render.")
        return

    # --- Apply filters ---
    nd = nodes_df.copy()
    if node_types_filter and "location_type" in nd.columns:
        nd = nd[nd["location_type"].astype(str).str.lower().isin(
            [t.lower() for t in node_types_filter]
        )].copy()

    ed = edges_df.copy()
    if edge_types_filter and "type" in ed.columns:
        ed = ed[ed["type"].astype(str).str.lower().isin(
            [t.lower() for t in edge_types_filter]
        )].copy()

    # Drop edges whose endpoints are no longer present after node filter
    if not nd.empty and not ed.empty:
        valid_ids = set(nd["id"].astype(str))
        ed = ed[ed["source"].astype(str).isin(valid_ids) & ed["target"].astype(str).isin(valid_ids)].copy()

    # Use filtered frames from here on
    nodes_df = nd
    edges_df = ed

    lat0, lon0, z0 = _autocenter(nodes_df, philippines_center, initial_zoom)

    ss = st.session_state
    last_centered = ss.get("map_last_centered_id")
    if selected_id and str(selected_id) != str(last_centered):
        sel = nodes_df.loc[nodes_df["id"].astype(str) == str(selected_id)]
        if not sel.empty:
            lat0 = float(sel["lat"].iloc[0])
            lon0 = float(sel["lon"].iloc[0])
            z0 = 8.5
            ss["map_last_centered_id"] = str(selected_id)

    m = folium.Map(
        location=[lat0, lon0],
        zoom_start=z0,
        tiles="CartoDB positron",
        control_scale=False,
        prefer_canvas=True,
    )

    if not edges_df.empty:
        nodes_idx = nodes_df.set_index("id")
        for _, e in edges_df.iterrows():
            u = str(e["source"]); v = str(e["target"])
            if u not in nodes_idx.index or v not in nodes_idx.index:
                continue
            urow = nodes_idx.loc[u]; vrow = nodes_idx.loc[v]
            lat1, lon1 = float(urow.get("lat")), float(urow.get("lon"))
            lat2, lon2 = float(vrow.get("lat")), float(vrow.get("lon"))
            if any(pd.isna([lat1, lon1, lat2, lon2])):
                continue
            etype = str(e.get("type", "default")).lower()
            color = EDGE_COLORS.get(etype, EDGE_COLORS["default"])
            folium.PolyLine(
                locations=[(lat1, lon1), (lat2, lon2)],
                color=color,
                weight=2,
                opacity=0.8,
            ).add_to(m)

    for _, r in nodes_df.iterrows():
        ntype = str(r.get("location_type", "default")).lower()
        color = NODE_COLORS.get(ntype, NODE_COLORS["default"])
        radius = NODE_RADIUS.get(ntype, NODE_RADIUS["default"])

        name = r.get("location_name", "")
        prov = r.get("province", "")
        tier = r.get("tier", "")
        pop  = r.get("population", "")
        tooltip_html = (
            f"<b>{name}</b><br>"
            f"Province: {prov}<br>"
            f"Tier: {tier}<br>"
            f"Population: {pop}"
        )

        folium.CircleMarker(
            location=(float(r["lat"]), float(r["lon"])),
            radius=max(3, int(radius)),
            color=None,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            weight=0,
            tooltip=folium.Tooltip(tooltip_html, sticky=True),
        ).add_to(m)

    st_folium(
        m,
        width=None,
        height=height,
        key="map_only",
        returned_objects=[],
    )


