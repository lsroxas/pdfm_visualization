
from __future__ import annotations
import streamlit as st
import pandas as pd

def header(title: str, subtitle: str | None = None):
    st.title(title)
    if subtitle:
        st.caption(subtitle)

def node_picker(nodes_df: pd.DataFrame, selected_id: str | None) -> str | None:
    if nodes_df.empty:
        st.info("No nodes loaded.")
        return None
    disp = nodes_df.copy()
    disp["label"] = disp["location_name"].astype(str) + " — " + disp["province"].astype(str)
    options = disp[["id","label"]].drop_duplicates().sort_values("label").to_records(index=False)
    labels = [lbl for _, lbl in options]
    id_by_label = {lbl: _id for _id, lbl in options}
    cur_label = None
    if selected_id is not None:
        row = disp.loc[disp["id"].astype(str) == str(selected_id)]
        if not row.empty:
            cur_label = (row["location_name"].iloc[0] + " — " + row["province"].iloc[0])
    picked_label = st.selectbox("Pick a location", labels, index=(labels.index(cur_label) if cur_label in labels else 0))
    return id_by_label.get(picked_label)

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
