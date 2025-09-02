# src/streamlit_network_explorer/styles.py
import streamlit as st

def palette():
    return {
        "selected": "#ff7f0e",
        "neighbor": "#1f77b4",
        "dim": "#c7c7c7",
        "edge": "#bbbbbb",
        "edge_highlight": "#ff7f0e",
    }

def inject_base_css():
    st.markdown("""
    <style>
    /* Add custom CSS tweaks here if desired */
    </style>
    """, unsafe_allow_html=True)
