# src/streamlit_network_explorer/state.py
import streamlit as st

_KEY_SELECTED = "selected_node"

def get_selected_node():
    return st.session_state.get(_KEY_SELECTED, None)

def set_selected_node(node_id):
    st.session_state[_KEY_SELECTED] = node_id
