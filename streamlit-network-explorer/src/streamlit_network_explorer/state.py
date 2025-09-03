
import streamlit as st

_SELECTED_KEY = "selected_node_id"

def get_selected_node() -> str | None:
    return st.session_state.get(_SELECTED_KEY)

def set_selected_node(node_id: str | None) -> None:
    if node_id is None:
        st.session_state.pop(_SELECTED_KEY, None)
    else:
        st.session_state[_SELECTED_KEY] = str(node_id)
