# # src/streamlit_network_explorer/state.py
# import streamlit as st

# _KEY_SELECTED = "selected_node"

# def get_selected_node():
#     return st.session_state.get(_KEY_SELECTED, None)

# def set_selected_node(node_id):
#     st.session_state[_KEY_SELECTED] = node_id


import streamlit as st

_KEY = "selected_node"

def get_selected_node() -> str | None:
    v = st.session_state.get(_KEY)
    return str(v) if v is not None else None

def set_selected_node(v: str | None) -> None:
    st.session_state[_KEY] = (str(v) if v is not None else None)
    print(f"Selected node set to: {st.session_state[_KEY]}")
