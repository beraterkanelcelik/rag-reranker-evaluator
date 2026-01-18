from __future__ import annotations

import streamlit as st

from frontend.api.client import APIClient


def render_system_section(client: APIClient) -> None:
    with st.expander("5. System Management", expanded=False):
        if "system_memory" not in st.session_state:
            st.session_state["system_memory"] = {}

        if st.button("Refresh memory", key="system_refresh"):
            status = _fetch_memory(client)
            st.session_state["system_memory"] = status

        memory = st.session_state.get("system_memory")
        if memory:
            col1, col2, col3 = st.columns(3)
            col1.metric("System Memory (MB)", memory.get("system_memory_mb", 0))
            col2.metric("Available (MB)", memory.get("available_memory_mb", 0))
            col3.metric("Loaded Models (MB)", memory.get("total_memory_mb", 0))
            st.json(memory.get("loaded_models", []))
        else:
            st.caption("Click refresh to load memory stats.")

        st.markdown("**Unload Models**")
        if st.button("Unload all models", type="primary", key="system_unload"):
            try:
                response = client.unload_models()
                st.success(response.get("message", "Models unloaded"))
                st.session_state["system_memory"] = _fetch_memory(client)
            except Exception as exc:
                st.error(str(exc))


def _fetch_memory(client: APIClient) -> dict:
    try:
        return client.get_system_memory()
    except Exception:
        return {}
