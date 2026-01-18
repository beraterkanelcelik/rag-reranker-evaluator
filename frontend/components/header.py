from __future__ import annotations

import streamlit as st

from frontend.api.client import APIClient
from frontend.config import APP_VERSION


def render_header(client: APIClient) -> None:
    col1, col2, col3 = st.columns([2.2, 1, 1])
    with col1:
        st.title("RAG + Reranker Evaluation Platform")
        st.caption(f"Version {APP_VERSION}")
        st.markdown(
            "A focused workspace to evaluate retrieval, reranking, and judge quality on Open RAGBench."
        )
    with col2:
        status = _fetch_health(client)
        db_status = status.get("database", "unknown")
        health_label = status.get("status", "unknown")
        if health_label == "healthy":
            st.success(f"Backend: {health_label}")
        else:
            st.error(f"Backend: {health_label}")
        st.caption(f"DB: {db_status}")
    with col3:
        if st.button("Refresh", key="header_refresh"):
            st.session_state["health_status"] = _fetch_health(client)
            st.session_state["memory_status"] = _fetch_memory(client)
        memory = _fetch_memory(client)
        if memory:
            st.metric("Loaded Models (MB)", memory.get("total_memory_mb", 0))
            st.caption(f"Available: {memory.get('available_memory_mb', 0)} MB")
        else:
            st.caption("Memory: unavailable")


def _fetch_health(client: APIClient) -> dict:
    if "health_status" in st.session_state:
        return st.session_state["health_status"]
    try:
        status = client.get_health()
        st.session_state["health_status"] = status
        return status
    except Exception:
        return {"status": "unavailable", "database": "unknown"}


def _fetch_memory(client: APIClient) -> dict:
    if "memory_status" in st.session_state:
        return st.session_state["memory_status"]
    try:
        status = client.get_system_memory()
        st.session_state["memory_status"] = status
        return status
    except Exception:
        return {}
