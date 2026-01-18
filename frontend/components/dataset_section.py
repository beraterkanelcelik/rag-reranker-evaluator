from __future__ import annotations

import streamlit as st

from frontend.api.client import APIClient


def render_dataset_section(client: APIClient) -> None:
    with st.expander("1. Dataset Setup", expanded=True):
        status = _fetch_status(client)
        status_label = status.get("status", "pending").upper()
        status_color = "green" if status.get("status") == "ready" else "orange"

        st.markdown(
            f"**Dataset:** {status.get('dataset_name', 'vectara/open_ragbench')}"
        )
        st.markdown(f"**Status:** :{status_color}[{status_label}]")

        col1, col2, col3 = st.columns(3)
        col1.metric("Documents", status.get("total_documents", 0))
        col2.metric("Sections", status.get("total_sections", 0))
        col3.metric("Queries", status.get("total_queries", 0))

        if status.get("completed_at"):
            st.caption(f"Last updated: {status.get('completed_at')}")

        if status.get("error_message"):
            st.error(status.get("error_message"))

        st.caption(
            "Download uses the Vectara Open RAGBench dataset (pdf/arxiv) and may take a few minutes."
        )

        if st.button("Download & Ingest Dataset", type="primary", key="dataset_ingest"):
            try:
                with st.spinner("Downloading dataset..."):
                    response = client.ingest_dataset("official/pdf/arxiv")
                st.success(response.get("message", "Ingestion started"))
            except Exception as exc:
                st.error(str(exc))


def _fetch_status(client: APIClient) -> dict:
    try:
        return client.get_dataset_status()
    except Exception:
        return {
            "status": "pending",
            "dataset_name": "vectara/open_ragbench",
            "total_documents": 0,
            "total_queries": 0,
            "total_sections": 0,
        }
