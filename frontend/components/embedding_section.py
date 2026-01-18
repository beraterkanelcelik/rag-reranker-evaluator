from __future__ import annotations

import streamlit as st

from frontend.api.client import APIClient


def render_embedding_section(client: APIClient) -> None:
    with st.expander("2. Embedding Configuration", expanded=False):
        models = _fetch_models(client)
        if models:
            st.subheader("Configured Models")
            for model in models:
                badge = "Ready" if model.get("status") == "ready" else model.get("status")
                st.write(
                    f"{model['id']}: {model['model_name']} ({model['model_source']}) - {badge}"
                )
        else:
            st.caption("No embedding models configured yet.")

        st.markdown("**Embedding Progress**")
        if st.button("Refresh progress", key="embed_refresh_progress"):
            try:
                progress = client.get_progress()
                embed_progress = progress.get("embedding", {})
                if embed_progress:
                    current = embed_progress.get("progress", 0)
                    total = embed_progress.get("total", 1)
                    percentage = int((current / total) * 100) if total > 0 else 0
                    status = embed_progress.get("status", "unknown")
                    st.write(f"Status: {status}")
                    st.progress(percentage / 100.0)
                    st.write(f"Processed {current}/{total} batches ({percentage}%)")
                    if status == "completed":
                        st.success("Embedding completed!")
                    elif status == "error":
                        st.error("Embedding failed!")
                else:
                    st.caption("No ongoing embedding process.")
            except Exception as exc:
                st.error(str(exc))


def _fetch_models(client: APIClient) -> list[dict]:
    try:
        return client.get_embedding_models().get("models", [])
    except Exception:
        return []
