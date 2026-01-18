from __future__ import annotations

import json
import streamlit as st

from frontend.api.client import APIClient


def render_results_section(client: APIClient) -> None:
    with st.expander("4. Results & History", expanded=False):
        if "results_payload" not in st.session_state:
            st.session_state["results_payload"] = {"results": []}
        runs = _fetch_runs(client)
        if not runs:
            st.caption("No evaluation runs yet.")
            return

        run_options = {
            f"#{run['id']} - {run.get('run_name') or 'Untitled'}": run for run in runs
        }
        selected = st.selectbox(
            "Select run", list(run_options.keys()), key="results_run_select"
        )
        run = run_options[selected]

        if st.button("Refresh run details", key="results_refresh_run"):
            run = _fetch_run_details(client, run["id"]) or run

        st.markdown("**Configuration**")
        config = run.get("config", {})
        st.json(config, expanded=True)

        metrics = run.get("metrics_summary") or {}
        if metrics:
            st.markdown("**Aggregate Metrics**")
            st.json(metrics, expanded=True)

        st.markdown("**Results**")
        limit = st.number_input(
            "Results limit", min_value=1, value=50, step=1, key="results_limit"
        )
        offset = st.number_input(
            "Results offset", min_value=0, value=0, step=1, key="results_offset"
        )
        if st.button("Load results", key="results_load"):
            results_payload = _fetch_results(client, run["id"], limit, offset)
            st.session_state["results_payload"] = results_payload

        results_payload = st.session_state.get("results_payload", {"results": []})
        for item in results_payload.get("results", []):
            _render_result_item(client, run["id"], item)

        if st.button("Export results JSON", key="results_export"):
            export_payload = client.export_results(run["id"])
            st.json(export_payload, expanded=False)

        st.caption("Results can be large; use filters and pagination for faster loads.")


def _render_result_item(client: APIClient, run_id: int, item: dict) -> None:
    query_uuid = item.get("query_uuid") or "unknown"
    header = f"{item.get('query_text', 'Query')} ({query_uuid})"
    st.markdown(f"### {header}")
    st.markdown("**Reference Answer**")
    st.write(item.get("reference_answer", ""))

    st.markdown("**Generated Answer**")
    st.write(item.get("generated_answer", ""))

    st.markdown("**Retrieval Metrics**")
    st.json(item.get("retrieval_metrics", {}), expanded=False)

    st.markdown("**Judge Scores**")
    st.json(item.get("scores", {}), expanded=False)

    if query_uuid != "unknown" and st.button(
        f"Load full details ({query_uuid})", key=f"result_detail_{query_uuid}"
    ):
        detail = client.get_result_query(run_id, query_uuid)
        st.markdown("**Final Context**")
        st.write(detail.get("final_context", ""))
        st.markdown("**Judge Responses**")
        st.json(detail.get("judge_responses", {}), expanded=False)
    st.divider()


def _fetch_runs(client: APIClient) -> list[dict]:
    try:
        return client.get_evaluation_runs().get("runs", [])
    except Exception:
        return []


def _fetch_run_details(client: APIClient, run_id: int) -> dict:
    try:
        return client.get_evaluation_run(run_id)
    except Exception:
        return {}


def _fetch_results(client: APIClient, run_id: int, limit: int, offset: int) -> dict:
    try:
        return client.get_results_details(run_id, limit=limit, offset=offset)
    except Exception:
        return {"results": [], "total": 0}
