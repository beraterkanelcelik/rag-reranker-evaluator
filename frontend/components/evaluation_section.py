from __future__ import annotations

import streamlit as st

from frontend.api.client import APIClient


def render_evaluation_section(client: APIClient) -> None:
    with st.expander("3. Evaluation Setup", expanded=False):
        models = _fetch_models(client)
        ready_models = [model for model in models if model.get("status") == "ready"]
        if not ready_models:
            st.warning("No ready embedding models found. Create one before running evaluation.")
        run_name = st.text_input("Run name", value="", key="eval_run_name")

        st.markdown("**RAG Settings**")
        model_label_map = {f"{model['id']} - {model['model_name']}": model for model in ready_models}
        model_choice = st.selectbox(
            "Embedding model",
            list(model_label_map.keys()) or ["No ready models"],
            key="eval_embedding_model",
        )
        retrieval_top_k = st.number_input(
            "Retrieval top-k", min_value=1, max_value=500, value=50, key="eval_retrieval_top_k"
        )

        st.markdown("**Reranker Settings (Optional)**")
        use_reranker = st.checkbox("Enable reranker", value=False, key="eval_use_reranker")
        reranker_model_name = st.text_input(
            "Reranker model",
            value="cross-encoder/ms-marco-MiniLM-L-6-v2",
            key="eval_reranker_model",
        )
        reranker_top_k = st.number_input(
            "Rerank top-k", min_value=1, max_value=50, value=5, key="eval_reranker_top_k"
        )

        st.markdown("**LLM Judge Settings**")
        judge_model = st.selectbox(
            "Judge model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"], key="eval_judge_model"
        )
        judge_api_key = st.text_input(
            "OpenAI API key", type="password", key="eval_judge_api_key"
        )
        judge_temperature = st.number_input(
            "Temperature", min_value=0.0, max_value=1.0, value=0.0, key="eval_judge_temp"
        )

        st.markdown("**Dataset Subset**")
        sample_size = st.number_input(
            "Sample size", min_value=1, value=100, step=1, key="eval_sample_size"
        )
        sample_seed = st.number_input(
            "Random seed", min_value=0, value=42, step=1, key="eval_sample_seed"
        )

        if st.button(
            "Start Evaluation", type="primary", disabled=not ready_models, key="eval_start"
        ):
            model_entry = model_label_map.get(model_choice)
            if not model_entry:
                st.error("Select a valid embedding model.")
                return
            payload = {
                "run_name": run_name or None,
                "embedding_model_id": model_entry["id"],
                "retrieval_top_k": retrieval_top_k,
                "use_reranker": use_reranker,
                "reranker_config": {
                    "model_name": reranker_model_name,
                    "top_k": reranker_top_k,
                }
                if use_reranker
                else None,
                "judge_config": {
                    "model_name": judge_model,
                    "api_key": judge_api_key,
                    "temperature": judge_temperature,
                },
                "sample_size": sample_size,
                "sample_seed": sample_seed,
            }
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                response = client.create_evaluation_run(payload)
                run_id = response.get('run_id')
                st.success(f"Evaluation started - Run ID: {run_id}")

                import threading
                import time

                def update_progress():
                    while True:
                        try:
                            progress = client.get_evaluation_progress(run_id)
                            progress_info = progress.get("progress", {})
                            percentage = progress_info.get("percentage", 0)
                            current = progress_info.get("current_query", 0)
                            total = progress_info.get("total_queries", 0)
                            progress_bar.progress(percentage / 100.0)
                            status_text.text(f"Processed {current}/{total} queries ({percentage:.1f}%) - {progress.get('status')}")
                            if progress.get("status") in ["completed", "error"]:
                                break
                            time.sleep(2)
                        except Exception:
                            # Stop polling on error (e.g., 404 run not found)
                            break

                progress_thread = threading.Thread(target=update_progress, daemon=True)
                progress_thread.start()
            except Exception as exc:
                st.error(str(exc))
                progress_bar.empty()
                status_text.empty()

        st.markdown("**Evaluation Progress**")
        runs = _fetch_runs(client)
        if not runs:
            st.caption("No evaluation runs yet.")
            return
        run_ids = [str(run["id"]) for run in runs]
        selected = st.selectbox("Select run", run_ids, key="eval_progress_run")
        if st.button("Refresh progress", key="eval_refresh_progress"):
            try:
                progress = client.get_evaluation_progress(int(selected))
                progress_info = progress.get("progress", {})
                st.write(f"Status: {progress.get('status')}")
                st.progress(progress_info.get("percentage", 0) / 100.0)
                st.write(
                    f"{progress_info.get('current_query', 0)}/{progress_info.get('total_queries', 0)} queries"
                )
                if progress.get("status") == "completed":
                    st.success("Evaluation completed!")
                elif progress.get("status") == "running":
                    st.info("Evaluation in progress...")
            except Exception as exc:
                st.error(str(exc))

        st.caption("Tip: keep sample sizes small until the workflow is validated.")


def _fetch_models(client: APIClient) -> list[dict]:
    try:
        return client.get_embedding_models().get("models", [])
    except Exception:
        return []


def _fetch_runs(client: APIClient) -> list[dict]:
    try:
        return client.get_evaluation_runs().get("runs", [])
    except Exception:
        return []
