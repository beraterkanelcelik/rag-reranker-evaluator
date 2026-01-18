from __future__ import annotations

import httpx

from frontend.config import BACKEND_URL


class APIClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or BACKEND_URL

    def get_health(self) -> dict:
        response = httpx.get(f"{self.base_url}/api/v1/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def get_dataset_status(self) -> dict:
        response = httpx.get(f"{self.base_url}/api/v1/dataset/status", timeout=30)
        response.raise_for_status()
        return response.json()

    def ingest_dataset(self, subset: str) -> dict:
        response = httpx.post(
            f"{self.base_url}/api/v1/dataset/ingest",
            json={"subset": subset},
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    def get_embedding_models(self) -> dict:
        response = httpx.get(f"{self.base_url}/api/v1/embedding/models", timeout=30)
        response.raise_for_status()
        return response.json()

    def create_embedding_model(self, payload: dict) -> dict:
        response = httpx.post(
            f"{self.base_url}/api/v1/embedding/models",
            json=payload,
            timeout=1200,  # 20 minutes for large embeddings
        )
        response.raise_for_status()
        return response.json()

    def create_evaluation_run(self, payload: dict) -> dict:
        response = httpx.post(
            f"{self.base_url}/api/v1/evaluation/runs",
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    def get_evaluation_runs(self) -> dict:
        response = httpx.get(f"{self.base_url}/api/v1/evaluation/runs", timeout=30)
        response.raise_for_status()
        return response.json()

    def get_evaluation_progress(self, run_id: int) -> dict:
        response = httpx.get(
            f"{self.base_url}/api/v1/evaluation/runs/{run_id}/progress", timeout=30
        )
        response.raise_for_status()
        return response.json()

    def get_evaluation_run(self, run_id: int) -> dict:
        response = httpx.get(f"{self.base_url}/api/v1/evaluation/runs/{run_id}", timeout=30)
        response.raise_for_status()
        return response.json()

    def get_results_details(self, run_id: int, limit: int = 100, offset: int = 0) -> dict:
        response = httpx.get(
            f"{self.base_url}/api/v1/results/{run_id}/details",
            params={"limit": limit, "offset": offset},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def get_result_query(self, run_id: int, query_uuid: str) -> dict:
        response = httpx.get(
            f"{self.base_url}/api/v1/results/{run_id}/query/{query_uuid}", timeout=60
        )
        response.raise_for_status()
        return response.json()

    def export_results(self, run_id: int) -> dict:
        response = httpx.get(f"{self.base_url}/api/v1/results/{run_id}/export", timeout=60)
        response.raise_for_status()
        return response.json()

    def get_system_memory(self) -> dict:
        response = httpx.get(f"{self.base_url}/api/v1/system/memory", timeout=30)
        response.raise_for_status()
        return response.json()

    def unload_models(self) -> dict:
        response = httpx.post(f"{self.base_url}/api/v1/system/unload-models", timeout=30)
        response.raise_for_status()
        return response.json()

    def get_progress(self) -> dict:
        response = httpx.get(f"{self.base_url}/api/v1/health/progress", timeout=10)
        response.raise_for_status()
        return response.json()
