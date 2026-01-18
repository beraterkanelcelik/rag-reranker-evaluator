from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer


@dataclass
class LoadedModel:
    name: str
    model_type: str
    model: Any
    memory_mb: float


class ModelManager:
    _shared_models: Dict[str, LoadedModel] = {}
    _instances: List["ModelManager"] = []

    def __init__(self) -> None:
        self._models = ModelManager._shared_models
        ModelManager._instances.append(self)

    def load_embedding_model(self, model_name: str, device: Optional[str] = None) -> LoadedModel:
        key = f"embedding::{model_name}"
        if key in self._models:
            return self._models[key]

        model = SentenceTransformer(model_name, device=device)
        memory_mb = self._estimate_model_size(model)
        loaded = LoadedModel(name=model_name, model_type="embedding", model=model, memory_mb=memory_mb)
        self._models[key] = loaded
        return loaded

    def load_reranker_model(self, model_name: str, device: Optional[str] = None) -> LoadedModel:
        key = f"reranker::{model_name}"
        if key in self._models:
            return self._models[key]

        model = CrossEncoder(model_name, device=device)
        memory_mb = self._estimate_model_size(model)
        loaded = LoadedModel(name=model_name, model_type="reranker", model=model, memory_mb=memory_mb)
        self._models[key] = loaded
        return loaded

    def unload(self, model_type: str, model_name: str) -> None:
        key = f"{model_type}::{model_name}"
        if key not in self._models:
            return
        del self._models[key]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_all(self) -> None:
        self._models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def unload_all_instances(cls) -> None:
        cls._shared_models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_loaded_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": model.model_type,
                "name": model.name,
                "memory_mb": model.memory_mb,
            }
            for model in self._models.values()
        ]

    def total_memory_mb(self) -> float:
        return sum(model.memory_mb for model in self._models.values())

    def _estimate_model_size(self, model: Any) -> float:
        if hasattr(model, "parameters"):
            total_bytes = sum(
                p.numel() * p.element_size() for p in model.parameters() if hasattr(p, "numel")
            )
            return total_bytes / (1024 * 1024)
        return 0.0
