from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
from openai import OpenAI

from backend.core.model_manager import ModelManager
from backend.core.progress import progress_store


@dataclass
class EmbeddingResult:
    embeddings: List[List[float]]
    dimension: int


class EmbeddingService:
    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager

    def embed_texts(
        self,
        model_name: str,
        texts: Iterable[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> EmbeddingResult:
        text_list = list(texts)
        total_batches = (len(text_list) + batch_size - 1) // batch_size
        progress_store["embedding"] = {"progress": 0, "total": total_batches, "status": "running"}
        print(f"[EMBEDDING] Starting embedding for {len(text_list)} texts with model {model_name}, batch_size {batch_size}")
        loaded = self.model_manager.load_embedding_model(model_name)
        all_embeddings = []
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"[EMBEDDING] Processing batch {batch_num}/{total_batches}, size {len(batch)}")
            progress_store["embedding"]["progress"] = batch_num
            batch_embeddings = loaded.model.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=normalize,
            )
            all_embeddings.extend(batch_embeddings)
        embeddings = np.asarray(all_embeddings)
        progress_store["embedding"]["progress"] = total_batches
        progress_store["embedding"]["status"] = "completed"
        print(f"[EMBEDDING] Completed embedding, shape: {embeddings.shape}")
        return EmbeddingResult(
            embeddings=embeddings.tolist(),
            dimension=embeddings.shape[1] if embeddings.ndim == 2 else 0,
        )

    def embed_texts_openai(
        self,
        model_name: str,
        texts: Iterable[str],
        api_key: str,
        batch_size: int = 100,
    ) -> EmbeddingResult:
        if not api_key:
            raise ValueError("OpenAI API key is required")
        client = OpenAI(api_key=api_key)
        vectors: List[List[float]] = []
        text_list = list(texts)
        for start in range(0, len(text_list), batch_size):
            batch = text_list[start : start + batch_size]
            response = client.embeddings.create(model=model_name, input=batch)
            vectors.extend([item.embedding for item in response.data])
        dimension = len(vectors[0]) if vectors else 0
        return EmbeddingResult(embeddings=vectors, dimension=dimension)
