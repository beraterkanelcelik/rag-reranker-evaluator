from __future__ import annotations

from dataclasses import dataclass
from typing import List

from backend.core.model_manager import ModelManager


@dataclass
class RerankResult:
    corpus_id: int
    doc_id: str
    section_id: int
    score: float


class RerankerService:
    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager

    def rerank(
        self,
        model_name: str,
        query: str,
        documents: List[dict],
        top_k: int = 5,
    ) -> List[RerankResult]:
        loaded = self.model_manager.load_reranker_model(model_name)
        pairs = [(query, doc["text"]) for doc in documents]
        scores = loaded.model.predict(pairs)
        scored = [
            RerankResult(
                corpus_id=doc["corpus_id"],
                doc_id=doc["doc_id"],
                section_id=doc["section_id"],
                score=float(score),
            )
            for doc, score in zip(documents, scores)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
