from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    gold_in_top_k: bool


class MetricsService:
    def compute_retrieval_metrics(
        self,
        retrieved: Sequence[dict],
        relevant: Iterable[dict],
        k: int,
    ) -> RetrievalMetrics:
        relevance_map = self._build_relevance_map(relevant)
        relevant_ids = set(relevance_map)
        recall_at_k = self._recall_at_k(retrieved, relevant_ids, k)
        mrr = self._mrr(retrieved, relevant_ids)
        ndcg_at_k = self._ndcg_at_k(retrieved, relevance_map, k)
        gold_in_top_k = self._gold_in_top_k(retrieved, relevant_ids, k)
        return RetrievalMetrics(
            recall_at_k=recall_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            gold_in_top_k=gold_in_top_k,
        )

    def aggregate_retrieval_metrics(self, metrics_list: Iterable[RetrievalMetrics]) -> dict:
        metrics = list(metrics_list)
        if not metrics:
            return {
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "ndcg_at_k": 0.0,
                "gold_in_top_k": 0.0,
            }
        return {
            "recall_at_k": self._average([item.recall_at_k for item in metrics]),
            "mrr": self._average([item.mrr for item in metrics]),
            "ndcg_at_k": self._average([item.ndcg_at_k for item in metrics]),
            "gold_in_top_k": self._average([1.0 if item.gold_in_top_k else 0.0 for item in metrics]),
        }

    def aggregate_track_a(self, scores_list: Iterable[dict]) -> dict:
        scores = list(scores_list)
        return {
            "avg_correctness": self._average(self._values(scores, ["correctness"])),
            "avg_completeness": self._average(self._values(scores, ["completeness"])),
            "avg_specificity": self._average(self._values(scores, ["specificity"])),
            "avg_clarity": self._average(self._values(scores, ["clarity"])),
            "avg_overall": self._average(self._values(scores, ["overall"])),
        }

    def aggregate_track_b(self, scores_list: Iterable[dict]) -> dict:
        scores = list(scores_list)
        return {
            "avg_context_support": self._average(self._values(scores, ["context_support"])),
            "avg_hallucination": self._average(self._values(scores, ["hallucination"])),
            "avg_citation_quality": self._average(self._values(scores, ["citation_quality"])),
            "avg_overall": self._average(
                self._values(scores, ["overall_groundedness", "overall"])
            ),
        }

    def _recall_at_k(self, retrieved: Sequence[dict], relevant_ids: set, k: int) -> float:
        if not relevant_ids:
            return 0.0
        hits = sum(
            1
            for item in retrieved[:k]
            if (item.get("doc_id"), item.get("section_id")) in relevant_ids
        )
        return hits / len(relevant_ids)

    def _mrr(self, retrieved: Sequence[dict], relevant_ids: set) -> float:
        for idx, item in enumerate(retrieved, start=1):
            if (item.get("doc_id"), item.get("section_id")) in relevant_ids:
                return 1.0 / idx
        return 0.0

    def _ndcg_at_k(
        self,
        retrieved: Sequence[dict],
        relevance_map: Mapping[tuple, int],
        k: int,
    ) -> float:
        dcg = 0.0
        for idx, item in enumerate(retrieved[:k], start=1):
            rel = relevance_map.get((item.get("doc_id"), item.get("section_id")), 0)
            if rel <= 0:
                continue
            dcg += (2**rel - 1) / math.log2(idx + 1)
        ideal_scores = sorted(relevance_map.values(), reverse=True)[:k]
        idcg = 0.0
        for idx, rel in enumerate(ideal_scores, start=1):
            if rel <= 0:
                continue
            idcg += (2**rel - 1) / math.log2(idx + 1)
        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    def _gold_in_top_k(self, retrieved: Sequence[dict], relevant_ids: set, k: int) -> bool:
        return any(
            (item.get("doc_id"), item.get("section_id")) in relevant_ids
            for item in retrieved[:k]
        )

    def _build_relevance_map(self, relevant: Iterable[dict]) -> dict:
        relevance_map: dict[tuple, int] = {}
        for item in relevant:
            key = (item.get("doc_id"), item.get("section_id"))
            relevance_map[key] = int(item.get("relevance_score", 1) or 1)
        return relevance_map

    def _values(self, scores: Iterable[dict], keys: Sequence[str]) -> list[float]:
        values: list[float] = []
        for score in scores:
            for key in keys:
                if key in score and score[key] is not None:
                    values.append(float(score[key]))
                    break
        return values

    def _average(self, values: Iterable[float]) -> float:
        values_list = list(values)
        if not values_list:
            return 0.0
        return sum(values_list) / len(values_list)
