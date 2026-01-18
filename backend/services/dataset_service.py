from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


@dataclass
class ParsedDataset:
    corpus: List[Dict[str, Any]]
    queries: List[Dict[str, Any]]
    qrels: List[Dict[str, Any]]
    answers: List[Dict[str, Any]]


class DatasetService:
    def __init__(self, dataset_name: str = "vectara/open_ragbench") -> None:
        self.dataset_name = dataset_name

    def download_dataset(self, subset: str) -> DatasetDict:
        resolved_subset = self.resolve_subset(subset)
        logger.info("Downloading dataset %s (%s)", self.dataset_name, resolved_subset)
        config_name = None if resolved_subset == "default" else resolved_subset
        dataset = load_dataset(self.dataset_name, config_name)
        if not isinstance(dataset, DatasetDict):
            raise ValueError("Expected DatasetDict from HuggingFace")
        return dataset

    def download_raw_snapshot(self, subset: str) -> ParsedDataset:
        resolved_subset = self.resolve_subset(subset)
        snapshot_path = self._resolve_snapshot_path(resolved_subset)
        if snapshot_path is None:
            logger.warning("Snapshot not in cache; triggering raw download.")
            self._download_raw_subset(resolved_subset)
            snapshot_path = self._resolve_snapshot_path(resolved_subset)
        if snapshot_path is None:
            raise ValueError("RAGBench snapshot not found in cache after download.")
        return self._parse_snapshot(snapshot_path)

    def parse_dataset(self, dataset: DatasetDict) -> ParsedDataset:
        corpus_split = self._require_split(dataset, ["corpus", "documents", "docs"])
        queries_split = self._require_split(dataset, ["queries", "questions"])
        qrels_split = self._require_split(dataset, ["qrels", "relevance"])
        answers_split = self._require_split(dataset, ["answers", "references"])

        corpus = [self._parse_corpus_row(row) for row in corpus_split]
        queries = [self._parse_query_row(row) for row in queries_split]
        qrels = [self._parse_qrel_row(row) for row in qrels_split]
        answers = [self._parse_answer_row(row) for row in answers_split]

        return ParsedDataset(corpus=corpus, queries=queries, qrels=qrels, answers=answers)

    def download_and_parse(self, subset: str) -> ParsedDataset:
        try:
            dataset = self.download_dataset(subset)
            return self.parse_dataset(dataset)
        except Exception as exc:
            logger.warning("Dataset parsing failed, falling back to snapshot parser: %s", exc)
            return self.download_raw_snapshot(subset)

    def resolve_subset(self, subset: str) -> str:
        if subset == "default":
            return "default"
        if subset == "official/pdf/arxiv":
            return "pdf/arxiv"
        return subset

    def _require_split(self, dataset: DatasetDict, names: List[str]) -> Dataset:
        for name in names:
            if name in dataset:
                return dataset[name]
        raise ValueError(f"Dataset missing required split. Expected one of: {names}")

    def _resolve_snapshot_path(self, subset: str) -> Optional[Path]:
        base = Path("/app/.cache/huggingface/hub/datasets--vectara--open_ragbench/snapshots")
        if not base.exists():
            return None
        snapshots = [p for p in base.iterdir() if p.is_dir()]
        if not snapshots:
            return None
        snapshot = snapshots[0]
        subset_path = snapshot / subset
        if not subset_path.exists():
            return None
        return subset_path

    def _download_raw_subset(self, subset: str) -> None:
        from huggingface_hub import snapshot_download

        repo_id = self.dataset_name
        snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=f"{subset}/**")

    def _parse_snapshot(self, subset_path: Path) -> ParsedDataset:
        queries_data = self._read_json(subset_path / "queries.json")
        answers_data = self._read_json(subset_path / "answers.json")
        qrels_data = self._read_json(subset_path / "qrels.json")
        corpus_dir = subset_path / "corpus"

        corpus = self._parse_corpus_files(corpus_dir)
        queries = self._parse_queries_dict(queries_data)
        answers = self._parse_answers_dict(answers_data)
        qrels = self._parse_qrels_dict(qrels_data)

        return ParsedDataset(corpus=corpus, queries=queries, qrels=qrels, answers=answers)

    def _parse_corpus_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "doc_id": self._require_value(row, ["doc_id", "document_id", "docid"], "doc_id"),
            "section_id": int(
                self._require_value(
                    row,
                    ["section_id", "section_index", "chunk_id", "section"],
                    "section_id",
                )
            ),
            "section_text": self._require_value(
                row, ["section_text", "text", "content", "section"], "section_text"
            ),
            "tables_markdown": self._optional_value(row, ["tables_markdown", "tables"], None),
            "has_images": bool(self._optional_value(row, ["has_images", "has_image"], False)),
            "image_count": int(self._optional_value(row, ["image_count", "images"], 0)),
            "metadata": self._optional_value(row, ["metadata", "meta"], None),
        }

    def _parse_query_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "query_uuid": self._require_value(
                row, ["query_uuid", "query_id", "qid", "id"], "query_uuid"
            ),
            "query_text": self._require_value(
                row, ["query_text", "query", "question"], "query_text"
            ),
            "query_type": self._optional_value(row, ["query_type", "type"], None),
            "source_type": self._optional_value(row, ["source_type", "source"], None),
            "metadata": self._optional_value(row, ["metadata", "meta"], None),
        }

    def _parse_qrel_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "query_uuid": self._require_value(
                row, ["query_uuid", "query_id", "qid", "id"], "query_uuid"
            ),
            "doc_id": self._require_value(row, ["doc_id", "document_id", "docid"], "doc_id"),
            "section_id": int(
                self._require_value(
                    row, ["section_id", "section_index", "chunk_id"], "section_id"
                )
            ),
            "relevance_score": int(self._optional_value(row, ["relevance_score", "score"], 1)),
        }

    def _parse_answer_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "query_uuid": self._require_value(
                row, ["query_uuid", "query_id", "qid", "id"], "query_uuid"
            ),
            "reference_answer": self._require_value(
                row, ["reference_answer", "answer", "response"], "reference_answer"
            ),
        }

    def _read_json(self, path: Path) -> Any:
        content = path.read_text(encoding="utf-8")
        return self._safe_json_loads(content)

    def _parse_queries_dict(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = []
        for query_uuid, payload in data.items():
            rows.append(
                {
                    "query_uuid": query_uuid,
                    "query_text": payload.get("query"),
                    "query_type": payload.get("type"),
                    "source_type": payload.get("source"),
                    "metadata": payload.get("metadata"),
                }
            )
        return rows

    def _parse_answers_dict(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"query_uuid": query_uuid, "reference_answer": answer}
            for query_uuid, answer in data.items()
        ]

    def _parse_qrels_dict(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = []
        for query_uuid, payload in data.items():
            rows.append(
                {
                    "query_uuid": query_uuid,
                    "doc_id": payload.get("doc_id"),
                    "section_id": int(payload.get("section_id")),
                    "relevance_score": int(payload.get("relevance_score", 1)),
                }
            )
        return rows

    def _parse_corpus_files(self, corpus_dir: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for corpus_file in corpus_dir.glob("*.json"):
            content = corpus_file.read_text(encoding="utf-8")
            data = self._safe_json_loads(content)
            doc_id = data.get("id") or corpus_file.stem
            metadata = {
                "title": data.get("title"),
                "authors": data.get("authors"),
                "categories": data.get("categories"),
            }
            for section in data.get("sections", []):
                rows.append(
                    {
                        "doc_id": doc_id,
                        "section_id": int(section.get("section_id", 0)),
                        "section_text": section.get("text", ""),
                        "tables_markdown": self._format_tables(section.get("tables")),
                        "has_images": bool(section.get("images")),
                        "image_count": len(section.get("images", []) or []),
                        "metadata": metadata,
                    }
                )
        return rows

    def _safe_json_loads(self, content: str) -> Any:
        import json

        return json.loads(content)

    def _format_tables(self, tables: Any) -> Optional[str]:
        if not tables:
            return None
        if isinstance(tables, list):
            return "\n\n".join(str(table) for table in tables if table is not None)
        return str(tables)

    def _require_value(self, row: Dict[str, Any], keys: Iterable[str], field: str) -> Any:
        for key in keys:
            if key in row and row[key] is not None:
                return row[key]
        raise ValueError(f"Missing required field {field} in dataset row")

    def _optional_value(self, row: Dict[str, Any], keys: Iterable[str], default: Any) -> Any:
        for key in keys:
            if key in row and row[key] is not None:
                return row[key]
        return default
