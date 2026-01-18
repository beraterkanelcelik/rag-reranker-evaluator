from __future__ import annotations

from datetime import datetime
from typing import List

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import Base, engine
from backend.models import database as models
from backend.services.dataset_service import DatasetService, ParsedDataset


class DatasetIngestionService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.dataset_service = DatasetService()

    async def ingest(self, subset: str) -> dict:
        await self._ensure_tables()
        parsed = self.dataset_service.download_and_parse(subset)
        filtered = self._filter_parsed(parsed)
        await self._upsert_status(filtered, subset, status="processing")

        await self._truncate_tables()
        await self._insert_corpus(filtered.corpus)
        await self._insert_queries(filtered.queries)
        await self.db.flush()
        await self._insert_qrels(filtered.qrels)
        await self._insert_answers(filtered.answers)

        await self._update_status_ready(filtered)
        await self.db.commit()

        return {
            "status": "ready",
            "total_documents": len({row["doc_id"] for row in filtered.corpus}),
            "total_sections": len(filtered.corpus),
            "total_queries": len(filtered.queries),
        }

    async def _ensure_tables(self) -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    def _filter_parsed(self, parsed: ParsedDataset) -> ParsedDataset:
        query_ids = {row["query_uuid"] for row in parsed.queries}
        corpus_index = {
            (row["doc_id"], row["section_id"]): row for row in parsed.corpus
        }
        answers = [row for row in parsed.answers if row["query_uuid"] in query_ids]
        qrels = [
            row
            for row in parsed.qrels
            if row["query_uuid"] in query_ids
            and (row["doc_id"], row["section_id"]) in corpus_index
        ]
        corpus = list(corpus_index.values())
        return ParsedDataset(corpus=corpus, queries=parsed.queries, qrels=qrels, answers=answers)

    async def _truncate_tables(self) -> None:
        for table in [models.Qrel, models.Answer, models.Corpus, models.Query]:
            await self.db.execute(delete(table))

    async def _insert_corpus(self, rows: List[dict]) -> None:
        self.db.add_all([models.Corpus(**row) for row in rows])

    async def _insert_queries(self, rows: List[dict]) -> None:
        self.db.add_all([models.Query(**row) for row in rows])

    async def _insert_qrels(self, rows: List[dict]) -> None:
        self.db.add_all([models.Qrel(**row) for row in rows])

    async def _insert_answers(self, rows: List[dict]) -> None:
        self.db.add_all([models.Answer(**row) for row in rows])

    async def _upsert_status(self, parsed: ParsedDataset, subset: str, status: str) -> None:
        total_documents = len({row["doc_id"] for row in parsed.corpus})
        total_sections = len(parsed.corpus)
        total_queries = len(parsed.queries)

        result = await self.db.execute(select(models.DatasetStatus).limit(1))
        current = result.scalar_one_or_none()
        if current is None:
            current = models.DatasetStatus()
            self.db.add(current)

        current.dataset_name = self.dataset_service.dataset_name
        current.subset_name = subset
        current.status = status
        current.total_documents = total_documents
        current.total_sections = total_sections
        current.total_queries = total_queries
        current.error_message = None
        current.started_at = current.started_at or datetime.utcnow()
        current.updated_at = datetime.utcnow()

    async def _update_status_ready(self, parsed: ParsedDataset) -> None:
        total_documents = len({row["doc_id"] for row in parsed.corpus})
        total_sections = len(parsed.corpus)
        total_queries = len(parsed.queries)

        result = await self.db.execute(select(models.DatasetStatus).limit(1))
        current = result.scalar_one_or_none()
        if current is None:
            current = models.DatasetStatus()
            self.db.add(current)

        current.dataset_name = self.dataset_service.dataset_name
        current.status = "ready"
        current.total_documents = total_documents
        current.total_sections = total_sections
        current.total_queries = total_queries
        current.error_message = None
        current.completed_at = datetime.utcnow()
        current.updated_at = datetime.utcnow()

    async def get_status(self) -> models.DatasetStatus | None:
        result = await self.db.execute(select(models.DatasetStatus).limit(1))
        return result.scalar_one_or_none()

    async def get_counts(self) -> dict:
        corpus_count = await self.db.execute(select(func.count(models.Corpus.id)))
        query_count = await self.db.execute(select(func.count(models.Query.id)))
        qrel_count = await self.db.execute(select(func.count(models.Qrel.id)))
        answer_count = await self.db.execute(select(func.count(models.Answer.id)))
        return {
            "corpus": corpus_count.scalar_one(),
            "queries": query_count.scalar_one(),
            "qrels": qrel_count.scalar_one(),
            "answers": answer_count.scalar_one(),
        }
