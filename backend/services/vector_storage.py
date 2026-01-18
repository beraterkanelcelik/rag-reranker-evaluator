from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from sqlalchemy import Column, Integer, MetaData, String, Table, Text, text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class VectorRow:
    corpus_id: int
    doc_id: str
    section_id: int
    embedding: List[float]


class VectorStorage:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.metadata = MetaData()

    async def create_vector_table(self, model_id: int, dimension: int) -> str:
        table_name = f"vectors_{model_id}"
        await self.db.execute(text(f"CREATE TABLE IF NOT EXISTS {table_name} (\n" +
                                   "id SERIAL PRIMARY KEY,\n" +
                                   "corpus_id INTEGER NOT NULL REFERENCES corpus(id),\n" +
                                   "doc_id VARCHAR(255) NOT NULL,\n" +
                                   "section_id INTEGER NOT NULL,\n" +
                                   f"embedding vector({dimension}),\n" +
                                   "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n" +
                                   "UNIQUE(corpus_id)\n" +
                                   ")"))
        await self.db.execute(text(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding "
            f"ON {table_name} USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        ))
        return table_name

    async def insert_vectors(self, table_name: str, rows: Iterable[VectorRow]) -> int:
        inserted = 0
        for row in rows:
            await self.db.execute(
                text(
                    f"INSERT INTO {table_name} (corpus_id, doc_id, section_id, embedding) "
                    "VALUES (:corpus_id, :doc_id, :section_id, :embedding)"
                ),
                {
                    "corpus_id": row.corpus_id,
                    "doc_id": row.doc_id,
                    "section_id": row.section_id,
                    "embedding": self._format_embedding(row.embedding),
                },
            )
            inserted += 1
        return inserted

    def _format_embedding(self, embedding: List[float]) -> str:
        values = ",".join(str(value) for value in embedding)
        return f"[{values}]"
