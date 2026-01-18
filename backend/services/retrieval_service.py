from __future__ import annotations

from typing import List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class RetrievalService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def similarity_search(
        self, table_name: str, query_embedding: List[float], top_k: int
    ) -> List[dict]:
        embedding_str = self._format_embedding(query_embedding)
        result = await self.db.execute(
            text(
                "SELECT corpus_id, doc_id, section_id, "
                "1 - (embedding <=> :embedding) AS score "
                f"FROM {table_name} "
                "ORDER BY embedding <=> :embedding "
                "LIMIT :limit"
            ),
            {"embedding": embedding_str, "limit": top_k},
        )
        rows = result.fetchall()
        return [
            {
                "corpus_id": row.corpus_id,
                "doc_id": row.doc_id,
                "section_id": row.section_id,
                "score": float(row.score),
            }
            for row in rows
        ]

    def _format_embedding(self, embedding: List[float]) -> str:
        values = ",".join(str(value) for value in embedding)
        return f"[{values}]"
