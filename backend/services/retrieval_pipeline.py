from __future__ import annotations

from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.model_manager import ModelManager
from sqlalchemy import select

from backend.models import database as models
from backend.services.embedding_service import EmbeddingService
from backend.services.reranker_service import RerankerService
from backend.services.retrieval_service import RetrievalService


class RetrievalPipeline:
    def __init__(self, db: AsyncSession, model_manager: ModelManager) -> None:
        self.db = db
        self.model_manager = model_manager
        self.embedding_service = EmbeddingService(model_manager)
        self.retrieval_service = RetrievalService(db)
        self.reranker_service = RerankerService(model_manager)

    async def retrieve(
        self,
        model_id: int,
        query_text: str,
        retrieval_top_k: int,
        use_reranker: bool = False,
        reranker_model_name: str | None = None,
        reranker_top_k: int = 5,
    ) -> dict:
        embedding_model = await self._get_embedding_model(model_id)
        embedding = self.embedding_service.embed_texts(
            embedding_model.model_name, [query_text]
        ).embeddings[0]

        retrieved = await self.retrieval_service.similarity_search(
            embedding_model.table_name, embedding, retrieval_top_k
        )

        if not use_reranker or not reranker_model_name:
            return {
                "retrieved": retrieved,
                "reranked": None,
            }

        documents = await self._fetch_documents(retrieved)
        reranked = self.reranker_service.rerank(
            reranker_model_name, query_text, documents, top_k=reranker_top_k
        )
        return {
            "retrieved": retrieved,
            "reranked": [
                {
                    "corpus_id": item.corpus_id,
                    "doc_id": item.doc_id,
                    "section_id": item.section_id,
                    "score": item.score,
                }
                for item in reranked
            ],
        }

    async def _get_embedding_model(self, model_id: int) -> models.EmbeddingModel:
        result = await self.db.execute(
            select(models.EmbeddingModel).where(models.EmbeddingModel.id == model_id)
        )
        model = result.scalar_one_or_none()
        if model is None:
            raise ValueError("Embedding model not found")
        return model

    async def _fetch_documents(self, retrieved: List[dict]) -> List[dict]:
        docs: List[dict] = []
        for item in retrieved:
            result = await self.db.execute(
                select(models.Corpus).where(
                    models.Corpus.doc_id == item["doc_id"],
                    models.Corpus.section_id == item["section_id"],
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                continue
            docs.append(
                {
                    "corpus_id": row.id,
                    "doc_id": row.doc_id,
                    "section_id": row.section_id,
                    "text": row.section_text,
                }
            )
        return docs
