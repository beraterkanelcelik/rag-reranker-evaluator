from __future__ import annotations

from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.core.model_manager import ModelManager
from backend.models import database as models
from backend.models.schemas import EmbeddingModelCreate, SearchRequest
from backend.services.embedding_service import EmbeddingService
from backend.services.retrieval_pipeline import RetrievalPipeline
from backend.services.retrieval_service import RetrievalService
from backend.services.reranker_service import RerankerService
from backend.services.vector_storage import VectorStorage, VectorRow


def _hash_api_key(api_key: str) -> str:
    return api_key[-4:]

router = APIRouter()


@router.get("/models")
async def list_models(db: AsyncSession = Depends(get_db)) -> dict:
    result = await db.execute(select(models.EmbeddingModel))
    models_list = result.scalars().all()
    return {
        "models": [
            {
                "id": row.id,
                "model_name": row.model_name,
                "model_source": row.model_source,
                "dimension": row.dimension,
                "status": row.status,
                "total_vectors": row.total_vectors,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in models_list
        ]
    }


@router.post("/models")
async def create_model(
    request: EmbeddingModelCreate, background: BackgroundTasks, db: AsyncSession = Depends(get_db)
) -> dict:
    manager = ModelManager()
    embedding_service = EmbeddingService(manager)

    model_entry = models.EmbeddingModel(
        model_name=request.model_name,
        model_source=request.model_source.value,
        dimension=request.dimension,
        table_name="",
        config=request.config,
        status="embedding",
    )
    db.add(model_entry)
    await db.flush()

    storage = VectorStorage(db)
    table_name = await storage.create_vector_table(model_entry.id, request.dimension)
    model_entry.table_name = table_name
    await db.commit()

    background.add_task(run_embedding, request, model_entry.id, db)

    return {
        "id": model_entry.id,
        "status": "embedding",
        "message": "Embedding started",
        "table_name": table_name,
    }


async def run_embedding(request: EmbeddingModelCreate, model_id: int, db: AsyncSession) -> None:
    try:
        manager = ModelManager()
        embedding_service = EmbeddingService(manager)

        corpus_rows = await db.execute(select(models.Corpus))
        corpus = corpus_rows.scalars().all()
        texts = [row.section_text for row in corpus]

        if request.model_source.value == "openai":
            if not request.api_key:
                raise ValueError("OpenAI API key required")
            result = embedding_service.embed_texts_openai(
                request.model_name, texts, request.api_key, request.config.get("batch_size", 100)
            )
            api_key_hash = _hash_api_key(request.api_key)
        else:
            result = embedding_service.embed_texts(
                request.model_name,
                texts,
                batch_size=request.config.get("batch_size", 32),
                normalize=request.config.get("normalize", True),
            )
            api_key_hash = None

        vector_rows = [
            VectorRow(
                corpus_id=corpus[idx].id,
                doc_id=corpus[idx].doc_id,
                section_id=corpus[idx].section_id,
                embedding=result.embeddings[idx],
            )
            for idx in range(len(corpus))
        ]

        storage = VectorStorage(db)
        model_result = await db.execute(
            select(models.EmbeddingModel).where(models.EmbeddingModel.id == model_id)
        )
        model_entry = model_result.scalar_one()
        inserted = await storage.insert_vectors(model_entry.table_name, vector_rows)
        model_entry.status = "ready"
        model_entry.total_vectors = inserted
        if api_key_hash:
            model_entry.api_key_hash = api_key_hash
        await db.commit()
    except Exception as exc:
        model_result = await db.execute(
            select(models.EmbeddingModel).where(models.EmbeddingModel.id == model_id)
        )
        model_entry = model_result.scalar_one()
        model_entry.status = "error"
        model_entry.error_message = str(exc)
        await db.commit()


@router.get("/models/{model_id}/status")
async def model_status(model_id: int, db: AsyncSession = Depends(get_db)) -> dict:
    result = await db.execute(
        select(models.EmbeddingModel).where(models.EmbeddingModel.id == model_id)
    )
    model = result.scalar_one_or_none()

    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    table_name = model.table_name
    deleted_vectors = model.total_vectors

    await db.execute(delete(models.EmbeddingModel).where(models.EmbeddingModel.id == model_id))
    if table_name:
        await db.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
    await db.commit()
    return {
        "message": "Model and vectors deleted",
        "deleted_vectors": deleted_vectors,
    }


@router.post("/search")
async def search_embeddings(
    request: SearchRequest, db: AsyncSession = Depends(get_db)
) -> dict:
    result = await db.execute(
        select(models.EmbeddingModel).where(models.EmbeddingModel.id == request.model_id)
    )
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    manager = ModelManager()
    pipeline = RetrievalPipeline(db, manager)
    pipeline_result = await pipeline.retrieve(
        model_id=request.model_id,
        query_text=request.query_text,
        retrieval_top_k=request.top_k,
        use_reranker=request.use_reranker,
        reranker_model_name=request.reranker_model_name,
        reranker_top_k=request.reranker_top_k,
    )

    results = pipeline_result["retrieved"]
    reranked = pipeline_result["reranked"]

    return {
        "results": [
            {
                "corpus_id": item["corpus_id"],
                "doc_id": item["doc_id"],
                "section_id": item["section_id"],
                "score": item["score"],
                "text_preview": "",
            }
            for item in results
        ],
        "reranked": reranked,
    }
