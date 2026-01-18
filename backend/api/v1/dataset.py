from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.models import database as models
from backend.models.schemas import DatasetIngestRequest
from backend.services.dataset_ingestion import DatasetIngestionService

router = APIRouter()


@router.get("/status")
async def get_status(db: AsyncSession = Depends(get_db)) -> dict:
    service = DatasetIngestionService(db)
    status = await service.get_status()
    if status is None:
        return {
            "status": "pending",
            "dataset_name": "vectara/open_ragbench",
            "total_documents": 0,
            "total_queries": 0,
            "total_sections": 0,
            "completed_at": None,
        }
    return {
        "status": status.status,
        "dataset_name": status.dataset_name,
        "total_documents": status.total_documents,
        "total_queries": status.total_queries,
        "total_sections": status.total_sections,
        "completed_at": status.completed_at.isoformat() if status.completed_at else None,
        "error_message": status.error_message,
    }


@router.post("/ingest")
async def ingest_dataset(
    request: DatasetIngestRequest, db: AsyncSession = Depends(get_db)
) -> dict:
    service = DatasetIngestionService(db)
    try:
        result = await service.ingest(request.subset)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "status": "downloading",
        "message": "Dataset ingestion started",
        "job_id": "dataset_ingest_1",
        "result": result,
    }


@router.get("/queries")
async def get_queries(
    limit: int = 100,
    offset: int = 0,
    query_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    query = select(models.Query)
    if query_type:
        query = query.where(models.Query.query_type == query_type)
    total_result = await db.execute(select(models.Query.id))
    total = len(total_result.scalars().all())
    result = await db.execute(query.offset(offset).limit(limit))
    rows = result.scalars().all()
    return {
        "queries": [
            {
                "query_uuid": row.query_uuid,
                "query_text": row.query_text,
                "query_type": row.query_type,
                "source_type": row.source_type,
            }
            for row in rows
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/corpus/{doc_id}/{section_id}")
async def get_corpus(
    doc_id: str, section_id: int, db: AsyncSession = Depends(get_db)
) -> dict:
    result = await db.execute(
        select(models.Corpus).where(
            models.Corpus.doc_id == doc_id, models.Corpus.section_id == section_id
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Corpus section not found")
    return {
        "doc_id": row.doc_id,
        "section_id": row.section_id,
        "section_text": row.section_text,
        "tables_markdown": row.tables_markdown,
        "has_images": row.has_images,
    }
