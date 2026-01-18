from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.models import database as models
from backend.models.schemas import EvaluationRunCreate
from backend.services.evaluation_service import EvaluationService

router = APIRouter()


@router.post("/runs")
async def create_run(
    request: EvaluationRunCreate, background: BackgroundTasks, db: AsyncSession = Depends(get_db)
) -> dict:
    service = EvaluationService(db)
    try:
        run = await service.create_run(request)
        await db.commit()
        background.add_task(service.run_evaluation_async, request, run.id)
        return {
            "run_id": run.id,
            "status": "running",
            "message": "Evaluation started",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/runs")
async def list_runs(
    limit: int = 50,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    query = select(models.EvaluationRun).order_by(desc(models.EvaluationRun.created_at))
    if status:
        query = query.where(models.EvaluationRun.status == status)
    result = await db.execute(query.limit(limit))
    runs = result.scalars().all()
    return {
        "runs": [
            {
                "id": run.id,
                "run_name": run.run_name,
                "status": run.status,
                "sample_size": run.sample_size,
                "use_reranker": run.use_reranker,
                "metrics_summary": run.metrics_summary,
                "created_at": run.created_at.isoformat() if run.created_at else None,
            }
            for run in runs
        ]
    }


@router.get("/runs/{run_id}")
async def get_run(run_id: int, db: AsyncSession = Depends(get_db)) -> dict:
    result = await db.execute(
        select(models.EvaluationRun).where(models.EvaluationRun.id == run_id)
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    embedding_result = await db.execute(
        select(models.EmbeddingModel).where(models.EmbeddingModel.id == run.embedding_model_id)
    )
    embedding_model = embedding_result.scalar_one_or_none()

    duration_seconds = None
    if run.started_at and run.completed_at:
        duration_seconds = int((run.completed_at - run.started_at).total_seconds())

    return {
        "id": run.id,
        "run_name": run.run_name,
        "status": run.status,
        "config": {
            "embedding_model": embedding_model.model_name if embedding_model else None,
            "retrieval_top_k": run.retrieval_top_k,
            "reranker_model": run.reranker_model_name,
            "reranker_top_k": run.reranker_top_k,
            "judge_model": run.judge_model_name,
            "sample_size": run.sample_size,
        },
        "metrics_summary": run.metrics_summary,
        "token_usage": {
            "total_judge_input": run.total_judge_input_tokens,
            "total_judge_output": run.total_judge_output_tokens,
            "estimated_cost_usd": 0.0,
        },
        "timing": {
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "duration_seconds": duration_seconds,
        },
    }


@router.get("/runs/{run_id}/progress")
async def get_progress(run_id: int, db: AsyncSession = Depends(get_db)) -> dict:
    result = await db.execute(
        select(models.EvaluationRun).where(models.EvaluationRun.id == run_id)
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    count_result = await db.execute(
        select(func.count(models.EvaluationResult.id)).where(
            models.EvaluationResult.run_id == run_id
        )
    )
    processed = int(count_result.scalar_one() or 0)
    total = run.sample_size or processed
    percentage = (processed / total * 100.0) if total else 0.0
    current_step = "completed" if run.status == "completed" else "running"

    return {
        "run_id": run_id,
        "status": run.status,
        "progress": {
            "current_query": processed,
            "total_queries": total,
            "percentage": percentage,
            "current_step": current_step,
            "current_query_text": None,
        },
        "estimated_remaining_seconds": None,
    }
