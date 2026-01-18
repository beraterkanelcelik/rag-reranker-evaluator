from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.models import database as models

router = APIRouter()


@router.get("/{run_id}/details")
async def get_results(
    run_id: int,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> dict:
    result = await db.execute(
        select(models.EvaluationResult)
        .where(models.EvaluationResult.run_id == run_id)
        .offset(offset)
        .limit(limit)
    )
    rows = result.scalars().all()
    total_result = await db.execute(
        select(models.EvaluationResult.id).where(models.EvaluationResult.run_id == run_id)
    )
    total = len(total_result.scalars().all())

    payload = []
    for row in rows:
        judge_result = await db.execute(
            select(models.JudgeScore).where(models.JudgeScore.result_id == row.id)
        )
        judge_score = judge_result.scalar_one_or_none()
        query_result = await db.execute(
            select(models.Query).where(models.Query.query_uuid == row.query_uuid)
        )
        query = query_result.scalar_one_or_none()
        answer_result = await db.execute(
            select(models.Answer).where(models.Answer.query_uuid == row.query_uuid)
        )
        answer = answer_result.scalar_one_or_none()

        payload.append(
            {
                "query_uuid": row.query_uuid,
                "query_text": query.query_text if query else "",
                "reference_answer": answer.reference_answer if answer else "",
                "generated_answer": row.generated_answer,
                "retrieved_docs": row.retrieved_ids,
                "reranked_docs": row.reranked_ids,
                "scores": {
                    "track_a": {
                        "correctness": judge_score.track_a_correctness if judge_score else None,
                        "completeness": judge_score.track_a_completeness if judge_score else None,
                        "specificity": judge_score.track_a_specificity if judge_score else None,
                        "clarity": judge_score.track_a_clarity if judge_score else None,
                        "overall": judge_score.track_a_overall if judge_score else None,
                        "reason": judge_score.track_a_reason if judge_score else None,
                    },
                    "track_b": {
                        "context_support": judge_score.track_b_context_support if judge_score else None,
                        "hallucination": judge_score.track_b_hallucination if judge_score else None,
                        "citation_quality": judge_score.track_b_citation_quality if judge_score else None,
                        "overall": judge_score.track_b_overall if judge_score else None,
                        "unsupported_claims": judge_score.track_b_unsupported_claims if judge_score else None,
                    },
                },
                "retrieval_metrics": {
                    "gold_in_top_k": row.gold_in_top_k,
                    "recall_at_k": row.retrieval_recall_at_k,
                    "mrr": row.retrieval_mrr,
                    "ndcg_at_k": row.retrieval_ndcg,
                },
            }
        )

    return {
        "run_id": run_id,
        "results": payload,
        "total": total,
    }


@router.get("/{run_id}/query/{query_uuid}")
async def get_query_result(
    run_id: int,
    query_uuid: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    result = await db.execute(
        select(models.EvaluationResult).where(
            models.EvaluationResult.run_id == run_id,
            models.EvaluationResult.query_uuid == query_uuid,
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Result not found")

    query_result = await db.execute(
        select(models.Query).where(models.Query.query_uuid == row.query_uuid)
    )
    query = query_result.scalar_one_or_none()
    answer_result = await db.execute(
        select(models.Answer).where(models.Answer.query_uuid == row.query_uuid)
    )
    answer = answer_result.scalar_one_or_none()
    judge_result = await db.execute(
        select(models.JudgeScore).where(models.JudgeScore.result_id == row.id)
    )
    judge_score = judge_result.scalar_one_or_none()

    return {
        "query": {
            "uuid": row.query_uuid,
            "text": query.query_text if query else "",
            "type": query.query_type if query else None,
        },
        "reference_answer": answer.reference_answer if answer else "",
        "generated_answer": row.generated_answer,
        "retrieved_documents": row.retrieved_ids,
        "reranked_documents": row.reranked_ids,
        "final_context": row.final_context_text,
        "judge_responses": {
            "track_a": {
                "scores": {
                    "correctness": judge_score.track_a_correctness if judge_score else None,
                    "completeness": judge_score.track_a_completeness if judge_score else None,
                    "specificity": judge_score.track_a_specificity if judge_score else None,
                    "clarity": judge_score.track_a_clarity if judge_score else None,
                    "overall": judge_score.track_a_overall if judge_score else None,
                    "reason": judge_score.track_a_reason if judge_score else None,
                },
                "raw_response": judge_score.track_a_raw_response if judge_score else None,
            },
            "track_b": {
                "scores": {
                    "context_support": judge_score.track_b_context_support if judge_score else None,
                    "hallucination": judge_score.track_b_hallucination if judge_score else None,
                    "citation_quality": judge_score.track_b_citation_quality if judge_score else None,
                    "overall": judge_score.track_b_overall if judge_score else None,
                    "unsupported_claims": judge_score.track_b_unsupported_claims if judge_score else None,
                },
                "raw_response": judge_score.track_b_raw_response if judge_score else None,
            },
        },
        "token_counts": {
            "context_tokens": row.context_tokens,
            "answer_tokens": row.answer_tokens,
            "judge_tokens": (judge_score.track_a_input_tokens if judge_score else 0)
            + (judge_score.track_a_output_tokens if judge_score else 0)
            + (judge_score.track_b_input_tokens if judge_score else 0)
            + (judge_score.track_b_output_tokens if judge_score else 0),
        },
    }


@router.get("/{run_id}/export")
async def export_results(run_id: int, db: AsyncSession = Depends(get_db)) -> dict:
    results_payload = await get_results(run_id=run_id, db=db, limit=100000, offset=0)
    return results_payload
