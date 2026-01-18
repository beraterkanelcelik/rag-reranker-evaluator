from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import random
from typing import Iterable, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.model_manager import ModelManager
from backend.models import database as models
from backend.models.schemas import EvaluationRunCreate
from backend.services.generation_service import GenerationService
from backend.services.judge_service import JudgeService
from backend.services.metrics_service import MetricsService, RetrievalMetrics
from backend.services.retrieval_pipeline import RetrievalPipeline


@dataclass
class EvaluationRunResult:
    run_id: int
    status: str
    metrics_summary: Optional[dict]
    results: List[dict]


class EvaluationService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.model_manager = ModelManager()
        self.retrieval_pipeline = RetrievalPipeline(db, self.model_manager)
        self.generation_service = GenerationService()
        self.judge_service = JudgeService()
        self.metrics_service = MetricsService()

    async def create_run(self, config: EvaluationRunCreate) -> models.EvaluationRun:
        return await self._create_run_entry(config)

    async def run_evaluation_async(self, config: EvaluationRunCreate, run_id: int) -> None:
        run_result = await self.db.execute(
            select(models.EvaluationRun).where(models.EvaluationRun.id == run_id)
        )
        run = run_result.scalar_one()
        retrieval_metrics_list: List[RetrievalMetrics] = []
        track_a_scores: List[dict] = []
        track_b_scores: List[dict] = []
        total_judge_input = 0
        total_judge_output = 0

        try:
            queries = await self._select_queries(config.sample_size, config.sample_seed)
            for query in queries:
                reference_answer = await self._get_reference_answer(query.query_uuid)
                qrels = await self._get_qrels(query.query_uuid)

                pipeline_result = await self.retrieval_pipeline.retrieve(
                    model_id=config.embedding_model_id,
                    query_text=query.query_text,
                    retrieval_top_k=config.retrieval_top_k,
                    use_reranker=config.use_reranker,
                    reranker_model_name=self._reranker_name(config),
                    reranker_top_k=self._reranker_top_k(config),
                )
                retrieved = pipeline_result["retrieved"]
                reranked = pipeline_result["reranked"]
                final_docs = reranked if reranked else retrieved

                context_entries = await self._fetch_context_entries(final_docs)
                contexts = [entry["text"] for entry in context_entries]
                context_text = "\n\n".join(contexts)

                generation_result = self.generation_service.generate_answer(
                    question=query.query_text,
                    contexts=contexts,
                    model_name=config.judge_config.model_name,
                    api_key=config.judge_config.api_key,
                    temperature=config.judge_config.temperature,
                )

                track_a_result = self.judge_service.judge_track_a(
                    question=query.query_text,
                    reference_answer=reference_answer,
                    model_answer=generation_result.answer,
                    model_name=config.judge_config.model_name,
                    api_key=config.judge_config.api_key,
                    temperature=config.judge_config.temperature,
                )
                track_b_result = self.judge_service.judge_track_b(
                    question=query.query_text,
                    model_answer=generation_result.answer,
                    contexts=contexts,
                    model_name=config.judge_config.model_name,
                    api_key=config.judge_config.api_key,
                    temperature=config.judge_config.temperature,
                )

                metrics = self.metrics_service.compute_retrieval_metrics(
                    retrieved=retrieved,
                    relevant=qrels,
                    k=config.retrieval_top_k,
                )

                evaluation_result = models.EvaluationResult(
                    run_id=run.id,
                    query_uuid=query.query_uuid,
                    retrieved_ids=retrieved,
                    reranked_ids=reranked,
                    final_context_ids=final_docs,
                    final_context_text=context_text,
                    generated_answer=generation_result.answer,
                    retrieval_recall_at_k=metrics.recall_at_k,
                    retrieval_mrr=metrics.mrr,
                    retrieval_ndcg=metrics.ndcg_at_k,
                    gold_in_top_k=metrics.gold_in_top_k,
                    context_tokens=generation_result.input_tokens,
                    answer_tokens=generation_result.output_tokens,
                )
                self.db.add(evaluation_result)
                await self.db.flush()

                track_a_data = track_a_result.scores
                track_b_data = track_b_result.scores

                judge_score = models.JudgeScore(
                    result_id=evaluation_result.id,
                    track_a_correctness=track_a_data.get("correctness"),
                    track_a_completeness=track_a_data.get("completeness"),
                    track_a_specificity=track_a_data.get("specificity"),
                    track_a_clarity=track_a_data.get("clarity"),
                    track_a_overall=track_a_data.get("overall"),
                    track_a_reason=track_a_data.get("short_reason") or track_a_data.get("reason"),
                    track_a_raw_response=track_a_result.raw_response,
                    track_b_context_support=track_b_data.get("context_support"),
                    track_b_hallucination=track_b_data.get("hallucination"),
                    track_b_citation_quality=track_b_data.get("citation_quality"),
                    track_b_overall=track_b_data.get("overall_groundedness") or track_b_data.get("overall"),
                    track_b_unsupported_claims=track_b_data.get("unsupported_claims"),
                    track_b_raw_response=track_b_result.raw_response,
                    track_a_input_tokens=track_a_result.input_tokens,
                    track_a_output_tokens=track_a_result.output_tokens,
                    track_b_input_tokens=track_b_result.input_tokens,
                    track_b_output_tokens=track_b_result.output_tokens,
                )
                self.db.add(judge_score)

                retrieval_metrics_list.append(metrics)
                track_a_scores.append(track_a_data)
                track_b_scores.append(track_b_data)
                total_judge_input += track_a_result.input_tokens + track_b_result.input_tokens
                total_judge_output += track_a_result.output_tokens + track_b_result.output_tokens

            run.metrics_summary = {
                "retrieval": self.metrics_service.aggregate_retrieval_metrics(
                    retrieval_metrics_list
                ),
                "track_a": self.metrics_service.aggregate_track_a(track_a_scores),
                "track_b": self.metrics_service.aggregate_track_b(track_b_scores),
            }
            run.total_judge_input_tokens = total_judge_input
            run.total_judge_output_tokens = total_judge_output
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            await self.db.commit()
        except Exception as exc:
            run.status = "error"
            run.error_message = str(exc)
            run.completed_at = datetime.utcnow()
            await self.db.commit()
            # Do not raise, since it's background

    async def _create_run_entry(self, config: EvaluationRunCreate) -> models.EvaluationRun:
        reranker_config = config.reranker_config.model_dump() if config.reranker_config else None
        run = models.EvaluationRun(
            run_name=config.run_name,
            embedding_model_id=config.embedding_model_id,
            retrieval_top_k=config.retrieval_top_k,
            use_reranker=config.use_reranker,
            reranker_model_name=self._reranker_name(config),
            reranker_top_k=self._reranker_top_k(config) if config.use_reranker else None,
            reranker_config=reranker_config,
            judge_model_name=config.judge_config.model_name,
            judge_config={
                "model_name": config.judge_config.model_name,
                "temperature": config.judge_config.temperature,
            },
            sample_size=config.sample_size,
            sample_seed=config.sample_seed,
            status="running",
            started_at=datetime.utcnow(),
        )
        self.db.add(run)
        await self.db.flush()
        return run

    async def _select_queries(self, sample_size: int, sample_seed: Optional[int]) -> List[models.Query]:
        result = await self.db.execute(select(models.Query))
        queries = result.scalars().all()
        if sample_size >= len(queries):
            return queries
        rng = random.Random(sample_seed)
        return rng.sample(queries, sample_size)

    async def _get_reference_answer(self, query_uuid: str) -> str:
        result = await self.db.execute(
            select(models.Answer).where(models.Answer.query_uuid == query_uuid)
        )
        row = result.scalar_one_or_none()
        return row.reference_answer if row else ""

    async def _get_qrels(self, query_uuid: str) -> List[dict]:
        result = await self.db.execute(
            select(models.Qrel).where(models.Qrel.query_uuid == query_uuid)
        )
        rows = result.scalars().all()
        return [
            {
                "doc_id": row.doc_id,
                "section_id": row.section_id,
                "relevance_score": row.relevance_score,
            }
            for row in rows
        ]

    async def _fetch_context_entries(self, items: Iterable[dict]) -> List[dict]:
        entries: List[dict] = []
        for item in items:
            result = await self.db.execute(
                select(models.Corpus).where(
                    models.Corpus.doc_id == item.get("doc_id"),
                    models.Corpus.section_id == item.get("section_id"),
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                continue
            entries.append(
                {
                    "doc_id": row.doc_id,
                    "section_id": row.section_id,
                    "text": self._format_context_text(row),
                }
            )
        return entries

    def _format_context_text(self, row: models.Corpus) -> str:
        if row.tables_markdown:
            return f"{row.section_text}\n\n{row.tables_markdown}"
        return row.section_text

    def _reranker_name(self, config: EvaluationRunCreate) -> Optional[str]:
        if not config.use_reranker:
            return None
        if not config.reranker_config:
            raise ValueError("Reranker config is required when use_reranker is true")
        return config.reranker_config.model_name

    def _reranker_top_k(self, config: EvaluationRunCreate) -> int:
        if not config.reranker_config:
            return config.retrieval_top_k
        return config.reranker_config.top_k
