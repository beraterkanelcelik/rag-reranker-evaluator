from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

from backend.core.database import Base


class DatasetStatus(Base):
    __tablename__ = "dataset_status"

    id = Column(Integer, primary_key=True)
    dataset_name = Column(String(255), nullable=False, server_default="vectara/open_ragbench")
    subset_name = Column(String(255))
    status = Column(String(50), nullable=False, server_default="pending")
    total_documents = Column(Integer, server_default="0")
    total_queries = Column(Integer, server_default="0")
    total_sections = Column(Integer, server_default="0")
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())


class Corpus(Base):
    __tablename__ = "corpus"
    __table_args__ = (
        UniqueConstraint("doc_id", "section_id"),
        Index("idx_corpus_doc_id", "doc_id"),
        Index("idx_corpus_section_id", "section_id"),
    )

    id = Column(Integer, primary_key=True)
    doc_id = Column(String(255), nullable=False)
    section_id = Column(Integer, nullable=False)
    section_text = Column(Text, nullable=False)
    tables_markdown = Column(Text)
    has_images = Column(Boolean, server_default="false")
    image_count = Column(Integer, server_default="0")
    metadata_json = Column("metadata", JSONB)
    created_at = Column(DateTime, server_default=func.now())


class Query(Base):
    __tablename__ = "queries"
    __table_args__ = (
        Index("idx_queries_uuid", "query_uuid"),
        Index("idx_queries_type", "query_type"),
    )

    id = Column(Integer, primary_key=True)
    query_uuid = Column(String(255), nullable=False, unique=True)
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50))
    source_type = Column(String(50))
    metadata_json = Column("metadata", JSONB)
    created_at = Column(DateTime, server_default=func.now())


class Qrel(Base):
    __tablename__ = "qrels"
    __table_args__ = (
        UniqueConstraint("query_uuid", "doc_id", "section_id"),
        ForeignKeyConstraint(["doc_id", "section_id"], ["corpus.doc_id", "corpus.section_id"]),
        Index("idx_qrels_query", "query_uuid"),
    )

    id = Column(Integer, primary_key=True)
    query_uuid = Column(String(255), ForeignKey("queries.query_uuid"), nullable=False)
    doc_id = Column(String(255), nullable=False)
    section_id = Column(Integer, nullable=False)
    relevance_score = Column(Integer, server_default="1")
    created_at = Column(DateTime, server_default=func.now())


class Answer(Base):
    __tablename__ = "answers"

    id = Column(Integer, primary_key=True)
    query_uuid = Column(String(255), ForeignKey("queries.query_uuid"), nullable=False, unique=True)
    reference_answer = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class EmbeddingModel(Base):
    __tablename__ = "embedding_models"
    __table_args__ = (
        Index("idx_embedding_models_name", "model_name"),
        Index("idx_embedding_models_status", "status"),
    )

    id = Column(Integer, primary_key=True)
    model_name = Column(String(255), nullable=False)
    model_source = Column(String(50), nullable=False)
    dimension = Column(Integer, nullable=False)
    table_name = Column(String(255), nullable=False, unique=True)
    config = Column(JSONB, nullable=False)
    status = Column(String(50), server_default="pending")
    total_vectors = Column(Integer, server_default="0")
    error_message = Column(Text)
    api_key_hash = Column(String(64))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (
        Index("idx_evaluation_runs_status", "status"),
        Index("idx_evaluation_runs_created", "created_at"),
    )

    id = Column(Integer, primary_key=True)
    run_name = Column(String(255))

    embedding_model_id = Column(Integer, ForeignKey("embedding_models.id"), nullable=False)
    retrieval_top_k = Column(Integer, nullable=False)

    use_reranker = Column(Boolean, server_default="false")
    reranker_model_name = Column(String(255))
    reranker_top_k = Column(Integer)
    reranker_config = Column(JSONB)

    judge_model_name = Column(String(255), nullable=False)
    judge_config = Column(JSONB)

    sample_size = Column(Integer, nullable=False)
    sample_seed = Column(Integer)

    status = Column(String(50), server_default="pending")
    error_message = Column(Text)

    metrics_summary = Column(JSONB)

    total_embedding_tokens = Column(Integer, server_default="0")
    total_judge_input_tokens = Column(Integer, server_default="0")
    total_judge_output_tokens = Column(Integer, server_default="0")

    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    __table_args__ = (
        Index("idx_eval_results_run", "run_id"),
        Index("idx_eval_results_query", "query_uuid"),
    )

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False)
    query_uuid = Column(String(255), ForeignKey("queries.query_uuid"), nullable=False)

    retrieved_ids = Column(JSONB, nullable=False)
    reranked_ids = Column(JSONB)

    final_context_ids = Column(JSONB, nullable=False)
    final_context_text = Column(Text)

    generated_answer = Column(Text, nullable=False)

    retrieval_recall_at_k = Column(Float)
    retrieval_mrr = Column(Float)
    retrieval_ndcg = Column(Float)
    gold_in_top_k = Column(Boolean)

    context_tokens = Column(Integer)
    answer_tokens = Column(Integer)

    created_at = Column(DateTime, server_default=func.now())


class JudgeScore(Base):
    __tablename__ = "judge_scores"
    __table_args__ = (Index("idx_judge_scores_result", "result_id"),)

    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey("evaluation_results.id", ondelete="CASCADE"), nullable=False)

    track_a_correctness = Column(Float)
    track_a_completeness = Column(Float)
    track_a_specificity = Column(Float)
    track_a_clarity = Column(Float)
    track_a_overall = Column(Float)
    track_a_reason = Column(Text)
    track_a_raw_response = Column(JSONB)

    track_b_context_support = Column(Float)
    track_b_hallucination = Column(Float)
    track_b_citation_quality = Column(Float)
    track_b_overall = Column(Float)
    track_b_unsupported_claims = Column(JSONB)
    track_b_raw_response = Column(JSONB)

    track_a_input_tokens = Column(Integer)
    track_a_output_tokens = Column(Integer)
    track_b_input_tokens = Column(Integer)
    track_b_output_tokens = Column(Integer)

    created_at = Column(DateTime, server_default=func.now())
