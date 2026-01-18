from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelSource(str, Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


class DatasetIngestRequest(BaseModel):
    subset: str = Field(default="official/pdf/arxiv", description="Dataset subset path")


class EmbeddingModelCreate(BaseModel):
    model_name: str = Field(..., description="Model name/path")
    model_source: ModelSource
    dimension: int = Field(..., gt=0, le=4096)
    config: Dict[str, Any] = Field(default_factory=dict)
    api_key: Optional[str] = Field(default=None, description="API key for OpenAI")


class RerankerConfig(BaseModel):
    model_name: str = Field(..., description="HuggingFace cross-encoder model")
    top_k: int = Field(default=5, ge=1, le=50)
    config: Dict[str, Any] = Field(default_factory=dict)


class JudgeConfig(BaseModel):
    model_name: str = Field(default="gpt-4o", description="OpenAI model name")
    api_key: str = Field(..., description="OpenAI API key")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class EvaluationRunCreate(BaseModel):
    run_name: Optional[str] = None
    embedding_model_id: int
    retrieval_top_k: int = Field(default=50, ge=1, le=500)
    use_reranker: bool = False
    reranker_config: Optional[RerankerConfig] = None
    judge_config: JudgeConfig
    sample_size: int = Field(default=100, ge=1, le=3045)
    sample_seed: Optional[int] = Field(default=None)


class SearchRequest(BaseModel):
    model_id: int
    query_text: str
    top_k: int = Field(default=10, ge=1, le=100)
    use_reranker: bool = False
    reranker_model_name: Optional[str] = None
    reranker_top_k: int = Field(default=5, ge=1, le=50)


class DatasetStatus(BaseModel):
    status: str
    dataset_name: str
    total_documents: int
    total_queries: int
    total_sections: int
    error_message: Optional[str] = None
    completed_at: Optional[str] = None


class EmbeddingModelResponse(BaseModel):
    id: int
    model_name: str
    model_source: str
    dimension: int
    status: str
    total_vectors: int
    table_name: str
    created_at: str


class EmbeddingProgress(BaseModel):
    id: int
    status: str
    progress: Dict[str, Any]
    estimated_remaining_seconds: Optional[int] = None


class MetricsSummary(BaseModel):
    retrieval: Dict[str, float]
    track_a: Dict[str, float]
    track_b: Dict[str, float]


class TokenUsage(BaseModel):
    total_judge_input: int
    total_judge_output: int
    estimated_cost_usd: float


class EvaluationRunResponse(BaseModel):
    id: int
    run_name: Optional[str]
    status: str
    config: Dict[str, Any]
    metrics_summary: Optional[MetricsSummary] = None
    token_usage: Optional[TokenUsage] = None
    created_at: str
    completed_at: Optional[str] = None


class RetrievedDocument(BaseModel):
    rank: int
    corpus_id: int
    doc_id: str
    section_id: int
    score: float
    text_preview: str


class JudgeScores(BaseModel):
    track_a: Dict[str, Any]
    track_b: Dict[str, Any]


class QueryResult(BaseModel):
    query_uuid: str
    query_text: str
    reference_answer: str
    generated_answer: str
    retrieved_docs: List[RetrievedDocument]
    reranked_docs: Optional[List[RetrievedDocument]] = None
    scores: JudgeScores
    retrieval_metrics: Dict[str, Any]


class MemoryStatus(BaseModel):
    loaded_models: List[Dict[str, Any]]
    total_memory_mb: int
    system_memory_mb: int
    available_memory_mb: int
