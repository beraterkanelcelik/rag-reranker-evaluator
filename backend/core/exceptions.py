from __future__ import annotations

from typing import Any, Dict, Optional


class RAGEvalException(Exception):
    def __init__(self, message: str, code: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


class DatasetError(RAGEvalException):
    pass


class EmbeddingError(RAGEvalException):
    pass


class RerankerError(RAGEvalException):
    pass


class JudgeError(RAGEvalException):
    pass


class APIKeyError(RAGEvalException):
    pass


class ModelNotFoundError(RAGEvalException):
    pass
