from __future__ import annotations

from fastapi import APIRouter

from backend.api.v1.dataset import router as dataset_router
from backend.api.v1.embedding import router as embedding_router
from backend.api.v1.evaluation import router as evaluation_router
from backend.api.v1.health import router as health_router
from backend.api.v1.results import router as results_router
from backend.api.v1.system import router as system_router


api_router = APIRouter()
api_router.include_router(health_router, prefix="/health", tags=["health"])
api_router.include_router(dataset_router, prefix="/dataset", tags=["dataset"])
api_router.include_router(embedding_router, prefix="/embedding", tags=["embedding"])
api_router.include_router(evaluation_router, prefix="/evaluation", tags=["evaluation"])
api_router.include_router(results_router, prefix="/results", tags=["results"])
api_router.include_router(system_router, prefix="/system", tags=["system"])
