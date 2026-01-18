from __future__ import annotations


from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.core.model_manager import ModelManager

router = APIRouter()


def _get_system_memory_mb() -> int:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return int(memory.total / (1024 * 1024))
    except Exception:
        return 0


def _get_available_memory_mb() -> int:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return int(memory.available / (1024 * 1024))
    except Exception:
        return 0


@router.get("/memory")
async def get_memory_status(db: AsyncSession = Depends(get_db)) -> dict:
    manager = ModelManager()
    loaded_models = manager.get_loaded_models()
    total_memory_mb = int(manager.total_memory_mb())
    return {
        "loaded_models": loaded_models,
        "total_memory_mb": total_memory_mb,
        "system_memory_mb": _get_system_memory_mb(),
        "available_memory_mb": _get_available_memory_mb(),
    }


@router.post("/unload-models")
async def unload_models(db: AsyncSession = Depends(get_db)) -> dict:
    manager = ModelManager()
    total_before = int(manager.total_memory_mb())
    ModelManager.unload_all_instances()
    return {
        "message": "All models unloaded",
        "freed_memory_mb": total_before,
    }
