from __future__ import annotations

from fastapi import FastAPI

from backend.api.router import api_router
from backend.config import settings
from backend.core.database import Base, engine
from backend.core.progress import progress_store
from backend.models import database  # noqa: F401


def create_app() -> FastAPI:
    app = FastAPI(title="RAG + Reranker Evaluation Platform", version=settings.app_version)
    app.include_router(api_router, prefix=f"/api/{settings.api_version}")

    @app.on_event("startup")
    async def startup() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    return app


app = create_app()
