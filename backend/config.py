from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://rageval:rageval_secret@db:5432/rageval"
    log_level: str = "INFO"
    api_version: str = "v1"
    app_version: str = "1.0.0"

    class Config:
        env_prefix = ""


settings = Settings()
