from __future__ import annotations

from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from environment variables or a ``.env`` file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Storage
    store_backend: Literal["postgres", "memory"] = "postgres"
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    # Embeddings
    embedding_provider: Literal["hashing", "openai"] = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 384

    # LLM
    llm_provider: Literal["extractive", "openai"] = "extractive"
    llm_model: str = "gpt-4o-mini"

    # OpenAI-compatible endpoint (OpenAI, Azure OpenAI, vLLM, Ollama, LiteLLM, ...)
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"

    # Chunking / retrieval
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5
    min_score: float = 0.1

    # API
    max_upload_mb: int = 20
    api_key: str | None = None

    @field_validator("openai_api_key", "api_key", mode="before")
    @classmethod
    def _blank_to_none(cls, value):
        return value or None

    @model_validator(mode="after")
    def _check_chunking(self):
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        return self
