from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: UUID
    filename: str
    content_type: str
    size_bytes: int
    num_chunks: int
    created_at: datetime


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, examples=["What is the refund policy?"])
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[UUID] | None = Field(
        default=None, description="Restrict retrieval to these documents."
    )


class SourceOut(BaseModel):
    citation: int = Field(description="Number used to cite this source in the answer, e.g. [1].")
    document_id: UUID
    filename: str
    chunk_index: int
    page: int | None
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    results: list[SourceOut]


class AskResponse(BaseModel):
    question: str
    answer: str
    grounded: bool = Field(description="False when no relevant context was found.")
    sources: list[SourceOut]


class HealthResponse(BaseModel):
    status: str
    version: str
    documents: int
