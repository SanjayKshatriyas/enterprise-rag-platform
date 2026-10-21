from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ChunkData:
    """A piece of a document ready to be embedded and stored."""

    index: int
    text: str
    page: int | None = None


@dataclass(frozen=True)
class DocumentInfo:
    id: str
    filename: str
    content_type: str
    size_bytes: int
    num_chunks: int
    created_at: datetime


@dataclass(frozen=True)
class SearchHit:
    """A retrieved chunk together with its similarity score and provenance."""

    document_id: str
    filename: str
    chunk_index: int
    page: int | None
    text: str
    score: float


@dataclass(frozen=True)
class Answer:
    answer: str
    sources: list[SearchHit]
    grounded: bool
