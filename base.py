from __future__ import annotations

from typing import Protocol

from app.core.models import ChunkData, DocumentInfo, SearchHit


class VectorStore(Protocol):
    """Storage + similarity search for documents and their embedded chunks."""

    def add_document(
        self,
        filename: str,
        content_type: str,
        size_bytes: int,
        chunks: list[ChunkData],
        embeddings: list[list[float]],
    ) -> DocumentInfo: ...

    def list_documents(self) -> list[DocumentInfo]: ...

    def get_document(self, document_id: str) -> DocumentInfo | None: ...

    def delete_document(self, document_id: str) -> bool: ...

    def search(
        self,
        embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]: ...

    def close(self) -> None: ...
