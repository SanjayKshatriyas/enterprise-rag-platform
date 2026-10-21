"""Document ingestion: parse -> chunk -> embed -> store."""
from __future__ import annotations

import os

from app.core.chunking import chunk_text
from app.core.embeddings import Embedder
from app.core.errors import EmptyDocumentError
from app.core.models import ChunkData, DocumentInfo
from app.core.parsing import extract_pages
from app.storage.base import VectorStore


class IngestionService:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, filename: str, content_type: str, data: bytes) -> DocumentInfo:
        safe_name = os.path.basename(filename or "document")[:255] or "document"

        chunks: list[ChunkData] = []
        for page_number, text in extract_pages(safe_name, data):
            for piece in chunk_text(text, self.chunk_size, self.chunk_overlap):
                chunks.append(ChunkData(index=len(chunks), text=piece, page=page_number))
        if not chunks:
            raise EmptyDocumentError("Document produced no text chunks.")

        embeddings = self.embedder.embed([c.text for c in chunks])
        return self.store.add_document(
            filename=safe_name,
            content_type=content_type or "application/octet-stream",
            size_bytes=len(data),
            chunks=chunks,
            embeddings=embeddings,
        )
