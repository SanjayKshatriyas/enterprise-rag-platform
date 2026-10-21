"""In-memory vector store (cosine similarity via NumPy). Used for tests and quick demos."""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

import numpy as np

from app.core.models import ChunkData, DocumentInfo, SearchHit


class InMemoryStore:
    def __init__(self) -> None:
        self._documents: dict[str, DocumentInfo] = {}
        self._chunks: dict[str, list[tuple[ChunkData, np.ndarray]]] = {}
        self._lock = threading.RLock()

    def add_document(self, filename, content_type, size_bytes, chunks, embeddings) -> DocumentInfo:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        info = DocumentInfo(
            id=str(uuid.uuid4()),
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            num_chunks=len(chunks),
            created_at=datetime.now(timezone.utc),
        )
        vectors = [np.asarray(e, dtype=np.float32) for e in embeddings]
        with self._lock:
            self._documents[info.id] = info
            self._chunks[info.id] = list(zip(chunks, vectors))
        return info

    def list_documents(self) -> list[DocumentInfo]:
        with self._lock:
            return sorted(self._documents.values(), key=lambda d: d.created_at, reverse=True)

    def get_document(self, document_id: str) -> DocumentInfo | None:
        with self._lock:
            return self._documents.get(document_id)

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            self._chunks.pop(document_id, None)
            return self._documents.pop(document_id, None) is not None

    def search(self, embedding, top_k, document_ids=None) -> list[SearchHit]:
        query = np.asarray(embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return []

        candidates: list[SearchHit] = []
        with self._lock:
            wanted = set(document_ids) if document_ids is not None else None
            for doc_id, items in self._chunks.items():
                if wanted is not None and doc_id not in wanted:
                    continue
                filename = self._documents[doc_id].filename
                for chunk, vector in items:
                    denom = query_norm * float(np.linalg.norm(vector))
                    score = float(np.dot(query, vector) / denom) if denom else 0.0
                    candidates.append(
                        SearchHit(doc_id, filename, chunk.index, chunk.page, chunk.text, score)
                    )
        candidates.sort(key=lambda hit: hit.score, reverse=True)
        return candidates[:top_k]

    def close(self) -> None:
        pass
