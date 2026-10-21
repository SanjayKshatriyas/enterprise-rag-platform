"""Retrieval-augmented generation: retrieve -> (guard) -> generate with citations."""
from __future__ import annotations

from app.core.embeddings import Embedder
from app.core.llm import LLM
from app.core.models import Answer, SearchHit
from app.storage.base import VectorStore

NO_ANSWER = "I couldn't find relevant information in the uploaded documents."


class RAGService:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        llm: LLM,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    def search(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        embedding = self.embedder.embed([query])[0]
        hits = self.store.search(embedding, top_k or self.top_k, document_ids)
        return [hit for hit in hits if hit.score >= self.min_score]

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> Answer:
        hits = self.search(question, top_k, document_ids)
        if not hits:
            return Answer(answer=NO_ANSWER, sources=[], grounded=False)
        return Answer(answer=self.llm.generate(question, hits), sources=hits, grounded=True)
