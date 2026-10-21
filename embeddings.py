"""Embedding providers.

* ``HashingEmbedder``   - deterministic, dependency-free, fully offline (feature hashing of
                          unigrams + bigrams). Lexical, not semantic: ideal for local
                          development, demos and tests.
* ``OpenAICompatibleEmbedder`` - any server speaking the OpenAI ``/embeddings`` API
                          (OpenAI, Azure OpenAI, vLLM, Ollama, LiteLLM, ...). Semantic.
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

from app.core.errors import ProviderError
from app.core.text import tokenize


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _add(self, vector: list[float], feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % self.dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        vector[index] += sign * weight

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        for token in tokens:
            self._add(vector, token, 1.0)
        for first, second in zip(tokens, tokens[1:]):
            self._add(vector, f"{first} {second}", 0.5)
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        dim: int,
        base_url: str = "https://api.openai.com/v1",
        batch_size: int = 64,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is required for the 'openai' embedding provider.")

        vectors: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                try:
                    response = client.post(
                        f"{self.base_url}/embeddings",
                        headers=headers,
                        json={"model": self.model, "input": batch},
                    )
                    response.raise_for_status()
                    data = sorted(response.json()["data"], key=lambda item: item["index"])
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    raise ProviderError(f"Embedding request failed: {exc}") from exc
                for item in data:
                    if len(item["embedding"]) != self.dim:
                        raise ProviderError(
                            f"Embedding has {len(item['embedding'])} dimensions but "
                            f"EMBEDDING_DIM={self.dim}. Set EMBEDDING_DIM to match the model."
                        )
                    vectors.append(item["embedding"])
        return vectors
