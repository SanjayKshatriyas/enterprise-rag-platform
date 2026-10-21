"""Answer generation.

* ``ExtractiveLLM``        - offline fallback that returns the most relevant sentences from the
                             retrieved context, each tagged with its source number.
* ``OpenAICompatibleLLM``  - chat-completions API (OpenAI, Azure OpenAI, vLLM, Ollama, ...).
"""
from __future__ import annotations

from typing import Protocol

from app.core.errors import ProviderError
from app.core.models import SearchHit
from app.core.text import split_sentences, tokenize

SYSTEM_PROMPT = (
    "You are an enterprise document assistant. Answer the user's question using ONLY the "
    "numbered context passages provided. Cite the passages you use with bracketed numbers "
    "like [1] or [2][3]. If the context does not contain the answer, say you could not find "
    "it in the uploaded documents. Never invent facts. Be concise."
)


def build_messages(question: str, hits: list[SearchHit]) -> list[dict[str, str]]:
    blocks = []
    for number, hit in enumerate(hits, start=1):
        location = f"{hit.filename}" + (f", page {hit.page}" if hit.page else "")
        blocks.append(f"[{number}] ({location})\n{hit.text}")
    context = "\n\n".join(blocks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


class LLM(Protocol):
    def generate(self, question: str, hits: list[SearchHit]) -> str: ...


class ExtractiveLLM:
    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        question_terms = set(tokenize(question))
        scored: list[tuple[int, int, int, str]] = []
        for number, hit in enumerate(hits, start=1):
            for position, sentence in enumerate(split_sentences(hit.text)):
                overlap = len(question_terms & set(tokenize(sentence)))
                if overlap:
                    scored.append((-overlap, number, position, sentence))
        if not scored:
            sentences = split_sentences(hits[0].text)
            return f"{sentences[0] if sentences else hits[0].text} [1]"

        best = sorted(scored)[: self.max_sentences]
        # Present the chosen sentences in document order (source, then position).
        best.sort(key=lambda item: (item[1], item[2]))
        return " ".join(f"{sentence} [{number}]" for _, number, _, sentence in best)


class OpenAICompatibleLLM:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        timeout: float = 90.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        import httpx

        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is required for the 'openai' LLM provider.")
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": self.temperature,
                    "messages": build_messages(question, hits),
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"LLM request failed: {exc}") from exc
