# Enterprise Document Intelligence & RAG Platform

Upload company documents, search them semantically, and ask questions that are answered by an
LLM **with source citations**. Built with **Python, FastAPI, PostgreSQL + pgvector, embeddings
and vector similarity search**.

```text
POST /documents  ->  parse -> chunk -> embed -> store (PostgreSQL + pgvector)
POST /ask        ->  embed question -> vector search -> LLM answer with [1][2] citations
```

## Features

- **Document ingestion** for PDF, DOCX, TXT and Markdown, with per-page tracking for PDFs.
- **Sentence-aware chunking** with configurable size and overlap.
- **Vector similarity search** using pgvector (HNSW index, cosine distance), optionally scoped to
  specific documents.
- **Grounded answers with traceability**: every answer returns numbered sources (file, page,
  chunk, similarity score, text). If nothing relevant is found, the API says so instead of guessing.
- **Pluggable providers**: any OpenAI-compatible API (OpenAI, Azure OpenAI, vLLM, Ollama,
  LiteLLM) for embeddings and chat, or fully **offline** defaults (hashing embedder + extractive
  answerer) so the project runs with zero API keys.
- **REST API** with OpenAPI docs at `/docs`, optional `X-API-Key` auth, upload size limits and
  clear error codes (415 / 413 / 422 / 502).
- **Docker Compose** setup (API + `pgvector/pgvector:pg16`) and GitHub Actions CI.

## Architecture

```mermaid
flowchart LR
    U[Client] -->|upload| API[FastAPI]
    API --> P[Parser<br/>PDF / DOCX / TXT / MD]
    P --> C[Chunker]
    C --> E[Embedder]
    E --> DB[(PostgreSQL + pgvector)]
    U -->|ask / search| API
    API --> Q[Embed query]
    Q --> DB
    DB -->|top-k chunks| R[RAG service]
    R --> L[LLM]
    L -->|answer + citations| U
```

The code is split so each stage can be swapped independently:

| Layer | Path | Responsibility |
|---|---|---|
| API | `app/main.py`, `app/schemas.py` | HTTP endpoints, validation, auth, error mapping |
| Core | `app/core/` | parsing, chunking, embeddings, LLM, RAG orchestration |
| Storage | `app/storage/` | `PostgresStore` (pgvector) and `InMemoryStore` behind one `VectorStore` interface |

## Quick start (Docker)

```bash
docker compose up --build
# API:  http://localhost:8000      Docs: http://localhost:8000/docs
```

Try it with the sample documents:

```bash
curl -F "file=@examples/hr_policy.md"       http://localhost:8000/documents
curl -F "file=@examples/security_policy.md" http://localhost:8000/documents

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How many vacation days do employees get?"}'
```

Example response (values illustrative):

```json
{
  "question": "How many vacation days do employees get?",
  "answer": "Employees receive 25 days of paid vacation per year. [1]",
  "grounded": true,
  "sources": [
    {
      "citation": 1,
      "document_id": "…",
      "filename": "hr_policy.md",
      "chunk_index": 0,
      "page": null,
      "score": 0.61,
      "text": "# Employee Handbook: Leave Policy Employees receive 25 days …"
    }
  ]
}
```

## Run locally without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Option A: no database needed (data is lost on restart)
STORE_BACKEND=memory uvicorn app.main:app --reload

# Option B: PostgreSQL with the pgvector extension available
cp .env.example .env        # edit DATABASE_URL if needed
uvicorn app.main:app --reload
```

## Using real semantic models

The offline defaults are lexical (keyword overlap), which is perfect for demos and tests but not
for production relevance. For semantic retrieval and fluent answers, point the app at an
OpenAI-compatible endpoint:

```bash
# .env  (OpenAI)
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

```bash
# .env  (local Ollama)
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768
LLM_PROVIDER=openai
LLM_MODEL=llama3.1
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
```

> `EMBEDDING_DIM` fixes the width of the pgvector column. If you change embedding models, use a
> fresh database (or drop the `chunks` table) and re-upload your documents.

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check (always public) |
| `POST` | `/documents` | Upload and index a file (`multipart/form-data`, field `file`) |
| `GET` | `/documents` | List indexed documents |
| `GET` | `/documents/{id}` | Get one document |
| `DELETE` | `/documents/{id}` | Delete a document and its chunks |
| `POST` | `/search` | Vector search: `{"query", "top_k"?, "document_ids"?}` |
| `POST` | `/ask` | Retrieve + generate an answer with citations (same body as `/search`) |

Set `API_KEY` to require an `X-API-Key` header on everything except `/health`.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `STORE_BACKEND` | `postgres` | `postgres` or `memory` |
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5432/rag` | PostgreSQL connection string |
| `EMBEDDING_PROVIDER` | `hashing` | `hashing` (offline) or `openai` (OpenAI-compatible) |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `text-embedding-3-small` / `384` | Model name and vector width |
| `LLM_PROVIDER` / `LLM_MODEL` | `extractive` / `gpt-4o-mini` | `extractive` (offline) or `openai` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | – / `https://api.openai.com/v1` | Credentials and endpoint |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Characters per chunk and overlap |
| `TOP_K` / `MIN_SCORE` | `5` / `0.1` | Chunks retrieved; minimum similarity to count as relevant |
| `MAX_UPLOAD_MB` | `20` | Upload size limit |
| `API_KEY` | – | Optional API key |

`MIN_SCORE` is the "I don't know" guard: with real embedding models, tune it for your data
(typical values are 0.2 to 0.4).

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite covers chunking, embeddings, parsing (including PDF and DOCX), the in-memory store,
the full ingest -> ask pipeline, and the HTTP API (upload limits, validation, auth, document
scoping). Tests use the in-memory store and offline providers, so they need no database or
network.

## Design decisions

- **One database for everything.** pgvector keeps documents, chunks and embeddings in
  PostgreSQL, so deletes cascade, filtering is plain SQL, and there is no extra service to run.
- **Retrieval guard before generation.** Chunks below `MIN_SCORE` are dropped and the LLM is
  never called with empty context, which prevents confident answers about nothing.
- **Citations by construction.** The prompt numbers each passage and the API returns the same
  numbering, so `[2]` in an answer maps directly to `sources[1]`.
- **Dependency injection.** `create_app(services)` and `build_services(...)` accept any store,
  embedder or LLM, which is how the tests run without external systems.

## Limitations and roadmap

- No OCR: scanned PDFs are rejected with a clear error.
- Ingestion is synchronous; large files would benefit from a background worker queue.
- Retrieval is dense-only; hybrid search (BM25 + vectors) and a re-ranker would improve recall.
- No streaming responses or multi-user document isolation yet.

## License

MIT
