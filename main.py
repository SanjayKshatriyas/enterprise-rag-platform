"""FastAPI application: document upload, semantic search and cited question answering."""
from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile

from app import __version__
from app.config import Settings
from app.core.embeddings import Embedder, HashingEmbedder, OpenAICompatibleEmbedder
from app.core.errors import (
    DocumentParsingError,
    EmptyDocumentError,
    ProviderError,
    UnsupportedFileTypeError,
)
from app.core.ingestion import IngestionService
from app.core.llm import LLM, ExtractiveLLM, OpenAICompatibleLLM
from app.core.models import DocumentInfo, SearchHit
from app.core.rag import RAGService
from app.schemas import (
    AskResponse,
    DocumentOut,
    HealthResponse,
    QueryRequest,
    SearchResponse,
    SourceOut,
)
from app.storage.base import VectorStore

logger = logging.getLogger("rag")


@dataclass
class Services:
    settings: Settings
    store: VectorStore
    ingestion: IngestionService
    rag: RAGService


def build_services(
    settings: Settings,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
) -> Services:
    """Wire up the dependency graph. Any component can be injected (used by the tests)."""
    if embedder is None:
        if settings.embedding_provider == "openai":
            embedder = OpenAICompatibleEmbedder(
                api_key=settings.openai_api_key,
                model=settings.embedding_model,
                dim=settings.embedding_dim,
                base_url=settings.openai_base_url,
            )
        else:
            embedder = HashingEmbedder(dim=settings.embedding_dim)

    if llm is None:
        if settings.llm_provider == "openai":
            llm = OpenAICompatibleLLM(
                api_key=settings.openai_api_key,
                model=settings.llm_model,
                base_url=settings.openai_base_url,
            )
        else:
            llm = ExtractiveLLM()

    if store is None:
        if settings.store_backend == "memory":
            from app.storage.memory import InMemoryStore

            store = InMemoryStore()
        else:
            from app.storage.postgres import PostgresStore

            store = PostgresStore(settings.database_url, dim=embedder.dim)

    return Services(
        settings=settings,
        store=store,
        ingestion=IngestionService(
            store, embedder, settings.chunk_size, settings.chunk_overlap
        ),
        rag=RAGService(store, embedder, llm, settings.top_k, settings.min_score),
    )


def _to_document(info: DocumentInfo) -> DocumentOut:
    return DocumentOut(**info.__dict__)


def _to_sources(hits: list[SearchHit]) -> list[SourceOut]:
    return [
        SourceOut(
            citation=number,
            document_id=hit.document_id,
            filename=hit.filename,
            chunk_index=hit.chunk_index,
            page=hit.page,
            score=round(hit.score, 4),
            text=hit.text,
        )
        for number, hit in enumerate(hits, start=1)
    ]


def create_app(services: Services | None = None) -> FastAPI:
    """Application factory. Pass ``services`` to inject components (tests); otherwise they are
    built from environment settings on startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = False
        if getattr(app.state, "services", None) is None:
            app.state.services = build_services(Settings())
            owned = True
        try:
            yield
        finally:
            if owned:
                app.state.services.store.close()

    app = FastAPI(
        title="Enterprise Document Intelligence & RAG Platform",
        version=__version__,
        description=(
            "Upload documents, search them semantically, and ask questions answered by an "
            "LLM with source citations (retrieval-augmented generation)."
        ),
        lifespan=lifespan,
    )
    app.state.services = services

    def get_services(request: Request) -> Services:
        return request.app.state.services

    def require_api_key(
        request: Request, x_api_key: str | None = Header(default=None)
    ) -> None:
        expected = request.app.state.services.settings.api_key
        if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
            raise HTTPException(401, "Invalid or missing X-API-Key.")

    protected = [Depends(require_api_key)]

    @app.exception_handler(UnsupportedFileTypeError)
    async def _unsupported(_: Request, exc: UnsupportedFileTypeError):
        return _error(415, str(exc))

    @app.exception_handler(EmptyDocumentError)
    async def _empty(_: Request, exc: EmptyDocumentError):
        return _error(422, str(exc))

    @app.exception_handler(DocumentParsingError)
    async def _parsing(_: Request, exc: DocumentParsingError):
        return _error(422, str(exc))

    @app.exception_handler(ProviderError)
    async def _provider(_: Request, exc: ProviderError):
        logger.error("Upstream provider error: %s", exc)
        return _error(502, str(exc))

    def _error(code: int, message: str):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=code, content={"detail": message})

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(svc: Services = Depends(get_services)):
        return HealthResponse(
            status="ok", version=__version__, documents=len(svc.store.list_documents())
        )

    @app.post(
        "/documents",
        response_model=DocumentOut,
        status_code=201,
        dependencies=protected,
        tags=["documents"],
        summary="Upload and index a document (PDF, DOCX, TXT, MD)",
    )
    def upload_document(file: UploadFile = File(...), svc: Services = Depends(get_services)):
        limit = svc.settings.max_upload_mb * 1024 * 1024
        data = file.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(
                413,
                f"File exceeds the {svc.settings.max_upload_mb} MB upload limit.",
            )
        info = svc.ingestion.ingest(file.filename or "document", file.content_type or "", data)
        return _to_document(info)

    @app.get("/documents", response_model=list[DocumentOut], dependencies=protected, tags=["documents"])
    def list_documents(svc: Services = Depends(get_services)):
        return [_to_document(d) for d in svc.store.list_documents()]

    @app.get("/documents/{document_id}", response_model=DocumentOut, dependencies=protected, tags=["documents"])
    def get_document(document_id: UUID, svc: Services = Depends(get_services)):
        info = svc.store.get_document(str(document_id))
        if info is None:
            raise HTTPException(404, "Document not found.")
        return _to_document(info)

    @app.delete(
        "/documents/{document_id}",
        status_code=204,
        dependencies=protected,
        tags=["documents"],
    )
    def delete_document(document_id: UUID, svc: Services = Depends(get_services)):
        if not svc.store.delete_document(str(document_id)):
            raise HTTPException(404, "Document not found.")

    @app.post("/search", response_model=SearchResponse, dependencies=protected, tags=["query"])
    def search(body: QueryRequest, svc: Services = Depends(get_services)):
        ids = [str(i) for i in body.document_ids] if body.document_ids is not None else None
        hits = svc.rag.search(body.query, body.top_k, ids)
        return SearchResponse(query=body.query, results=_to_sources(hits))

    @app.post("/ask", response_model=AskResponse, dependencies=protected, tags=["query"],
              summary="Ask a question; the answer cites the retrieved sources")
    def ask(body: QueryRequest, svc: Services = Depends(get_services)):
        ids = [str(i) for i in body.document_ids] if body.document_ids is not None else None
        result = svc.rag.ask(body.query, body.top_k, ids)
        return AskResponse(
            question=body.query,
            answer=result.answer,
            grounded=result.grounded,
            sources=_to_sources(result.sources),
        )

    return app


app = create_app()
