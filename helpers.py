from app.core.embeddings import HashingEmbedder
from app.core.ingestion import IngestionService
from app.core.llm import ExtractiveLLM
from app.core.rag import RAGService
from app.storage.memory import InMemoryStore

HR_POLICY = (
    "Employees receive 25 days of paid vacation per year. Vacation requests must be "
    "submitted to the line manager at least two weeks in advance. Unused vacation days "
    "can be carried over until March 31 of the following year."
)
SECURITY_POLICY = (
    "All laptops must use full disk encryption. Passwords must be at least 14 characters "
    "long and rotated every 90 days. Security incidents must be reported to the security "
    "team within one hour."
)


def make_pipeline(min_score: float = 0.1):
    store = InMemoryStore()
    embedder = HashingEmbedder(dim=384)
    ingestion = IngestionService(store, embedder, chunk_size=400, chunk_overlap=60)
    rag = RAGService(store, embedder, ExtractiveLLM(), top_k=3, min_score=min_score)
    return store, ingestion, rag
