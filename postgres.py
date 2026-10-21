"""PostgreSQL + pgvector store.

Documents and chunks live in regular tables; chunk embeddings use the ``vector`` type with an
HNSW index (cosine distance), so retrieval is a single indexed SQL query.
"""
from __future__ import annotations

import uuid

import numpy as np

from app.core.models import ChunkData, DocumentInfo, SearchHit


def _schema_statements(dim: int) -> list[str]:
    dim = int(dim)
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        """
        CREATE TABLE IF NOT EXISTS documents (
            id           UUID PRIMARY KEY,
            filename     TEXT        NOT NULL,
            content_type TEXT        NOT NULL,
            size_bytes   BIGINT      NOT NULL,
            num_chunks   INTEGER     NOT NULL,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id          BIGSERIAL PRIMARY KEY,
            document_id UUID    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            page        INTEGER,
            content     TEXT    NOT NULL,
            embedding   vector({dim}) NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id)",
        "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
        "ON chunks USING hnsw (embedding vector_cosine_ops)",
    ]


def _to_info(row) -> DocumentInfo:
    return DocumentInfo(
        id=str(row[0]),
        filename=row[1],
        content_type=row[2],
        size_bytes=row[3],
        num_chunks=row[4],
        created_at=row[5],
    )


class PostgresStore:
    def __init__(self, dsn: str, dim: int) -> None:
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg_pool import ConnectionPool

        self.dim = int(dim)

        # The extension must exist before the pool registers the vector type on connections.
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        self._pool = ConnectionPool(
            dsn, min_size=1, max_size=10, configure=register_vector, open=True
        )
        with self._pool.connection() as conn:
            for statement in _schema_statements(self.dim):
                conn.execute(statement)

    def add_document(self, filename, content_type, size_bytes, chunks, embeddings) -> DocumentInfo:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        document_id = uuid.uuid4()
        with self._pool.connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    """
                    INSERT INTO documents (id, filename, content_type, size_bytes, num_chunks)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id, filename, content_type, size_bytes, num_chunks, created_at
                    """,
                    (document_id, filename, content_type, size_bytes, len(chunks)),
                ).fetchone()
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO chunks (document_id, chunk_index, page, content, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        [
                            (document_id, c.index, c.page, c.text, np.asarray(e, dtype=np.float32))
                            for c, e in zip(chunks, embeddings)
                        ],
                    )
        return _to_info(row)

    def list_documents(self) -> list[DocumentInfo]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, filename, content_type, size_bytes, num_chunks, created_at "
                "FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [_to_info(r) for r in rows]

    def get_document(self, document_id: str) -> DocumentInfo | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, filename, content_type, size_bytes, num_chunks, created_at "
                "FROM documents WHERE id = %s",
                (uuid.UUID(document_id),),
            ).fetchone()
        return _to_info(row) if row else None

    def delete_document(self, document_id: str) -> bool:
        with self._pool.connection() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = %s", (uuid.UUID(document_id),))
            return cur.rowcount > 0

    def search(self, embedding, top_k, document_ids=None) -> list[SearchHit]:
        query = np.asarray(embedding, dtype=np.float32)
        ids = [uuid.UUID(i) for i in document_ids] if document_ids is not None else None
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT c.document_id, d.filename, c.chunk_index, c.page, c.content,
                       1 - (c.embedding <=> %(q)s) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE (%(ids)s::uuid[] IS NULL OR c.document_id = ANY(%(ids)s::uuid[]))
                ORDER BY c.embedding <=> %(q)s
                LIMIT %(k)s
                """,
                {"q": query, "ids": ids, "k": top_k},
            ).fetchall()
        return [
            SearchHit(str(r[0]), r[1], r[2], r[3], r[4], float(r[5])) for r in rows
        ]

    def close(self) -> None:
        self._pool.close()
