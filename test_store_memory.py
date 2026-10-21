from app.core.embeddings import HashingEmbedder
from app.core.models import ChunkData
from app.storage.memory import InMemoryStore


def add(store, embedder, name, texts):
    chunks = [ChunkData(i, t, page=i + 1) for i, t in enumerate(texts)]
    return store.add_document(name, "text/plain", 10, chunks, embedder.embed(texts))


def test_add_list_get_delete():
    store, emb = InMemoryStore(), HashingEmbedder()
    doc = add(store, emb, "a.txt", ["alpha beta", "gamma delta"])
    assert doc.num_chunks == 2
    assert store.get_document(doc.id) == doc
    assert [d.id for d in store.list_documents()] == [doc.id]
    assert store.delete_document(doc.id) is True
    assert store.delete_document(doc.id) is False
    assert store.get_document(doc.id) is None
    assert store.search(emb.embed(["alpha"])[0], 5) == []


def test_search_ranks_and_limits():
    store, emb = InMemoryStore(), HashingEmbedder()
    add(store, emb, "a.txt", ["apples and oranges are fruit", "cars and trucks are vehicles"])
    hits = store.search(emb.embed(["which fruit is mentioned"])[0], top_k=1)
    assert len(hits) == 1
    assert hits[0].filename == "a.txt" and hits[0].page == 1
    assert "fruit" in hits[0].text


def test_document_filter():
    store, emb = InMemoryStore(), HashingEmbedder()
    a = add(store, emb, "a.txt", ["shared topic about encryption"])
    b = add(store, emb, "b.txt", ["shared topic about encryption"])
    hits = store.search(emb.embed(["encryption"])[0], 5, document_ids=[b.id])
    assert {h.document_id for h in hits} == {b.id}
    assert a.id not in {h.document_id for h in hits}
    assert store.search(emb.embed(["encryption"])[0], 5, document_ids=[]) == []
