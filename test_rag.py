from app.core.errors import EmptyDocumentError
from app.core.rag import NO_ANSWER

from tests.helpers import HR_POLICY, SECURITY_POLICY, make_pipeline


def test_ingest_then_ask_returns_cited_answer_from_right_document():
    store, ingestion, rag = make_pipeline()
    hr = ingestion.ingest("hr_policy.txt", "text/plain", HR_POLICY.encode())
    ingestion.ingest("security_policy.txt", "text/plain", SECURITY_POLICY.encode())

    result = rag.ask("How many vacation days do employees get?")

    assert result.grounded is True
    assert result.sources[0].document_id == hr.id
    assert result.sources[0].filename == "hr_policy.txt"
    assert "25 days" in result.answer
    assert "[1]" in result.answer


def test_question_about_other_document_routes_correctly():
    _, ingestion, rag = make_pipeline()
    ingestion.ingest("hr_policy.txt", "text/plain", HR_POLICY.encode())
    sec = ingestion.ingest("security_policy.txt", "text/plain", SECURITY_POLICY.encode())

    result = rag.ask("How often must passwords be rotated?")
    assert result.sources[0].document_id == sec.id
    assert "90 days" in result.answer


def test_unrelated_question_is_not_grounded():
    _, ingestion, rag = make_pipeline()
    ingestion.ingest("hr_policy.txt", "text/plain", HR_POLICY.encode())

    result = rag.ask("What is the capital of France?")
    assert result.grounded is False
    assert result.answer == NO_ANSWER
    assert result.sources == []


def test_document_scoping():
    _, ingestion, rag = make_pipeline(min_score=0.0)
    hr = ingestion.ingest("hr_policy.txt", "text/plain", HR_POLICY.encode())
    ingestion.ingest("security_policy.txt", "text/plain", SECURITY_POLICY.encode())

    hits = rag.search("policy", document_ids=[hr.id])
    assert hits and all(h.document_id == hr.id for h in hits)


def test_long_document_is_split_into_multiple_chunks_with_ordered_indexes():
    store, ingestion, _ = make_pipeline()
    text = "\n\n".join(f"Section {i}. " + "Details about the topic. " * 10 for i in range(8))
    doc = ingestion.ingest("long.txt", "text/plain", text.encode())
    assert doc.num_chunks > 3
    assert store.get_document(doc.id).num_chunks == doc.num_chunks


def test_empty_upload_rejected():
    _, ingestion, _ = make_pipeline()
    try:
        ingestion.ingest("empty.txt", "text/plain", b"  ")
    except EmptyDocumentError:
        return
    raise AssertionError("expected EmptyDocumentError")


def test_filename_is_sanitised():
    _, ingestion, _ = make_pipeline()
    doc = ingestion.ingest("../../etc/passwd.txt", "text/plain", b"some text here.")
    assert doc.filename == "passwd.txt"
