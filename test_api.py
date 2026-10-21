"""End-to-end API tests using FastAPI's TestClient with an in-memory store."""
import uuid

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import build_services, create_app
from tests.helpers import HR_POLICY, SECURITY_POLICY


def make_client(**overrides):
    settings = Settings(store_backend="memory", min_score=0.1, **overrides)
    return TestClient(create_app(build_services(settings)))


def upload(client, name, text, headers=None):
    return client.post(
        "/documents",
        files={"file": (name, text.encode(), "text/plain")},
        headers=headers or {},
    )


def test_health():
    r = make_client().get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_upload_list_get_delete_flow():
    client = make_client()
    created = upload(client, "hr_policy.txt", HR_POLICY)
    assert created.status_code == 201
    doc = created.json()
    assert doc["filename"] == "hr_policy.txt" and doc["num_chunks"] >= 1

    assert [d["id"] for d in client.get("/documents").json()] == [doc["id"]]
    assert client.get(f"/documents/{doc['id']}").json()["id"] == doc["id"]
    assert client.delete(f"/documents/{doc['id']}").status_code == 204
    assert client.get(f"/documents/{doc['id']}").status_code == 404
    assert client.delete(f"/documents/{doc['id']}").status_code == 404


def test_ask_returns_answer_with_sources():
    client = make_client()
    upload(client, "hr_policy.txt", HR_POLICY)
    upload(client, "security_policy.txt", SECURITY_POLICY)

    r = client.post("/ask", json={"query": "How many vacation days do employees get?"})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True
    assert "25 days" in body["answer"]
    assert body["sources"][0]["filename"] == "hr_policy.txt"
    assert body["sources"][0]["citation"] == 1


def test_ask_unrelated_question_not_grounded():
    client = make_client()
    upload(client, "hr_policy.txt", HR_POLICY)
    body = client.post("/ask", json={"query": "What is the capital of France?"}).json()
    assert body["grounded"] is False and body["sources"] == []


def test_search_with_document_filter():
    client = make_client()
    hr = upload(client, "hr_policy.txt", HR_POLICY).json()
    upload(client, "security_policy.txt", SECURITY_POLICY)
    r = client.post("/search", json={"query": "policy", "document_ids": [hr["id"]], "top_k": 5})
    assert r.status_code == 200
    assert all(item["document_id"] == hr["id"] for item in r.json()["results"])


def test_validation_errors():
    client = make_client()
    assert client.post("/ask", json={"query": ""}).status_code == 422
    assert client.post("/ask", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.get("/documents/not-a-uuid").status_code == 422
    assert client.get(f"/documents/{uuid.uuid4()}").status_code == 404


def test_unsupported_and_empty_files():
    client = make_client()
    r = client.post("/documents", files={"file": ("a.exe", b"x", "application/octet-stream")})
    assert r.status_code == 415
    assert upload(client, "empty.txt", "   ").status_code == 422


def test_upload_size_limit():
    client = make_client(max_upload_mb=1)
    r = client.post(
        "/documents", files={"file": ("big.txt", b"a" * (1024 * 1024 + 1), "text/plain")}
    )
    assert r.status_code == 413


def test_api_key_protection():
    client = make_client(api_key="s3cret")
    assert client.get("/health").status_code == 200  # health stays public
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert upload(client, "a.txt", HR_POLICY, {"X-API-Key": "s3cret"}).status_code == 201
