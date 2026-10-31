"""HTTP surface, with no Qdrant and no Ollama: the document and config endpoints."""

import os

os.environ.setdefault("QDRANT_URL", ":memory:")
os.environ.setdefault("LOG_QUERIES", "false")
os.environ.setdefault("OLLAMA_URL", "http://127.0.0.1:1")  # refused fast, no 5s wait

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_livez(client):
    assert client.get("/livez").json() == {"ok": True}


def test_health_reports_what_is_missing_instead_of_crashing(client):
    resp = client.get("/health")
    assert resp.status_code == 503
    assert "unreachable" in resp.json()["ollama"]


def test_config_has_no_secrets(client):
    body = client.get("/config").json()
    assert "rerank" in body and "top_k_final" in body
    assert not any("key" in k.lower() for k in body)


def test_toc_lists_chapters_and_appendices(client):
    docs = client.get("/toc").json()
    if not docs:
        pytest.skip("no documents in data/processed")
    doc = docs[0]
    assert doc["chapters"] and doc["appendices"]
    assert any(a["article"] == "Điều 1" for c in doc["chapters"] for a in c["articles"])


def test_article_returns_full_text(client):
    docs = client.get("/toc").json()
    if not docs:
        pytest.skip("no documents in data/processed")
    doc_id = docs[0]["doc_id"]
    body = client.get("/article", params={"doc_id": doc_id, "article": "Điều 5"}).json()
    assert body["article"] == "Điều 5" and body["sections"]
    assert body["page"] <= body["page_end"]


def test_unknown_article_is_404(client):
    docs = client.get("/toc").json()
    if not docs:
        pytest.skip("no documents in data/processed")
    resp = client.get("/article", params={"doc_id": docs[0]["doc_id"], "article": "Điều 999"})
    assert resp.status_code == 404


def test_source_pdf_is_not_a_path_traversal(client):
    assert client.get("/source/..%2F..%2Fetc%2Fpasswd").status_code in (404, 400)


def test_feedback_is_accepted(client):
    resp = client.post("/feedback", json={"vote": "up", "question": "x", "answer": "y"})
    assert resp.status_code == 202


def test_feedback_rejects_a_bad_vote(client):
    assert client.post("/feedback", json={"vote": "maybe", "question": "x"}).status_code == 422


def test_pages_are_served(client):
    for path in ("/", "/browse", "/debug"):
        assert client.get(path).status_code == 200
