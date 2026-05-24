"""
Tests for the RAG Q&A API v2.
Covers: health, stats, upload, ingest, chat, streaming, session management.
Uses mock chains — no real LLM or FAISS index needed.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_mock_chain(answer="The document talks about AI.", source="data/docs/test.pdf"):
    mock = MagicMock()
    mock.invoke.return_value = {
        "answer": answer,
        "source_documents": [
            MagicMock(metadata={"source": source, "page": 1})
        ],
    }
    return mock


# ─── System endpoints ─────────────────────────────────────────────────────────

def test_health_no_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["index_ready"] is False
    assert "active_sessions" in data
    assert "num_chunks" in data


def test_stats(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert "index_ready" in data
    assert "num_chunks" in data
    assert "model" in data
    assert "embedding_model" in data


# ─── Upload ───────────────────────────────────────────────────────────────────

def test_upload_invalid_type(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = client.post(
        "/upload",
        files={"file": ("malware.exe", b"bad", "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "Unsupported" in r.json()["detail"]


@pytest.mark.parametrize("filename,ctype", [
    ("sample.txt",  "text/plain"),
    ("report.pdf",  "application/pdf"),
    ("notes.md",    "text/markdown"),
    ("data.csv",    "text/csv"),
])
def test_upload_valid_types(tmp_path, monkeypatch, filename, ctype):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "docs").mkdir(parents=True, exist_ok=True)
    r = client.post(
        "/upload",
        files={"file": (filename, b"Hello content.", ctype)},
    )
    assert r.status_code == 200
    assert "Uploaded" in r.json()["message"]


# ─── Chat ─────────────────────────────────────────────────────────────────────

def test_chat_no_index():
    with patch("app.main.get_vectorstore", return_value=None):
        r = client.post("/chat", json={"question": "Hello?"})
    assert r.status_code == 503


def test_chat_auto_session():
    mock_chain = make_mock_chain()
    with patch("app.main.get_or_create_chain", return_value=mock_chain):
        r = client.post("/chat", json={"question": "What is this about?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "sources" in body
    assert "session_id" in body
    assert body["session_id"]  # non-empty


def test_chat_with_session_id():
    mock_chain = make_mock_chain()
    with patch("app.main.get_or_create_chain", return_value=mock_chain):
        r = client.post("/chat", json={"question": "Explain section 2.", "session_id": "test-session-abc"})
    assert r.status_code == 200
    assert r.json()["session_id"] == "test-session-abc"


def test_chat_source_format():
    mock_chain = make_mock_chain(source="data/docs/report.pdf")
    with patch("app.main.get_or_create_chain", return_value=mock_chain):
        r = client.post("/chat", json={"question": "Summary?"})
    body = r.json()
    assert isinstance(body["sources"], list)
    if body["sources"]:
        src = body["sources"][0]
        assert "file" in src
        assert "path" in src
        assert src["file"] == "report.pdf"


def test_chat_question_too_long():
    r = client.post("/chat", json={"question": "x" * 2001})
    assert r.status_code == 422


def test_chat_empty_question():
    r = client.post("/chat", json={"question": ""})
    assert r.status_code == 422


# ─── Sessions ─────────────────────────────────────────────────────────────────

def test_list_sessions():
    r = client.get("/sessions")
    assert r.status_code == 200
    body = r.json()
    assert "active_sessions" in body
    assert "sessions" in body


def test_delete_nonexistent_session():
    r = client.delete("/sessions/nonexistent-session-xyz")
    assert r.status_code == 404


def test_clear_all_sessions():
    r = client.delete("/sessions")
    assert r.status_code == 200
    assert "Cleared" in r.json()["message"]


def test_reset_deprecated():
    """Legacy /reset should still work."""
    r = client.delete("/reset")
    assert r.status_code == 200


# ─── Streaming ────────────────────────────────────────────────────────────────

def test_stream_no_index():
    with patch("app.main.get_vectorstore", return_value=None):
        r = client.post("/chat/stream", json={"question": "Hello?"})
    assert r.status_code == 503


# ─── Ingest ───────────────────────────────────────────────────────────────────

def test_ingest_no_docs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = client.post("/ingest")
    assert r.status_code == 422


def test_ingest_full_no_docs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = client.post("/ingest/full")
    assert r.status_code == 422
