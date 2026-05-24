"""
FastAPI application — RAG Document Q&A API (v2)

New in v2:
  - Multi-session support: each session_id gets independent memory
  - Streaming /chat/stream endpoint (Server-Sent Events)
  - Incremental /ingest (only re-embeds changed files)
  - /ingest/full forces a full rebuild
  - /sessions management endpoints
  - Rich source metadata (filename + page number) in responses
  - /stats endpoint for index info
  - Proper CORS and error handling
"""

import uuid
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.logger import get_logger
from app.session_manager import session_store
from app.rag_pipeline import (
    load_documents,
    load_new_documents,
    chunk_documents,
    build_index,
    update_index,
    load_index,
    load_chunks,
    build_qa_chain,
    answer_query,
    answer_query_stream,
    INDEX_PATH,
    DOCS_PICKLE,
    MANIFEST_PATH,
)

logger = get_logger(__name__)

# ─── Shared vectorstore (rebuilt on ingest, read-only during chat) ────────────
_vectorstore = None
_chunks_cache: list = []


def get_vectorstore():
    return _vectorstore


def _build_chain_for_session(session_id: str):
    """Build a fresh chain for a session using the current vectorstore."""
    vs = get_vectorstore()
    if vs is None:
        return None
    chain = build_qa_chain(vs, _chunks_cache)
    session_store.set(session_id, chain)
    return chain


def get_or_create_chain(session_id: str):
    chain = session_store.get(session_id)
    if chain is None and get_vectorstore() is not None:
        chain = _build_chain_for_session(session_id)
    return chain


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vectorstore, _chunks_cache
    logger.info("Starting up RAG Q&A API v2...")
    if INDEX_PATH.exists():
        try:
            _vectorstore = load_index()
            _chunks_cache = load_chunks()
            logger.info("Index loaded on startup.")
        except Exception as e:
            logger.warning(f"Could not load index on startup: {e}")
    else:
        logger.warning("No index on disk — call /ingest first.")
    yield
    logger.info("Shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Document Q&A API",
    description=(
        "Upload documents, build a FAISS vector index, "
        "and query them with multi-turn conversational memory. "
        "Supports multi-session, streaming, incremental indexing, and more."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend UI at /
@app.get("/", include_in_schema=False)
async def serve_frontend():
    frontend = Path("frontend/index.html")
    if frontend.exists():
        return FileResponse(str(frontend))
    return {"message": "RAG Q&A API v2 — see /docs for API reference"}


# ─── Schemas ──────────────────────────────────────────────────────────────────
class IngestResponse(BaseModel):
    message: str
    num_chunks: int
    num_docs: int
    mode: str  # "incremental" or "full"


class SourceInfo(BaseModel):
    file: str
    path: str
    page: Optional[int] = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, example="What is this document about?")
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation continuity. Auto-generated if omitted.",
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    session_id: str


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    active_sessions: int
    num_chunks: int


class StatsResponse(BaseModel):
    index_ready: bool
    num_chunks: int
    num_files: int
    active_sessions: int
    model: str
    embedding_model: str


# ─── System routes ────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return {
        "status": "ok",
        "index_ready": INDEX_PATH.exists(),
        "active_sessions": session_store.active_count,
        "num_chunks": len(_chunks_cache),
    }


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def stats():
    num_files = 0
    if MANIFEST_PATH.exists():
        import json
        manifest = json.loads(MANIFEST_PATH.read_text())
        num_files = len(manifest)

    return {
        "index_ready": INDEX_PATH.exists(),
        "num_chunks": len(_chunks_cache),
        "num_files": num_files,
        "active_sessions": session_store.active_count,
        "model": settings.LLM_MODEL,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }


# ─── Ingestion routes ─────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, tags=["Indexing"])
async def ingest_incremental():
    """
    Incremental ingest: only re-embeds new or changed files.
    Much faster than a full rebuild for large document sets.
    Call /ingest/full to force a complete rebuild.
    """
    global _vectorstore, _chunks_cache

    docs_dir = "data/docs"
    if not Path(docs_dir).exists() or not any(Path(docs_dir).rglob("*.*")):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No documents found in {docs_dir}. Upload files via /upload first.",
        )

    new_docs, updated_manifest = load_new_documents(docs_dir)

    if not new_docs and INDEX_PATH.exists():
        return {
            "message": "No changes detected. Index is already up to date.",
            "num_docs": 0,
            "num_chunks": len(_chunks_cache),
            "mode": "incremental (no changes)",
        }

    if not new_docs:
        raise HTTPException(status_code=422, detail="No readable content found.")

    chunks = chunk_documents(new_docs)
    _vectorstore = update_index(chunks, updated_manifest)
    _chunks_cache = load_chunks()

    # Invalidate all sessions so they pick up the new index
    session_store.clear_all()

    return {
        "message": "Index updated successfully.",
        "num_docs": len(new_docs),
        "num_chunks": len(_chunks_cache),
        "mode": "incremental",
    }


@app.post("/ingest/full", response_model=IngestResponse, tags=["Indexing"])
async def ingest_full():
    """
    Full rebuild: re-embeds ALL documents from scratch.
    Use this after deleting files or to fix a corrupted index.
    """
    global _vectorstore, _chunks_cache

    docs_dir = "data/docs"
    if not Path(docs_dir).exists() or not any(Path(docs_dir).rglob("*.*")):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No documents found in {docs_dir}.",
        )

    docs = load_documents(docs_dir)
    if not docs:
        raise HTTPException(status_code=422, detail="No readable content found.")

    chunks = chunk_documents(docs)
    _vectorstore = build_index(chunks)
    _chunks_cache = chunks

    # Save manifest so incremental ingest works from here
    import json, hashlib
    manifest = {}
    for p in Path(docs_dir).rglob("*.*"):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        manifest[str(p)] = h.hexdigest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    session_store.clear_all()

    return {
        "message": "Full index rebuild complete.",
        "num_docs": len(docs),
        "num_chunks": len(chunks),
        "mode": "full",
    }


@app.post("/upload", tags=["Indexing"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, DOCX, MD, HTML, CSV) to data/docs/.
    After uploading, call /ingest to update the index.
    """
    allowed = {".pdf", ".txt", ".docx", ".md", ".html", ".htm", ".csv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(allowed)}",
        )

    dest = Path("data/docs") / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"message": f"Uploaded '{file.filename}'. Call /ingest to update index."}


@app.get("/documents", tags=["Indexing"])
async def list_documents():
    """
    List all documents currently in data/docs/ with metadata.
    """
    import json
    docs_dir = Path("data/docs")
    if not docs_dir.exists():
        return {"files": []}

    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())

    files = []
    allowed = {".pdf", ".txt", ".docx", ".md", ".html", ".htm", ".csv"}
    for p in sorted(docs_dir.rglob("*.*")):
        if p.suffix.lower() not in allowed:
            continue
        stat = p.stat()
        indexed = str(p) in manifest
        files.append({
            "filename": p.name,
            "size_bytes": stat.st_size,
            "indexed": indexed,
            "modified": stat.st_mtime,
        })
    return {"files": files}


@app.delete("/document/{filename}", tags=["Indexing"])
async def delete_document(filename: str):
    """
    Delete a document from data/docs/ and trigger a full index rebuild.
    """
    global _vectorstore, _chunks_cache

    target = Path("data/docs") / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    target.unlink()

    # Rebuild index without the deleted file
    docs_dir = "data/docs"
    remaining = list(Path(docs_dir).rglob("*.*"))
    if remaining:
        docs = load_documents(docs_dir)
        chunks = chunk_documents(docs)
        _vectorstore = build_index(chunks)
        _chunks_cache = chunks
    else:
        # No docs left — wipe index
        import shutil as _shutil
        if INDEX_PATH.exists():
            _shutil.rmtree(INDEX_PATH)
        if DOCS_PICKLE.exists():
            DOCS_PICKLE.unlink()
        _vectorstore = None
        _chunks_cache = []

    session_store.clear_all()
    return {"message": f"Deleted '{filename}' and rebuilt index."}


# ─── Chat routes ──────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["QA"])
async def chat(req: ChatRequest):
    """
    Multi-turn Q&A. Pass session_id to continue a conversation;
    omit it to start a new session (one will be auto-generated).
    """
    session_id = req.session_id or str(uuid.uuid4())
    chain = get_or_create_chain(session_id)

    if chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Index not ready. Call /ingest first.",
        )

    try:
        answer, sources = answer_query(chain, req.question)
    except Exception as e:
        logger.error(f"Chain error (session={session_id}): {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": answer, "sources": sources, "session_id": session_id}


@app.post("/chat/stream", tags=["QA"])
async def chat_stream(req: ChatRequest):
    """
    Streaming Q&A via Server-Sent Events.
    Tokens are streamed as they are generated — much lower perceived latency.

    Usage (JavaScript):
      const es = new EventSource('/chat/stream');
      // POST is not natively supported by EventSource; use fetch with ReadableStream.
    """
    session_id = req.session_id or str(uuid.uuid4())
    chain = get_or_create_chain(session_id)

    if chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Index not ready. Call /ingest first.",
        )

    async def token_generator() -> AsyncGenerator[str, None]:
        yield f"data: {{\"session_id\": \"{session_id}\"}}\n\n"
        try:
            async for token in answer_query_stream(chain, req.question):
                # Escape newlines for SSE format
                safe = token.replace("\n", "\\n")
                yield f"data: {safe}\n\n"
        except Exception as e:
            logger.error(f"Streaming error (session={session_id}): {e}")
            yield f"data: [ERROR] {e}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# ─── Session routes ───────────────────────────────────────────────────────────

@app.get("/sessions", tags=["Sessions"])
async def list_sessions():
    """List all active sessions with idle time and TTL info."""
    return {
        "active_sessions": session_store.active_count,
        "sessions": session_store.list_sessions(),
    }


@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str):
    """Delete a specific session (clears its conversation history)."""
    deleted = session_store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return {"message": f"Session '{session_id}' deleted."}


@app.delete("/sessions", tags=["Sessions"])
async def clear_all_sessions():
    """Clear ALL active sessions (reset all conversation histories)."""
    count = session_store.active_count
    session_store.clear_all()
    return {"message": f"Cleared {count} sessions."}


# ─── Legacy reset ─────────────────────────────────────────────────────────────

@app.delete("/reset", tags=["System"], deprecated=True)
async def reset():
    """Deprecated: use DELETE /sessions instead."""
    session_store.clear_all()
    return {"message": "All sessions reset."}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
