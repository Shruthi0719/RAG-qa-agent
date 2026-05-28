"""
FastAPI application — RAG Document Q&A API v3
Each session gets its own isolated docs folder + FAISS index.
No user can see another user's documents.
"""
from __future__ import annotations

import uuid
import os
import gc
import json
import shutil
import hashlib
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.logger import get_logger
from app.session_manager import session_store
from app.rag_pipeline import (
    load_new_documents,
    load_documents,
    chunk_documents,
    build_index,
    update_index,
    load_index,
    load_chunks,
    build_qa_chain,
    answer_query,
    answer_query_stream,
)

logger = get_logger(__name__)

# ─── Per-user storage root ────────────────────────────────────────────────────
# Each user (identified by session_id) gets:
#   /tmp/rag_users/<session_id>/docs/        ← their uploaded files
#   /tmp/rag_users/<session_id>/faiss_index/ ← their personal FAISS index
#   /tmp/rag_users/<session_id>/docs.pkl
#   /tmp/rag_users/<session_id>/manifest.json

def _user_root(session_id: str) -> Path:
    p = Path("/tmp/rag_users") / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def _user_docs_dir(session_id: str) -> Path:
    p = _user_root(session_id) / "docs"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _user_index_path(session_id: str) -> Path:
    return _user_root(session_id) / "faiss_index"

def _user_pickle(session_id: str) -> Path:
    return _user_root(session_id) / "docs.pkl"

def _user_manifest(session_id: str) -> Path:
    return _user_root(session_id) / "manifest.json"


# ─── In-memory per-user vectorstore cache ────────────────────────────────────
_user_vectorstores: dict[str, object] = {}
_user_chunks: dict[str, list] = {}
# Track ingest status per user: "idle" | "indexing" | "done" | "error: ..."
_ingest_status: dict[str, str] = {}


def _get_user_vectorstore(session_id: str):
    if session_id not in _user_vectorstores:
        idx = _user_index_path(session_id)
        if idx.exists():
            try:
                vs = load_index(str(idx))
                _user_vectorstores[session_id] = vs
                _user_chunks[session_id] = load_chunks(str(_user_pickle(session_id)))
            except Exception as e:
                logger.warning(f"Could not load index for {session_id}: {e}")
                return None
        else:
            return None
    return _user_vectorstores.get(session_id)


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG Q&A API starting — per-user isolation enabled")
    yield
    logger.info("Shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Q&A Agent", version="3.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", include_in_schema=False)
async def serve_frontend():
    f = Path("frontend/index.html")
    return FileResponse(str(f)) if f.exists() else {"message": "RAG Q&A API"}


# ─── Schemas ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None

class SourceInfo(BaseModel):
    file: str
    path: str
    page: Optional[int] = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    session_id: str


# ─── System endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/stats")
async def stats(session_id: Optional[str] = None):
    if session_id:
        chunks = _user_chunks.get(session_id, [])
        manifest_path = _user_manifest(session_id)
        num_files = len(json.loads(manifest_path.read_text())) if manifest_path.exists() else 0
        index_ready = _user_index_path(session_id).exists()
        ingest_status = _ingest_status.get(session_id, "idle")
    else:
        chunks, num_files, index_ready, ingest_status = [], 0, False, "idle"
    return {
        "index_ready": index_ready,
        "num_chunks": len(chunks),
        "num_files": num_files,
        "active_sessions": len(_user_vectorstores),
        "model": settings.LLM_MODEL,
        "embedding_model": "cohere/embed-english-light-v3.0",
        "ingest_status": ingest_status,
    }

@app.get("/ingest/status")
async def ingest_status_endpoint(session_id: str):
    return {"status": _ingest_status.get(session_id, "idle")}


# ─── Upload ───────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
):
    # Create session if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    allowed = {".pdf", ".txt", ".docx", ".md", ".html", ".htm", ".csv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {ext}")

    # Check file size — reject over 25MB
    contents = await file.read()
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 25MB.")

    docs_dir = _user_docs_dir(session_id)
    dest = docs_dir / file.filename
    with open(dest, "wb") as f:
        f.write(contents)
    del contents  # free memory immediately
    gc.collect()

    # Kick off background ingest for this user only
    _ingest_status[session_id] = "indexing"
    if background_tasks:
        background_tasks.add_task(_background_ingest_user, session_id)

    return {
        "message": f"Uploaded '{file.filename}'. Indexing in background.",
        "session_id": session_id,
        "filename": file.filename,
        "indexing": True,
    }


# ─── Per-user background ingest ───────────────────────────────────────────────
async def _background_ingest_user(session_id: str):
    try:
        docs_dir = str(_user_docs_dir(session_id))
        index_path = str(_user_index_path(session_id))
        pickle_path = str(_user_pickle(session_id))
        manifest_path = str(_user_manifest(session_id))

        new_docs, updated_manifest = load_new_documents(
            docs_dir, manifest_path=manifest_path
        )

        if not new_docs:
            _ingest_status[session_id] = "done"
            return

        chunks = chunk_documents(new_docs)
        vs = update_index(
            chunks, updated_manifest,
            index_path=index_path,
            pickle_path=pickle_path,
            manifest_path=manifest_path,
        )
        _user_vectorstores[session_id] = vs
        _user_chunks[session_id] = load_chunks(pickle_path)

        # Rebuild chain for this session
        chain = build_qa_chain(vs, _user_chunks[session_id])
        session_store.set(session_id, chain)

        _ingest_status[session_id] = "done"
        logger.info(f"[{session_id[:8]}] Ingest done: {len(_user_chunks[session_id])} chunks")

    except Exception as e:
        _ingest_status[session_id] = f"error: {e}"
        logger.error(f"[{session_id[:8]}] Ingest failed: {e}")


# ─── Documents list ───────────────────────────────────────────────────────────
@app.get("/documents")
async def list_documents(session_id: Optional[str] = None):
    if not session_id:
        return {"files": []}
    docs_dir = _user_docs_dir(session_id)
    manifest_path = _user_manifest(session_id)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    allowed = {".pdf", ".txt", ".docx", ".md", ".html", ".htm", ".csv"}
    files = []
    for p in sorted(docs_dir.rglob("*.*")):
        if p.suffix.lower() not in allowed:
            continue
        stat = p.stat()
        files.append({
            "filename": p.name,
            "size_bytes": stat.st_size,
            "indexed": str(p) in manifest,
            "modified": stat.st_mtime,
        })
    return {"files": files}


# ─── Delete document ──────────────────────────────────────────────────────────
@app.delete("/document/{filename}")
async def delete_document(filename: str, session_id: Optional[str] = None):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    target = _user_docs_dir(session_id) / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    target.unlink()

    # Rebuild index without this file
    _ingest_status[session_id] = "indexing"
    remaining = list(_user_docs_dir(session_id).rglob("*.*"))
    if remaining:
        await _background_ingest_user(session_id)
    else:
        # No files left — clear everything
        idx = _user_index_path(session_id)
        if idx.exists():
            shutil.rmtree(idx)
        pk = _user_pickle(session_id)
        if pk.exists():
            pk.unlink()
        _user_vectorstores.pop(session_id, None)
        _user_chunks.pop(session_id, None)
        session_store.delete(session_id)
        _ingest_status[session_id] = "idle"

    return {"message": f"Deleted '{filename}'.", "session_id": session_id}


# ─── Manual ingest ────────────────────────────────────────────────────────────
@app.post("/ingest")
async def ingest_incremental(session_id: Optional[str] = None, background_tasks: BackgroundTasks = None):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    docs_dir = _user_docs_dir(session_id)
    if not any(docs_dir.rglob("*.*")):
        raise HTTPException(status_code=422, detail="No documents found. Upload files first.")
    _ingest_status[session_id] = "indexing"
    if background_tasks:
        background_tasks.add_task(_background_ingest_user, session_id)
    return {"message": "Indexing started.", "session_id": session_id}

@app.post("/ingest/full")
async def ingest_full(session_id: Optional[str] = None, background_tasks: BackgroundTasks = None):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    # Delete manifest to force full rebuild
    mp = _user_manifest(session_id)
    if mp.exists():
        mp.unlink()
    _ingest_status[session_id] = "indexing"
    if background_tasks:
        background_tasks.add_task(_background_ingest_user, session_id)
    return {"message": "Full rebuild started.", "session_id": session_id}


# ─── Chat ─────────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    vs = _get_user_vectorstore(session_id)
    if vs is None:
        raise HTTPException(status_code=503, detail="Index not ready. Upload and ingest files first.")
    chain = session_store.get(session_id)
    if chain is None:
        chain = build_qa_chain(vs, _user_chunks.get(session_id, []))
        session_store.set(session_id, chain)
    try:
        answer, sources = answer_query(chain, req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"answer": answer, "sources": sources, "session_id": session_id}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    vs = _get_user_vectorstore(session_id)
    if vs is None:
        raise HTTPException(status_code=503, detail="Index not ready. Upload and ingest files first.")
    chain = session_store.get(session_id)
    if chain is None:
        chain = build_qa_chain(vs, _user_chunks.get(session_id, []))
        session_store.set(session_id, chain)

    async def token_generator() -> AsyncGenerator[str, None]:
        yield f"data: {{\"session_id\": \"{session_id}\"}}\n\n"
        try:
            async for token in answer_query_stream(chain, req.question):
                safe = token.replace("\n", "\\n")
                yield f"data: {safe}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Sessions ─────────────────────────────────────────────────────────────────
@app.get("/sessions")
async def list_sessions():
    return {"active_sessions": len(_user_vectorstores)}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    _user_vectorstores.pop(session_id, None)
    _user_chunks.pop(session_id, None)
    _ingest_status.pop(session_id, None)
    session_store.delete(session_id)
    return {"message": f"Session {session_id} cleared."}
