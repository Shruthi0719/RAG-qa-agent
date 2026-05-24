# RAG Q&A Agent

A production-ready Retrieval-Augmented Generation (RAG) document Q&A system built with FastAPI, LangChain, FAISS, and Groq. Upload your PDFs, Word docs, Markdown files, and more — then chat with them.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![LangChain](https://img.shields.io/badge/LangChain-0.2-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Features

- **Multi-format document support** — PDF, TXT, DOCX, Markdown, HTML, CSV
- **Hybrid retrieval** — FAISS (dense) + BM25 (sparse) ensemble for better recall
- **Streaming answers** — tokens stream in real time via Server-Sent Events
- **Source citations** — every answer links back to the exact filename and page number
- **Multi-turn memory** — sliding window conversation history per session
- **Incremental indexing** — only re-embeds changed files (SHA-256 manifest)
- **Document management UI** — see, upload, and delete files directly from the sidebar
- **Query routing** — detects off-topic questions instead of hallucinating
- **Railway deploy ready** — one-command cloud deploy

---

## What's new in v3

### 1. Document Management UI
- Live file list in the sidebar with size and index status (indexed / not indexed)
- One-click delete — removes the file and automatically rebuilds the index
- Pending upload queue shows files staged for upload separately from indexed docs

### 2. Query Routing (stops hallucinations)
- Before answering, checks whether the question is covered by your documents
- Off-topic questions get a clear warning instead of a hallucinated answer
- Uses a lightweight LLM call against retrieved context — fast and accurate

### 3. Streaming sources fix
- Tokens stream immediately as generated
- Source citations appear below the answer after streaming completes
- No more missing sources on streamed responses

### 4. Railway deploy support
- `railway.toml` and `nixpacks.toml` included
- `$PORT` env var support for Railway/Render compatibility

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/rag-qa-agent.git
cd rag-qa-agent

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env — set your GROQ_API_KEY

# 5. Run
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000, upload documents, click **Ingest**, then start chatting.

---

## Docker

```bash
cp .env.example .env
# Add GROQ_API_KEY to .env

docker-compose up --build
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | required | Get from [console.groq.com](https://console.groq.com) |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `LLM_TEMPERATURE` | `0.0` | Response temperature |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `RETRIEVER_TOP_K` | `4` | Chunks retrieved per query |
| `MEMORY_WINDOW_K` | `5` | Conversation turns to remember |
| `SESSION_TTL_SECONDS` | `3600` | Session expiry |
| `MAX_SESSIONS` | `100` | Max concurrent sessions |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Frontend UI |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Index statistics |
| `POST` | `/upload` | Upload a document |
| `GET` | `/documents` | List all documents with metadata |
| `POST` | `/ingest` | Incremental index update |
| `POST` | `/ingest/full` | Full index rebuild |
| `DELETE` | `/document/{filename}` | Delete a file and rebuild index |
| `POST` | `/chat` | Q&A (non-streaming) |
| `POST` | `/chat/stream` | Q&A (streaming SSE) |
| `GET` | `/sessions` | List active sessions |
| `DELETE` | `/sessions/{id}` | Delete a session |

---

## Architecture

```
Browser ──► FastAPI ──► Query Router ──► off-topic? → reject early
                │               │
                │        Hybrid Retriever
                │      (FAISS + BM25 ensemble)
                │               │
                └──────────► Groq LLM ──► Streamed answer + Sources
```

---

## Deploy to Railway

```bash
npm install -g @railway/cli
railway login
railway init
railway variables set GROQ_API_KEY=gsk_your_key_here
railway variables set LLM_MODEL=llama-3.1-8b-instant
railway up
railway open
```

Add a Volume mounted at `/app/data` to persist your index across deploys.
See [DEPLOY.md](./DEPLOY.md) for full instructions.

---

## Supported File Types

| Format | Extension |
|--------|-----------|
| PDF | `.pdf` |
| Word | `.docx` |
| Plain text | `.txt` |
| Markdown | `.md` |
| HTML | `.html`, `.htm` |
| CSV | `.csv` |

---

## Tech Stack

- **Backend** — FastAPI, LangChain, LangChain-Groq
- **LLM** — Groq (llama-3.1-8b-instant or any Groq model)
- **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (local, no API cost)
- **Vector store** — FAISS
- **Sparse retrieval** — BM25 (rank-bm25)
- **Frontend** — Vanilla JS, single HTML file, no build step