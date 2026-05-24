# RAG Q&A Agent v3

A production-ready Retrieval-Augmented Generation (RAG) document Q&A system.
Upload your PDFs, Word docs, Markdown files, and more — then chat with them.

## What's new in v3

### 1. Document Management UI
- **File list with live status** — see every file in `data/docs/` with its size and index status (indexed / not indexed)
- **One-click delete** — remove any document from the sidebar; the index is automatically rebuilt
- **Pending upload queue** — files waiting to be uploaded show separately from already-indexed docs

### 2. Query Routing (stop hallucinations)
- Before answering, the system checks whether the question is actually covered by your documents
- Off-topic questions get a clear `⚠️ This question doesn't appear to be covered by your uploaded documents.` response instead of a hallucinated answer
- Routing uses a lightweight LLM call against retrieved context — fast and accurate

### 3. Sources after streaming (fixed)
- Streaming tokens now appear immediately as they're generated
- Source citations appear below the answer *after* streaming completes (no more missing sources)
- A separate background call fetches source metadata without blocking the stream

### 4. Railway deploy
- `railway.toml` and updated `Dockerfile` included for one-command deploy
- `$PORT` env var support for Railway/Render compatibility
- See [DEPLOY.md](./DEPLOY.md) for full instructions

## Quick Start (local)

```bash
# 1. Copy env and fill in your Groq API key
cp .env.example .env
# Edit .env: GROQ_API_KEY=gsk_...

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn app.main:app --reload --port 8000

# 4. Open http://localhost:8000 in your browser
```

## Docker

```bash
docker-compose up --build
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Frontend UI |
| GET | `/health` | Health check |
| GET | `/stats` | Index statistics |
| POST | `/upload` | Upload a document |
| **GET** | **`/documents`** | **List all documents with metadata** *(new)* |
| POST | `/ingest` | Incremental index update |
| POST | `/ingest/full` | Full index rebuild |
| DELETE | `/document/{filename}` | Delete a file + rebuild index |
| POST | `/chat` | Q&A (non-streaming) |
| POST | `/chat/stream` | Q&A (streaming SSE) |
| GET | `/sessions` | List active sessions |
| DELETE | `/sessions/{id}` | Delete a session |

## Architecture

```
Browser  ──►  FastAPI  ──►  Query Router  ──►  (off-topic? → reject)
                  │                │
                  │           Hybrid Retriever
                  │         (FAISS + BM25 ensemble)
                  │                │
                  └──────────►  Groq LLM  ──►  Answer + Sources
```

## Supported File Types

PDF · TXT · DOCX · Markdown · HTML · CSV

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | required | Your Groq API key |
| `LLM_MODEL` | `llama3-8b-8192` | Groq model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `RETRIEVER_TOP_K` | `4` | Chunks retrieved per query |
| `MEMORY_WINDOW_K` | `5` | Conversation turns to remember |

## Deploy to Railway

See [DEPLOY.md](./DEPLOY.md) for step-by-step instructions.

```bash
# One-command deploy
railway init && railway variables set GROQ_API_KEY=gsk_... && railway up
```
