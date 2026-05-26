"""
RAG Pipeline — Core retrieval-augmented generation logic.
Uses Cohere's free Embed API for embeddings (no local model = no OOM on Render).
"""

import gc
import hashlib
import json
import pickle
import os as _os
from pathlib import Path
from typing import AsyncGenerator, List, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    CSVLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

# ─── Paths (writable fallback chain for Render) ───────────────────────────────

def _writable(rel: str) -> Path:
    for base in [".", "/app", "/tmp"]:
        p = Path(base) / rel
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            test = p.parent / ".write_test"
            test.touch(); test.unlink()
            return p
        except OSError:
            continue
    return Path("/tmp") / rel

INDEX_PATH    = _writable(_os.environ.get("FAISS_INDEX_PATH",  "data/faiss_index"))
DOCS_PICKLE   = _writable(_os.environ.get("DOCS_PICKLE_PATH",  "data/docs.pkl"))
MANIFEST_PATH = _writable(_os.environ.get("MANIFEST_PATH",     "data/file_manifest.json"))

EMBED_BATCH_SIZE = 48   # Cohere allows up to 96 per call; 48 is safe


# ─── Embeddings (Cohere free API — zero local RAM overhead) ──────────────────

_embeddings_cache = None

def get_embeddings():
    """
    CohereEmbeddings via API — no local model, no ONNX, no PyTorch.
    RAM cost: ~10MB (just the HTTP client). Works on Render 512MB free tier.
    Free tier: 1000 calls/min, no credit card required.
    Requires COHERE_API_KEY env var (get free key at dashboard.cohere.com).
    """
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache

    cohere_key = _os.environ.get("COHERE_API_KEY", "")
    if not cohere_key:
        raise RuntimeError(
            "COHERE_API_KEY is not set. "
            "Get a free key at https://dashboard.cohere.com/api-keys "
            "and add it to your Render environment variables."
        )

    try:
        from langchain_cohere import CohereEmbeddings
    except ImportError:
        raise RuntimeError(
            "langchain-cohere not installed. It's in requirements.txt — "
            "make sure you've pushed and Render has redeployed."
        )

    _embeddings_cache = CohereEmbeddings(
        cohere_api_key=cohere_key,
        model="embed-english-light-v3.0",   # fastest + lightest free model
    )
    logger.info("Cohere embeddings initialized (embed-english-light-v3.0)")
    return _embeddings_cache


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


# ─── Document Loading ─────────────────────────────────────────────────────────

def _load_pdf_lazy(file_path: Path) -> list:
    """Load PDF page by page to avoid holding full file in RAM."""
    docs = []
    try:
        for page in PyPDFLoader(str(file_path)).lazy_load():
            docs.append(page)
    except Exception:
        docs = PyPDFLoader(str(file_path)).load()
    return docs


def load_new_documents(data_dir: str = "data/docs") -> Tuple[list, dict]:
    manifest = _load_manifest()
    new_docs = []
    updated_manifest = {}

    data_path = Path(data_dir)
    ext_to_loader = {
        ".pdf":  PyPDFLoader,
        ".txt":  TextLoader,
        ".docx": Docx2txtLoader,
        ".csv":  CSVLoader,
    }
    try:
        from langchain_community.document_loaders import (
            UnstructuredHTMLLoader,
            UnstructuredMarkdownLoader,
        )
        ext_to_loader[".md"]   = UnstructuredMarkdownLoader
        ext_to_loader[".html"] = UnstructuredHTMLLoader
        ext_to_loader[".htm"]  = UnstructuredHTMLLoader
    except ImportError:
        pass

    for file_path in sorted(data_path.rglob("*")):
        if file_path.suffix.lower() not in ext_to_loader:
            continue
        sha = _file_sha256(file_path)
        updated_manifest[str(file_path)] = sha
        if manifest.get(str(file_path)) == sha:
            logger.debug(f"Skipping unchanged: {file_path.name}")
            continue

        try:
            if file_path.suffix.lower() == ".pdf":
                docs = _load_pdf_lazy(file_path)
            else:
                docs = ext_to_loader[file_path.suffix.lower()](str(file_path)).load()
            new_docs.extend(docs)
            logger.info(f"Loaded: {file_path.name} ({len(docs)} pages)")
        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {e}")

    return new_docs, updated_manifest


def load_documents(data_dir: str = "data/docs") -> list:
    docs, _ = load_new_documents(data_dir)
    return docs


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks (size={settings.CHUNK_SIZE})")
    return chunks


# ─── Batched FAISS builder ────────────────────────────────────────────────────

def _build_faiss_batched(chunks: list, embeddings) -> FAISS:
    """
    Embed chunks in batches to stay within API rate limits and memory.
    Cohere free tier: 1000 calls/min — batching at 48 keeps us well clear.
    """
    if not chunks:
        raise ValueError("No chunks to embed.")

    total_batches = -(-len(chunks) // EMBED_BATCH_SIZE)
    logger.info(f"Embedding {len(chunks)} chunks in {total_batches} batches of {EMBED_BATCH_SIZE}…")

    vectorstore = None
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i: i + EMBED_BATCH_SIZE]
        batch_num = i // EMBED_BATCH_SIZE + 1
        logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)")

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.merge_from(FAISS.from_documents(batch, embeddings))

        gc.collect()   # free memory between batches

    return vectorstore


# ─── Index Build / Load ───────────────────────────────────────────────────────

def build_index(chunks: list) -> FAISS:
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()
    vectorstore = _build_faiss_batched(chunks, embeddings)
    vectorstore.save_local(str(INDEX_PATH))
    with open(DOCS_PICKLE, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Index built — {len(chunks)} chunks")
    return vectorstore


def update_index(new_chunks: list, manifest: dict) -> FAISS:
    embeddings = get_embeddings()

    if INDEX_PATH.exists() and DOCS_PICKLE.exists():
        logger.info("Merging into existing index…")
        vectorstore = FAISS.load_local(
            str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True
        )
        if new_chunks:
            vectorstore.merge_from(_build_faiss_batched(new_chunks, embeddings))
        with open(DOCS_PICKLE, "rb") as f:
            existing = pickle.load(f)
        all_chunks = existing + new_chunks
    else:
        logger.info("Building index from scratch…")
        if not new_chunks:
            raise ValueError("No documents to index. Upload files first.")
        vectorstore = _build_faiss_batched(new_chunks, embeddings)
        all_chunks = new_chunks

    vectorstore.save_local(str(INDEX_PATH))
    with open(DOCS_PICKLE, "wb") as f:
        pickle.dump(all_chunks, f)
    _save_manifest(manifest)
    logger.info(f"Index updated — {len(all_chunks)} total chunks")
    return vectorstore


def load_index() -> FAISS:
    if not INDEX_PATH.exists():
        raise FileNotFoundError("No index found. Run /ingest first.")
    embeddings = get_embeddings()
    vs = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
    logger.info("FAISS index loaded.")
    return vs


def load_chunks() -> list:
    if not DOCS_PICKLE.exists():
        return []
    with open(DOCS_PICKLE, "rb") as f:
        return pickle.load(f)


# ─── Hybrid Retriever ─────────────────────────────────────────────────────────

def build_hybrid_retriever(vectorstore: FAISS, chunks: list):
    faiss_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.RETRIEVER_TOP_K, "fetch_k": settings.RETRIEVER_TOP_K * 3},
    )
    if not chunks:
        return faiss_retriever
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = settings.RETRIEVER_TOP_K
    return EnsembleRetriever(retrievers=[bm25, faiss_retriever], weights=[0.4, 0.6])


# ─── Prompts ──────────────────────────────────────────────────────────────────

QA_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant answering questions from uploaded documents.
Rules:
- Answer ONLY from the provided context.
- If the answer is not in the context, say "I don't have enough information to answer that."
- Do NOT make up facts. Cite source filename and page when relevant.

Context:
{context}

Question: {question}
Answer:"""
)

CONDENSE_PROMPT = PromptTemplate.from_template(
    """Given the conversation and a follow-up question, rephrase it as a standalone question.

Chat History:
{chat_history}

Follow-up: {question}
Standalone question:"""
)

ROUTING_PROMPT = PromptTemplate.from_template(
    """You are a query router. Given retrieved document excerpts and a question,
respond with ONLY one word: "RELEVANT" or "OFF_TOPIC".

Context:
{context}

Question: {question}
Decision:"""
)


# ─── Query Routing ────────────────────────────────────────────────────────────

def check_query_relevance(question: str, context_docs: list) -> bool:
    preview = "\n---\n".join(d.page_content[:400] for d in context_docs[:4])
    if not preview.strip():
        return False
    try:
        llm = ChatGroq(model=settings.LLM_MODEL, temperature=0.0, groq_api_key=settings.GROQ_API_KEY)
        result = llm.invoke(ROUTING_PROMPT.format(context=preview, question=question))
        return "RELEVANT" in result.content.strip().upper()
    except Exception as e:
        logger.warning(f"Routing check failed (defaulting RELEVANT): {e}")
        return True


# ─── QA Chain ─────────────────────────────────────────────────────────────────

def build_qa_chain(vectorstore: FAISS, chunks: list = None) -> ConversationalRetrievalChain:
    llm = ChatGroq(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        groq_api_key=settings.GROQ_API_KEY,
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", return_messages=True,
        output_key="answer", k=settings.MEMORY_WINDOW_K,
    )
    if chunks is None:
        chunks = load_chunks()
    retriever = build_hybrid_retriever(vectorstore, chunks)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory,
        return_source_documents=True,
        condense_question_prompt=CONDENSE_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        verbose=False,
    )


# ─── Answer ───────────────────────────────────────────────────────────────────

def answer_query(chain, question: str) -> Tuple[str, List[dict]]:
    try:
        vs = chain.retriever.retrievers[1].vectorstore
    except (AttributeError, IndexError):
        try: vs = chain.retriever.vectorstore
        except AttributeError: vs = None

    if vs is not None:
        probe = vs.similarity_search(question, k=3)
        if not check_query_relevance(question, probe):
            return ("⚠️ This question doesn't appear to be covered by your uploaded documents.", [])

    result = chain.invoke({"question": question})
    seen, sources = set(), []
    for doc in result.get("source_documents", []):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = f"{src}:{page}"
        if key not in seen:
            seen.add(key)
            sources.append({"file": Path(src).name, "path": src, "page": page})
    return result["answer"], sources


async def answer_query_stream(chain, question: str) -> AsyncGenerator[str, None]:
    try:
        vs = chain.retriever.retrievers[1].vectorstore
    except (AttributeError, IndexError):
        try: vs = chain.retriever.vectorstore
        except AttributeError: vs = None

    if vs is not None:
        probe = vs.similarity_search(question, k=3)
        if not check_query_relevance(question, probe):
            yield "⚠️ This question doesn't appear to be covered by your uploaded documents."
            return

    async for event in chain.astream_events({"question": question}, version="v1"):
        if event.get("event") == "on_chat_model_stream":
            content = getattr(event["data"]["chunk"], "content", "")
            if content:
                yield content
