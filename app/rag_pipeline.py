"""
RAG Pipeline — per-user isolated version.
All functions accept explicit paths so each user's index is stored separately.
"""
from __future__ import annotations

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
    Docx2txtLoader, PyPDFLoader, TextLoader, CSVLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

EMBED_BATCH_SIZE = 48

# ─── Default paths (kept for backward compat) ────────────────────────────────
def _writable(rel: str) -> Path:
    for base in [".", "/app", "/tmp"]:
        p = Path(base) / rel
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            t = p.parent / ".wt"; t.touch(); t.unlink()
            return p
        except OSError:
            continue
    return Path("/tmp") / rel

INDEX_PATH    = _writable("data/faiss_index")
DOCS_PICKLE   = _writable("data/docs.pkl")
MANIFEST_PATH = _writable("data/file_manifest.json")


# ─── Embeddings ───────────────────────────────────────────────────────────────
_embeddings_cache = None

def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache
    cohere_key = _os.environ.get("COHERE_API_KEY", "")
    if not cohere_key:
        raise RuntimeError(
            "COHERE_API_KEY is not set. "
            "Get a free key at https://dashboard.cohere.com/api-keys"
        )
    from langchain_cohere import CohereEmbeddings
    _embeddings_cache = CohereEmbeddings(
        cohere_api_key=cohere_key,
        model="embed-english-light-v3.0",
    )
    logger.info("Cohere embeddings ready.")
    return _embeddings_cache


# ─── Manifest helpers ─────────────────────────────────────────────────────────
def _load_manifest(manifest_path: str = None) -> dict:
    p = Path(manifest_path) if manifest_path else MANIFEST_PATH
    return json.loads(p.read_text()) if p.exists() else {}

def _save_manifest(manifest: dict, manifest_path: str = None):
    p = Path(manifest_path) if manifest_path else MANIFEST_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2))

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Document loading ─────────────────────────────────────────────────────────
def _load_pdf_lazy(path: Path) -> list:
    docs = []
    try:
        for page in PyPDFLoader(str(path)).lazy_load():
            docs.append(page)
    except Exception:
        docs = PyPDFLoader(str(path)).load()
    return docs

EXT_LOADERS = {
    ".pdf": PyPDFLoader, ".txt": TextLoader,
    ".docx": Docx2txtLoader, ".csv": CSVLoader,
}
try:
    from langchain_community.document_loaders import UnstructuredHTMLLoader, UnstructuredMarkdownLoader
    EXT_LOADERS[".md"] = UnstructuredMarkdownLoader
    EXT_LOADERS[".html"] = UnstructuredHTMLLoader
    EXT_LOADERS[".htm"] = UnstructuredHTMLLoader
except ImportError:
    pass


def load_new_documents(data_dir: str, manifest_path: str = None) -> Tuple[list, dict]:
    manifest = _load_manifest(manifest_path)
    new_docs, updated_manifest = [], {}

    for fp in sorted(Path(data_dir).rglob("*")):
        if fp.suffix.lower() not in EXT_LOADERS:
            continue
        sha = _sha256(fp)
        updated_manifest[str(fp)] = sha
        if manifest.get(str(fp)) == sha:
            continue
        try:
            docs = _load_pdf_lazy(fp) if fp.suffix.lower() == ".pdf" else EXT_LOADERS[fp.suffix.lower()](str(fp)).load()
            new_docs.extend(docs)
            logger.info(f"Loaded: {fp.name} ({len(docs)} pages)")
        except Exception as e:
            logger.warning(f"Failed: {fp.name}: {e}")

    return new_docs, updated_manifest


def load_documents(data_dir: str, manifest_path: str = None) -> list:
    docs, _ = load_new_documents(data_dir, manifest_path)
    return docs


# ─── Chunking ─────────────────────────────────────────────────────────────────
def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


# ─── Batched FAISS ────────────────────────────────────────────────────────────
def _build_faiss_batched(chunks: list, embeddings) -> FAISS:
    if not chunks:
        raise ValueError("No chunks to embed.")
    vs = None
    total = -(-len(chunks) // EMBED_BATCH_SIZE)
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i: i + EMBED_BATCH_SIZE]
        logger.info(f"  Embedding batch {i//EMBED_BATCH_SIZE+1}/{total} ({len(batch)} chunks)")
        if vs is None:
            vs = FAISS.from_documents(batch, embeddings)
        else:
            vs.merge_from(FAISS.from_documents(batch, embeddings))
        gc.collect()
    return vs


# ─── Index build / load / update (accept custom paths) ───────────────────────
def build_index(chunks: list, index_path: str = None, pickle_path: str = None) -> FAISS:
    ip = Path(index_path) if index_path else INDEX_PATH
    pp = Path(pickle_path) if pickle_path else DOCS_PICKLE
    ip.mkdir(parents=True, exist_ok=True)
    vs = _build_faiss_batched(chunks, get_embeddings())
    vs.save_local(str(ip))
    with open(pp, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Index built — {len(chunks)} chunks → {ip}")
    return vs


def update_index(new_chunks: list, manifest: dict,
                 index_path: str = None, pickle_path: str = None,
                 manifest_path: str = None) -> FAISS:
    ip = Path(index_path) if index_path else INDEX_PATH
    pp = Path(pickle_path) if pickle_path else DOCS_PICKLE
    emb = get_embeddings()

    if ip.exists() and pp.exists():
        vs = FAISS.load_local(str(ip), emb, allow_dangerous_deserialization=True)
        if new_chunks:
            vs.merge_from(_build_faiss_batched(new_chunks, emb))
        with open(pp, "rb") as f:
            all_chunks = pickle.load(f) + new_chunks
    else:
        if not new_chunks:
            raise ValueError("No documents to index.")
        vs = _build_faiss_batched(new_chunks, emb)
        all_chunks = new_chunks

    ip.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(ip))
    with open(pp, "wb") as f:
        pickle.dump(all_chunks, f)
    _save_manifest(manifest, manifest_path)
    logger.info(f"Index updated — {len(all_chunks)} chunks")
    return vs


def load_index(index_path: str = None) -> FAISS:
    ip = Path(index_path) if index_path else INDEX_PATH
    if not ip.exists():
        raise FileNotFoundError("No index found.")
    return FAISS.load_local(str(ip), get_embeddings(), allow_dangerous_deserialization=True)


def load_chunks(pickle_path: str = None) -> list:
    pp = Path(pickle_path) if pickle_path else DOCS_PICKLE
    if not pp.exists():
        return []
    with open(pp, "rb") as f:
        return pickle.load(f)


# ─── Retriever ────────────────────────────────────────────────────────────────
def build_hybrid_retriever(vs: FAISS, chunks: list):
    faiss_r = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.RETRIEVER_TOP_K, "fetch_k": settings.RETRIEVER_TOP_K * 3},
    )
    if not chunks:
        return faiss_r
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = settings.RETRIEVER_TOP_K
    return EnsembleRetriever(retrievers=[bm25, faiss_r], weights=[0.4, 0.6])


# ─── Prompts ──────────────────────────────────────────────────────────────────
QA_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant answering questions from uploaded documents.
- Answer ONLY from the provided context.
- If not in context, say "I don't have enough information to answer that."
- Cite source filename and page when relevant.

Context:
{context}

Question: {question}
Answer:""")

CONDENSE_PROMPT = PromptTemplate.from_template(
    """Rephrase the follow-up as a standalone question.
Chat History: {chat_history}
Follow-up: {question}
Standalone question:""")

ROUTING_PROMPT = PromptTemplate.from_template(
    """Respond ONLY with "RELEVANT" or "OFF_TOPIC".
Context: {context}
Question: {question}
Decision:""")


# ─── Query routing ────────────────────────────────────────────────────────────
def check_query_relevance(question: str, context_docs: list) -> bool:
    preview = "\n---\n".join(d.page_content[:400] for d in context_docs[:4])
    if not preview.strip():
        return False
    try:
        llm = ChatGroq(model=settings.LLM_MODEL, temperature=0.0, groq_api_key=settings.GROQ_API_KEY)
        result = llm.invoke(ROUTING_PROMPT.format(context=preview, question=question))
        return "RELEVANT" in result.content.strip().upper()
    except Exception as e:
        logger.warning(f"Routing failed (defaulting RELEVANT): {e}")
        return True


# ─── QA Chain ─────────────────────────────────────────────────────────────────
def build_qa_chain(vs: FAISS, chunks: list = None) -> ConversationalRetrievalChain:
    llm = ChatGroq(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        groq_api_key=settings.GROQ_API_KEY,
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", return_messages=True,
        output_key="answer", k=settings.MEMORY_WINDOW_K,
    )
    retriever = build_hybrid_retriever(vs, chunks or [])
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
        except: vs = None

    if vs:
        probe = vs.similarity_search(question, k=3)
        if not check_query_relevance(question, probe):
            return "⚠️ This question doesn't appear to be covered by your uploaded documents.", []

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
        except: vs = None

    if vs:
        probe = vs.similarity_search(question, k=3)
        if not check_query_relevance(question, probe):
            yield "⚠️ This question doesn't appear to be covered by your uploaded documents."
            return

    async for event in chain.astream_events({"question": question}, version="v1"):
        if event.get("event") == "on_chat_model_stream":
            content = getattr(event["data"]["chunk"], "content", "")
            if content:
                yield content
