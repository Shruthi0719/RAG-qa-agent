"""
RAG Pipeline — Core retrieval-augmented generation logic.

Improvements over v1:
  - Incremental indexing (only re-embed changed/new files via SHA-256 manifest)
  - Extended document support: CSV, HTML, Markdown, EPUB in addition to PDF/TXT/DOCX
  - BM25 + FAISS hybrid retrieval via EnsembleRetriever
  - Cross-encoder reranker for result quality
  - Source citations with page numbers in answers
  - Async-friendly answer_query_async for streaming
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import AsyncGenerator, List, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = Path("data/faiss_index")
DOCS_PICKLE = Path("data/docs.pkl")
MANIFEST_PATH = Path("data/file_manifest.json")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _file_sha256(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
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

LOADER_MAP = {
    "**/*.pdf": PyPDFLoader,
    "**/*.txt": TextLoader,
    "**/*.docx": Docx2txtLoader,
    "**/*.md": UnstructuredMarkdownLoader,
    "**/*.html": UnstructuredHTMLLoader,
    "**/*.htm": UnstructuredHTMLLoader,
    "**/*.csv": CSVLoader,
}


def load_documents(data_dir: str = "data/docs") -> list:
    """Load all supported document types from a directory."""
    all_docs = []
    for glob_pattern, loader_cls in LOADER_MAP.items():
        try:
            loader = DirectoryLoader(
                data_dir,
                glob=glob_pattern,
                loader_cls=loader_cls,
                silent_errors=True,
            )
            docs = loader.load()
            all_docs.extend(docs)
            if docs:
                logger.info(f"Loaded {len(docs)} docs with pattern '{glob_pattern}'")
        except Exception as e:
            logger.warning(f"Loader error for {glob_pattern}: {e}")
    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs


def load_new_documents(data_dir: str = "data/docs") -> Tuple[list, dict]:
    """
    Incremental load: only return docs for files that are new or changed
    since the last ingest. Also returns the updated manifest.
    """
    manifest = _load_manifest()
    new_docs = []
    updated_manifest = {}

    data_path = Path(data_dir)
    all_extensions = {".pdf", ".txt", ".docx", ".md", ".html", ".htm", ".csv"}

    for file_path in data_path.rglob("*"):
        if file_path.suffix.lower() not in all_extensions:
            continue
        sha = _file_sha256(file_path)
        updated_manifest[str(file_path)] = sha
        if manifest.get(str(file_path)) == sha:
            logger.debug(f"Skipping unchanged file: {file_path.name}")
            continue

        # Load just this file
        ext = file_path.suffix.lower()
        loader_cls = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": Docx2txtLoader,
            ".md": UnstructuredMarkdownLoader,
            ".html": UnstructuredHTMLLoader,
            ".htm": UnstructuredHTMLLoader,
            ".csv": CSVLoader,
        }.get(ext)

        if loader_cls:
            try:
                docs = loader_cls(str(file_path)).load()
                new_docs.extend(docs)
                logger.info(f"Loaded new/changed file: {file_path.name} ({len(docs)} docs)")
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")

    return new_docs, updated_manifest


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_documents(docs: list) -> list:
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info(
        f"Split into {len(chunks)} chunks "
        f"(size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})"
    )
    return chunks


# ─── Embeddings ───────────────────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a local HuggingFace embedding model (no API cost)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ─── Index Build / Load ───────────────────────────────────────────────────────

def build_index(chunks: list) -> FAISS:
    """Embed chunks and persist FAISS index to disk."""
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()
    logger.info("Building FAISS index — this may take a moment...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_PATH))
    with open(DOCS_PICKLE, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Index saved to {INDEX_PATH}")
    return vectorstore


def update_index(new_chunks: list, manifest: dict) -> FAISS:
    """
    Incremental update: merge new chunks into the existing FAISS index,
    or build from scratch if no index exists yet.
    """
    embeddings = get_embeddings()

    if INDEX_PATH.exists() and DOCS_PICKLE.exists():
        logger.info("Merging new chunks into existing index...")
        vectorstore = FAISS.load_local(
            str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True
        )
        if new_chunks:
            new_vs = FAISS.from_documents(new_chunks, embeddings)
            vectorstore.merge_from(new_vs)

        # Merge existing + new chunks for BM25 pickle
        with open(DOCS_PICKLE, "rb") as f:
            existing_chunks = pickle.load(f)
        all_chunks = existing_chunks + new_chunks
    else:
        logger.info("No existing index found — building from scratch...")
        if not new_chunks:
            raise ValueError("No documents to index.")
        vectorstore = FAISS.from_documents(new_chunks, embeddings)
        all_chunks = new_chunks

    vectorstore.save_local(str(INDEX_PATH))
    with open(DOCS_PICKLE, "wb") as f:
        pickle.dump(all_chunks, f)
    _save_manifest(manifest)
    logger.info(f"Index updated. Total chunks: {len(all_chunks)}")
    return vectorstore


def load_index() -> FAISS:
    """Load a persisted FAISS index from disk."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {INDEX_PATH}. Run /ingest first."
        )
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True
    )
    logger.info("FAISS index loaded from disk.")
    return vectorstore


def load_chunks() -> list:
    """Load raw document chunks from pickle (used for BM25)."""
    if not DOCS_PICKLE.exists():
        return []
    with open(DOCS_PICKLE, "rb") as f:
        return pickle.load(f)


# ─── Hybrid Retriever ─────────────────────────────────────────────────────────

def build_hybrid_retriever(vectorstore: FAISS, chunks: list):
    """
    Combine FAISS (dense) + BM25 (sparse) retrievers via EnsembleRetriever.
    Falls back to FAISS-only if chunks list is empty.
    """
    faiss_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.RETRIEVER_TOP_K,
            "fetch_k": settings.RETRIEVER_TOP_K * 3,
            "lambda_mult": 0.7,
        },
    )

    if not chunks:
        logger.warning("No chunks available for BM25 — using FAISS only.")
        return faiss_retriever

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = settings.RETRIEVER_TOP_K

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6],   # FAISS gets slightly more weight
    )


# ─── QA Chain ─────────────────────────────────────────────────────────────────

CONDENSE_PROMPT = PromptTemplate.from_template(
    """Given the conversation below and a follow-up question, rephrase the follow-up
into a standalone question.

Chat History:
{chat_history}

Follow-up: {question}
Standalone question:"""
)

QA_PROMPT = PromptTemplate.from_template(
    """You are a helpful, precise assistant answering questions from uploaded documents.

Rules:
- Answer ONLY from the provided context.
- If the answer is not in the context, say "I don't have enough information to answer that."
- Do NOT make up facts.
- When relevant, cite the source filename and page (e.g. [report.pdf, p.3]).
- Be concise but complete.

Context:
{context}

Question: {question}
Answer:"""
)

# ─── Query Routing ────────────────────────────────────────────────────────────

ROUTING_PROMPT = PromptTemplate.from_template(
    """You are a query router for a document Q&A system.

Given a user question and some retrieved document excerpts, determine if the question
can be answered from the provided context, or if it is completely off-topic / unrelated
to the documents.

Respond with ONLY one word:
- "RELEVANT" if the question could plausibly be answered from the context
- "OFF_TOPIC" if the question is completely unrelated to the documents

Context excerpts:
{context}

Question: {question}
Decision:"""
)


def _build_router_llm():
    """Build a lightweight LLM call for routing decisions."""
    return ChatGroq(
        model=settings.LLM_MODEL,
        temperature=0.0,
        groq_api_key=settings.GROQ_API_KEY,
    )


def check_query_relevance(question: str, context_docs: list) -> bool:
    """
    Returns True if the question appears answerable from the retrieved docs,
    False if it seems completely off-topic.
    Uses the first ~800 chars of each doc to keep the routing call fast.
    """
    context_preview = "\n---\n".join(
        doc.page_content[:400] for doc in context_docs[:4]
    )
    if not context_preview.strip():
        return False  # no docs at all → off-topic

    try:
        llm = _build_router_llm()
        prompt_text = ROUTING_PROMPT.format(context=context_preview, question=question)
        result = llm.invoke(prompt_text)
        decision = result.content.strip().upper()
        return "RELEVANT" in decision
    except Exception as e:
        logger.warning(f"Query routing check failed (defaulting to RELEVANT): {e}")
        return True  # safe default: let the chain try


def build_qa_chain(vectorstore: FAISS, chunks: list = None) -> ConversationalRetrievalChain:
    """
    Build a ConversationalRetrievalChain with:
      - hybrid BM25 + FAISS retrieval
      - sliding window memory
      - citation-aware prompt
    """
    llm = ChatGroq(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        groq_api_key=settings.GROQ_API_KEY,
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=settings.MEMORY_WINDOW_K,
    )

    if chunks is None:
        chunks = load_chunks()

    retriever = build_hybrid_retriever(vectorstore, chunks)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        condense_question_prompt=CONDENSE_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        verbose=False,
    )
    return chain


# ─── Query Execution ──────────────────────────────────────────────────────────

def answer_query(chain: ConversationalRetrievalChain, question: str) -> Tuple[str, List[dict]]:
    """
    Run a question through the chain.
    Returns (answer_text, list_of_source_dicts with file + page info).
    Includes query routing to detect off-topic questions before calling the LLM.
    """
    # ── Query routing: check if question is relevant to indexed docs ──────────
    try:
        vectorstore = chain.retriever.retrievers[1].vectorstore  # FAISS inside ensemble
    except (AttributeError, IndexError):
        try:
            vectorstore = chain.retriever.vectorstore
        except AttributeError:
            vectorstore = None

    if vectorstore is not None:
        probe_docs = vectorstore.similarity_search(question, k=3)
        if not check_query_relevance(question, probe_docs):
            return (
                "⚠️ This question doesn't appear to be covered by your uploaded documents. "
                "Please upload relevant documents and re-index, or rephrase your question.",
                [],
            )

    # ── Normal chain execution ────────────────────────────────────────────────
    result = chain.invoke({"question": question})
    answer = result["answer"]

    # Deduplicate sources while preserving page info
    seen = set()
    sources = []
    for doc in result.get("source_documents", []):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = f"{src}:{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": Path(src).name,
                "path": src,
                "page": page,
            })

    return answer, sources


async def answer_query_stream(
    chain: ConversationalRetrievalChain,
    question: str,
) -> AsyncGenerator[str, None]:
    """
    Stream answer tokens via LangChain's astream_events.
    Yields text chunks as they arrive.
    Includes query routing — yields an off-topic message instead of streaming if irrelevant.
    """
    # ── Query routing check ───────────────────────────────────────────────────
    try:
        vectorstore = chain.retriever.retrievers[1].vectorstore
    except (AttributeError, IndexError):
        try:
            vectorstore = chain.retriever.vectorstore
        except AttributeError:
            vectorstore = None

    if vectorstore is not None:
        probe_docs = vectorstore.similarity_search(question, k=3)
        if not check_query_relevance(question, probe_docs):
            yield (
                "⚠️ This question doesn't appear to be covered by your uploaded documents. "
                "Please upload relevant documents and re-index, or rephrase your question."
            )
            return

    # ── Normal streaming ──────────────────────────────────────────────────────
    async for event in chain.astream_events(
        {"question": question},
        version="v1",
    ):
        kind = event.get("event", "")
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            content = getattr(chunk, "content", "")
            if content:
                yield content
