"""
scripts/ingest.py — standalone ingestion script.

Usage:
    python scripts/ingest.py --docs_dir data/docs          # incremental (default)
    python scripts/ingest.py --docs_dir data/docs --full   # full rebuild
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag_pipeline import (
    load_documents,
    load_new_documents,
    chunk_documents,
    build_index,
    update_index,
)
from app.logger import get_logger

logger = get_logger("ingest")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from documents.")
    parser.add_argument("--docs_dir", default="data/docs", help="Directory containing documents")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full rebuild (default: incremental)",
    )
    args = parser.parse_args()

    if args.full:
        logger.info(f"Full rebuild from: {args.docs_dir}")
        docs = load_documents(args.docs_dir)
        if not docs:
            logger.error("No documents found. Exiting.")
            sys.exit(1)
        chunks = chunk_documents(docs)
        build_index(chunks)
        logger.info(f"✅ Done. {len(chunks)} chunks indexed from {len(docs)} documents.")
    else:
        logger.info(f"Incremental ingest from: {args.docs_dir}")
        new_docs, manifest = load_new_documents(args.docs_dir)
        if not new_docs:
            logger.info("✅ No changes detected. Index is already up to date.")
            return
        chunks = chunk_documents(new_docs)
        update_index(chunks, manifest)
        logger.info(f"✅ Done. {len(chunks)} new chunks indexed from {len(new_docs)} documents.")


if __name__ == "__main__":
    main()
