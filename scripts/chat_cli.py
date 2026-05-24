"""
scripts/chat_cli.py — interactive terminal chat against the RAG index.

Usage:
    python scripts/chat_cli.py
    python scripts/chat_cli.py --session my-session-id
"""

import sys
import argparse
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag_pipeline import load_index, load_chunks, build_qa_chain, answer_query
from app.logger import get_logger

logger = get_logger("chat_cli")

COMMANDS = {
    "/help": "Show this help",
    "/sources": "Show sources from last answer",
    "/reset": "Clear conversation history (start new session)",
    "/exit": "Quit",
}


def print_help():
    print("\nCommands:")
    for cmd, desc in COMMANDS.items():
        print(f"  {cmd:12} — {desc}")
    print()


def main():
    parser = argparse.ArgumentParser(description="RAG Q&A CLI")
    parser.add_argument("--session", default=str(uuid.uuid4()), help="Session ID")
    args = parser.parse_args()

    print(f"\n🔍 RAG Q&A CLI v2  |  Session: {args.session}")
    print("Type /help for commands, /exit to quit.\n")

    try:
        vs = load_index()
        chunks = load_chunks()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    chain = build_qa_chain(vs, chunks)
    print("✅ Index loaded. Ask away!\n")

    last_sources = []

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue

        if question.lower() in {"/exit", "/quit", "exit", "quit", "q"}:
            print("Bye!")
            break

        if question == "/help":
            print_help()
            continue

        if question == "/sources":
            if last_sources:
                print("\nSources from last answer:")
                for s in last_sources:
                    page = f", page {s['page']}" if s.get("page") is not None else ""
                    print(f"  • {s['file']}{page}")
            else:
                print("  (no sources from last answer)")
            print()
            continue

        if question == "/reset":
            chain = build_qa_chain(vs, chunks)
            last_sources = []
            print("  🔄 Conversation history cleared.\n")
            continue

        answer, last_sources = answer_query(chain, question)
        print(f"\nAssistant: {answer}")
        if last_sources:
            parts = []
            for s in last_sources:
                page = f" p.{s['page']}" if s.get("page") is not None else ""
                parts.append(f"{s['file']}{page}")
            print(f"Sources: {', '.join(parts)}")
        print()


if __name__ == "__main__":
    main()
