"""
app/session_manager.py — Multi-user session management.

Each session gets its own ConversationalRetrievalChain with independent
conversation memory. Sessions expire after SESSION_TTL_SECONDS of inactivity
and are evicted when MAX_SESSIONS is reached (LRU policy).
"""

import time
from collections import OrderedDict
from typing import Optional

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


class SessionStore:
    """
    Thread-safe (single-process) session store with TTL + LRU eviction.

    Stores:  session_id -> {"chain": chain_obj, "last_used": float}
    """

    def __init__(self, ttl: int = None, max_sessions: int = None):
        self._ttl = ttl or settings.SESSION_TTL_SECONDS
        self._max = max_sessions or settings.MAX_SESSIONS
        self._store: OrderedDict = OrderedDict()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evict_expired(self):
        now = time.time()
        expired = [sid for sid, v in self._store.items() if now - v["last_used"] > self._ttl]
        for sid in expired:
            del self._store[sid]
            logger.info(f"Session expired and evicted: {sid}")

    def _evict_lru(self):
        """Remove least-recently-used entry when over capacity."""
        while len(self._store) >= self._max:
            sid, _ = self._store.popitem(last=False)
            logger.info(f"Session evicted (LRU, capacity={self._max}): {sid}")

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, session_id: str) -> Optional[object]:
        self._evict_expired()
        entry = self._store.get(session_id)
        if entry is None:
            return None
        entry["last_used"] = time.time()
        self._store.move_to_end(session_id)   # mark as recently used
        return entry["chain"]

    def set(self, session_id: str, chain: object):
        self._evict_expired()
        self._evict_lru()
        self._store[session_id] = {"chain": chain, "last_used": time.time()}
        self._store.move_to_end(session_id)
        logger.info(f"Session created/updated: {session_id}")

    def delete(self, session_id: str) -> bool:
        if session_id in self._store:
            del self._store[session_id]
            logger.info(f"Session deleted: {session_id}")
            return True
        return False

    def clear_all(self):
        count = len(self._store)
        self._store.clear()
        logger.info(f"All {count} sessions cleared.")

    @property
    def active_count(self) -> int:
        self._evict_expired()
        return len(self._store)

    def list_sessions(self) -> list:
        self._evict_expired()
        now = time.time()
        return [
            {
                "session_id": sid,
                "idle_seconds": round(now - v["last_used"]),
                "expires_in": round(self._ttl - (now - v["last_used"])),
            }
            for sid, v in self._store.items()
        ]


# ── Singleton ─────────────────────────────────────────────────────────────────
session_store = SessionStore()
