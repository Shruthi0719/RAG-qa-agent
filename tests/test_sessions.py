"""Tests for SessionStore."""
import time
import pytest
from app.session_manager import SessionStore


def test_set_and_get():
    store = SessionStore(ttl=60, max_sessions=10)
    store.set("s1", "chain1")
    assert store.get("s1") == "chain1"


def test_get_missing():
    store = SessionStore(ttl=60, max_sessions=10)
    assert store.get("nope") is None


def test_expiry():
    store = SessionStore(ttl=1, max_sessions=10)
    store.set("s1", "chain1")
    time.sleep(1.1)
    assert store.get("s1") is None


def test_lru_eviction():
    store = SessionStore(ttl=60, max_sessions=3)
    store.set("s1", "c1")
    store.set("s2", "c2")
    store.set("s3", "c3")
    # Access s1 and s2 to mark s3 as LRU
    store.get("s1"); store.get("s2")
    store.set("s4", "c4")  # should evict s3
    assert store.get("s3") is None
    assert store.get("s1") == "c1"


def test_delete():
    store = SessionStore(ttl=60, max_sessions=10)
    store.set("s1", "c1")
    assert store.delete("s1") is True
    assert store.get("s1") is None
    assert store.delete("s1") is False


def test_clear_all():
    store = SessionStore(ttl=60, max_sessions=10)
    store.set("s1", "c1")
    store.set("s2", "c2")
    store.clear_all()
    assert store.active_count == 0


def test_list_sessions():
    store = SessionStore(ttl=60, max_sessions=10)
    store.set("s1", "c1")
    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "s1"
    assert "idle_seconds" in sessions[0]
    assert "expires_in" in sessions[0]
