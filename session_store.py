"""
session_store.py
=================
Fixes the "RAG silently stops working" bug: `sessions` in app.py is a plain
in-memory dict, and Render's free tier wipes process memory whenever the
instance spins down after ~15min idle (or restarts for any other reason).
The client-side file chip still says "ready", but the next request hits a
fresh process where `session["rag_files"]`/`["rag_chunks"]` are just gone —
so retrieval quietly returns nothing.

This module mirrors only the RAG-relevant slice of each session (file
metadata + chunks/vectors, NOT raw file bytes, NOT chat history) to Redis,
and rehydrates it into a session dict the first time that session is seen
by a process that doesn't have it in memory. Conversation history stays
exactly as it was (in-memory, session-scoped) — out of scope for this fix.

If REDIS_URL isn't set, or the `redis` package/connection isn't available,
every function below is a no-op and behavior is identical to before this
change. To actually fix the bug in production, set REDIS_URL on Render to
a real Redis instance (e.g. a free Upstash database).
"""
import os
import json
import logging

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL")
_client = None

if REDIS_URL:
    try:
        import redis  # requires `redis` in requirements.txt
        _client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
        _client.ping()
        logger.info("session_store: Redis persistence enabled for RAG session data")
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("session_store: REDIS_URL set but Redis unreachable (%s); "
                        "falling back to in-memory-only RAG storage", exc)
        _client = None
else:
    logger.info("session_store: REDIS_URL not set; RAG data is in-memory only "
                "and will be lost on process restart/spin-down")

_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


def _key(session_id: str) -> str:
    return f"ailoops:rag:{session_id}"


def save_rag_state(session: dict) -> None:
    """Persist session['rag_files']/['rag_chunks'] to Redis. No-op without Redis."""
    if not _client:
        return
    session_id = session.get("_session_id")
    if not session_id:
        return
    try:
        files = {
            fid: {k: v for k, v in rec.items() if not k.startswith("_")}
            for fid, rec in (session.get("rag_files") or {}).items()
        }
        payload = json.dumps({"rag_files": files, "rag_chunks": session.get("rag_chunks") or []})
        _client.set(_key(session_id), payload, ex=_TTL_SECONDS)
    except Exception as exc:
        logger.warning("session_store: failed to persist RAG state for %s: %s", session_id, exc)


def load_rag_state(session: dict) -> None:
    """Hydrate rag_files/rag_chunks into `session` if this process doesn't already
    have them (e.g. a fresh instance after a cold start). Safe to call on every
    session lookup — it's a cheap no-op once the session already has files."""
    if not _client or session.get("rag_files"):
        return
    session_id = session.get("_session_id")
    if not session_id:
        return
    try:
        raw = _client.get(_key(session_id))
        if raw:
            data = json.loads(raw)
            session["rag_files"] = data.get("rag_files") or {}
            session["rag_chunks"] = data.get("rag_chunks") or []
    except Exception as exc:
        logger.warning("session_store: failed to hydrate RAG state for %s: %s", session_id, exc)


def clear_rag_state(session_id: str) -> None:
    if not _client:
        return
    try:
        _client.delete(_key(session_id))
    except Exception as exc:
        logger.warning("session_store: failed to clear RAG state for %s: %s", session_id, exc)
