"""core/schema_cache.py

Per-supplier extraction memory. For every offer we record WHICH method handled
it (one of the deterministic parsers, or the LLM) and how many products it
produced, keyed by the sender's domain. This turns "coverage" into something
observable and self-correcting:

  * a supplier consistently handled by a deterministic parser is pinned and its
    count is known, so a future drift (different count) stands out;
  * a supplier that keeps falling to the LLM surfaces as "needs a parser" — a
    ranked backlog instead of guesswork.

Redis-backed (best-effort; a failure here never affects extraction). Read it
with /debug/schema-cache.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from core.redis_client import redis_manager

logger = logging.getLogger(__name__)

_HKEY = "schema_cache:v1"      # hash: domain -> json entry
_MAX_HISTORY = 5


def _domain(sender_email: Optional[str]) -> str:
    s = (sender_email or "").strip().lower()
    return s.split("@")[-1] if "@" in s else (s or "unknown")


def _client():
    try:
        if redis_manager.use_redis and redis_manager.client:
            return redis_manager.client
    except Exception:
        pass
    return None


def record(sender_email: Optional[str], method: Optional[str], count: int,
           subject: Optional[str] = None,
           flagged: Optional[bool] = None) -> None:
    """Remember how this supplier's latest offer was extracted."""
    r = _client()
    if not r:
        return
    try:
        dom = _domain(sender_email)
        prev_raw = r.hget(_HKEY, dom)
        prev = {}
        if prev_raw:
            try:
                prev = json.loads(prev_raw.decode() if isinstance(prev_raw, bytes) else prev_raw)
            except Exception:
                prev = {}
        hist = (prev.get("history") or [])[-(_MAX_HISTORY - 1):]
        hist.append({"method": method, "count": count, "ts": int(time.time()),
                     "subject": (subject or "")[:120], "flagged": bool(flagged)})
        entry = {
            "domain": dom,
            "last_method": method,
            "last_count": count,
            "seen": (prev.get("seen") or 0) + 1,
            "llm_path": method == "llm",
            "history": hist,
        }
        r.hset(_HKEY, dom, json.dumps(entry))
    except Exception as e:  # never break extraction over telemetry
        logger.warning(f"[schema_cache] record failed: {e}")


def all_entries() -> Dict[str, Any]:
    r = _client()
    if not r:
        return {}
    try:
        out = {}
        for k, v in (r.hgetall(_HKEY) or {}).items():
            key = k.decode() if isinstance(k, bytes) else k
            val = v.decode() if isinstance(v, bytes) else v
            try:
                out[key] = json.loads(val)
            except Exception:
                continue
        return out
    except Exception as e:
        logger.warning(f"[schema_cache] read failed: {e}")
        return {}
