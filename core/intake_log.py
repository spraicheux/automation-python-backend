"""core/intake_log.py

Lightweight, Redis-backed intake status so every inbound message is visible in
the dashboard the instant it arrives — status moves received/queued →
processing → done | failed, with the error captured on failure. This is what
makes a file "show instantly, extraction in progress", and it is also how we
identify files that failed (Samuel's visible-status requirement).

Deliberately Redis-only: no SourceFileDB schema change (no prod DDL). Entries
expire after a week. Safe no-ops when Redis is unavailable.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from core.redis_client import redis_manager

logger = logging.getLogger(__name__)

_INDEX = "intake:index"       # sorted set: job_id -> ts
_ENTRY = "intake:entry:"      # + job_id -> json blob
_TTL = 7 * 24 * 3600
_MAX = 300


def _r():
    if redis_manager.use_redis and redis_manager.client:
        return redis_manager.client
    return None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def record_intake(job_id: str, *, channel: Optional[str],
                  source_filename: Optional[str], sender_name: Optional[str] = None,
                  sender_email: Optional[str] = None,
                  supplier_name: Optional[str] = None,
                  attachment_count: int = 0, has_text: bool = False,
                  status: str = "queued") -> None:
    r = _r()
    if not r:
        return
    try:
        entry = {
            "job_id": job_id, "channel": channel,
            "source_filename": source_filename,
            "sender_name": sender_name, "sender_email": sender_email,
            "supplier_name": supplier_name,
            "attachment_count": attachment_count, "has_text": has_text,
            "status": status, "products": None, "error": None,
            "received_at": _now(), "updated_at": _now(),
        }
        r.set(_ENTRY + job_id, json.dumps(entry), ex=_TTL)
        r.zadd(_INDEX, {job_id: time.time()})
        n = r.zcard(_INDEX)
        if n > _MAX:
            old = r.zrange(_INDEX, 0, n - _MAX - 1)
            if old:
                r.zrem(_INDEX, *old)
    except Exception as e:
        logger.warning(f"record_intake failed for {job_id}: {e}")


def update_intake(job_id: str, *, status: Optional[str] = None,
                  products: Optional[int] = None, error: Optional[Any] = None,
                  clear_error: bool = False) -> None:
    """Update a live intake row. `error=None` means "leave the error as-is"
    (so a plain status bump doesn't wipe a previously-recorded failure); pass
    clear_error=True to explicitly remove a stale error — e.g. when a parked
    'OpenAI out of credits' job resumes and completes, so the done row doesn't
    keep showing the old outage message."""
    r = _r()
    if not r:
        return
    try:
        raw = r.get(_ENTRY + job_id)
        entry = json.loads(raw) if raw else {"job_id": job_id, "received_at": _now()}
        if status is not None:
            entry["status"] = status
        if products is not None:
            entry["products"] = products
        if clear_error:
            entry["error"] = None
        elif error is not None:
            entry["error"] = str(error)[:500]
        entry["updated_at"] = _now()
        r.set(_ENTRY + job_id, json.dumps(entry), ex=_TTL)
    except Exception as e:
        logger.warning(f"update_intake failed for {job_id}: {e}")


def clear_intakes(statuses: Optional[List[str]] = None,
                  filename: Optional[str] = None,
                  clear_all: bool = False) -> int:
    """Remove intake-feed entries. Matches by status (e.g. stale
    'processing'/'queued' left by a killed worker), by source_filename
    substring, or all. Returns how many were removed."""
    r = _r()
    if not r:
        return 0
    removed = 0
    try:
        ids = r.zrange(_INDEX, 0, -1)
        for jid in ids:
            jid = jid.decode() if isinstance(jid, bytes) else jid
            raw = r.get(_ENTRY + jid)
            if not raw:
                r.zrem(_INDEX, jid)
                continue
            entry = json.loads(raw)
            match = clear_all
            if not match and statuses and entry.get("status") in statuses:
                match = True
            if not match and filename and filename.lower() in (entry.get("source_filename") or "").lower():
                match = True
            if match:
                r.delete(_ENTRY + jid)
                r.zrem(_INDEX, jid)
                removed += 1
        return removed
    except Exception as e:
        logger.warning(f"clear_intakes failed: {e}")
        return removed


def list_intakes(limit: int = 50) -> List[Dict[str, Any]]:
    r = _r()
    if not r:
        return []
    try:
        ids = r.zrevrange(_INDEX, 0, max(0, limit - 1))
        out: List[Dict[str, Any]] = []
        for jid in ids:
            jid = jid.decode() if isinstance(jid, bytes) else jid
            raw = r.get(_ENTRY + jid)
            if raw:
                out.append(json.loads(raw))
        return out
    except Exception as e:
        logger.warning(f"list_intakes failed: {e}")
        return []
