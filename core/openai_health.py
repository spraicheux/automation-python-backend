"""core/openai_health.py

OpenAI credit/health status + credit-exhaustion handling.

- The backend learns OpenAI's status for free, from its own extraction calls:
  every successful call marks it OK; a call that fails with "insufficient_quota /
  no credits" marks it EXHAUSTED (with the error). The dashboard reads this.
- When extraction hits an exhausted balance, the job is NOT silently finished
  with 0 products — it raises OpenAICreditsExhausted, and the worker parks it in
  a Redis "pending credits" list and marks the intake NO CREDITS.
- The in-app scheduler drains the parked jobs (re-queues them) as soon as a
  probe shows credits are back, so processing auto-resumes.

OpenAI does not expose a dollar balance via the API key, so "live credits" is a
reliable OK / EXHAUSTED status rather than an amount.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

from core.redis_client import redis_manager

logger = logging.getLogger(__name__)

_STATUS_KEY = "openai:status"
_PENDING_KEY = "openai:pending_credits"   # list of {job_id, payload}
_PENDING_MAX = 1000


class OpenAICreditsExhausted(Exception):
    """Raised when an OpenAI call fails because the account is out of credits.
    Propagates out of extraction so the job is parked (not finished with 0)."""


def _r():
    if redis_manager.use_redis and redis_manager.client:
        return redis_manager.client
    return None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def is_credit_error(exc: Exception) -> bool:
    """True when an OpenAI error means the balance is exhausted (vs a normal
    rate limit, which is transient and should NOT park the job)."""
    s = str(exc).lower()
    code = getattr(getattr(exc, "body", None), "get", lambda *_: None)("code") if hasattr(exc, "body") else None
    return (
        "insufficient_quota" in s
        or "credit_balance_exhausted" in s
        or "no credits remaining" in s
        or "exceeded your current quota" in s
        or "billing" in s and "credit" in s
        or code in ("insufficient_quota", "credit_balance_exhausted")
    )


def mark_openai_ok() -> None:
    r = _r()
    if not r:
        return
    try:
        r.set(_STATUS_KEY, json.dumps({"ok": True, "error": None, "checked_at": _now()}))
    except Exception:
        pass


def mark_openai_exhausted(msg: Any) -> None:
    r = _r()
    if not r:
        return
    try:
        r.set(_STATUS_KEY, json.dumps({"ok": False, "error": str(msg)[:300], "checked_at": _now()}))
    except Exception:
        pass


def get_openai_status() -> Dict[str, Any]:
    r = _r()
    if r:
        try:
            raw = r.get(_STATUS_KEY)
            if raw:
                d = json.loads(raw)
                d["pending_jobs"] = pending_count()
                return d
        except Exception:
            pass
    return {"ok": None, "error": None, "checked_at": None, "pending_jobs": pending_count()}


# ─── parked-job queue (jobs waiting for credits) ──────────────────────────────

def queue_pending_credit_job(job_id: str, payload: Dict[str, Any]) -> None:
    r = _r()
    if not r:
        return
    try:
        r.rpush(_PENDING_KEY, json.dumps({"job_id": job_id, "payload": payload}))
        r.ltrim(_PENDING_KEY, -_PENDING_MAX, -1)
    except Exception as e:
        logger.warning(f"queue_pending_credit_job failed for {job_id}: {e}")


def pending_count() -> int:
    r = _r()
    if not r:
        return 0
    try:
        return int(r.llen(_PENDING_KEY))
    except Exception:
        return 0


def clear_pending() -> int:
    """Drop all parked jobs (so they do NOT auto-resume). Returns how many."""
    r = _r()
    if not r:
        return 0
    try:
        n = int(r.llen(_PENDING_KEY) or 0)
        r.delete(_PENDING_KEY)
        return n
    except Exception as e:
        logger.warning(f"clear_pending failed: {e}")
        return 0


def probe_openai() -> bool:
    """Minimal (1-token) call to check whether credits are back. Updates the
    stored status. Returns True if OpenAI is usable."""
    try:
        from openai import OpenAI
        c = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        c.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "ok"}], max_tokens=1)
        mark_openai_ok()
        return True
    except Exception as e:
        if is_credit_error(e):
            mark_openai_exhausted(e)
            return False
        # Some other error — don't claim exhausted; report not-ok conservatively.
        logger.warning(f"probe_openai non-credit error: {e}")
        return False


def drain_pending_credit_jobs() -> int:
    """Re-queue every parked job. Returns how many were re-enqueued."""
    r = _r()
    if not r:
        return 0
    n = 0
    try:
        from workers.celery_tasks import process_document_task
        from core.intake_log import update_intake
        while True:
            raw = r.lpop(_PENDING_KEY)
            if not raw:
                break
            item = json.loads(raw)
            update_intake(item["job_id"], status="queued", error=None)
            process_document_task.delay(item["job_id"], item["payload"])
            n += 1
        if n:
            logger.info(f"drained {n} credit-parked job(s) back into the queue")
    except Exception as e:
        logger.warning(f"drain_pending_credit_jobs failed: {e}")
    return n
