"""core/job_store.py

Remembers each ingest job's payload (now tiny — it references a saved file, not
raw bytes) so a job can be re-queued later: by the credit-resume scheduler, by
the /reprocess action, or after any failure. Stored in Redis with a 30-day TTL.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from core.redis_client import redis_manager

logger = logging.getLogger(__name__)

_PREFIX = "job:payload:"
_TTL = 30 * 24 * 3600


def _r():
    if redis_manager.use_redis and redis_manager.client:
        return redis_manager.client
    return None


def save_job_payload(job_id: str, payload: Dict[str, Any]) -> None:
    r = _r()
    if not r:
        return
    try:
        r.set(_PREFIX + job_id, json.dumps(payload), ex=_TTL)
    except Exception as e:
        logger.warning(f"save_job_payload {job_id} failed: {e}")


def load_job_payload(job_id: str) -> Optional[Dict[str, Any]]:
    r = _r()
    if not r:
        return None
    try:
        raw = r.get(_PREFIX + job_id)
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"load_job_payload {job_id} failed: {e}")
    return None
