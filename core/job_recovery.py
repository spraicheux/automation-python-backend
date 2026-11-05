"""core/job_recovery.py

Self-healing safety net for the intake pipeline.

The failure it fixes
---------------------
`enqueue_offer` records a dashboard-visible "queued" intake row and then hands
the job to the Celery broker. If that hand-off never lands (a transient Redis/
broker blip), the row used to sit at "queued" forever — the file was silently
lost, never processed, never retried. That is exactly what stranded Samuel's
09:39–09:51 batch.

The fix
-------
`enqueue_offer` now flags a job "delivered" the instant the broker accepts it.
This sweeper runs every couple of minutes (from the in-app scheduler in
main.py, like the credit resumer) and re-enqueues any job that is:

  * still "queued" (or failed purely because the enqueue could not reach the
    queue), AND
  * NOT flagged delivered — so it never actually reached the broker and cannot
    already be running (recovery can never double-run a live job → no
    duplicates, no wasted OpenAI credits), AND
  * older than a short grace (a delivered job flips to "processing" in
    seconds), AND
  * still has its saved payload, AND
  * under the per-job attempt cap.

It also tidies terminal rows: a job that was parked "OpenAI out of credits" and
later resumed to "done" should not keep showing the stale outage error.
"""
from __future__ import annotations

import calendar
import logging
import time
from typing import Dict

from core.redis_client import redis_manager
from core.intake_log import list_intakes, update_intake
from core.ingest_service import was_delivered, mark_delivered

logger = logging.getLogger(__name__)

_SWEEP_ATTEMPTS_KEY = "intake:sweep_attempts:"
_SWEEP_ATTEMPTS_TTL = 7 * 24 * 3600
_MAX_ATTEMPTS = 5
# A delivered job moves queued → processing within seconds; wait a little longer
# than a worker poll before deciding a non-delivered "queued" row is an orphan.
_ORPHAN_GRACE_SECONDS = 180


def _r():
    if redis_manager.use_redis and redis_manager.client:
        return redis_manager.client
    return None


def _age_seconds(iso: str) -> float:
    try:
        t = calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ"))
        return max(0.0, time.time() - t)
    except Exception:
        return 0.0


def _attempts(job_id: str) -> int:
    r = _r()
    if not r:
        return 0
    try:
        return int(r.get(_SWEEP_ATTEMPTS_KEY + job_id) or 0)
    except Exception:
        return 0


def _bump_attempts(job_id: str) -> None:
    r = _r()
    if not r:
        return
    try:
        r.incr(_SWEEP_ATTEMPTS_KEY + job_id)
        r.expire(_SWEEP_ATTEMPTS_KEY + job_id, _SWEEP_ATTEMPTS_TTL)
    except Exception:
        pass


def _is_enqueue_failure(entry: Dict) -> bool:
    """A 'failed' row we created ourselves because the broker was unreachable —
    safe to auto-retry. A genuine extraction failure is left alone (Celery's own
    retry already covered it; we don't silently replay real errors)."""
    err = (entry.get("error") or "").lower()
    return "could not reach the processing queue" in err


def sweep_stuck_jobs() -> Dict[str, int]:
    """Re-enqueue orphaned jobs and tidy stale terminal errors. Returns a small
    summary {scanned, requeued, cleaned}. Safe to call repeatedly."""
    from core.job_store import load_job_payload

    scanned = requeued = cleaned = 0
    try:
        from workers.celery_tasks import process_document_task
    except Exception as e:
        logger.warning(f"job sweep: cannot import task: {e}")
        return {"scanned": 0, "requeued": 0, "cleaned": 0}

    for entry in list_intakes(limit=300):
        scanned += 1
        job_id = entry.get("job_id")
        status = entry.get("status")
        if not job_id:
            continue

        # Tidy: a resumed/retried job that finished should not keep an old error.
        if status == "done" and entry.get("error"):
            update_intake(job_id, clear_error=True)
            cleaned += 1
            continue

        # Only queued orphans, or our own "could not reach the queue" failures.
        if status == "queued":
            pass
        elif status == "failed" and _is_enqueue_failure(entry):
            pass
        else:
            continue

        # Delivered to the broker already ⇒ it is in flight or will be redelivered
        # by Celery (acks_late); re-enqueuing would risk a duplicate. Skip.
        if was_delivered(job_id):
            continue

        if _age_seconds(entry.get("updated_at") or entry.get("received_at") or "") < _ORPHAN_GRACE_SECONDS:
            continue

        if _attempts(job_id) >= _MAX_ATTEMPTS:
            continue

        payload = load_job_payload(job_id)
        if not payload:
            # Nothing to retry from (expired, or predates file storage).
            continue

        _bump_attempts(job_id)
        try:
            # Same job_id ⇒ updates the SAME intake row (no clutter). Never ran
            # and never delivered, so this is its first real trip through the
            # worker — no duplicate, no wasted credits.
            process_document_task.delay(job_id, payload)
            mark_delivered(job_id)
            update_intake(job_id, status="queued", clear_error=True)
            requeued += 1
            logger.info(f"job sweep: recovered orphaned job {job_id} "
                        f"({entry.get('source_filename')})")
        except Exception as e:
            logger.warning(f"job sweep: re-enqueue failed for {job_id}: {e}")

    if requeued or cleaned:
        logger.info(f"job sweep: scanned={scanned} requeued={requeued} cleaned={cleaned}")
    return {"scanned": scanned, "requeued": requeued, "cleaned": cleaned}
