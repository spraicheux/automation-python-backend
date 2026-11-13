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
import re
import time
from typing import Dict, Optional, Set

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
# Before auto-retrying a genuinely FAILED job, give Celery's own in-task retry
# (max_retries=2, backoff up to ~2 min) time to finish so we never double-run.
_FAILED_RETRY_GRACE_SECONDS = 300
# A job stuck in "processing" this long has outlived the 95-min hard time-limit
# AND the 100-min broker (acks_late) redelivery window — the worker was hard-
# killed (OOM / Azure recycle) and nothing recovered it. Re-enqueue it (idempotent
# re-processing rewrites one clean set). Kept past both windows so we never
# re-run a job that is still legitimately in flight or being redelivered.
_STUCK_PROCESSING_SECONDS = 130 * 60
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _active_job_ids(celery_app) -> Optional[Set[str]]:
    """job_ids currently running on a LIVE worker, or None when that can't be
    determined (broker hiccup). None ⇒ we must NOT touch 'processing' jobs this
    round, since we can't prove no worker is on them."""
    try:
        insp = celery_app.control.inspect(timeout=3)
        active = insp.active()
    except Exception:
        return None
    if active is None:
        return None
    ids: Set[str] = set()
    for tasks in active.values():
        for t in (tasks or []):
            a = t.get("args")
            if isinstance(a, (list, tuple)) and a:
                ids.add(str(a[0]))
            elif isinstance(a, str):
                m = _UUID_RE.search(a)
                if m:
                    ids.add(m.group(0))
    return ids


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


# A slow/timeout failure replays the SAME long extraction and will likely time
# out again, hogging the single worker from everyone else — so it gets far fewer
# retries than a transient crash.
_TIMEOUT_MAX_ATTEMPTS = 2


def _retry_cap(entry: Dict) -> int:
    err = (entry.get("error") or "").lower()
    if "timed out" in err or "time limit" in err:
        return _TIMEOUT_MAX_ATTEMPTS
    return _MAX_ATTEMPTS


def sweep_stuck_jobs() -> Dict[str, int]:
    """Re-enqueue orphaned jobs and tidy stale terminal errors. Returns a small
    summary {scanned, requeued, cleaned}. Safe to call repeatedly."""
    from core.job_store import load_job_payload

    scanned = requeued = cleaned = 0
    try:
        from workers.celery_tasks import process_document_task
        from core.celery_app import celery_app
    except Exception as e:
        logger.warning(f"job sweep: cannot import task: {e}")
        return {"scanned": 0, "requeued": 0, "cleaned": 0}

    # Snapshot what live workers are actually running, so we never re-enqueue a
    # job that is still in flight. None ⇒ unknown (broker hiccup) ⇒ leave
    # "processing" jobs alone this round.
    active_ids = _active_job_ids(celery_app)

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

        age = _age_seconds(entry.get("updated_at") or entry.get("received_at") or "")

        # Decide whether, and why, this job should be re-enqueued. Every path is
        # safe to replay: idempotent re-processing clears any prior rows for the
        # job_id and writes ONE clean set (see workers.processor.process_offer).
        reason: Optional[str] = None
        if status == "queued":
            # Orphan: the broker hand-off never landed. Only safe to replay when
            # it was never delivered (else it is in flight / will be redelivered).
            if was_delivered(job_id) or age < _ORPHAN_GRACE_SECONDS:
                continue
            reason = "orphaned-queued"
        elif status == "failed" and _is_enqueue_failure(entry):
            # Our own "could not reach the queue" failure — same as an orphan.
            if was_delivered(job_id) or age < _ORPHAN_GRACE_SECONDS:
                continue
            reason = "enqueue-failed"
        elif status in ("failed", "error"):
            # A genuine extraction failure (timeout, soft-limit, exhausted Celery
            # retries, or a crash). Never abandon it — auto-retry, bounded by the
            # attempt cap. Wait out Celery's own in-task retry first so we don't
            # double-run, and never replay a timeout past the cap (it would just
            # time out again and hog the single worker).
            if age < _FAILED_RETRY_GRACE_SECONDS:
                continue
            reason = "failed-retry"
        elif status == "processing":
            # Stuck past the hard limit AND the broker redelivery window, with no
            # live worker on it ⇒ hard-killed and unrecovered. Re-enqueue.
            if age < _STUCK_PROCESSING_SECONDS:
                continue
            if active_ids is None or job_id in active_ids:
                continue  # still (or might still be) running — don't double-run
            reason = "stuck-processing"
        else:
            continue

        if _attempts(job_id) >= _retry_cap(entry):
            # Tried hard and it still won't complete — stop replaying (don't hog
            # the worker / burn credits forever). A stuck "processing" row that
            # has exhausted its retries is parked as "failed" so it stops being
            # re-scanned and is visibly actionable, instead of looking live.
            if status == "processing":
                redis_manager.set_job_status(job_id, "failed")
                update_intake(job_id, status="failed",
                              error="stuck in processing; auto-recovery exhausted — needs manual review")
            continue

        payload = load_job_payload(job_id)
        if not payload:
            # Nothing to retry from (expired, or predates file storage).
            continue

        _bump_attempts(job_id)
        try:
            # Same job_id ⇒ the SAME intake row is updated (no clutter) and the
            # idempotent write path guarantees exactly one clean result set.
            process_document_task.delay(job_id, payload)
            mark_delivered(job_id)
            update_intake(job_id, status="queued", clear_error=True)
            requeued += 1
            logger.info(f"job sweep: recovered {reason} job {job_id} "
                        f"({entry.get('source_filename')}) attempt "
                        f"{_attempts(job_id)}/{_MAX_ATTEMPTS}")
        except Exception as e:
            logger.warning(f"job sweep: re-enqueue failed for {job_id}: {e}")

    if requeued or cleaned:
        logger.info(f"job sweep: scanned={scanned} requeued={requeued} cleaned={cleaned}")
    return {"scanned": scanned, "requeued": requeued, "cleaned": cleaned}
