from dotenv import load_dotenv
load_dotenv()

import logging
import requests
import traceback
import asyncio
from datetime import datetime
from pydantic import ValidationError
import logging
from core.logging_utils import setup_logging

# Initialize global logging to stdout for Azure visibility
setup_logging(level=logging.INFO)

from celery.exceptions import SoftTimeLimitExceeded

from core.celery_app import celery_app
from workers.processor import process_offer
from core.redis_client import redis_manager

logger = logging.getLogger(__name__)


def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()


# soft_time_limit raises SoftTimeLimitExceeded inside the worker → our try/except
# catches it, marks the job "failed", and stops retrying.
# time_limit is the hard kill switch so a truly-stuck worker can't monopolise the
# celery slot on the B1 single-vCPU box.
# M2 file sizes: W23 221 rows ≈ 12 min, Cosmetics 268–281 ≈ 16 min, W39 761 ≈ 50 min,
# STOCK LIST 1043 ≈ 70 min. 90 min soft / 95 min hard covers all of them with buffer.
# Zwolle (2054, held) would need ~2.5h — bump before running that file.
@celery_app.task(bind=True, max_retries=2, soft_time_limit=5400, time_limit=5700)
def process_document_task(self, job_id: str, payload_dict: dict):
    try:
        logger.info(f"Celery processing started for JobID: {job_id}")
        redis_manager.set_job_status(job_id, "processing")

        from schemas.ingest import IngestRequest
        
        try:
            payload = IngestRequest(**payload_dict)
        except ValidationError as e:
            logger.error(f"Failed to parse payload for JobID {job_id}: {e}")
            raise

        loop = get_or_create_eventloop()

        # Hard timeout on the whole extraction (90 min). asyncio.wait_for works
        # regardless of celery pool type, unlike soft_time_limit which is a
        # no-op under --pool=solo. On timeout the task marks the job "failed"
        # and celery is free to pick up the next queued job. Sized for M2:
        # STOCK LIST 1043 rows ≈ 70 min, W39 761 ≈ 50 min. Zwolle 2054 (held)
        # would need ~2.5h — bump this before running it.
        try:
            loop.run_until_complete(
                asyncio.wait_for(process_offer(payload, job_id), timeout=5400.0)
            )
        except asyncio.TimeoutError:
            logger.error(f"Extraction timed out (>90min) for JobID: {job_id}. Marking failed, moving on.")
            redis_manager.set_job_status(job_id, "failed")
            return

        result = redis_manager.get_job_result(job_id)
        if result:
            status = result.get("status")
            total = len(result.get("products", []))
            logger.info(f"Job {job_id} finished processing with status: {status} ({total} products written to Excel)")
        else:
            logger.error(f"Job {job_id} finished but no result was found in RedisManager.")

    except SoftTimeLimitExceeded:
        # 10-minute soft limit hit — extraction was too slow. Mark failed and stop.
        logger.error(f"Celery soft time limit exceeded for JobID: {job_id}. Marking failed, no retry.")
        redis_manager.set_job_status(job_id, "failed")
        return
    except Exception as exc:
        logger.error(f"Processing failed in celery for JobID: {job_id}. Err: {exc}\n{traceback.format_exc()}")
        redis_manager.set_job_status(job_id, "failed")
        raise self.retry(exc=exc, countdown=backoff(self.request.retries))


def backoff(retries: int) -> int:
    return 30 * (2 ** retries)


def _poll_lock(name: str, ttl: int) -> bool:
    """Best-effort single-flight lock so overlapping scheduled polls (e.g. a
    poll queued behind a long ingest) don't double-read the mailbox. Returns
    True if we acquired it. If Redis is unavailable we proceed (dedup still
    protects us)."""
    try:
        if redis_manager.use_redis and redis_manager.client:
            return bool(redis_manager.client.set(name, "1", nx=True, ex=ttl))
    except Exception as e:
        logger.warning(f"poll lock {name} unavailable: {e}")
    return True


@celery_app.task(name="workers.celery_tasks.poll_email_offers")
def poll_email_offers():
    """Pull new offer emails from the IMAP mailbox and enqueue each for
    processing. Runs on the Celery beat schedule. Per-message failures are
    logged and skipped so one bad email never stops the rest; duplicates
    (same filename or same bytes) are refused by core.ingest_service."""
    import hashlib
    from core.imap_client import fetch_new_messages, ImapConfigError
    from core.ingest_service import enqueue_offer, buffer_attachment, DuplicateIngest

    if not _poll_lock("lock:poll_email_offers", ttl=110):
        logger.info("email poll already running; skipping this tick")
        return

    try:
        messages = fetch_new_messages()
    except ImapConfigError as e:
        logger.warning(f"email poll skipped — {e}")
        return
    except Exception as e:
        logger.error(f"email poll failed to fetch: {e}\n{traceback.format_exc()}")
        return

    enqueued = skipped = failed = 0
    for m in messages:
        atts = m.get("attachments") or []
        if not atts:
            logger.info(f"email uid {m['uid']} from {m.get('from_email')} has no "
                        f"attachment — skipping (not an offer file)")
            skipped += 1
            continue
        try:
            built = [buffer_attachment(a["fileName"], a["contentType"], a["bytes"])
                     for a in atts]
            content_hash = hashlib.sha256(atts[0]["bytes"]).hexdigest()
            res = enqueue_offer(
                source_channel="email",
                source_message_id=m.get("message_id") or f"imap-uid-{m['uid']}",
                sender_email=m.get("from_email"),
                sender_name=m.get("from_name"),
                supplier_email=m.get("from_email"),
                subject=m.get("subject"),
                text_body=m.get("text_body"),
                attachments=built,
                content_hash=content_hash,
                check_content_hash=True,
            )
            enqueued += 1
            logger.info(f"email uid {m['uid']} enqueued job {res['job_id']} "
                        f"({len(built)} file(s): {[a['fileName'] for a in atts]})")
        except DuplicateIngest as e:
            skipped += 1
            logger.info(f"email uid {m['uid']} duplicate "
                        f"({e.detail.get('error')}) — skipped")
        except Exception as e:
            failed += 1
            logger.error(f"email uid {m['uid']} enqueue failed: {e}")

    logger.info(f"email poll done: {len(messages)} new, {enqueued} enqueued, "
                f"{skipped} skipped, {failed} failed")
