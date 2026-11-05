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
from core.intake_log import update_intake
from core.openai_health import OpenAICreditsExhausted, queue_pending_credit_job

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
        update_intake(job_id, status="processing")

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
            update_intake(job_id, status="failed", error="extraction timed out (>90min)")
            return

        result = redis_manager.get_job_result(job_id)
        if result:
            status = result.get("status")
            total = len(result.get("products", []))
            logger.info(f"Job {job_id} finished processing with status: {status} ({total} products written to Excel)")
            # clear_error: a job that was parked 'OpenAI out of credits' and then
            # resumed must not keep showing the stale outage error once it's done.
            update_intake(job_id, status="done", products=total, clear_error=True)
        else:
            logger.error(f"Job {job_id} finished but no result was found in RedisManager.")
            update_intake(job_id, status="done", products=0, clear_error=True)

    except OpenAICreditsExhausted as exc:
        # Out of OpenAI credits — park the job (don't mark done/failed). The
        # in-app scheduler re-queues it automatically once credits are back.
        logger.warning(f"Job {job_id} parked — OpenAI out of credits: {exc}")
        redis_manager.set_job_status(job_id, "no_credits")
        update_intake(job_id, status="no_credits",
                      error="OpenAI out of credits — will auto-retry when topped up")
        queue_pending_credit_job(job_id, payload_dict)
        return
    except SoftTimeLimitExceeded:
        # 10-minute soft limit hit — extraction was too slow. Mark failed and stop.
        logger.error(f"Celery soft time limit exceeded for JobID: {job_id}. Marking failed, no retry.")
        redis_manager.set_job_status(job_id, "failed")
        update_intake(job_id, status="failed", error="soft time limit exceeded")
        return
    except Exception as exc:
        logger.error(f"Processing failed in celery for JobID: {job_id}. Err: {exc}\n{traceback.format_exc()}")
        redis_manager.set_job_status(job_id, "failed")
        update_intake(job_id, status="failed", error=exc)
        raise self.retry(exc=exc, countdown=backoff(self.request.retries))


def backoff(retries: int) -> int:
    return 30 * (2 ** retries)


@celery_app.task(name="workers.celery_tasks.poll_email_offers")
def poll_email_offers():
    """Celery-beat entry point for the IMAP poll. The actual logic lives in
    core.email_poll.run_email_poll so the in-app scheduler (main.py) can run
    the exact same poll without depending on beat being started on prod."""
    from core.email_poll import run_email_poll
    return run_email_poll()
