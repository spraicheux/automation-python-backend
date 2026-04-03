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

from core.celery_app import celery_app
from workers.processor import process_offer
from core.redis_client import redis_manager

logger = logging.getLogger(__name__)

WEBHOOK_URL = "https://hook.eu1.make.com/gxhv22brpghf60o7rjff8l8kxuoagybx"


def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()


@celery_app.task(bind=True, max_retries=5)
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
        
        # We reuse the existing core processing logic which returns nothing but updates Redis with the final "done" status and "result_data" mapping
        loop.run_until_complete(process_offer(payload, job_id))

        result = redis_manager.get_job_result(job_id)
        if result:
            status = result.get("status")
            logger.info(f"Job {job_id} finished processing with status: {status}")
            
            # Send a final 'job_completed' summary webhook
            from core.webhook_client import send_consolidated_webhook
            send_consolidated_webhook(
                job_id=job_id,
                payload_type="job_summary",
                data={"status": status, "total_extracted": len(result.get("products", []))}
            )
        else:
            logger.error(f"Job {job_id} finished but no result was found in RedisManager.")

    except Exception as exc:
        logger.error(f"Processing failed in celery for JobID: {job_id}. Err: {exc}\n{traceback.format_exc()}")
        redis_manager.set_job_status(job_id, "failed")
        raise self.retry(exc=exc, countdown=backoff(self.request.retries))


# @celery_app.task(bind=True, max_retries=6)
# def send_webhook_with_retry(self, job_id: str, result: dict = None):
#     """Legacy/Fallback task - now mostly handled sequentially in processor.py"""
#     from core.webhook_client import send_consolidated_webhook
#     success = send_consolidated_webhook(
#         job_id=job_id,
#         payload_type="batch_retry",
#         data={"results": result} if result else {}
#     )
#     if not success:
#         raise self.retry(countdown=30 * (2 ** self.request.retries))

def backoff(retries: int) -> int:
    return 30 * (2 ** retries)
