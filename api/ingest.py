from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks
from schemas.ingest import IngestRequest
from workers.processor import process_offer
from core.redis_client import redis_manager


router = APIRouter()


@router.post("/ingest")
async def ingest(payload: IngestRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid4())

    redis_manager.set_job_status(job_id, "processing")

    background_tasks.add_task(
        process_offer,
        payload,
        job_id
    )

    return {
        "status": "accepted",
        "job_id": job_id,
        "message": f"Job {job_id} has been queued for processing"
    }