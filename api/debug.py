from fastapi import APIRouter
from core.redis_client import redis_manager

router = APIRouter()


@router.get("/debug/job/{job_id}")
async def debug_job(job_id: str):
    status = redis_manager.get_job_status(job_id)
    result = redis_manager.get_job_result(job_id)

    return {
        "job_id": job_id,
        "status": status,
        "result": result,
        "exists": redis_manager.job_exists(job_id)
    }


@router.get("/debug/jobs")
async def debug_all_jobs():
    return {
        "message": "For Redis, use SCAN command. For in-memory storage, all jobs are listed.",
        "storage_type": "redis" if redis_manager.use_redis else "memory"
    }