import os
from fastapi import APIRouter, HTTPException
import redis
from core.redis_client import redis_manager

router = APIRouter()



@router.get("/result/{job_id}")
async def get_result(job_id: str):
    if not redis_manager.job_exists(job_id):
        raise HTTPException(status_code=404, detail="Unknown job")

    status = redis_manager.get_job_status(job_id)

    if status != "done":
        return {
            "status": status,
            "job_id": job_id
        }

    result = redis_manager.get_job_result(job_id)

    if not result:
        return {
            "status": "processing",
            "job_id": job_id,
            "message": "Result not yet available"
        }

    if "products" in result:
        return {
            "status": "done",
            "job_id": job_id,
            "total_products": result.get("total_products", 0),
            "products": result["products"]
        }
    else:
        return {
            "status": "done",
            "job_id": job_id,
            "data": result
        }