from fastapi import APIRouter, HTTPException
from workers.state import JOB_RESULTS

router = APIRouter()

from workers.state import JOB_RESULTS, JOB_STATUS

@router.get("/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in JOB_STATUS:
        raise HTTPException(status_code=404, detail="Unknown job")

    if JOB_STATUS[job_id] != "done":
        return {
            "status": JOB_STATUS[job_id]
        }

    return {
        "status": "done",
        "data": JOB_RESULTS[job_id]
    }

