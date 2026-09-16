import os
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


def _mask(value: str, keep: int = 6) -> str:
    if not value:
        return None
    if len(value) <= keep:
        return "*" * len(value)
    return value[:keep] + "…"


@router.get("/debug/status")
async def debug_status():
    """Diagnostic snapshot for deploy verification — no secrets returned."""
    out = {"redis": {}, "db": {}, "excel": {}, "env": {}}

    # Redis
    try:
        out["redis"]["connected"] = bool(redis_manager.use_redis and redis_manager.client)
        if redis_manager.use_redis and redis_manager.client:
            out["redis"]["ping"] = redis_manager.client.ping()
    except Exception as e:
        out["redis"]["error"] = str(e)

    # Database
    try:
        from core.database import get_engine
        from sqlalchemy import text
        eng = get_engine()
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        out["db"]["reachable"] = True
        out["db"]["url_host"] = (os.getenv("DATABASE_URL") or "").split("@")[-1].split("/")[0] or None
    except Exception as e:
        out["db"]["reachable"] = False
        out["db"]["error"] = str(e)[:200]

    # Excel env vars (presence only, not values)
    for key in ("MS_CLIENT_ID", "MS_AUTHORITY", "EXCEL_WORKBOOK_PATH",
                "EXCEL_WORKSHEET", "MSAL_CACHE_KEY", "OPENAI_API_KEY", "REDIS_URL", "DATABASE_URL"):
        val = os.getenv(key)
        out["env"][key] = {
            "set": bool(val),
            "preview": _mask(val) if val else None,
        }

    # MSAL cache present?
    try:
        r = redis_manager.client if (redis_manager.use_redis and redis_manager.client) else None
        cache_key = os.getenv("MSAL_CACHE_KEY", "msal:cache:onedrive")
        if r:
            blob = r.get(cache_key)
            out["excel"]["msal_cache_present"] = bool(blob)
            out["excel"]["msal_cache_bytes"] = len(blob) if blob else 0
        else:
            out["excel"]["msal_cache_present"] = False
    except Exception as e:
        out["excel"]["msal_cache_error"] = str(e)[:200]

    # Can we acquire a Graph access token silently?
    try:
        from core.excel_client import get_excel_client
        c = get_excel_client()
        tok = c.get_access_token()
        out["excel"]["token_acquired"] = bool(tok)
        out["excel"]["token_len"] = len(tok) if tok else 0
    except Exception as e:
        out["excel"]["token_acquired"] = False
        out["excel"]["token_error"] = str(e)[:200]

    # Can we probe the workbook?
    if out["excel"].get("token_acquired"):
        try:
            import requests
            from core.excel_client import get_excel_client
            c = get_excel_client()
            r = requests.get(c._workbook_url(), headers=c._headers(), timeout=15)
            out["excel"]["workbook_probe_status"] = r.status_code
            out["excel"]["workbook_probe_ok"] = r.ok
            if not r.ok:
                out["excel"]["workbook_probe_body"] = r.text[:300]
        except Exception as e:
            out["excel"]["workbook_probe_error"] = str(e)[:200]

    return out