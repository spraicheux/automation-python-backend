from dotenv import load_dotenv
load_dotenv()

from fastapi.responses import FileResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from core.logging_utils import setup_logging

# Initialize global logging to stdout for Azure visibility
setup_logging(level=logging.DEBUG)

from api.ingest import router as ingest_router
from api.results import router as results_router
from api.debug import router as debug_router
from api.records import router as records_router
from api.sources import router as sources_router
from api.auth import router as auth_router
from api.offers_write import router as offers_write_router
from api.alerts import router as alerts_router
from api.events import router as events_router
from api.admin import router as admin_router
from api.whatsapp import router as whatsapp_router


app = FastAPI(
    title="Python Ingestion Backend",
    version="1.0.0",
    description="Async backend for Make.com-driven Email & WhatsApp offer processing",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router, prefix="/api", tags=["Ingest"])
app.include_router(results_router, prefix="/api", tags=["Result"])
app.include_router(records_router, prefix="/api", tags=["Records"])
app.include_router(debug_router, prefix="/debug", tags=["debug"])
app.include_router(sources_router, prefix="/api", tags=["Sources"])
app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
app.include_router(offers_write_router, prefix="/api", tags=["Offers-Write"])
app.include_router(alerts_router, prefix="/api", tags=["Alerts"])
app.include_router(events_router, prefix="/api", tags=["Events"])
app.include_router(admin_router, prefix="/api", tags=["Admin"])
app.include_router(whatsapp_router, prefix="/api", tags=["WhatsApp"])


@app.on_event("startup")
async def startup_event():
    """Initialise the database and create tables if they don't exist."""
    try:
        from core.database import init_db
        init_db()
    except Exception as e:
        logging.getLogger(__name__).error(f"DB init failed on startup: {e}")


@app.on_event("startup")
async def start_email_poller():
    """Beat-independent IMAP poll. Prod starts from a Procfile with no celery
    beat, so we run the poll from the web app (which is always up). The Redis
    single-flight lock in run_email_poll makes it safe across the 2 gunicorn
    workers (and alongside beat, if it ever runs). Only active when IMAP_HOST
    is configured. imaplib is blocking, so it runs in a thread executor."""
    import os
    import asyncio

    if not os.getenv("IMAP_HOST"):
        return
    interval = float(os.getenv("IMAP_POLL_SECONDS", "120"))
    log = logging.getLogger(__name__)

    async def _loop():
        from core.email_poll import run_email_poll
        loop = asyncio.get_running_loop()
        await asyncio.sleep(20)  # let the app finish booting
        while True:
            try:
                await loop.run_in_executor(None, run_email_poll)
            except Exception as e:
                log.error(f"email poll loop error: {e}")
            await asyncio.sleep(interval)

    asyncio.create_task(_loop())
    logging.getLogger(__name__).info(
        f"email poll scheduler started (every {interval}s)")


@app.on_event("startup")
async def start_credit_resumer():
    """Auto-resume jobs parked for lack of OpenAI credits. Every ~2 min, if any
    jobs are parked, probe OpenAI; once credits are back, re-queue them. Runs
    regardless of IMAP config. A Redis lock keeps the 2 gunicorn workers from
    probing/draining at once."""
    import asyncio
    log = logging.getLogger(__name__)

    async def _loop():
        from core.redis_client import redis_manager
        from core.openai_health import pending_count, probe_openai, drain_pending_credit_jobs
        loop = asyncio.get_running_loop()
        await asyncio.sleep(30)
        while True:
            try:
                if pending_count() > 0:
                    lock_ok = True
                    try:
                        if redis_manager.use_redis and redis_manager.client:
                            lock_ok = bool(redis_manager.client.set(
                                "lock:credit_resume", "1", nx=True, ex=100))
                    except Exception:
                        lock_ok = True
                    if lock_ok:
                        ok = await loop.run_in_executor(None, probe_openai)
                        if ok:
                            n = await loop.run_in_executor(None, drain_pending_credit_jobs)
                            if n:
                                log.info(f"credit resumer: re-queued {n} parked job(s)")
            except Exception as e:
                log.error(f"credit resumer error: {e}")
            await asyncio.sleep(120)

    asyncio.create_task(_loop())
    logging.getLogger(__name__).info("OpenAI credit-resume scheduler started")


@app.on_event("startup")
async def start_job_sweeper():
    """Self-healing recovery for stranded jobs. Every ~2 min, re-enqueue any
    intake row stuck 'queued' because its broker hand-off never landed (how
    files silently went missing), and clear stale errors on resumed/done rows.
    A Redis lock keeps the 2 gunicorn workers from sweeping at once."""
    import asyncio
    log = logging.getLogger(__name__)

    async def _loop():
        from core.redis_client import redis_manager
        from core.job_recovery import sweep_stuck_jobs
        loop = asyncio.get_running_loop()
        await asyncio.sleep(45)  # after boot, and offset from the other loops
        while True:
            try:
                lock_ok = True
                try:
                    if redis_manager.use_redis and redis_manager.client:
                        lock_ok = bool(redis_manager.client.set(
                            "lock:job_sweep", "1", nx=True, ex=100))
                except Exception:
                    lock_ok = True
                if lock_ok:
                    await loop.run_in_executor(None, sweep_stuck_jobs)
            except Exception as e:
                log.error(f"job sweeper error: {e}")
            await asyncio.sleep(120)

    asyncio.create_task(_loop())
    logging.getLogger(__name__).info("job-recovery sweeper started")


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "service": "automation-python-backend",
        "version": "1.0.0",
    }

@app.get("/", include_in_schema=False)
async def serve_dashboard():
    return FileResponse("dashboard.html")

