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

