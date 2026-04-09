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


@app.on_event("startup")
async def startup_event():
    """Initialise the database and create tables if they don't exist."""
    try:
        from core.database import init_db
        init_db()
    except Exception as e:
        logging.getLogger(__name__).error(f"DB init failed on startup: {e}")


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

