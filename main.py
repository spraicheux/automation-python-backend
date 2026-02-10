from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.ingest import router as ingest_router
from api.results import router as results_router
from api.debug import router as debug_router


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
app.include_router(debug_router, prefix="/debug", tags=["debug"])


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "service": "automation-python-backend",
        "version": "1.0.0",
    }
