#!/bin/bash
# Production startup for the single Azure App Service container.
#
# Replicates the exact inline appCommandLine params (celery solo worker +
# gunicorn), but SUPERVISES the Celery worker: if it ever exits — a transient
# Redis blip, an OOM, or a code error — it is restarted instead of being left
# silently dead while gunicorn keeps serving. On 2026-11-08 an unsupervised
# `celery … &` died at boot and ingestion was down (jobs queued, nothing
# processed) until a manual redeploy. This loop prevents that class of outage.
#
# Azure appCommandLine is set to: bash start-prod.sh

set -u

# ── Supervised Celery worker (background) ───────────────────────────────────
(
  while true; do
    echo "[start-prod] starting celery worker..."
    celery -A core.celery_app.celery_app worker --loglevel=info --pool=solo --concurrency=1 || true
    echo "[start-prod] celery worker exited (code $?); restarting in 3s"
    sleep 3
  done
) &

# ── Gunicorn / FastAPI (foreground — keeps the container alive) ──────────────
exec gunicorn main:app -k uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000 --workers=1
