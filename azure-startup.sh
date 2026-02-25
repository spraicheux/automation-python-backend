#!/bin/bash

# Start Celery worker in the background
# We use --loglevel=info and logs will be captured by Azure through stdout
echo "Starting Celery worker..."
python -m celery -A core.celery_app worker --loglevel=info &

# Start Gunicorn (FastAPI) as the main process
echo "Starting Gunicorn server..."
gunicorn main:app -k uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000 --workers=2
