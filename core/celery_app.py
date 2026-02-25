import os
import ssl
from celery import Celery

broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", broker_url)

celery_app = Celery(
    "automation_backend",
    broker=broker_url,
    backend=result_backend,
    include=["workers.celery_tasks"]
)

# SSL configuration for Azure Cache for Redis
# Note: Use rediss:// in your environment variables to trigger this
ssl_conf = None
if broker_url.startswith('rediss://'):
    ssl_conf = {
        'ssl_cert_reqs': ssl.CERT_NONE
    }

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    broker_connection_retry_on_startup=True,
    broker_use_ssl=ssl_conf,
    redis_backend_use_ssl=ssl_conf,
)