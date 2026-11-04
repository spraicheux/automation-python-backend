import os
import ssl
import socket
from celery import Celery


broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", broker_url)

celery_app = Celery(
    "automation_backend",
    broker=broker_url,
    backend=result_backend,
    include=["workers.celery_tasks"]
)

ssl_conf = None
if broker_url.startswith("rediss://"):
    ssl_conf = {
        "ssl_cert_reqs": ssl.CERT_NONE
    }

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # Task cap must be ≥ the per-task time_limit on process_document_task
    # (currently 5700 for M2). An hour was too tight for M2's larger files.
    task_time_limit=5700,
    broker_connection_retry_on_startup=True,
    broker_use_ssl=ssl_conf,
    redis_backend_use_ssl=ssl_conf,

    broker_heartbeat=0,
    broker_heartbeat_checkrate=2,

    broker_transport_options={
        # visibility_timeout must be ≥ the longest task time_limit so a long-
        # running M2 ingest (STOCK LIST ≈ 70 min) is not re-queued mid-run.
        "visibility_timeout": 6000,
        "socket_keepalive": True,
        "socket_keepalive_options": {
            # socket.TCP_KEEPIDLE: 60,
            socket.TCP_KEEPINTVL: 10,
            socket.TCP_KEEPCNT: 6,
        },
        "socket_connect_timeout": 10,
        "socket_timeout": 30,
        "retry_on_timeout": True,
    },

    redis_socket_keepalive=True,

    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=10,
    # Process multiple files at once (any file type). Env-tunable and reversible
    # without a code change; 2 is safe on the B1 box (extraction is I/O-bound,
    # waiting on OpenAI). Set CELERY_WORKER_CONCURRENCY=1 to revert to serial.
    worker_concurrency=int(os.getenv("CELERY_WORKER_CONCURRENCY", "2")),
)

# Local / isolated-run escape hatch: when CELERY_TASK_ALWAYS_EAGER=1, tasks run
# synchronously in-process, so you don't need a separate worker. Prod leaves it
# unset and the worker keeps running tasks via the broker as normal.
if os.getenv("CELERY_TASK_ALWAYS_EAGER") == "1":
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True

# Email intake runs on the Celery beat schedule (a single scheduler process —
# unlike an in-app timer, which would double-fire across the 2 gunicorn
# workers). Only registered when IMAP is configured, so the beat stays quiet
# otherwise. `expires` drops a poll that could not run within its interval
# (e.g. queued behind a long ingest) instead of letting ticks pile up.
if os.getenv("IMAP_HOST"):
    _IMAP_POLL_SECONDS = float(os.getenv("IMAP_POLL_SECONDS", "120"))
    celery_app.conf.beat_schedule = {
        "poll-email-offers": {
            "task": "workers.celery_tasks.poll_email_offers",
            "schedule": _IMAP_POLL_SECONDS,
            "options": {"expires": _IMAP_POLL_SECONDS},
        }
    }
