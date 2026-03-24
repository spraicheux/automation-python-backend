import os
import sys
import ssl
import socket
from celery import Celery

# Ensure the project root is on sys.path so that `models`, `workers`, `core`,
# etc. are all importable inside Celery worker processes regardless of how
# the worker was launched (Procfile, Azure container, local shell, etc.).
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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
    task_time_limit=3600,
    broker_connection_retry_on_startup=True,
    broker_use_ssl=ssl_conf,
    redis_backend_use_ssl=ssl_conf,

    broker_heartbeat=0,
    broker_heartbeat_checkrate=2,

    broker_transport_options={
        "visibility_timeout": 3600,
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
    worker_concurrency=1,
)