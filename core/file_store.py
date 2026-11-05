"""core/file_store.py

Durable storage for the raw bytes of every inbound offer file, so a file is
never lost and any job can be retried from it — even after a restart, a deploy,
a credit outage, or (for WhatsApp) after the 360dialog media has expired.

On Azure App Service, /home is persistent (Azure Files) and survives restarts
and deploys, so that is the default location. Off-Azure it falls back to a local
dir. Set OFFER_FILE_DIR to override.
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


def _dir() -> str:
    d = os.getenv("OFFER_FILE_DIR")
    if d:
        return d
    # Azure App Service: /home is persistent. Elsewhere, a local dir.
    return "/home/offer_files" if os.path.isdir("/home") else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".offer_files")


def _ensure() -> Optional[str]:
    d = _dir()
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except Exception as e:
        logger.warning(f"offer file dir {d} not writable: {e}")
        return None


def save_file(filename: Optional[str], data: bytes, file_id: Optional[str] = None) -> Optional[str]:
    """Persist raw bytes durably; return the stored path (or None on failure)."""
    d = _ensure()
    if not d or not data:
        return None
    fid = file_id or uuid.uuid4().hex
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", (filename or "file"))[:120] or "file"
    path = os.path.join(d, f"{fid}__{safe}")
    try:
        with open(path, "wb") as f:
            f.write(data)
        logger.info(f"saved offer file {path} ({len(data)} bytes)")
        return path
    except Exception as e:
        logger.warning(f"save_file failed for {filename!r}: {e}")
        return None


def load_file(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"load_file failed {path}: {e}")
        return None


def file_exists(path: str) -> bool:
    try:
        return bool(path) and os.path.isfile(path)
    except Exception:
        return False
