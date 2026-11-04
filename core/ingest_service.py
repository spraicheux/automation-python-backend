"""core/ingest_service.py

Shared offer-intake plumbing used by every inbound channel:

  - api/ingest.py           — HTTP /ingest (JSON or multipart upload)
  - core/outlook_client.py  — Outlook/Graph email poller
  - api/whatsapp.py         — 360dialog inbound WhatsApp webhook

Builds the IngestRequest payload, applies the SAME source-filename and
content-hash dedup rules, and enqueues `process_document_task`. Keeping this
in one place means a fix to the dedup / filename logic lands for every channel
at once (Ali's rule: fix permanently, in one place, for all ingest paths).

Attachment carriers
-------------------
Two shapes feed the downstream resolver (core.file_download.resolve_attachment_bytes):
  * `buffer_attachment(...)`  — the bytes travel inside the payload
      (`data={"type":"Buffer","data":[...]}`). Used for email attachments and
      multipart uploads, where we already hold the bytes.
  * `media_attachment(...)`   — only the 360dialog media id travels
      (`data={"id": media_id}`); the bytes are fetched lazily at processing time.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class DuplicateIngest(Exception):
    """A file (by name) or its exact bytes (by content hash) were already
    ingested. api/ingest maps this to HTTP 400; the email poller and WhatsApp
    webhook catch it and skip the message (and still mark it handled)."""

    def __init__(self, detail: dict):
        self.detail = detail
        super().__init__(detail.get("message", "duplicate"))


# ─── filename + dedup (one copy, shared by all channels) ──────────────────────

def generate_source_filename(source_filename, attachments, supplier_email,
                             supplier_name, source_message_id, job_id):
    if source_filename:
        return source_filename
    if attachments:
        first = attachments[0]
        name = (first.get("fileName") if isinstance(first, dict)
                else getattr(first, "fileName", None))
        if name:
            return name
    base = supplier_email or supplier_name or "unknown_source"
    base = re.sub(r"[^a-zA-Z0-9@._-]", "_", base).lower()
    unique_part = source_message_id or job_id
    return f"{base}_{unique_part}.txt"


def check_duplicate_filename(filename: str) -> None:
    if not filename:
        return
    try:
        from core.database import get_session_factory
        from models.offer_item import OfferItemDB

        db = get_session_factory()()
        try:
            existing = (db.query(OfferItemDB.uid)
                          .filter(OfferItemDB.source_filename == filename)
                          .first())
            if existing:
                raise DuplicateIngest({
                    "error": "duplicate_file",
                    "message": f"File '{filename}' already processed.",
                    "filename": filename,
                })
        finally:
            db.close()
    except DuplicateIngest:
        raise
    except Exception as e:  # DB hiccup must not block intake
        logger.warning(f"Duplicate filename check failed: {e}")


def check_duplicate_content_hash(content_hash: str) -> None:
    if not content_hash:
        return
    try:
        from core.database import get_session_factory
        from models.source_file import SourceFileDB

        db = get_session_factory()()
        try:
            existing = (db.query(SourceFileDB.source_filename)
                          .filter(SourceFileDB.content_hash == content_hash)
                          .first())
            if existing:
                raise DuplicateIngest({
                    "error": "duplicate_content",
                    "message": (
                        f"File contents already ingested "
                        f"(previously as {existing[0]!r}). "
                        "Re-upload of the exact same bytes is refused. "
                        "If this is a genuinely new supplier offer, "
                        "the file bytes must differ."
                    ),
                    "existing_filename": existing[0],
                    "content_hash": content_hash,
                })
        finally:
            db.close()
    except DuplicateIngest:
        raise
    except Exception as e:
        logger.warning(f"Content-hash duplicate check failed: {e}")


# ─── attachment builders ──────────────────────────────────────────────────────

def buffer_attachment(file_name: str, content_type: Optional[str],
                      file_bytes: bytes) -> Dict[str, Any]:
    """An attachment that carries its own bytes (email / multipart upload)."""
    return {
        "fileName": file_name,
        "contentType": content_type or "application/octet-stream",
        "checksum": hashlib.sha256(file_bytes).hexdigest(),
        "contentId": None,
        "fileSize": len(file_bytes),
        "data": {"type": "Buffer", "data": list(file_bytes)},
    }


def media_attachment(file_name: str, content_type: Optional[str], media_id: str,
                     file_size: int = 0) -> Dict[str, Any]:
    """An attachment whose bytes are fetched lazily from 360dialog by media id."""
    return {
        "fileName": file_name,
        "contentType": content_type or "application/octet-stream",
        "checksum": "",
        "contentId": media_id,
        "fileSize": file_size,
        "data": {"id": media_id},
    }


# ─── the single enqueue path ──────────────────────────────────────────────────

def enqueue_offer(*, source_channel: Optional[str] = None,
                  source_message_id: Optional[str] = None,
                  source_filename: Optional[str] = None,
                  supplier_email: Optional[str] = None,
                  supplier_name: Optional[str] = None,
                  sender_email: Optional[str] = None,
                  sender_name: Optional[str] = None,
                  subject: Optional[str] = None,
                  text_body: Optional[str] = None,
                  attachments: Optional[List[Dict[str, Any]]] = None,
                  content_hash: Optional[str] = None,
                  check_content_hash: bool = True) -> Dict[str, Any]:
    """Validate, dedup, and enqueue ONE offer for processing.

    Returns {"status":"accepted","job_id":..., "source_filename":...}.
    Raises DuplicateIngest on a filename or content-hash collision.

    `check_content_hash=False` stores the hash on the payload but skips the
    content-hash dedup query — used by the JSON /ingest path to preserve its
    historical behaviour (filename dedup only).
    """
    from schemas.ingest import IngestRequest
    from core.redis_client import redis_manager

    attachments = attachments or []
    job_id = str(uuid4())

    dedup_filename = generate_source_filename(
        source_filename=source_filename, attachments=attachments,
        supplier_email=supplier_email, supplier_name=supplier_name,
        source_message_id=source_message_id, job_id=job_id)

    check_duplicate_filename(dedup_filename)
    if check_content_hash:
        check_duplicate_content_hash(content_hash)

    payload_dict = {
        "source_channel": source_channel,
        "source_message_id": source_message_id,
        "source_filename": dedup_filename,
        "supplier_email": supplier_email,
        "supplier_name": supplier_name,
        "sender_email": sender_email,
        "sender_name": sender_name,
        "subject": subject,
        "text_body": text_body,
        "attachments": attachments,
        "content_hash": content_hash,
    }
    # Fail fast on a malformed payload here, not deep inside the worker.
    IngestRequest(**payload_dict)

    redis_manager.set_job_status(job_id, "processing")
    # Lazy import breaks the cycle: workers.celery_tasks → (poller) → this module.
    from workers.celery_tasks import process_document_task
    process_document_task.delay(job_id, payload_dict)

    logger.info(
        f"Enqueued offer job {job_id} via {source_channel or 'unknown'} "
        f"({len(attachments)} attachment(s), file={dedup_filename})")
    return {"status": "accepted", "job_id": job_id,
            "source_filename": dedup_filename}
