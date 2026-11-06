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
from datetime import datetime
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


# ─── broker-delivery flag ─────────────────────────────────────────────────────
# Set the instant a job is successfully handed to the Celery broker. The
# job_recovery sweeper uses it to tell a genuine orphan (recorded "queued" but
# the enqueue never reached the broker — how files silently went missing) from
# a job that IS in the broker and just waiting its turn. Only jobs WITHOUT this
# flag are safe to re-enqueue, so recovery can never double-run a live job.
_DELIVERED_KEY = "intake:delivered:"
_DELIVERED_TTL = 7 * 24 * 3600


def mark_delivered(job_id: str) -> None:
    from core.redis_client import redis_manager
    try:
        if redis_manager.use_redis and redis_manager.client:
            redis_manager.client.set(_DELIVERED_KEY + job_id, "1", ex=_DELIVERED_TTL)
    except Exception:
        pass


def was_delivered(job_id: str) -> bool:
    from core.redis_client import redis_manager
    try:
        if redis_manager.use_redis and redis_manager.client:
            return bool(redis_manager.client.get(_DELIVERED_KEY + job_id))
    except Exception:
        pass
    return False


# ─── supplier detection (mirrors the Make normalization prompts) ──────────────
# Supplier is derived ONLY from the message/email text (and, for email, the
# subject) using explicit markers/patterns — NEVER from the sender, the
# filename, or the attachment. If nothing matches, supplier is None (no
# fallback), exactly as the Make scenario produced it.

# An explicit supplier designation the user typed in the message ("Supplier: X",
# "Fournisseur - X", "Vendor = X", …). This is Samuel manually naming the
# commercial supplier on a forward and it MUST win over any quoted From:/De:
# header of the original sender. Multi-language + colon/dash/equals separators.
_SUPPLIER_LINE = re.compile(
    r'^[>\s]*(?:supplier|fournisseur|fournisseure|lieferant|fornitore|proveedor|'
    r'vendor|seller|supp)\s*[:\-=]\s*(.+)$',
    re.IGNORECASE)
_SUPPLIER_PATTERNS = [
    re.compile(r'\boffer\s+from\s+(.+)$', re.IGNORECASE),
    re.compile(r'\boffre\s+de\s+(.+)$', re.IGNORECASE),
    re.compile(r'\boffree\s+de\s+(.+)$', re.IGNORECASE),
    # Forwarded-email headers inside the body — "From:" (EN), "De:" (FR),
    # "Von:" (DE). A leading ">" quote marker is common on a forward. The
    # original supplier lives here, distinct from the person forwarding.
    re.compile(r'^[>\s]*from\s*:\s*(.+)$', re.IGNORECASE),
    re.compile(r'^[>\s]*de\s*:\s*(.+)$', re.IGNORECASE),
    re.compile(r'^[>\s]*von\s*:\s*(.+)$', re.IGNORECASE),
]


def _clean_supplier(name: str) -> Optional[str]:
    name = (name or "").strip().strip('"\'')
    # "FBC Trades <connect@mail.fbctradeholdings.com>" → "FBC Trades": keep the
    # display name, drop the <email>. A bare "<email>" (no name) becomes "".
    name = re.sub(r'<[^>]*>', '', name).strip().strip('"\'')
    name = re.sub(r'[\.,;:]+$', '', name).strip()
    # Reject a left-over email address or an obviously-not-a-name capture.
    if not name or "@" in name or len(name) > 120:
        return None
    return name


def detect_supplier_from_text(*texts: Optional[str]) -> Optional[str]:
    """Detect a supplier name from one or more text blocks (e.g. body then
    subject), honoring the Make rules: an explicit `SUPPLIER:` marker wins,
    otherwise the `OFFER FROM` / `Offre de` / `From:` patterns, taking ALL
    words that follow. Returns None when nothing is clearly identifiable."""
    blocks = [t for t in texts if t]
    # Pass 1: explicit SUPPLIER: marker, anywhere.
    for t in blocks:
        for line in t.splitlines():
            m = _SUPPLIER_LINE.match(line)
            if m:
                name = _clean_supplier(m.group(1))
                if name:
                    return name
    # Pass 2: OFFER FROM / Offre de / From: patterns.
    for t in blocks:
        for line in t.splitlines():
            for pat in _SUPPLIER_PATTERNS:
                m = pat.search(line)
                if m:
                    name = _clean_supplier(m.group(1))
                    if name:
                        return name
    return None


# ─── filename + dedup (one copy, shared by all channels) ──────────────────────

# Client §2/§7 (2026-11-06): a signature / decorative image (Outlook inline
# "image001.gif", a logo, a footer banner) is NOT an offer source. It must
# never become the source_filename and must never be sent for extraction —
# the ElixeMarket offers were wrongly attributed to image001.gif (the LOXO
# signature logo) because it happened to be attachments[0].
_SIG_NAME_RE = re.compile(
    r"(?i)^(?:image|img|logo|signature|sig|footer|banner|icon|header|"
    r"outlook-[-\w]*)[-_ ]?\d*\.(?:gif|png|jpe?g|bmp|svg|tiff?)$"
)


def _att_get(att, key, default=None):
    """Read a field from an attachment that may be a dict or an object."""
    if isinstance(att, dict):
        return att.get(key, default)
    return getattr(att, key, default)


def is_signature_attachment(att) -> bool:
    """True when an attachment is a signature / decorative image rather than a
    real offer document. Conservative: a genuine offer photo (a reasonably
    sized jpg/png with no inline Content-ID and a non-boilerplate name) is NOT
    treated as a signature, so it still reaches vision extraction."""
    name = str(_att_get(att, "fileName", "") or "").strip()
    ctype = str(_att_get(att, "contentType", "") or "").lower()
    size = _att_get(att, "fileSize", 0) or 0
    cid = _att_get(att, "contentId")
    lname = name.lower()
    is_image = ctype.startswith("image/") or lname.endswith(
        (".gif", ".png", ".jpg", ".jpeg", ".bmp", ".svg", ".tif", ".tiff"))
    if not is_image:
        return False
    # Auto-generated inline names (Outlook image001.gif, logo.png, …).
    if _SIG_NAME_RE.match(name):
        return True
    # GIFs are effectively never a product offer — signatures/banners are.
    if ctype == "image/gif" or lname.endswith(".gif"):
        return True
    # Inline image (has a Content-ID) that is small → decorative signature.
    try:
        if cid and size and int(size) < 50_000:
            return True
    except (TypeError, ValueError):
        pass
    return False


def generate_source_filename(source_filename, attachments, supplier_email,
                             supplier_name, source_message_id, job_id):
    if source_filename:
        return source_filename
    # Skip signature/decorative images so a logo never poses as the source.
    for att in (attachments or []):
        if is_signature_attachment(att):
            continue
        name = _att_get(att, "fileName")
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


def file_attachment(file_name: str, content_type: Optional[str], path: str,
                    file_size: int = 0, checksum: str = "") -> Dict[str, Any]:
    """An attachment whose bytes live in durable server storage (core.file_store).
    The job carries only the path, so payloads stay tiny and the file survives
    restarts / deploys / credit outages and can be retried anytime."""
    return {
        "fileName": file_name,
        "contentType": content_type or "application/octet-stream",
        "checksum": checksum or "",
        "contentId": None,
        "fileSize": file_size,
        "data": {"path": path},
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
        # client §6 — stamp arrival ONCE so a reprocess keeps the original
        # receipt time (age must not restart when the saved payload is re-run).
        "received_at": datetime.utcnow().isoformat(),
    }
    # Fail fast on a malformed payload here, not deep inside the worker.
    IngestRequest(**payload_dict)

    redis_manager.set_job_status(job_id, "processing")

    # Remember the payload (tiny now — it references saved files) so the job can
    # be re-queued later: credit-resume, /reprocess, or after any failure.
    try:
        from core.job_store import save_job_payload
        save_job_payload(job_id, payload_dict)
    except Exception as e:
        logger.warning(f"save_job_payload failed for {job_id}: {e}")

    # Instant, dashboard-visible intake record (received → processing → done/failed).
    try:
        from core.intake_log import record_intake
        record_intake(job_id, channel=source_channel,
                      source_filename=dedup_filename, sender_name=sender_name,
                      sender_email=sender_email, supplier_name=supplier_name,
                      attachment_count=len(attachments),
                      has_text=bool(text_body), status="queued")
    except Exception as e:  # visibility must never block ingest
        logger.warning(f"intake record failed for {job_id}: {e}")

    # Hand the job to the Celery worker. A transient broker/Redis blip must NOT
    # silently orphan the job at "queued" (which is exactly how files went
    # missing): retry the enqueue a few times, flag it delivered on success, and
    # if it still fails mark the intake FAILED (visible) instead of leaving a
    # fake "queued". The payload is already saved, so the job_recovery sweeper
    # re-enqueues it either way — no file is ever lost.
    # Lazy import breaks the cycle: workers.celery_tasks → (poller) → this module.
    from workers.celery_tasks import process_document_task
    try:
        process_document_task.apply_async(
            args=[job_id, payload_dict],
            retry=True,
            retry_policy={"max_retries": 3, "interval_start": 0.3,
                          "interval_step": 0.5, "interval_max": 2.0},
        )
        mark_delivered(job_id)
    except Exception as e:
        logger.error(f"enqueue to worker failed for {job_id} ({dedup_filename}): {e}")
        try:
            from core.intake_log import update_intake
            update_intake(job_id, status="failed",
                          error="could not reach the processing queue — will auto-retry")
        except Exception:
            pass
        # Deliberately not re-raised: the saved payload + sweeper recover it.

    logger.info(
        f"Enqueued offer job {job_id} via {source_channel or 'unknown'} "
        f"({len(attachments)} attachment(s), text={bool(text_body)}, "
        f"file={dedup_filename})")
    return {"status": "accepted", "job_id": job_id,
            "source_filename": dedup_filename}
