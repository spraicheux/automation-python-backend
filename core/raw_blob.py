"""core/raw_blob.py

Durable, database-backed archive of the raw bytes of every inbound offer file.

Why this exists (Samuel 2026-11-11): /home on this App Service is ephemeral —
WEBSITES_ENABLE_APP_SERVICE_STORAGE is unset — so core.file_store's
/home/offer_files is wiped on every restart and deploy. We redeployed ~6 times
on 10 Nov, which is why the 8 CHELSEA images and the Alder NE410 PDF could not
be recovered. Text/body sources already survive in source_files.source_text;
this closes the gap for BINARY attachments by storing the bytes in Postgres,
which is durable across restarts, deploys, credit outages and media expiry.

Best-effort: a failure here NEVER breaks ingestion — the extractor still runs
on the in-memory bytes; only the durable copy is skipped, with a warning.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Guard: perfume sheets / PDFs are small; refuse to archive anything absurd so a
# rogue upload can't bloat the DB. 25 MB comfortably covers real offer files.
_MAX_BYTES = 25 * 1024 * 1024


def save_blob(job_id: Optional[str], filename: Optional[str],
              content_type: Optional[str], data: Optional[bytes]) -> Optional[str]:
    """Persist raw bytes durably in Postgres; return the row id (or None).

    Idempotent per (job_id, sha256): re-processing the same file does not create
    a second copy."""
    if not data:
        return None
    if len(data) > _MAX_BYTES:
        logger.warning(f"raw_blob: {filename!r} too large ({len(data)} bytes) — not archived")
        return None
    try:
        from core.database import get_session_factory
        from models.raw_blob import RawBlobDB
        sha = hashlib.sha256(data).hexdigest()
        db = get_session_factory()()
        try:
            existing = (db.query(RawBlobDB)
                        .filter(RawBlobDB.sha256 == sha,
                                RawBlobDB.job_id == job_id)
                        .first())
            if existing:
                return existing.id
            row = RawBlobDB(job_id=job_id, source_filename=filename,
                            content_type=content_type, size_bytes=len(data),
                            sha256=sha, data=data)
            db.add(row)
            db.commit()
            logger.info(f"raw_blob archived {filename!r} ({len(data)} bytes) "
                        f"job={job_id} id={row.id}")
            return row.id
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"raw_blob save failed for {filename!r}: {e}")
        return None


def load_blob(blob_id: Optional[str] = None, job_id: Optional[str] = None,
              filename: Optional[str] = None) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """Return (data, content_type, filename) for the most recent matching blob,
    or (None, None, None)."""
    try:
        from core.database import get_session_factory
        from models.raw_blob import RawBlobDB
        db = get_session_factory()()
        try:
            q = db.query(RawBlobDB)
            if blob_id:
                q = q.filter(RawBlobDB.id == blob_id)
            if job_id:
                q = q.filter(RawBlobDB.job_id == job_id)
            if filename:
                q = q.filter(RawBlobDB.source_filename.ilike(f"%{filename}%"))
            row = q.order_by(RawBlobDB.created_at.desc()).first()
            if not row:
                return (None, None, None)
            return (row.data, row.content_type, row.source_filename)
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"raw_blob load failed: {e}")
        return (None, None, None)


def list_blobs(limit: int = 300, pattern: Optional[str] = None) -> list:
    """Metadata for archived blobs (no bytes), newest first."""
    try:
        from core.database import get_session_factory
        from models.raw_blob import RawBlobDB
        db = get_session_factory()()
        try:
            rows = (db.query(RawBlobDB)
                    .order_by(RawBlobDB.created_at.desc())
                    .limit(limit).all())
            out = []
            for r in rows:
                if pattern and pattern.lower() not in (r.source_filename or "").lower():
                    continue
                out.append(r.to_meta())
            return out
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"raw_blob list failed: {e}")
        return []
