from uuid import uuid4

import logging
from fastapi import APIRouter, Body, HTTPException
from fastapi import UploadFile, File, Form, Request
from typing import List, Optional
from schemas.ingest import IngestRequest
from workers.celery_tasks import process_document_task
from core.redis_client import redis_manager

logger = logging.getLogger(__name__)


router = APIRouter()


def _check_duplicate_filename(filename: str) -> None:
    """
    Query the database for an existing record with this source_filename.
    Raises HTTP 400 immediately if a duplicate is found — BEFORE any Celery
    task is dispatched, so no AI tokens are consumed.
    """
    if not filename:
        return  # blank filenames are not tracked for uniqueness

    try:
        from core.database import get_session_factory
        from models.offer_item import OfferItemDB

        factory = get_session_factory()
        db = factory()
        try:
            existing = (
                db.query(OfferItemDB.uid)
                .filter(OfferItemDB.source_filename == filename)
                .first()
            )
            if existing:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "duplicate_file",
                        "message": f"File '{filename}' has already been processed. "
                                   "Use GET /api/records?filename=<name> to retrieve the existing data.",
                        "filename": filename,
                    },
                )
        finally:
            db.close()
    except HTTPException:
        raise  # re-raise 400 as-is
    except Exception as e:
        # DB unavailable — log and continue (degrade gracefully)
        logger.warning(f"Duplicate filename check failed (DB unavailable?): {e}. Proceeding without check.")


@router.post("/ingest")
async def ingest(
    request: Request,
    payload: Optional[dict] = Body(None),

    source_channel: Optional[str] = Form(None),
    source_message_id: Optional[str] = Form(None),
    source_filename: Optional[str] = Form(None),
    supplier_email: Optional[str] = Form(None),
    supplier_name: Optional[str] = Form(None),
    sender_email: Optional[str] = Form(None),
    sender_name: Optional[str] = Form(None),
    subject: Optional[str] = Form(None),
    text_body: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
):
    job_id = str(uuid4())

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        if payload is None:
            return {"error": "Invalid JSON body"}

        validated_payload = IngestRequest(**payload)

        # ── Duplicate check (JSON path) ──────────────────────────────────────
        check_filename = validated_payload.source_filename
        if not check_filename and validated_payload.attachments:
            # Use the first attachment filename as the dedup key
            first = validated_payload.attachments[0]
            check_filename = getattr(first, "fileName", None)
        _check_duplicate_filename(check_filename)

        redis_manager.set_job_status(job_id, "processing")
        process_document_task.delay(job_id, validated_payload.model_dump())

        return {
            "status": "accepted",
            "job_id": job_id,
            "mode": "json"
        }

    attachments = []

    if files:
        for file in files:
            file_bytes = await file.read()
            if not file_bytes or not file.filename or file.filename.lower() in ("empty", ""):
                logger.info(f"Skipping empty file attachment: {file.filename!r}")
                continue
            attachments.append({
                "fileName": file.filename,
                "contentType": file.content_type,
                "checksum": "",
                "contentId": None,
                "fileSize": len(file_bytes),
                "data": {
                    "type": "Buffer",
                    "data": list(file_bytes)
                }
            })

    # ── Duplicate check (multipart path) ────────────────────────────────────
    # Check source_filename first; fall back to first attachment filename
    dedup_filename = source_filename
    if not dedup_filename and attachments:
        dedup_filename = attachments[0].get("fileName")
    _check_duplicate_filename(dedup_filename)

    payload_dict = {
        "source_channel": source_channel,
        "source_message_id": source_message_id,
        "source_filename": source_filename,
        "supplier_email": supplier_email,
        "supplier_name": supplier_name,
        "sender_email": sender_email,
        "sender_name": sender_name,
        "subject": subject,
        "text_body": text_body,
        "attachments": attachments
    }

    redis_manager.set_job_status(job_id, "processing")
    process_document_task.delay(job_id, payload_dict)

    return {
        "status": "accepted",
        "job_id": job_id,
        "mode": "multipart"
    }