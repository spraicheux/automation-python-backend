from uuid import uuid4
import logging
import re

from fastapi import APIRouter, Body, HTTPException
from fastapi import UploadFile, File, Form, Request
from typing import List, Optional

from schemas.ingest import IngestRequest
from workers.celery_tasks import process_document_task
from core.redis_client import redis_manager

logger = logging.getLogger(__name__)
router = APIRouter()


def generate_source_filename(
    source_filename,
    attachments,
    supplier_email,
    supplier_name,
    source_message_id,
    job_id
):
    if source_filename:
        return source_filename

    if attachments:
        return attachments[0].get("fileName")

    base = supplier_email or supplier_name or "unknown_source"

    base = re.sub(r"[^a-zA-Z0-9@._-]", "_", base).lower()

    unique_part = source_message_id or job_id

    return f"{base}_{unique_part}.txt"


def _check_duplicate_filename(filename: str) -> None:
    if not filename:
        return

    try:
        from core.database import get_session_factory
        from models.offer_item import OfferItemDB

        db = get_session_factory()()
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
                        "message": f"File '{filename}' already processed.",
                        "filename": filename,
                    },
                )
        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Duplicate check failed: {e}")


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

        dedup_filename = generate_source_filename(
            source_filename=validated_payload.source_filename,
            attachments=validated_payload.attachments,
            supplier_email=validated_payload.supplier_email,
            supplier_name=validated_payload.supplier_name,
            source_message_id=validated_payload.source_message_id,
            job_id=job_id
        )

        _check_duplicate_filename(dedup_filename)

        payload_dict = validated_payload.model_dump()
        payload_dict["source_filename"] = dedup_filename

        redis_manager.set_job_status(job_id, "processing")
        process_document_task.delay(job_id, payload_dict)

        return {
            "status": "accepted",
            "job_id": job_id,
            "mode": "json",
            "source_filename": dedup_filename
        }

    attachments = []

    if files:
        for file in files:
            file_bytes = await file.read()

            if not file_bytes or not file.filename or file.filename.lower() in ("empty", ""):
                logger.info(f"Skipping empty file: {file.filename}")
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

    dedup_filename = generate_source_filename(
        source_filename=source_filename,
        attachments=attachments,
        supplier_email=supplier_email,
        supplier_name=supplier_name,
        source_message_id=source_message_id,
        job_id=job_id
    )

    _check_duplicate_filename(dedup_filename)

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
        "attachments": attachments
    }

    redis_manager.set_job_status(job_id, "processing")
    process_document_task.delay(job_id, payload_dict)

    return {
        "status": "accepted",
        "job_id": job_id,
        "mode": "multipart",
        "source_filename": dedup_filename
    }