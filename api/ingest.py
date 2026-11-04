import hashlib
import logging

from fastapi import APIRouter, Body, HTTPException
from fastapi import UploadFile, File, Form, Request
from typing import List, Optional

from schemas.ingest import IngestRequest
from core.ingest_service import (
    enqueue_offer,
    buffer_attachment,
    DuplicateIngest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


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
    content_type = request.headers.get("content-type", "")

    # ── JSON body (Make-style trigger, or any upstream that already has the
    #    attachment bytes/media ids) ───────────────────────────────────────────
    if "application/json" in content_type:
        if payload is None:
            return {"error": "Invalid JSON body"}

        validated_payload = IngestRequest(**payload)
        try:
            # Historical behaviour of the JSON path: filename dedup only
            # (content-hash is stored but not deduped here).
            result = enqueue_offer(
                source_channel=validated_payload.source_channel,
                source_message_id=validated_payload.source_message_id,
                source_filename=validated_payload.source_filename,
                supplier_email=validated_payload.supplier_email,
                supplier_name=validated_payload.supplier_name,
                sender_email=validated_payload.sender_email,
                sender_name=validated_payload.sender_name,
                subject=validated_payload.subject,
                text_body=validated_payload.text_body,
                attachments=[a.dict() for a in (validated_payload.attachments or [])],
                content_hash=validated_payload.content_hash,
                check_content_hash=False,
            )
        except DuplicateIngest as e:
            raise HTTPException(status_code=400, detail=e.detail)

        return {**result, "mode": "json"}

    # ── multipart upload (direct file drop) ─────────────────────────────────────
    attachments = []
    content_hash = None  # sha256 of the first (usually only) attachment

    if files:
        for file in files:
            file_bytes = await file.read()
            if not file_bytes or not file.filename or file.filename.lower() in ("empty", ""):
                logger.info(f"Skipping empty file: {file.filename}")
                continue
            if content_hash is None:
                content_hash = hashlib.sha256(file_bytes).hexdigest()
            attachments.append(
                buffer_attachment(file.filename, file.content_type, file_bytes)
            )

    try:
        result = enqueue_offer(
            source_channel=source_channel,
            source_message_id=source_message_id,
            source_filename=source_filename,
            supplier_email=supplier_email,
            supplier_name=supplier_name,
            sender_email=sender_email,
            sender_name=sender_name,
            subject=subject,
            text_body=text_body,
            attachments=attachments,
            content_hash=content_hash,
            check_content_hash=True,
        )
    except DuplicateIngest as e:
        raise HTTPException(status_code=400, detail=e.detail)

    return {**result, "mode": "multipart"}
