from uuid import uuid4

from fastapi import APIRouter, Body
from fastapi import UploadFile, File, Form, Request
from typing import List, Optional
from schemas.ingest import IngestRequest
from workers.celery_tasks import process_document_task
from core.redis_client import redis_manager


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
    job_id = str(uuid4())

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        if payload is None:
            return {"error": "Invalid JSON body"}

        validated_payload = IngestRequest(**payload)

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