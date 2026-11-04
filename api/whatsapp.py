"""api/whatsapp.py

Inbound WhatsApp webhook for 360dialog (new number +33 7 83 77 64 70).

360dialog delivers each inbound WhatsApp message here; we turn every media
message (document / image / video / audio) into an ingest job. The media bytes
are NOT in the webhook — only a media id — so we attach the id and let
core.file_download.resolve_attachment_bytes fetch it from 360dialog at
processing time (needs D360_API_KEY for the active number).

Robustness (Samuel's requirements):
  * Security — the endpoint requires a shared secret (WHATSAPP_WEBHOOK_TOKEN),
    accepted as `?token=` or `X-Webhook-Token`. The URL you paste into
    360dialog carries the token.
  * No double-processing — 360dialog retries a webhook until it gets a 200, and
    may redeliver. Each message id (wamid) is recorded in a Redis seen-set and
    also encoded into the source filename, so a redelivery is refused as a
    duplicate rather than ingested twice.
  * Error isolation — one malformed message never fails the batch; we log it and
    still return 200 so 360dialog stops retrying the good ones.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List

from fastapi import APIRouter, Request, Response, Query, Header
from fastapi.responses import PlainTextResponse, JSONResponse

from core.redis_client import redis_manager
from core.ingest_service import enqueue_offer, media_attachment, DuplicateIngest

logger = logging.getLogger(__name__)
router = APIRouter()

_MEDIA_TYPES = {"document", "image", "video", "audio"}
_SEEN_PREFIX = "wa:seen:"          # + wamid
_SEEN_TTL = 7 * 24 * 3600          # a week is plenty to cover retries


def _token_ok(token_q: str | None, token_h: str | None) -> bool:
    expected = os.getenv("WHATSAPP_WEBHOOK_TOKEN")
    if not expected:
        # Not configured yet — accept but shout, so prod never runs open silently.
        logger.warning("WHATSAPP_WEBHOOK_TOKEN is not set — webhook is UNAUTHENTICATED")
        return True
    return (token_q == expected) or (token_h == expected)


def _already_seen(wamid: str) -> bool:
    if not wamid:
        return False
    try:
        if redis_manager.use_redis and redis_manager.client:
            # set NX: True means we just claimed it (not seen before)
            claimed = redis_manager.client.set(_SEEN_PREFIX + wamid, "1",
                                                nx=True, ex=_SEEN_TTL)
            return not bool(claimed)
    except Exception as e:
        logger.warning(f"wa seen-set check failed for {wamid}: {e}")
    return False


_EXT_BY_MIME = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xls",
    "text/csv": "csv",
    "image/jpeg": "jpg",
    "image/png": "png",
}


def _collect_messages(payload: Dict[str, Any]) -> (List[Dict[str, Any]], Dict[str, str]):
    """Return (messages, contacts_by_waid). Handles both the Cloud-API
    entry/changes envelope (360dialog v2) and a flat {messages, contacts}."""
    messages: List[Dict[str, Any]] = []
    contacts: Dict[str, str] = {}

    def _ingest_value(value: Dict[str, Any]):
        for c in value.get("contacts", []) or []:
            waid = c.get("wa_id")
            name = (c.get("profile") or {}).get("name")
            if waid and name:
                contacts[waid] = name
        messages.extend(value.get("messages", []) or [])

    if isinstance(payload.get("entry"), list):
        for entry in payload["entry"]:
            for change in entry.get("changes", []) or []:
                _ingest_value(change.get("value", {}) or {})
    else:
        _ingest_value(payload)  # flat shape
    return messages, contacts


def _plan_message(msg: Dict[str, Any], contacts: Dict[str, str]) -> Dict[str, Any]:
    """Pure mapping of one inbound WhatsApp message to an enqueue plan.
    Returns {"status": "media"|"skip"|"error", "wamid", ...}. No side effects
    (no seen-set, no enqueue) so it is unit-testable."""
    wamid = msg.get("id") or ""
    mtype = msg.get("type")
    sender = msg.get("from") or ""
    sender_name = contacts.get(sender) or sender or None

    if mtype not in _MEDIA_TYPES:
        return {"status": "skip", "wamid": wamid, "reason": f"type={mtype}"}

    media = msg.get(mtype) or {}
    media_id = media.get("id")
    if not media_id:
        return {"status": "error", "wamid": wamid, "reason": "no media id"}

    mime = media.get("mime_type") or ""
    orig = media.get("filename")
    if not orig:
        ext = _EXT_BY_MIME.get(mime.split(";")[0].strip(), "bin")
        orig = f"{mtype}.{ext}"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", orig)
    short = re.sub(r"[^A-Za-z0-9]", "", wamid)[-12:] or "nowamid"
    source_filename = f"wa-{short}-{safe}"

    return {
        "status": "media",
        "wamid": wamid,
        "sender": sender,
        "kwargs": dict(
            source_channel="whatsapp",
            source_message_id=wamid,
            source_filename=source_filename,
            sender_name=sender_name,
            supplier_name=sender_name,
            text_body=media.get("caption"),
            attachments=[media_attachment(orig, mime, media_id)],
            check_content_hash=False,
        ),
    }


@router.get("/webhook/whatsapp")
async def whatsapp_verify(
    request: Request,
    token: str | None = Query(None),
    x_webhook_token: str | None = Header(None),
):
    """Meta/360dialog verification handshake (hub.challenge) used when first
    registering the webhook URL."""
    params = request.query_params
    mode = params.get("hub.mode")
    verify_token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    expected = os.getenv("WHATSAPP_WEBHOOK_TOKEN")
    if mode == "subscribe" and challenge is not None and (not expected or verify_token == expected):
        return PlainTextResponse(challenge)
    if not _token_ok(token, x_webhook_token):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return PlainTextResponse("ok")


@router.post("/webhook/whatsapp")
async def whatsapp_inbound(
    request: Request,
    token: str | None = Query(None),
    x_webhook_token: str | None = Header(None),
):
    if not _token_ok(token, x_webhook_token):
        return JSONResponse({"error": "forbidden"}, status_code=403)

    try:
        payload = await request.json()
    except Exception as e:
        logger.warning(f"wa webhook: invalid JSON body: {e}")
        return JSONResponse({"status": "ignored", "reason": "bad json"}, status_code=200)

    messages, contacts = _collect_messages(payload)
    enqueued = skipped = failed = 0

    for msg in messages:
        try:
            plan = _plan_message(msg, contacts)
            if plan["status"] == "skip":
                skipped += 1
                logger.info(f"wa {plan['wamid'] or '?'}: {plan['reason']} — skipped")
                continue
            if plan["status"] == "error":
                failed += 1
                logger.error(f"wa {plan['wamid'] or '?'}: {plan['reason']}")
                continue
            # media
            if _already_seen(plan["wamid"]):
                skipped += 1
                logger.info(f"wa {plan['wamid']}: already seen — skipped (retry)")
                continue
            res = enqueue_offer(**plan["kwargs"])
            enqueued += 1
            logger.info(f"wa {plan['wamid']} from {plan['sender']}: enqueued job "
                        f"{res['job_id']} ({plan['kwargs']['source_filename']})")
        except DuplicateIngest as e:
            skipped += 1
            logger.info(f"wa {msg.get('id','?')}: duplicate "
                        f"({e.detail.get('error')}) — skipped")
        except Exception as e:
            failed += 1
            logger.error(f"wa {msg.get('id','?')}: enqueue failed: {e}")

    logger.info(f"wa webhook: {len(messages)} msg(s), {enqueued} enqueued, "
                f"{skipped} skipped, {failed} failed")
    # Always 200 so 360dialog stops retrying the batch.
    return {"status": "ok", "enqueued": enqueued, "skipped": skipped, "failed": failed}
