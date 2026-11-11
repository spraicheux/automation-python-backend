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
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Request, Response, Query, Header
from fastapi.responses import PlainTextResponse, JSONResponse

from core.redis_client import redis_manager
from core.intake_log import record_intake, update_intake
from core.ingest_service import (enqueue_offer, media_attachment,
                                 detect_supplier_from_text, DuplicateIngest)

logger = logging.getLogger(__name__)
router = APIRouter()

_MEDIA_TYPES = {"document", "image", "video", "audio"}
_MIN_TEXT_OFFER_CHARS = 25         # ignore greetings / one-word texts
_SEEN_PREFIX = "wa:seen:"          # + wamid
_SEEN_TTL = 7 * 24 * 3600          # a week is plenty to cover retries
_HINT_PREFIX = "wa:supplier_hint:" # + sender; links a text instruction to media
_HINT_TTL = 30 * 60                # images usually follow within minutes

# A standalone instruction naming the supplier for the images that follow, e.g.
# "offres de CHELSEA HK" / "offer from CHELSEA HK" / "fournisseur: CHELSEA HK".
_SUPPLIER_INSTRUCTION_RE = re.compile(
    r'^\s*(?:offres?\s+(?:de|pour|from)|offer\s+from|supplier|fournisseur|from)'
    r'\s*[:\-]?\s*(.+?)\s*$', re.IGNORECASE)


def _supplier_instruction(body: str):
    """Return the supplier named by a short standalone instruction, else None."""
    if not body:
        return None
    b = str(body).strip()
    if len(b) > 60 or "\n" in b:
        return None  # a real offer, not a one-line instruction
    m = _SUPPLIER_INSTRUCTION_RE.match(b)
    if m:
        name = m.group(1).strip(" . ")
        if 2 <= len(name) <= 40:
            return name
    return None


def _set_supplier_hint(sender: str, name: str) -> None:
    try:
        from core.redis_client import redis_manager
        if sender and name and redis_manager.use_redis and redis_manager.client:
            redis_manager.client.set(_HINT_PREFIX + sender, name, ex=_HINT_TTL)
            logger.info(f"wa: supplier hint '{name}' stashed for sender {sender}")
    except Exception as e:
        logger.warning(f"wa: could not stash supplier hint: {e}")


def _get_supplier_hint(sender: str):
    try:
        from core.redis_client import redis_manager
        if sender and redis_manager.use_redis and redis_manager.client:
            v = redis_manager.client.get(_HINT_PREFIX + sender)
            return v.decode() if isinstance(v, bytes) else v
    except Exception:
        return None
    return None


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
    Returns {"status": "ingest"|"skip"|"error", "wamid", ...}. No side effects
    (no seen-set, no enqueue) so it is unit-testable.

    Make rules honored: supplier comes ONLY from the message text (caption/body),
    never from the sender; sender_email is always None for WhatsApp; subject "".
    Media messages carry the file; substantive text-only messages are ingested
    as text offers; greetings and non-media/non-text are skipped.
    """
    wamid = msg.get("id") or ""
    mtype = msg.get("type")
    sender = msg.get("from") or ""
    sender_name = contacts.get(sender) or sender or None
    short = re.sub(r"[^A-Za-z0-9]", "", wamid)[-12:] or "nowamid"

    if mtype in _MEDIA_TYPES:
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
        caption = media.get("caption")
        return {
            "status": "ingest", "wamid": wamid, "sender": sender,
            "kwargs": dict(
                source_channel="whatsapp", source_message_id=wamid,
                source_filename=f"wa-{short}-{safe}",
                sender_name=sender_name, sender_email=None,
                supplier_name=detect_supplier_from_text(caption),
                supplier_email=None, subject="", text_body=caption,
                attachments=[media_attachment(orig, mime, media_id)],
                check_content_hash=False,
            ),
        }

    if mtype == "text":
        body = (msg.get("text") or {}).get("body") or ""
        if len(body.strip()) < _MIN_TEXT_OFFER_CHARS:
            return {"status": "skip", "wamid": wamid, "reason": "short text"}
        return {
            "status": "ingest", "wamid": wamid, "sender": sender,
            "kwargs": dict(
                source_channel="whatsapp", source_message_id=wamid,
                source_filename=f"wa-{short}-text.txt",
                sender_name=sender_name, sender_email=None,
                supplier_name=detect_supplier_from_text(body),
                supplier_email=None, subject="", text_body=body,
                attachments=[], check_content_hash=False,
            ),
        }

    return {"status": "skip", "wamid": wamid, "reason": f"type={mtype}"}


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
            # A standalone supplier instruction ("offres de CHELSEA HK") names
            # the supplier for the images that follow it. Stash it per sender so
            # the (caption-less) media inherit it — even across webhook deliveries
            # (Samuel 2026-11-11: 8 CHELSEA images landed with no supplier).
            if msg.get("type") == "text":
                _instr = _supplier_instruction((msg.get("text") or {}).get("body") or "")
                if _instr:
                    _set_supplier_hint(msg.get("from") or "", _instr)

            plan = _plan_message(msg, contacts)
            # Apply a pending supplier hint to media that carries no caption-derived
            # supplier, so an image batch is attributed to the named supplier.
            if plan.get("status") == "ingest":
                _k = plan["kwargs"]
                if (_k.get("attachments") and
                        (_k.get("supplier_name") in (None, "", "Not Found"))):
                    _hint = _get_supplier_hint(plan.get("sender") or "")
                    if _hint:
                        _k["supplier_name"] = _hint
                        logger.info(f"wa {plan['wamid']}: applied supplier hint "
                                    f"'{_hint}' to media")
            if plan["status"] == "skip":
                skipped += 1
                logger.info(f"wa {plan['wamid'] or '?'}: {plan['reason']} — skipped")
                continue
            if plan["status"] == "error":
                failed += 1
                logger.error(f"wa {plan['wamid'] or '?'}: {plan['reason']}")
                continue
            # ingest (media or substantive text)
            if _already_seen(plan["wamid"]):
                skipped += 1
                logger.info(f"wa {plan['wamid']}: already seen — skipped (retry)")
                continue

            kwargs = plan["kwargs"]
            # Download the 360dialog media NOW and persist it, so it's never lost
            # and survives the media's expiry. Also lets us content-hash dedup.
            _atts = kwargs.get("attachments") or []
            if _atts and (_atts[0].get("data") or {}).get("id"):
                try:
                    import hashlib as _h
                    from schemas.ingest import Attachment
                    from core.file_download import resolve_attachment_bytes
                    from core.file_store import save_file
                    from core.ingest_service import file_attachment
                    _att = _atts[0]
                    _data = await resolve_attachment_bytes(Attachment(**_att))
                    if _data:
                        _path = save_file(_att["fileName"], _data)
                        if _path:
                            _ck = _h.sha256(_data).hexdigest()
                            kwargs["attachments"] = [file_attachment(
                                _att["fileName"], _att["contentType"], _path,
                                len(_data), checksum=_ck)]
                            kwargs["content_hash"] = _ck
                            kwargs["check_content_hash"] = True
                            logger.info(f"wa {plan['wamid']}: media saved ({len(_data)} bytes) {_path}")
                except Exception as _dl_exc:
                    logger.warning(f"wa {plan['wamid']}: media pre-download failed, "
                                   f"deferring to lazy fetch: {_dl_exc}")

            res = enqueue_offer(**kwargs)
            enqueued += 1
            logger.info(f"wa {plan['wamid']} from {plan['sender']}: enqueued job "
                        f"{res['job_id']} ({plan['kwargs']['source_filename']})")
        except DuplicateIngest as e:
            skipped += 1
            k = plan.get("kwargs", {})  # plan is set before enqueue_offer raises
            jid = str(uuid.uuid4())
            record_intake(jid, channel="whatsapp",
                          source_filename=k.get("source_filename") or msg.get("id"),
                          sender_name=k.get("sender_name"),
                          supplier_name=k.get("supplier_name"),
                          attachment_count=len(k.get("attachments") or []),
                          has_text=bool(k.get("text_body")), status="duplicate")
            update_intake(jid, error=f"already ingested ({e.detail.get('error')})")
            logger.info(f"wa {msg.get('id','?')}: duplicate "
                        f"({e.detail.get('error')}) — recorded as duplicate")
        except Exception as e:
            failed += 1
            logger.error(f"wa {msg.get('id','?')}: enqueue failed: {e}")

    logger.info(f"wa webhook: {len(messages)} msg(s), {enqueued} enqueued, "
                f"{skipped} skipped, {failed} failed")
    # Always 200 so 360dialog stops retrying the batch.
    return {"status": "ok", "enqueued": enqueued, "skipped": skipped, "failed": failed}
