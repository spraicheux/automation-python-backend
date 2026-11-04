"""core/email_poll.py

Pull new offer emails from the IMAP mailbox and enqueue each for processing.

Callable from two places so it runs regardless of how prod is started:
  - the in-app scheduler in main.py (the web app is always up), and
  - the Celery beat task (when beat runs).
A Redis single-flight lock makes concurrent callers a no-op, so running from
both — or from 2 gunicorn workers — is safe.

Handles:
  * multi-attachment emails  → one job carrying all files (none lost),
  * text-only offers         → no attachment but a substantive body is enqueued
                               for text extraction,
  * supplier detection       → from the body/subject text only (Make rules),
                               never from the sender.
"""
from __future__ import annotations

import hashlib
import logging
import re
import traceback
from typing import Any, Dict

from core.redis_client import redis_manager
from core.ingest_service import (enqueue_offer, buffer_attachment,
                                 detect_supplier_from_text, DuplicateIngest)

logger = logging.getLogger(__name__)

_MIN_TEXT_OFFER_CHARS = 25  # a body shorter than this, with no file, is not an offer


def _lock(name: str, ttl: int) -> bool:
    try:
        if redis_manager.use_redis and redis_manager.client:
            return bool(redis_manager.client.set(name, "1", nx=True, ex=ttl))
    except Exception as e:
        logger.warning(f"email poll lock unavailable: {e}")
    return True


def _text_source_filename(m: Dict[str, Any]) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", (m.get("from_email") or "email"))[:40]
    mid = re.sub(r"[^A-Za-z0-9]", "", (m.get("message_id") or str(m.get("uid", ""))))[-12:] or "nomid"
    return f"email-{stem}-{mid}.txt"


def run_email_poll() -> Dict[str, Any]:
    from core.imap_client import fetch_new_messages, ImapConfigError

    if not _lock("lock:poll_email_offers", ttl=110):
        logger.info("email poll already running; skipping this tick")
        return {"skipped": "locked"}

    try:
        messages = fetch_new_messages()
    except ImapConfigError as e:
        logger.warning(f"email poll skipped — {e}")
        return {"skipped": str(e)}
    except Exception as e:
        logger.error(f"email poll fetch failed: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}

    enqueued = skipped = failed = 0
    for m in messages:
        atts = m.get("attachments") or []
        body = (m.get("text_body") or "")
        subject = (m.get("subject") or "")
        msg_id = m.get("message_id") or f"imap-uid-{m.get('uid')}"
        try:
            supplier = detect_supplier_from_text(body, subject)  # text only, never sender
            if atts:
                built = [buffer_attachment(a["fileName"], a["contentType"], a["bytes"])
                         for a in atts]
                content_hash = hashlib.sha256(atts[0]["bytes"]).hexdigest()
                res = enqueue_offer(
                    source_channel="email", source_message_id=msg_id,
                    sender_email=m.get("from_email"), sender_name=m.get("from_name"),
                    supplier_name=supplier, subject=subject, text_body=body or None,
                    attachments=built, content_hash=content_hash,
                    check_content_hash=True)
                enqueued += 1
                logger.info(f"email uid {m.get('uid')} enqueued job {res['job_id']} "
                            f"({len(built)} file(s): {[a['fileName'] for a in atts]})")
            elif len(body.strip()) >= _MIN_TEXT_OFFER_CHARS:
                res = enqueue_offer(
                    source_channel="email", source_message_id=msg_id,
                    source_filename=_text_source_filename(m),
                    sender_email=m.get("from_email"), sender_name=m.get("from_name"),
                    supplier_name=supplier, subject=subject, text_body=body,
                    attachments=[], check_content_hash=False)
                enqueued += 1
                logger.info(f"email uid {m.get('uid')} enqueued job {res['job_id']} "
                            f"(text-only offer, no attachment)")
            else:
                skipped += 1
                logger.info(f"email uid {m.get('uid')} from {m.get('from_email')} "
                            f"has no attachment and no substantive body — skipped")
        except DuplicateIngest as e:
            skipped += 1
            logger.info(f"email uid {m.get('uid')} duplicate "
                        f"({e.detail.get('error')}) — skipped")
        except Exception as e:
            failed += 1
            logger.error(f"email uid {m.get('uid')} enqueue failed: {e}")

    logger.info(f"email poll done: {len(messages)} new, {enqueued} enqueued, "
                f"{skipped} skipped, {failed} failed")
    return {"new": len(messages), "enqueued": enqueued,
            "skipped": skipped, "failed": failed}
