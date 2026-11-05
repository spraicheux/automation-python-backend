"""core/imap_client.py

Reads new supplier-offer emails from the offers mailbox over IMAP
(offers@loxo-france.com) and returns them as structured messages with ALL
attachments decoded. The email poller (workers.celery_tasks.poll_email_offers)
turns each into an ingest job.

Why IMAP (not Graph): the offers mailbox is already served over IMAP, and IMAP
works the same whatever host backs the domain — no per-provider app
registration. The Make scenario reads the same mailbox during the parallel
backup period; to coexist cleanly this client:
  * opens the folder READ-ONLY (never sets/clears \\Seen), so Make's view is
    untouched and we don't fight over flags, and
  * tracks its own high-water UID in Redis, so "new" is independent of
    read/unread state.
Content-hash dedup in core.ingest_service is the backstop if both paths grab
the same message.

Env:
    IMAP_HOST            e.g. imap.loxo-france.com  (or the provider's host)
    IMAP_USER            offers@loxo-france.com
    IMAP_PASSWORD        mailbox or app password
    IMAP_PORT            default 993
    IMAP_FOLDER          default "INBOX"
    IMAP_SSL             default "1" (SSL); "0" for STARTTLS on 143
    IMAP_INGEST_BACKLOG  default 0 — on the very first poll, how many of the most
                         recent existing messages to ingest. 0 = baseline only
                         (only mail that arrives AFTER first poll is ingested),
                         so deploying against a full mailbox doesn't replay history.
"""
from __future__ import annotations

import email
import imaplib
import logging
import os
import re
from email.header import decode_header, make_header
from email.utils import parseaddr
from typing import Any, Dict, List, Optional

from core.redis_client import redis_manager

logger = logging.getLogger(__name__)

_LAST_UID_KEY = "imap:last_uid"
_UIDVALIDITY_KEY = "imap:uidvalidity"


class ImapConfigError(RuntimeError):
    pass


def _cfg() -> Dict[str, Any]:
    host = os.getenv("IMAP_HOST")
    user = os.getenv("IMAP_USER")
    password = os.getenv("IMAP_PASSWORD")
    missing = [n for n, v in (("IMAP_HOST", host), ("IMAP_USER", user),
                              ("IMAP_PASSWORD", password)) if not v]
    if missing:
        raise ImapConfigError(f"IMAP not configured — missing: {', '.join(missing)}")
    return {
        "host": host, "user": user, "password": password,
        "port": int(os.getenv("IMAP_PORT", "993")),
        "folder": os.getenv("IMAP_FOLDER", "INBOX"),
        "ssl": os.getenv("IMAP_SSL", "1") != "0",
        "backlog": int(os.getenv("IMAP_INGEST_BACKLOG", "0")),
    }


def _redis():
    if redis_manager.use_redis and redis_manager.client:
        return redis_manager.client
    return None


def _decode(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def _state_key(base: str, mailbox: str) -> str:
    return f"{base}:{mailbox}"


def _is_inline_signature_image(part, filename: str, disposition: str) -> bool:
    """True for an embedded body image — an Outlook signature logo
    (image001.gif/png), a tracking pixel, a CID-referenced inline image. These
    are part of the body, NOT offer files: treating one as the attachment
    corrupts the filename, the dedup hash and wastes a vision call, and pushes
    the real offers (in the body) to the side. A genuinely-attached image
    (Content-Disposition: attachment) is kept."""
    if part.get_content_maintype() != "image":
        return False
    if "attachment" in disposition:
        return False
    content_id = part.get("Content-ID") or part.get("Content-Id")
    is_inline = "inline" in disposition
    looks_like_sig = bool(re.match(
        r"^image\d+\.(gif|png|jpe?g|bmp|tiff?)$", filename or "", re.IGNORECASE))
    return bool(is_inline or content_id or looks_like_sig)


def _extract_attachments(msg) -> List[Dict[str, Any]]:
    """Every real attachment part, decoded. Captures PDFs, spreadsheets,
    images and any file with a name or an attachment disposition — so a
    multi-file offer email loses nothing. Inline body images (signature logos
    like image001.gif) are skipped: they are not offer files."""
    out: List[Dict[str, Any]] = []
    for part in msg.walk():
        if part.get_content_maintype() == "multipart":
            continue
        filename = _decode(part.get_filename())
        disposition = (part.get("Content-Disposition") or "").lower()
        if not filename and "attachment" not in disposition:
            continue
        if _is_inline_signature_image(part, filename, disposition):
            logger.info(f"skipping inline/embedded image {filename!r} "
                        f"(signature/logo, not an offer file)")
            continue
        try:
            payload = part.get_payload(decode=True)
        except Exception as e:
            logger.warning(f"Could not decode attachment {filename!r}: {e}")
            payload = None
        if not payload:
            continue
        if not filename:
            ext = (part.get_content_subtype() or "bin")
            filename = f"attachment.{ext}"
        out.append({
            "fileName": filename,
            "contentType": part.get_content_type(),
            "bytes": payload,
        })
    return out


def _html_to_text(html: str) -> str:
    """Flatten an HTML email body into readable lines. Offers often live in an
    HTML <table>, so rows and cells become newlines/tabs (preserving one offer
    per line) before tags are stripped — otherwise a whole table collapses into
    one blob and the per-line extractor misses offers."""
    if not html:
        return ""
    import html as _htmlmod
    s = html
    s = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</(p|div|tr|li|h[1-6]|table)\s*>", "\n", s)
    s = re.sub(r"(?i)</(td|th)\s*>", "\t", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = _htmlmod.unescape(s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def _extract_text_body(msg) -> str:
    plain, html = "", ""
    for part in msg.walk():
        if part.get_content_maintype() == "multipart":
            continue
        disposition = (part.get("Content-Disposition") or "").lower()
        if "attachment" in disposition:
            continue
        ctype = part.get_content_type()
        try:
            payload = part.get_payload(decode=True)
        except Exception:
            payload = None
        if not payload:
            continue
        charset = part.get_content_charset() or "utf-8"
        try:
            text = payload.decode(charset, errors="replace")
        except (LookupError, Exception):
            text = payload.decode("utf-8", errors="replace")
        if ctype == "text/plain" and not plain:
            plain = text
        elif ctype == "text/html" and not html:
            html = text
    plain = (plain or "").strip()
    html = (html or "").strip()
    # Prefer a substantive plain-text alternative; otherwise flatten the HTML
    # (the offers may only exist in the HTML table).
    if plain and len(plain) >= 40:
        return plain
    converted = _html_to_text(html)
    if len(converted) >= len(plain):
        return converted
    return plain


def fetch_new_messages() -> List[Dict[str, Any]]:
    """Return new messages since the last poll, each as:
        {uid, message_id, from_email, from_name, subject, date,
         text_body, attachments: [{fileName, contentType, bytes}]}
    Advances the stored high-water UID. Read-only on the mailbox.
    """
    cfg = _cfg()
    mailbox = cfg["user"]
    r = _redis()

    if cfg["ssl"]:
        conn = imaplib.IMAP4_SSL(cfg["host"], cfg["port"])
    else:
        conn = imaplib.IMAP4(cfg["host"], cfg["port"])
        conn.starttls()
    try:
        conn.login(cfg["user"], cfg["password"])
        # READ-ONLY select: never touch \Seen (Make still sees the mail).
        typ, data = conn.select(cfg["folder"], readonly=True)
        if typ != "OK":
            raise ImapConfigError(f"IMAP select {cfg['folder']} failed: {data}")

        # UIDVALIDITY guards against the server renumbering UIDs under us.
        uidvalidity = None
        tv, vd = conn.response("UIDVALIDITY")
        if vd and vd[0]:
            uidvalidity = vd[0].decode() if isinstance(vd[0], bytes) else str(vd[0])

        last_uid = 0
        stored_validity = None
        if r:
            lu = r.get(_state_key(_LAST_UID_KEY, mailbox))
            sv = r.get(_state_key(_UIDVALIDITY_KEY, mailbox))
            last_uid = int(lu) if lu else 0
            stored_validity = sv.decode() if isinstance(sv, bytes) else sv
        if uidvalidity and stored_validity and uidvalidity != stored_validity:
            logger.warning(f"IMAP UIDVALIDITY changed ({stored_validity} → "
                           f"{uidvalidity}); re-baselining.")
            last_uid = 0

        typ, sdata = conn.uid("search", None, "ALL")
        all_uids = [int(x) for x in sdata[0].split()] if sdata and sdata[0] else []
        all_uids.sort()

        first_run = (last_uid == 0 and stored_validity is None)
        if first_run:
            # Baseline: remember where we are, optionally ingest a small backlog.
            new_uids = all_uids[-cfg["backlog"]:] if cfg["backlog"] > 0 else []
            logger.info(f"IMAP first poll on {mailbox}: {len(all_uids)} existing, "
                        f"ingesting backlog={len(new_uids)}, baselining.")
        else:
            new_uids = [u for u in all_uids if u > last_uid]

        messages: List[Dict[str, Any]] = []
        for uid in new_uids:
            typ, mdata = conn.uid("fetch", str(uid), "(RFC822)")
            if typ != "OK" or not mdata or not mdata[0]:
                logger.warning(f"IMAP fetch uid {uid} failed")
                continue
            raw = mdata[0][1]
            msg = email.message_from_bytes(raw)
            from_name, from_email = parseaddr(_decode(msg.get("From")))
            messages.append({
                "uid": uid,
                "message_id": (msg.get("Message-ID") or "").strip(),
                "from_email": from_email or None,
                "from_name": from_name or None,
                "subject": _decode(msg.get("Subject")),
                "date": _decode(msg.get("Date")),
                "text_body": _extract_text_body(msg),
                "attachments": _extract_attachments(msg),
            })

        # Advance the high-water mark (even on a baseline run with no messages).
        new_high = max(all_uids) if all_uids else last_uid
        if r:
            r.set(_state_key(_LAST_UID_KEY, mailbox), str(new_high))
            if uidvalidity:
                r.set(_state_key(_UIDVALIDITY_KEY, mailbox), uidvalidity)

        return messages
    finally:
        try:
            conn.logout()
        except Exception:
            pass
