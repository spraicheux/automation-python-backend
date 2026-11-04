"""Native email (IMAP) + WhatsApp (360dialog) ingestion.

Covers the two new inbound channels that replace the Make.com trigger:
  * WhatsApp message → enqueue plan (media types, dedup key, multi-message,
    both payload envelopes).
  * IMAP email → all attachments decoded, body + encoded headers.
  * Shared ingest service → filename rule, attachment shapes, dedup, enqueue.

No network, no OpenAI, no live mailbox — parsing/mapping only.
"""
import os
import sys
import pathlib
from email.message import EmailMessage

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used-in-unit-tests")


# ─── WhatsApp (360dialog) ─────────────────────────────────────────────────────

class TestWhatsAppPlan:
    def _contacts(self):
        return {"33783776470": "ACME Spirits"}

    def _doc(self, wamid="wamid.AAA", fname="pricelist.pdf"):
        return {"id": wamid, "from": "33783776470", "type": "document",
                "document": {"id": "MEDIA123", "mime_type": "application/pdf",
                             "filename": fname, "caption": "March offer"}}

    def test_document_message_plans_enqueue(self):
        from api.whatsapp import _plan_message
        p = _plan_message(self._doc(), self._contacts())
        assert p["status"] == "media"
        k = p["kwargs"]
        assert k["source_channel"] == "whatsapp"
        assert k["source_message_id"] == "wamid.AAA"
        assert k["sender_name"] == "ACME Spirits"
        assert k["text_body"] == "March offer"
        assert len(k["attachments"]) == 1
        att = k["attachments"][0]
        assert att["fileName"] == "pricelist.pdf"
        assert att["data"] == {"id": "MEDIA123"}        # bytes fetched lazily
        assert "pricelist.pdf" in k["source_filename"]
        assert k["source_filename"].startswith("wa-")

    def test_same_wamid_gives_same_filename(self):
        # 360dialog redelivery of the SAME wamid → identical source_filename →
        # filename dedup refuses it. No double-processing.
        from api.whatsapp import _plan_message
        a = _plan_message(self._doc(wamid="wamid.X"), self._contacts())
        b = _plan_message(self._doc(wamid="wamid.X"), self._contacts())
        assert a["kwargs"]["source_filename"] == b["kwargs"]["source_filename"]

    def test_distinct_wamid_same_name_distinct_filename(self):
        # Two suppliers each send "offer.pdf" → must NOT collide/false-dedup.
        from api.whatsapp import _plan_message
        a = _plan_message(self._doc(wamid="wamid.1", fname="offer.pdf"), self._contacts())
        b = _plan_message(self._doc(wamid="wamid.2", fname="offer.pdf"), self._contacts())
        assert a["kwargs"]["source_filename"] != b["kwargs"]["source_filename"]

    def test_text_message_skipped(self):
        from api.whatsapp import _plan_message
        msg = {"id": "wamid.T", "from": "x", "type": "text", "text": {"body": "hi"}}
        assert _plan_message(msg, {})["status"] == "skip"

    def test_missing_media_id_is_error(self):
        from api.whatsapp import _plan_message
        msg = {"id": "wamid.E", "from": "x", "type": "document",
               "document": {"mime_type": "application/pdf"}}
        assert _plan_message(msg, {})["status"] == "error"

    def test_image_without_filename_synthesizes_name(self):
        from api.whatsapp import _plan_message
        msg = {"id": "wamid.I", "from": "x", "type": "image",
               "image": {"id": "M", "mime_type": "image/jpeg"}}
        p = _plan_message(msg, {})
        assert p["status"] == "media"
        assert p["kwargs"]["attachments"][0]["fileName"].endswith(".jpg")

    def test_collect_messages_entry_envelope(self):
        from api.whatsapp import _collect_messages
        payload = {"entry": [{"changes": [{"value": {
            "contacts": [{"wa_id": "1", "profile": {"name": "N"}}],
            "messages": [{"id": "a", "type": "text"}, {"id": "b", "type": "document"}],
        }}]}]}
        msgs, contacts = _collect_messages(payload)
        assert len(msgs) == 2 and contacts == {"1": "N"}

    def test_collect_messages_flat_envelope(self):
        from api.whatsapp import _collect_messages
        payload = {"contacts": [{"wa_id": "2", "profile": {"name": "M"}}],
                   "messages": [{"id": "c", "type": "text"}]}
        msgs, contacts = _collect_messages(payload)
        assert len(msgs) == 1 and contacts == {"2": "M"}


# ─── IMAP email ───────────────────────────────────────────────────────────────

class TestImapParsing:
    def _email(self):
        msg = EmailMessage()
        msg["From"] = "ACME <sales@acme.example>"
        msg["Subject"] = "Spring offer"
        msg["Message-ID"] = "<abc@acme>"
        msg.set_content("Please find two files attached.")
        msg.add_attachment(b"%PDF-1.4 fake", maintype="application",
                           subtype="pdf", filename="list1.pdf")
        msg.add_attachment(
            b"xlsxbytes", maintype="application",
            subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename="list2.xlsx")
        return msg

    def test_extracts_all_attachments(self):
        from core.imap_client import _extract_attachments
        atts = _extract_attachments(self._email())
        names = sorted(a["fileName"] for a in atts)
        assert names == ["list1.pdf", "list2.xlsx"]       # nothing dropped
        assert all(a["bytes"] for a in atts)

    def test_extracts_text_body(self):
        from core.imap_client import _extract_text_body
        assert "two files" in _extract_text_body(self._email())

    def test_decode_encoded_header(self):
        from core.imap_client import _decode
        assert _decode("=?utf-8?q?Caf=C3=A9?=") == "Café"

    def test_body_only_email_has_no_attachments(self):
        from core.imap_client import _extract_attachments
        msg = EmailMessage()
        msg.set_content("just text, no files")
        assert _extract_attachments(msg) == []


# ─── shared ingest service ────────────────────────────────────────────────────

class TestIngestService:
    def _session_factory(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from core.database import Base
        from models.offer_item import OfferItemDB   # noqa: F401
        from models.source_file import SourceFileDB  # noqa: F401
        eng = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(eng)
        return sessionmaker(bind=eng)

    def test_generate_source_filename(self):
        from core.ingest_service import generate_source_filename
        assert generate_source_filename("x.pdf", [], None, None, None, "job") == "x.pdf"
        assert generate_source_filename(None, [{"fileName": "a.pdf"}], None, None,
                                        None, "job") == "a.pdf"
        assert generate_source_filename(None, [], "s@e.com", None, None,
                                        "job").startswith("s@e.com")

    def test_attachment_shapes(self):
        from core.ingest_service import buffer_attachment, media_attachment
        b = buffer_attachment("f.pdf", "application/pdf", b"abc")
        assert b["data"]["type"] == "Buffer" and b["fileSize"] == 3 and b["checksum"]
        m = media_attachment("f.pdf", "application/pdf", "MID")
        assert m["data"] == {"id": "MID"} and m["contentId"] == "MID"

    def test_duplicate_filename_raises(self, monkeypatch):
        import core.ingest_service as svc
        from models.offer_item import OfferItemDB
        Session = self._session_factory()
        s = Session()
        s.add(OfferItemDB(uid="u1", job_id="j", source_filename="dup.pdf"))
        s.commit(); s.close()
        monkeypatch.setattr("core.database.get_session_factory", lambda: Session)
        with pytest.raises(svc.DuplicateIngest):
            svc.check_duplicate_filename("dup.pdf")
        svc.check_duplicate_filename("other.pdf")  # different name → no raise

    def test_enqueue_offer_happy_path(self, monkeypatch):
        import core.ingest_service as svc
        import workers.celery_tasks as ct
        from core.redis_client import redis_manager
        Session = self._session_factory()
        monkeypatch.setattr("core.database.get_session_factory", lambda: Session)
        monkeypatch.setattr(redis_manager, "set_job_status", lambda *a, **k: None)

        captured = {}

        class FakeTask:
            def delay(self, job_id, payload):
                captured["job_id"] = job_id
                captured["payload"] = payload

        monkeypatch.setattr(ct, "process_document_task", FakeTask())

        res = svc.enqueue_offer(
            source_channel="email", sender_email="s@e.com",
            attachments=[svc.buffer_attachment("a.pdf", "application/pdf", b"xx")],
            content_hash="hash123", check_content_hash=True)

        assert res["status"] == "accepted"
        assert captured["payload"]["source_channel"] == "email"
        assert captured["payload"]["source_filename"] == "a.pdf"
        assert len(captured["payload"]["attachments"]) == 1
