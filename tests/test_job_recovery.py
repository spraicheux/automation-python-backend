"""tests/test_job_recovery.py

Regression tests for the two fixes behind Samuel's 2026-11-05 report:

  1. Jobs silently stranded at "queued" (the broker hand-off never landed) must
     be auto-recovered — without ever double-running a job that IS in the broker
     (no duplicates, no wasted OpenAI credits). Plus the stale
     "OpenAI out of credits" error on a resumed/done job must be cleared.
  2. An inline Outlook signature image (image001.gif) must NOT be treated as the
     offer file; the body (where the offers are) must be extracted — including
     when the offers live only in an HTML table.
"""
import email
import json
import time

import pytest


# ─── a tiny in-memory Redis (only the ops these modules use) ──────────────────

class FakeRedis:
    def __init__(self):
        self.kv = {}
        self.z = {}

    def set(self, k, v, ex=None, nx=False):
        if nx and k in self.kv:
            return False
        self.kv[k] = v
        return True

    def setex(self, k, ttl, v):
        self.kv[k] = v

    def get(self, k):
        return self.kv.get(k)

    def delete(self, *ks):
        n = 0
        for k in ks:
            if self.kv.pop(k, None) is not None:
                n += 1
            if self.z.pop(k, None) is not None:
                n += 1
        return n

    def incr(self, k):
        self.kv[k] = str(int(self.kv.get(k) or 0) + 1)
        return int(self.kv[k])

    def expire(self, k, ttl):
        return True

    def zadd(self, k, mapping):
        self.z.setdefault(k, {}).update(mapping)

    def zcard(self, k):
        return len(self.z.get(k, {}))

    def _sorted(self, k, reverse=False):
        return [m for m, _ in sorted(self.z.get(k, {}).items(),
                                     key=lambda x: x[1], reverse=reverse)]

    def zrange(self, k, start, end):
        members = self._sorted(k)
        return members[start:(None if end == -1 else end + 1)]

    def zrevrange(self, k, start, end):
        members = self._sorted(k, reverse=True)
        return members[start:(None if end == -1 else end + 1)]

    def zrem(self, k, *members):
        d = self.z.get(k, {})
        for m in members:
            d.pop(m, None)


@pytest.fixture
def fake_redis(monkeypatch):
    from core.redis_client import redis_manager
    fr = FakeRedis()
    monkeypatch.setattr(redis_manager, "use_redis", True)
    monkeypatch.setattr(redis_manager, "client", fr)
    return fr


# ─── 1a. clearing a stale error ───────────────────────────────────────────────

class TestClearError:
    def test_error_persists_on_plain_status_bump(self, fake_redis):
        from core import intake_log as il
        il.record_intake("j1", channel="email", source_filename="f.xlsx")
        il.update_intake("j1", status="no_credits", error="OpenAI out of credits")
        # A plain status change must NOT wipe the recorded error.
        il.update_intake("j1", status="processing")
        entry = json.loads(fake_redis.get("intake:entry:j1"))
        assert entry["error"] == "OpenAI out of credits"

    def test_clear_error_removes_it(self, fake_redis):
        from core import intake_log as il
        il.record_intake("j1", channel="email", source_filename="f.xlsx")
        il.update_intake("j1", status="no_credits", error="OpenAI out of credits")
        il.update_intake("j1", status="done", products=1357, clear_error=True)
        entry = json.loads(fake_redis.get("intake:entry:j1"))
        assert entry["status"] == "done"
        assert entry["products"] == 1357
        assert entry["error"] is None


# ─── 1b. the orphan sweeper ───────────────────────────────────────────────────

class _FakeTask:
    def __init__(self):
        self.calls = []

    def delay(self, job_id, payload):
        self.calls.append((job_id, payload))


def _write_entry(fake_redis, job_id, *, status, updated_at, error=None):
    entry = {"job_id": job_id, "channel": "email", "source_filename": f"{job_id}.xlsx",
             "status": status, "products": None, "error": error,
             "received_at": updated_at, "updated_at": updated_at}
    fake_redis.set("intake:entry:" + job_id, json.dumps(entry))
    fake_redis.zadd("intake:index", {job_id: time.time()})


def _old_iso(seconds_ago):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - seconds_ago))


class TestSweeper:
    def _patch_task_and_payload(self, monkeypatch, payloads):
        import workers.celery_tasks as ct
        task = _FakeTask()
        monkeypatch.setattr(ct, "process_document_task", task)
        monkeypatch.setattr("core.job_store.load_job_payload",
                            lambda jid: payloads.get(jid))
        return task

    def test_requeues_undelivered_orphan(self, fake_redis, monkeypatch):
        from core.job_recovery import sweep_stuck_jobs
        from core.ingest_service import was_delivered
        task = self._patch_task_and_payload(monkeypatch, {"orphan": {"source_channel": "email"}})
        # Stuck "queued", old enough, never delivered, payload exists.
        _write_entry(fake_redis, "orphan", status="queued", updated_at=_old_iso(600))

        out = sweep_stuck_jobs()

        assert out["requeued"] == 1
        assert task.calls and task.calls[0][0] == "orphan"
        assert was_delivered("orphan")  # flagged so it won't be swept again

    def test_skips_delivered_job(self, fake_redis, monkeypatch):
        from core.job_recovery import sweep_stuck_jobs
        from core.ingest_service import mark_delivered
        task = self._patch_task_and_payload(monkeypatch, {"live": {"x": 1}})
        _write_entry(fake_redis, "live", status="queued", updated_at=_old_iso(600))
        mark_delivered("live")  # it IS in the broker — must not double-run

        out = sweep_stuck_jobs()

        assert out["requeued"] == 0
        assert task.calls == []

    def test_skips_fresh_queued_job(self, fake_redis, monkeypatch):
        from core.job_recovery import sweep_stuck_jobs
        task = self._patch_task_and_payload(monkeypatch, {"fresh": {"x": 1}})
        _write_entry(fake_redis, "fresh", status="queued", updated_at=_old_iso(5))

        out = sweep_stuck_jobs()

        assert out["requeued"] == 0  # within grace — give the worker time
        assert task.calls == []

    def test_skips_orphan_without_payload(self, fake_redis, monkeypatch):
        from core.job_recovery import sweep_stuck_jobs
        task = self._patch_task_and_payload(monkeypatch, {})  # no payload
        _write_entry(fake_redis, "nopayload", status="queued", updated_at=_old_iso(600))

        out = sweep_stuck_jobs()

        assert out["requeued"] == 0
        assert task.calls == []

    def test_does_not_resurrect_real_extraction_failure(self, fake_redis, monkeypatch):
        from core.job_recovery import sweep_stuck_jobs
        task = self._patch_task_and_payload(monkeypatch, {"boom": {"x": 1}})
        _write_entry(fake_redis, "boom", status="failed", updated_at=_old_iso(600),
                     error="extraction blew up")

        out = sweep_stuck_jobs()

        assert out["requeued"] == 0  # only enqueue-failures are auto-retried
        assert task.calls == []

    def test_retries_enqueue_failure(self, fake_redis, monkeypatch):
        from core.job_recovery import sweep_stuck_jobs
        task = self._patch_task_and_payload(monkeypatch, {"blip": {"x": 1}})
        _write_entry(fake_redis, "blip", status="failed", updated_at=_old_iso(600),
                     error="could not reach the processing queue — will auto-retry")

        out = sweep_stuck_jobs()

        assert out["requeued"] == 1
        assert task.calls and task.calls[0][0] == "blip"

    def test_respects_attempt_cap(self, fake_redis, monkeypatch):
        from core.job_recovery import sweep_stuck_jobs, _MAX_ATTEMPTS, _SWEEP_ATTEMPTS_KEY
        task = self._patch_task_and_payload(monkeypatch, {"stubborn": {"x": 1}})
        _write_entry(fake_redis, "stubborn", status="queued", updated_at=_old_iso(600))
        fake_redis.set(_SWEEP_ATTEMPTS_KEY + "stubborn", str(_MAX_ATTEMPTS))

        out = sweep_stuck_jobs()

        assert out["requeued"] == 0
        assert task.calls == []

    def test_cleans_stale_error_on_done_row(self, fake_redis, monkeypatch):
        from core.job_recovery import sweep_stuck_jobs
        self._patch_task_and_payload(monkeypatch, {})
        _write_entry(fake_redis, "resumed", status="done", updated_at=_old_iso(600),
                     error="OpenAI out of credits — will auto-retry when topped up")

        out = sweep_stuck_jobs()

        assert out["cleaned"] == 1
        entry = json.loads(fake_redis.get("intake:entry:resumed"))
        assert entry["error"] is None


# ─── 2. inline signature image must not become the offer file ─────────────────

def _email_with_inline_gif_and_body(body_html, attach=None):
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
    from email.mime.application import MIMEApplication

    msg = MIMEMultipart("related")
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(body_html, "html"))
    msg.attach(alt)

    # Outlook signature logo: inline image with a Content-ID.
    img = MIMEImage(b"GIF89a-fake-bytes", _subtype="gif")
    img.add_header("Content-ID", "<image001.gif@01D.XYZ>")
    img.add_header("Content-Disposition", "inline", filename="image001.gif")
    msg.attach(img)

    if attach:
        name, data = attach
        part = MIMEApplication(data, _subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        part.add_header("Content-Disposition", "attachment", filename=name)
        msg.attach(part)
    return msg


class TestInlineImageFilter:
    def test_inline_signature_gif_is_dropped(self):
        from core.imap_client import _extract_attachments
        msg = _email_with_inline_gif_and_body("<p>hi</p>")
        atts = _extract_attachments(msg)
        names = [a["fileName"] for a in atts]
        assert "image001.gif" not in names
        assert atts == []  # nothing but the signature → no offer file

    def test_real_attachment_survives_alongside_signature(self):
        from core.imap_client import _extract_attachments
        msg = _email_with_inline_gif_and_body("<p>hi</p>", attach=("OFFER.xlsx", b"PK\x03\x04data"))
        names = [a["fileName"] for a in _extract_attachments(msg)]
        assert names == ["OFFER.xlsx"]

    def test_attached_image_is_kept(self):
        # A genuinely-attached image (disposition: attachment) is an offer file.
        from email.mime.image import MIMEImage
        from email.mime.multipart import MIMEMultipart
        from core.imap_client import _extract_attachments
        msg = MIMEMultipart()
        img = MIMEImage(b"JPEGdata", _subtype="jpeg")
        img.add_header("Content-Disposition", "attachment", filename="pricelist_photo.jpg")
        msg.attach(img)
        names = [a["fileName"] for a in _extract_attachments(msg)]
        assert names == ["pricelist_photo.jpg"]


class TestHtmlBody:
    def test_html_table_becomes_lines(self):
        from core.imap_client import _html_to_text
        html = ("<table><tr><td>Vodka 1L</td><td>12.50</td></tr>"
                "<tr><td>Gin 0.7L</td><td>9.90</td></tr></table>")
        text = _html_to_text(html)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        assert any("Vodka 1L" in ln for ln in lines)
        assert any("Gin 0.7L" in ln for ln in lines)
        # Two offers → two distinct lines (not one collapsed blob).
        assert len(lines) >= 2

    def test_body_prefers_html_when_plain_is_empty(self):
        from core.imap_client import _extract_text_body
        msg = _email_with_inline_gif_and_body(
            "<table><tr><td>Rum 1L</td><td>15.00</td></tr></table>")
        body = _extract_text_body(msg)
        assert "Rum 1L" in body
