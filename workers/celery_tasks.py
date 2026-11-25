from dotenv import load_dotenv
load_dotenv()

import logging
import requests
import traceback
import asyncio
from datetime import datetime
from pydantic import ValidationError
import logging
from core.logging_utils import setup_logging

# Initialize global logging to stdout for Azure visibility
setup_logging(level=logging.INFO)

from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import worker_ready

from core.celery_app import celery_app
from workers.processor import process_offer
from core.redis_client import redis_manager
from core.intake_log import update_intake
from core.openai_health import OpenAICreditsExhausted, queue_pending_credit_job
from core.openai_client import usage_begin, usage_finish

logger = logging.getLogger(__name__)


# ── Deploy-overlap gate ──────────────────────────────────────────────────────
# Azure App Service keeps the OLD container — and its Celery worker — running
# and consuming the shared Redis queue while a new deploy warms up. So a
# re-ingest enqueued after a deploy can be picked up by a worker still running
# OLD code, which made fixes look like they never took (Samuel 2026-11-18).
# Each worker claims a monotonically-increasing epoch on startup; the newest
# container holds the highest. A worker that is no longer the newest defers the
# job back to the queue so the CURRENT worker runs it — with a small bounce
# limit so a job is never lost if the newest worker has since died.
_MY_EPOCH = None


@worker_ready.connect
def _register_worker_epoch(**_kw):
    global _MY_EPOCH
    try:
        r = redis_manager.client
        if not (redis_manager.use_redis and r):
            return
        _MY_EPOCH = int(r.incr("deploy:worker_epoch"))
        r.set("deploy:current_epoch", _MY_EPOCH)
        logger.info(f"[deploy-gate] worker epoch {_MY_EPOCH} registered as current")
    except Exception as e:
        logger.warning(f"[deploy-gate] epoch registration failed: {e}")


def _newer_worker_live() -> bool:
    """True when a strictly-newer worker container is live — this (older) worker
    should defer the job, not run it on stale code."""
    try:
        if _MY_EPOCH is None:
            return False
        r = redis_manager.client
        if not (redis_manager.use_redis and r):
            return False
        cur = r.get("deploy:current_epoch")
        return cur is not None and int(cur) > _MY_EPOCH
    except Exception:
        return False


def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()


# soft_time_limit raises SoftTimeLimitExceeded inside the worker → our try/except
# catches it, marks the job "failed", and stops retrying.
# time_limit is the hard kill switch so a truly-stuck worker can't monopolise the
# celery slot on the B1 single-vCPU box.
# M2 file sizes: W23 221 rows ≈ 12 min, Cosmetics 268–281 ≈ 16 min, W39 761 ≈ 50 min,
# STOCK LIST 1043 ≈ 70 min. 90 min soft / 95 min hard covers all of them with buffer.
# Zwolle (2054, held) would need ~2.5h — bump before running that file.
@celery_app.task(bind=True, max_retries=2, soft_time_limit=5400, time_limit=5700)
def process_document_task(self, job_id: str, payload_dict: dict):
    # Deploy-overlap gate (see _register_worker_epoch): a worker that is no
    # longer the newest re-enqueues the job so the CURRENT worker runs it on
    # current code, up to a small bounce limit (then it runs it anyway so a job
    # is never lost). Bounce count lives in Redis, keyed by job_id, so the task
    # payload is never mutated (schema-safe).
    if _newer_worker_live():
        _r = redis_manager.client if (redis_manager.use_redis and redis_manager.client) else None
        _bkey = f"deploy:bounce:{job_id}"
        _bounce = 0
        try:
            _bounce = int(_r.get(_bkey) or 0) if _r else 0
        except Exception:
            _bounce = 0
        if _bounce < 3:
            try:
                if _r:
                    _r.incr(_bkey)
                    _r.expire(_bkey, 600)
            except Exception:
                pass
            logger.warning(f"[deploy-gate] stale worker (epoch {_MY_EPOCH}) deferring "
                           f"job {job_id}; bounce {_bounce + 1}")
            process_document_task.apply_async(args=[job_id, payload_dict], countdown=10)
            return {"deferred_stale_worker": True, "job_id": job_id, "bounce": _bounce + 1}
    try:
        logger.info(f"Celery processing started for JobID: {job_id}")
        redis_manager.set_job_status(job_id, "processing")
        update_intake(job_id, status="processing")

        from schemas.ingest import IngestRequest
        
        try:
            payload = IngestRequest(**payload_dict)
        except ValidationError as e:
            logger.error(f"Failed to parse payload for JobID {job_id}: {e}")
            raise

        loop = get_or_create_eventloop()

        # Per-file token/cost metering: bracket the extraction and persist the
        # per-job usage to Redis (read-only accounting; see core.openai_client).
        usage_begin(job_id)
        def _persist_usage():
            try:
                _u = usage_finish(job_id)
                if not _u:
                    return
                logger.info(f"[usage] job={job_id} calls={_u.get('calls')} "
                            f"prompt={_u.get('prompt_tokens')} completion={_u.get('completion_tokens')} "
                            f"cost_usd={_u.get('cost_usd')} by_model={_u.get('by_model')}")
                if redis_manager.use_redis and redis_manager.client:
                    import json as _json
                    redis_manager.client.setex(f"usage:{job_id}", 86400, _json.dumps(_u))
            except Exception as _e:
                logger.error(f"[usage] persist failed job={job_id}: {_e}")

        # Hard timeout on the whole extraction (90 min). asyncio.wait_for works
        # regardless of celery pool type, unlike soft_time_limit which is a
        # no-op under --pool=solo. On timeout the task marks the job "failed"
        # and celery is free to pick up the next queued job. Sized for M2:
        # STOCK LIST 1043 rows ≈ 70 min, W39 761 ≈ 50 min. Zwolle 2054 (held)
        # would need ~2.5h — bump this before running it.
        try:
            loop.run_until_complete(
                asyncio.wait_for(process_offer(payload, job_id), timeout=5400.0)
            )
        except asyncio.TimeoutError:
            logger.error(f"Extraction timed out (>90min) for JobID: {job_id}. Marking failed, moving on.")
            redis_manager.set_job_status(job_id, "failed")
            update_intake(job_id, status="failed", error="extraction timed out (>90min)")
            _persist_usage()
            return

        _persist_usage()
        result = redis_manager.get_job_result(job_id)
        if result:
            status = result.get("status")
            total = len(result.get("products", []))
            logger.info(f"Job {job_id} finished processing with status: {status} ({total} products written to Excel)")
            # clear_error: a job that was parked 'OpenAI out of credits' and then
            # resumed must not keep showing the stale outage error once it's done.
            if total > 0:
                update_intake(job_id, status="done", products=total, clear_error=True)
            else:
                # 0 offers: process_offer's own finalize already set the correct
                # terminal status — a buy-side 'request', or 'no_offer_content'
                # with its reason. Do NOT clobber it back to a green 'done'/0:
                # that both hid RFQs and was the "DONE – 0 products" bug (Samuel).
                logger.info(f"Job {job_id}: 0 offers — keeping the intake status "
                            f"process_offer set (request / no_offer_content)")
        else:
            logger.error(f"Job {job_id} finished but no result was found in RedisManager.")
            update_intake(job_id, status="done", products=0, clear_error=True)

    except OpenAICreditsExhausted as exc:
        # Out of OpenAI credits — park the job (don't mark done/failed). The
        # in-app scheduler re-queues it automatically once credits are back.
        logger.warning(f"Job {job_id} parked — OpenAI out of credits: {exc}")
        redis_manager.set_job_status(job_id, "no_credits")
        update_intake(job_id, status="no_credits",
                      error="OpenAI out of credits — will auto-retry when topped up")
        queue_pending_credit_job(job_id, payload_dict)
        return
    except SoftTimeLimitExceeded:
        # 10-minute soft limit hit — extraction was too slow. Mark failed and stop.
        logger.error(f"Celery soft time limit exceeded for JobID: {job_id}. Marking failed, no retry.")
        redis_manager.set_job_status(job_id, "failed")
        update_intake(job_id, status="failed", error="soft time limit exceeded")
        return
    except Exception as exc:
        logger.error(f"Processing failed in celery for JobID: {job_id}. Err: {exc}\n{traceback.format_exc()}")
        redis_manager.set_job_status(job_id, "failed")
        update_intake(job_id, status="failed", error=exc)
        raise self.retry(exc=exc, countdown=backoff(self.request.retries))


def backoff(retries: int) -> int:
    return 30 * (2 ** retries)


@celery_app.task(name="workers.celery_tasks.poll_email_offers")
def poll_email_offers():
    """Celery-beat entry point for the IMAP poll. The actual logic lives in
    core.email_poll.run_email_poll so the in-app scheduler (main.py) can run
    the exact same poll without depending on beat being started on prod."""
    from core.email_poll import run_email_poll
    return run_email_poll()


@celery_app.task(name="workers.celery_tasks.worker_code_probe")
def worker_code_probe():
    """Behavioural fingerprint of the CODE the worker is actually running, so a
    deploy that recycled the worker to a NEW hostname but STILL old code can be
    detected without a wasteful re-ingest (Samuel 2026-11-18: web had the new
    code but the Celery worker lagged a whole commit). `currency_invented` is
    False only on the field-accuracy build (032bed6+): clean_product_data no
    longer stamps EUR on a no-price / no-currency row."""
    import socket
    from core.openai_client import clean_product_data
    probe = clean_product_data({"product_name": "Probe RTD",
                                "price_per_unit": None, "price_per_case": None,
                                "currency": None})
    # Bump BUILD on every deploy whose worker behaviour must be confirmed before
    # a re-ingest. "procfix1" = the process_offer corruption fixes (dual-currency
    # over-propagation, EUR-forcing on no-price rows, dropped commercial_conditions).
    _cur_epoch = None
    try:
        _ce = redis_manager.client.get("deploy:current_epoch") if (redis_manager.use_redis and redis_manager.client) else None
        _cur_epoch = int(_ce) if _ce is not None else None
    except Exception:
        pass
    return {
        "host": socket.gethostname(),
        "build": "recon50",
        "my_epoch": _MY_EPOCH,
        "current_epoch": _cur_epoch,
        "is_current": (_MY_EPOCH is not None and _MY_EPOCH == _cur_epoch),
        "currency_invented": probe.get("currency") is not None,
        "probe_currency": probe.get("currency"),
    }


@celery_app.task(name="workers.celery_tasks.probe_extract")
def probe_extract(pattern: str):
    """Run the deterministic extractor + cleaner ON THE WORKER for a saved
    file and report the fields, so we can see exactly what the worker produces
    (no LLM, no persistence). Settles whether a wrong stored value comes from
    extraction or from a later pipeline step (Samuel 2026-11-18)."""
    import os, glob, socket, traceback
    out = {"host": socket.gethostname()}
    try:
        from core.file_store import _dir
        d = _dir()
        matches = sorted(f for f in glob.glob(os.path.join(d, "*"))
                         if pattern.lower() in os.path.basename(f).lower())
        out["dir"] = d
        out["matches"] = [os.path.basename(m) for m in matches]
        if not matches:
            out["error"] = "no file"
            return out
        from core.xlsx_structured import extract_structured_products
        from core.openai_client import clean_product_data, extract_from_file
        prods = extract_structured_products(matches[0])
        out["structured_count"] = None if prods is None else len(prods)
        rows = []
        for p in (prods or []):
            nm = (p.get("product_name") or "")
            if any(w in nm.lower() for w in ("oréal", "oreal", "cody", "beefeater", "nishane")):
                c = clean_product_data(dict(p))
                rows.append({"name": nm[:26], "cur": c.get("currency"),
                             "ppu": c.get("price_per_unit"), "g": c.get("size_weight_g"),
                             "ml": c.get("unit_volume_ml"),
                             "cc": (c.get("commercial_conditions") or "")[:24]})
        out["rows"] = rows
        # Also run the FULL extract_from_file exactly as process_offer calls it
        # (it may differ from extract_structured_products on the worker).
        import asyncio
        try:
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            ef = loop.run_until_complete(extract_from_file(
                matches[0],
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
            out["eff_source_type"] = ef.get("source_type")
            out["eff_rows"] = [
                {"name": ((pp.get("brand") or "") + " " + (pp.get("product_name") or "")).strip()[:26],
                 "cur": pp.get("currency")}
                for pp in (ef.get("products") or [])
                if any(w in ((pp.get("product_name") or "") + (pp.get("brand") or "")).lower()
                       for w in ("oréal", "oreal", "cody", "beefeater"))]
        except Exception as _e:
            out["eff_error"] = f"{type(_e).__name__}: {_e}"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["tb"] = traceback.format_exc()[-800:]
    return out


@celery_app.task(name="workers.celery_tasks.probe_pipeline")
def probe_pipeline(job_id: str):
    """Trace the post-extraction pipeline for a saved job: show what the
    document-level detectors return and whether apply_deterministic_defaults
    changes a no-price row's currency / a row's incoterm+conditions. No persist."""
    import asyncio, socket
    out = {"host": socket.gethostname()}
    try:
        from core.job_store import load_job_payload
        from core.deterministic_rules import (
            apply_deterministic_defaults, detect_document_currency,
            detect_document_incoterm, detect_document_location,
            detect_document_commercial_conditions)
        pl = load_job_payload(job_id)
        if not pl:
            return {**out, "error": "no payload"}
        tb = pl.get("text_body") or ""
        atts = pl.get("attachments") or []
        out["text_body_len"] = len(tb)
        out["attachments"] = [a.get("fileName") for a in atts]
        # Build products: attachment (structured) or text (LLM-free: use detectors only)
        products = []
        src_text = tb
        if atts:
            import os, glob
            from core.file_store import _dir
            from core.openai_client import extract_from_file
            fn = atts[0].get("fileName", "")
            m = sorted(f for f in glob.glob(os.path.join(_dir(), "*")) if fn.lower() in os.path.basename(f).lower())
            if m:
                loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                ef = loop.run_until_complete(extract_from_file(m[0], atts[0].get("contentType", "")))
                products = ef.get("products") or []
                if ef.get("source_text"):
                    src_text = (tb + "\n\n" + ef["source_text"]) if tb else ef["source_text"]
        header_text = (tb or "")
        if atts:
            header_text = f"Subject: {pl.get('subject') or ''}\nFrom: {pl.get('sender_email') or ''}\n\n" + header_text
        out["doc_currency"] = detect_document_currency(header_text)
        out["doc_incoterm"] = detect_document_incoterm(header_text)
        out["doc_location"] = detect_document_location(header_text)
        out["doc_conditions"] = (detect_document_commercial_conditions(header_text) or "")[:90]
        # also test the raw text_body directly
        out["tb_incoterm"] = detect_document_incoterm(tb)
        out["tb_conditions"] = (detect_document_commercial_conditions(tb) or "")[:90]
        # trace a no-price row's currency before/after defaults
        def pick(ps):
            for p in ps:
                if 'cody' in ((p.get('brand') or '')+(p.get('product_name') or '')).lower():
                    return p
            return ps[0] if ps else {}
        cody_before = pick(products).get("currency")
        corrected, _ = apply_deterministic_defaults(products, header_text,
                        source_filename=pl.get("source_filename"), sender_email=pl.get("sender_email"))
        cp = pick(corrected)
        out["cody_currency_before_defaults"] = cody_before
        out["cody_currency_after_defaults"] = cp.get("currency")
        out["row0_incoterm_after"] = (corrected[0].get("incoterm") if corrected else None)
        out["row0_conditions_after"] = (corrected[0].get("commercial_conditions") if corrected else None)
    except Exception as e:
        import traceback
        out["error"] = f"{type(e).__name__}: {e}"; out["tb"] = traceback.format_exc()[-700:]
    return out
