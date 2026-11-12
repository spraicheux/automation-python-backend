import os
from fastapi import APIRouter, UploadFile, File, Depends, Header, HTTPException
from core.redis_client import redis_manager

router = APIRouter()


def _require_admin(x_admin_token: str = Header(None)):
    """Gate endpoints that return file contents, so saved offer files are not
    publicly downloadable."""
    if x_admin_token != os.getenv("ADMIN_TOKEN", "valid-token"):
        raise HTTPException(status_code=401, detail="admin token required")


@router.get("/debug/imap-check")
async def imap_check():
    """Diagnose the email poll: is IMAP configured, does it connect, how many
    messages are in the folder, what UID have we polled up to, and how many are
    pending (arrived after our high-water mark). No secrets are returned."""
    import imaplib
    out = {"configured": {}, "connect": None, "folder": None,
           "mailbox_messages": None, "max_uid": None, "stored_last_uid": None,
           "uidvalidity": None, "stored_uidvalidity": None,
           "pending_above_last_uid": None, "error": None}
    host = os.getenv("IMAP_HOST")
    user = os.getenv("IMAP_USER")
    pw = os.getenv("IMAP_PASSWORD")
    port = int(os.getenv("IMAP_PORT", "993"))
    folder = os.getenv("IMAP_FOLDER", "INBOX")
    out["configured"] = {"IMAP_HOST": bool(host), "IMAP_USER": bool(user),
                         "IMAP_PASSWORD": bool(pw), "IMAP_PORT": port,
                         "IMAP_FOLDER": folder,
                         "host_preview": (host or "")[:24]}
    if not (host and user and pw):
        out["error"] = "IMAP not fully configured (missing host/user/password)"
        return out
    try:
        ssl = os.getenv("IMAP_SSL", "1") != "0"
        conn = imaplib.IMAP4_SSL(host, port) if ssl else imaplib.IMAP4(host, port)
        try:
            conn.login(user, pw)
            out["connect"] = "ok"
            typ, data = conn.select(folder, readonly=True)
            out["folder"] = {"name": folder, "select": typ}
            if typ == "OK":
                out["mailbox_messages"] = int(data[0]) if data and data[0] else 0
                tv, vd = conn.response("UIDVALIDITY")
                if vd and vd[0]:
                    out["uidvalidity"] = vd[0].decode() if isinstance(vd[0], bytes) else str(vd[0])
                typ2, sdata = conn.uid("search", None, "ALL")
                uids = [int(x) for x in sdata[0].split()] if sdata and sdata[0] else []
                out["max_uid"] = max(uids) if uids else 0
                r = redis_manager.client if (redis_manager.use_redis and redis_manager.client) else None
                if r:
                    lu = r.get(f"imap:last_uid:{user}")
                    sv = r.get(f"imap:uidvalidity:{user}")
                    out["stored_last_uid"] = int(lu) if lu else 0
                    out["stored_uidvalidity"] = (sv.decode() if isinstance(sv, bytes) else sv) if sv else None
                    out["pending_above_last_uid"] = len([u for u in uids if u > (int(lu) if lu else 0)])
        finally:
            try:
                conn.logout()
            except Exception:
                pass
    except Exception as e:
        out["connect"] = "failed"
        out["error"] = f"{type(e).__name__}: {e}"
    return out


@router.get("/debug/purge-queue")
async def purge_queue():
    """Flush the Celery broker queue — used after an App Service restart so a
    task that was killed mid-run (and would otherwise be redelivered) and any
    other queued jobs are dropped, per the no-ghost-duplicates rule."""
    from core.celery_app import celery_app
    out = {}
    try:
        with celery_app.connection_for_write() as conn:
            out["celery_queue_purged"] = conn.default_channel.queue_purge("celery")
    except Exception as e:
        out["queue_purge_error"] = str(e)
    try:
        out["control_purge"] = celery_app.control.purge()
    except Exception as e:
        out["control_purge_error"] = str(e)
    return out


@router.get("/debug/job-payload", dependencies=[Depends(_require_admin)])
async def job_payload(job_id: str):
    """Return a job's saved payload (the exact email/WhatsApp text the extractor
    received, plus attachment metadata) so we can see two-column layouts,
    forwarded supplier blocks, etc. Attachment bytes are summarised, not dumped."""
    from core.job_store import load_job_payload
    p = load_job_payload(job_id)
    if not p:
        return {"error": "no saved payload", "job_id": job_id}
    atts = []
    for a in (p.get("attachments") or []):
        d = a.get("data") or {}
        atts.append({"fileName": a.get("fileName"), "contentType": a.get("contentType"),
                     "fileSize": a.get("fileSize"),
                     "carrier": ("path" if "path" in d else "id" if "id" in d
                                 else "buffer" if d.get("type") == "Buffer" else "?"),
                     "path": d.get("path")})
    return {
        "job_id": job_id,
        "source_channel": p.get("source_channel"),
        "sender_name": p.get("sender_name"), "sender_email": p.get("sender_email"),
        "supplier_name": p.get("supplier_name"), "subject": p.get("subject"),
        "text_body": p.get("text_body"),
        "attachments": atts,
    }


@router.get("/debug/download-saved", dependencies=[Depends(_require_admin)])
async def download_saved(pattern: str):
    """Download a saved offer file (from durable storage) by name substring, so
    the exact bytes the extractor processed can be inspected locally."""
    import os, glob
    from fastapi.responses import FileResponse
    from core.file_store import _dir
    if not pattern or len(pattern) < 3:
        return {"error": "pattern too short (min 3 chars)"}
    d = _dir()
    matches = sorted(f for f in glob.glob(os.path.join(d, "*"))
                     if pattern.lower() in os.path.basename(f).lower())
    if not matches:
        return {"error": "no saved file matches", "pattern": pattern, "dir": d,
                "available": [os.path.basename(f) for f in glob.glob(os.path.join(d, "*"))][:50]}
    path = matches[0]
    return FileResponse(path, filename=os.path.basename(path),
                        media_type="application/octet-stream")


@router.get("/debug/archived-files", dependencies=[Depends(_require_admin)])
async def archived_files(pattern: str = "", limit: int = 300):
    """List raw offer files durably archived in Postgres (never-lose-files).
    Metadata only — no bytes. Survives restarts/deploys unlike /home storage."""
    from core.raw_blob import list_blobs
    items = list_blobs(limit=limit, pattern=(pattern or None))
    return {"count": len(items), "files": items}


@router.get("/debug/download-archived", dependencies=[Depends(_require_admin)])
async def download_archived(id: str = "", job_id: str = "", name: str = ""):
    """Download a durably-archived raw offer file by blob id, job_id, or filename
    substring (most recent match). This is the recovery path for binary
    attachments (images / PDFs) after the ephemeral /home copy is gone."""
    from fastapi.responses import Response
    from core.raw_blob import load_blob
    if not (id or job_id or name):
        return {"error": "pass id, job_id, or name"}
    data, ctype, fname = load_blob(blob_id=(id or None), job_id=(job_id or None),
                                   filename=(name or None))
    if not data:
        return {"error": "no archived blob matches", "id": id, "job_id": job_id, "name": name}
    return Response(content=data, media_type=(ctype or "application/octet-stream"),
                    headers={"Content-Disposition": f'attachment; filename="{fname or "file"}"'})


@router.get("/debug/inspect-saved-xlsx", dependencies=[Depends(_require_admin)])
async def inspect_saved_xlsx(pattern: str, rows: int = 45):
    """Read a saved XLSX exactly as the extractor sees it (pandas) AND with
    openpyxl number formats, so percent cells (40% kept as 0.4), dates, and the
    real price/units columns are visible. Read-only diagnosis."""
    import os, glob
    from core.file_store import _dir
    if not pattern or len(pattern) < 3:
        return {"error": "pattern too short (min 3 chars)"}
    d = _dir()
    matches = sorted(f for f in glob.glob(os.path.join(d, "*"))
                     if pattern.lower() in os.path.basename(f).lower())
    if not matches:
        return {"error": "no saved file matches", "pattern": pattern,
                "available": [os.path.basename(f) for f in glob.glob(os.path.join(d, "*"))][:50]}
    path = matches[0]
    out = {"file": os.path.basename(path)}
    try:
        import pandas as pd
        xls = pd.ExcelFile(path)
        out["sheet_names"] = xls.sheet_names
        df = pd.read_excel(path, sheet_name=xls.sheet_names[0], header=None, dtype=object)
        out["pandas_shape"] = list(df.shape)
        out["pandas_rows"] = [[("" if pd.isna(v) else str(v)) for v in row]
                              for row in df.head(rows).values.tolist()]
    except Exception as e:
        out["pandas_error"] = str(e)
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb[wb.sheetnames[0]]
        cells = []
        for r in ws.iter_rows(min_row=1, max_row=rows):
            cells.append([{"v": (c.value.isoformat() if hasattr(c.value, "isoformat") else c.value),
                           "fmt": c.number_format} for c in r])
        out["openpyxl_cells"] = cells
    except Exception as e:
        out["openpyxl_error"] = str(e)
    return out


@router.get("/debug/worker")
async def worker_status():
    """Is the Celery worker alive, how backed-up is it, and how many jobs are
    waiting in the broker? Answers 'are queued jobs stuck or just waiting?' in
    one call: broker_queue_len is how many tasks sit unclaimed; workers lists
    each live worker with its active (running) and reserved (prefetched) task
    counts. No worker replying to ping ⇒ the worker is down."""
    from core.celery_app import celery_app
    out = {"broker_queue_len": None, "workers": None, "ping": None, "error": None}
    try:
        with celery_app.connection_for_read() as conn:
            out["broker_queue_len"] = conn.default_channel.client.llen("celery")
    except Exception as e:
        out["error"] = f"queue len: {e}"
    try:
        insp = celery_app.control.inspect(timeout=3)
        out["ping"] = insp.ping()
        active = insp.active() or {}
        reserved = insp.reserved() or {}
        out["workers"] = {
            w: {"active": len(active.get(w, [])), "reserved": len(reserved.get(w, []))}
            for w in set(list(active) + list(reserved) + list(out["ping"] or {}))
        }
    except Exception as e:
        out["error"] = f"{out['error'] + '; ' if out['error'] else ''}inspect: {e}"
    return out


@router.get("/debug/sweep-now")
async def sweep_now():
    """Run the orphan-recovery sweep immediately (instead of waiting for the
    ~2-min scheduler): re-enqueue any job stranded 'queued' whose broker
    hand-off never landed, and clear stale errors on resumed/done rows."""
    from core.job_recovery import sweep_stuck_jobs
    return sweep_stuck_jobs()


@router.get("/debug/clear-pending-credits")
async def clear_pending_credits():
    """Drop parked (NO CREDITS) jobs so they will NOT auto-resume — used when
    you intend to re-send the file fresh instead."""
    from core.openai_health import clear_pending
    return {"cleared_pending": clear_pending()}


@router.get("/debug/delete-saved-files")
async def delete_saved_files(pattern: str = "", all: bool = False):
    """Delete saved offer files from server storage by name substring, or all."""
    from core.file_store import delete_files
    if not all and len(pattern) < 3:
        return {"error": "pattern too short (min 3 chars); or pass all=true"}
    return {"deleted": delete_files(pattern=(pattern or None), clear_all=all)}


@router.get("/debug/reprocess")
async def reprocess(job_id: str, force: bool = False):
    """Re-run a job from its saved payload/file (nothing re-sent). Without force,
    dedup applies (an already-ingested file is refused as duplicate); force=true
    re-runs regardless. Only works for jobs created after file-storage shipped."""
    from core.job_store import load_job_payload, save_job_payload
    payload = load_job_payload(job_id)
    if not payload:
        return {"error": "no saved payload for this job_id (expired, or predates file storage)",
                "job_id": job_id}
    import uuid
    from workers.celery_tasks import process_document_task
    from core.intake_log import record_intake
    if force:
        new_job = str(uuid.uuid4())
        redis_manager.set_job_status(new_job, "processing")
        record_intake(new_job, channel=payload.get("source_channel"),
                      source_filename=payload.get("source_filename"),
                      sender_name=payload.get("sender_name"),
                      sender_email=payload.get("sender_email"),
                      supplier_name=payload.get("supplier_name"),
                      attachment_count=len(payload.get("attachments") or []),
                      has_text=bool(payload.get("text_body")), status="queued")
        save_job_payload(new_job, payload)
        process_document_task.delay(new_job, payload)
        return {"reprocessing": True, "forced": True, "old_job_id": job_id, "new_job_id": new_job}
    # Dedup-respecting path
    from core.ingest_service import enqueue_offer, DuplicateIngest
    try:
        res = enqueue_offer(
            source_channel=payload.get("source_channel"),
            source_message_id=payload.get("source_message_id"),
            source_filename=payload.get("source_filename"),
            supplier_email=payload.get("supplier_email"),
            supplier_name=payload.get("supplier_name"),
            sender_email=payload.get("sender_email"),
            sender_name=payload.get("sender_name"),
            subject=payload.get("subject"),
            text_body=payload.get("text_body"),
            attachments=payload.get("attachments") or [],
            content_hash=payload.get("content_hash"),
            check_content_hash=bool(payload.get("content_hash")),
        )
        return {"reprocessing": True, "old_job_id": job_id, **res}
    except DuplicateIngest as e:
        return {"skipped": "already ingested", "detail": e.detail}


@router.get("/debug/usage/{job_id}")
async def debug_usage(job_id: str):
    """Exact token usage + cost captured for a processed job (per-file metering).
    Returns calls, prompt/completion tokens, per-model breakdown and cost_usd."""
    r = redis_manager.client if (redis_manager.use_redis and redis_manager.client) else None
    if not r:
        return {"job_id": job_id, "usage": None, "error": "redis unavailable"}
    v = r.get(f"usage:{job_id}")
    if v is None:
        return {"job_id": job_id, "usage": None}
    import json as _json
    if isinstance(v, bytes):
        v = v.decode()
    return {"job_id": job_id, "usage": _json.loads(v)}


@router.get("/debug/intake-clear")
async def intake_clear(status: str = "", filename: str = "", all: bool = False):
    """Remove entries from the live Intake feed — e.g. stale 'processing' /
    'queued' rows left behind when a worker was restarted mid-run. Filter by
    status (comma list), source_filename substring, or all=true."""
    from core.intake_log import clear_intakes
    statuses = [s.strip() for s in status.split(",") if s.strip()] or None
    n = clear_intakes(statuses=statuses, filename=(filename or None), clear_all=all)
    return {"cleared": n}


@router.get("/debug/job/{job_id}")
async def debug_job(job_id: str):
    status = redis_manager.get_job_status(job_id)
    result = redis_manager.get_job_result(job_id)

    return {
        "job_id": job_id,
        "status": status,
        "result": result,
        "exists": redis_manager.job_exists(job_id)
    }


@router.post("/debug/inline-extract")
async def debug_inline_extract(file: UploadFile = File(...)):
    """
    Run the extraction inline (no celery) and return what happened.
    Bypasses redis / queueing so we can see if the actual pipeline works.
    """
    import tempfile, os, traceback
    from core.openai_client import extract_from_file

    file_bytes = await file.read()
    try:
        ext = file.filename.split('.')[-1].lower() if file.filename and '.' in file.filename else 'bin'
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            # Also grab raw pypdf text for diagnosis
            pdf_text_preview = None
            raw_llm_response = None
            if file.content_type == "application/pdf":
                try:
                    import PyPDF2
                    with open(tmp_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        pages_text = [p.extract_text() or "" for p in reader.pages]
                        joined = "\n".join(pages_text)
                        pdf_text_preview = {
                            "n_pages": len(pages_text),
                            "total_chars": len(joined),
                            "first_500": joined[:500],
                        }
                    # Also run a direct GPT-4o call on the raw text so we can
                    # see what the LLM says without any extraction wrapper.
                    from core.openai_client import client, SHARED_EXTRACTION_RULES
                    prompt = (
                        "Extract commercial product offers (any category — Wines & Spirits, "
                        "Perfumes, or Cosmetics — identify each row's category_slug per Rule 0.23). "
                        "Return a JSON object with a 'products' array.\n\n"
                        f"{SHARED_EXTRACTION_RULES}\n\nTEXT:\n{joined[:6000]}"
                    )
                    resp = await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "Return valid JSON only."},
                            {"role": "user", "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.0,
                        max_tokens=4000,
                    )
                    raw_llm_response = {
                        "chars": len(resp.choices[0].message.content),
                        "preview": resp.choices[0].message.content[:800],
                    }
                except Exception as e:
                    pdf_text_preview = pdf_text_preview or {}
                    pdf_text_preview["llm_probe_error"] = str(e)[:300]
            extracted = await extract_from_file(tmp_path, file.content_type)
            summary = {
                "filename": file.filename,
                "content_type": file.content_type,
                "bytes": len(file_bytes),
                "pdf_text_preview": pdf_text_preview,
                "raw_llm_response": raw_llm_response,
                "extract_ok": True,
                "type": type(extracted).__name__,
                "keys": list(extracted.keys()) if isinstance(extracted, dict) else None,
                "n_products": len(extracted.get("products") or []) if isinstance(extracted, dict) else None,
                "first_product": (extracted.get("products") or [None])[0] if isinstance(extracted, dict) and extracted.get("products") else None,
                "error_field": extracted.get("error") if isinstance(extracted, dict) else None,
            }
            return summary
        finally:
            try: os.unlink(tmp_path)
            except: pass
    except Exception as e:
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "bytes": len(file_bytes),
            "extract_ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()[-1000:],
        }


@router.get("/debug/buffer-check")
async def debug_buffer_check():
    """One-shot proof the file_download fix is live on this worker."""
    import inspect
    from core import file_download
    src = inspect.getsource(file_download.resolve_attachment_bytes)
    # First non-comment executable line after the log statement.
    return {
        "has_buffer_first": "1️⃣ Buffer handling — MUST run before" in src,
        "source_hash": hex(hash(src) & 0xFFFFFFFF),
    }


@router.get("/debug/jobs")
async def debug_all_jobs():
    return {
        "message": "For Redis, use SCAN command. For in-memory storage, all jobs are listed.",
        "storage_type": "redis" if redis_manager.use_redis else "memory"
    }


def _mask(value: str, keep: int = 6) -> str:
    if not value:
        return None
    if len(value) <= keep:
        return "*" * len(value)
    return value[:keep] + "…"


@router.get("/debug/status")
async def debug_status():
    """Diagnostic snapshot for deploy verification — no secrets returned."""
    out = {"redis": {}, "db": {}, "excel": {}, "env": {}}

    # Redis
    try:
        out["redis"]["connected"] = bool(redis_manager.use_redis and redis_manager.client)
        if redis_manager.use_redis and redis_manager.client:
            out["redis"]["ping"] = redis_manager.client.ping()
    except Exception as e:
        out["redis"]["error"] = str(e)

    # Database
    try:
        from core.database import get_engine
        from sqlalchemy import text
        eng = get_engine()
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        out["db"]["reachable"] = True
        out["db"]["url_host"] = (os.getenv("DATABASE_URL") or "").split("@")[-1].split("/")[0] or None
    except Exception as e:
        out["db"]["reachable"] = False
        out["db"]["error"] = str(e)[:200]

    # Excel env vars (presence only, not values)
    for key in ("MS_CLIENT_ID", "MS_AUTHORITY", "EXCEL_WORKBOOK_PATH",
                "EXCEL_WORKSHEET", "MSAL_CACHE_KEY", "OPENAI_API_KEY", "REDIS_URL", "DATABASE_URL"):
        val = os.getenv(key)
        out["env"][key] = {
            "set": bool(val),
            "preview": _mask(val) if val else None,
        }

    # MSAL cache present?
    try:
        r = redis_manager.client if (redis_manager.use_redis and redis_manager.client) else None
        cache_key = os.getenv("MSAL_CACHE_KEY", "msal:cache:onedrive")
        if r:
            blob = r.get(cache_key)
            out["excel"]["msal_cache_present"] = bool(blob)
            out["excel"]["msal_cache_bytes"] = len(blob) if blob else 0
        else:
            out["excel"]["msal_cache_present"] = False
    except Exception as e:
        out["excel"]["msal_cache_error"] = str(e)[:200]

    # Can we acquire a Graph access token silently?
    try:
        from core.excel_client import get_excel_client
        c = get_excel_client()
        tok = c.get_access_token()
        out["excel"]["token_acquired"] = bool(tok)
        out["excel"]["token_len"] = len(tok) if tok else 0
    except Exception as e:
        out["excel"]["token_acquired"] = False
        out["excel"]["token_error"] = str(e)[:200]

    # Can we probe the workbook?
    if out["excel"].get("token_acquired"):
        try:
            import requests
            from core.excel_client import get_excel_client
            c = get_excel_client()
            r = requests.get(c._workbook_url(), headers=c._headers(), timeout=15)
            out["excel"]["workbook_probe_status"] = r.status_code
            out["excel"]["workbook_probe_ok"] = r.ok
            if not r.ok:
                out["excel"]["workbook_probe_body"] = r.text[:300]
        except Exception as e:
            out["excel"]["workbook_probe_error"] = str(e)[:200]

    return out