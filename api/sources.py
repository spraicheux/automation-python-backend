import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.database import get_db
from models.offer_item import OfferItemDB
from models.source_file import SourceFileDB
from sqlalchemy import func


logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/openai-status")
async def openai_status(probe: bool = False):
    """Live OpenAI health for the dashboard: ok True/False/None, the last error
    (e.g. 'no credits remaining'), when it was last checked, and how many jobs
    are currently parked waiting for credits. OpenAI exposes no dollar balance
    via the API key, so this is a reliable OK / EXHAUSTED signal, not an amount.

    Probes OpenAI when the status is unknown or older than 5 min (or ?probe=true)
    so the chip reflects reality even when nothing is extracting. A probe is a
    1-token call when credits exist, and free (rejected) when they don't."""
    import time
    import calendar
    import asyncio
    from core.openai_health import get_openai_status, probe_openai
    st = get_openai_status()
    stale = st.get("ok") is None
    if not stale and st.get("checked_at"):
        try:
            t = calendar.timegm(time.strptime(st["checked_at"], "%Y-%m-%dT%H:%M:%SZ"))
            stale = (time.time() - t) > 300
        except Exception:
            stale = True
    if probe or stale:
        try:
            await asyncio.get_running_loop().run_in_executor(None, probe_openai)
            st = get_openai_status()
        except Exception:
            pass
    return st


@router.get("/intake-status")
async def intake_status(limit: int = 50):
    """Live intake feed (Redis-backed): every inbound email/WhatsApp message the
    instant it arrives, with status received/queued → processing → done | failed
    and the error on failure. Lets the dashboard show a file immediately while
    extraction is still running, and surfaces failed files."""
    from core.intake_log import list_intakes
    items = list_intakes(limit=limit)
    counts = {}
    for it in items:
        counts[it.get("status")] = counts.get(it.get("status"), 0) + 1
    return {"total": len(items), "counts": counts, "intakes": items}


@router.get("/sources")
async def get_sources(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    total = db.query(SourceFileDB).count()

    sources = (
        db.query(SourceFileDB)
        .order_by(SourceFileDB.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    if not sources:
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "sources": []
        }

    source_ids = [s.id for s in sources]

    counts = (
        db.query(
            OfferItemDB.source_file_id,
            func.count(OfferItemDB.uid)
        )
        .filter(OfferItemDB.source_file_id.in_(source_ids))
        .group_by(OfferItemDB.source_file_id)
        .all()
    )

    count_map = {sid: count for sid, count in counts}

    result = []
    for source in sources:
        data = source.to_dict()
        data["product_count"] = count_map.get(source.id, 0)
        result.append(data)

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "sources": result
    }