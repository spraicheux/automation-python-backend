"""
R-12 persistent trading events.

/api/events returns the full history of detected best-price / competitive-
supplier events, newest first. Each event has a stable id so Samuel's
acceptance review can reference them by id and replay the lineage
"A01 €25 → B01 €24 → C01 €23.50" even after the latest batch is loaded.
"""
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from core.database import get_db
from models.price_event import PriceEventDB

router = APIRouter()


def _event_to_dict(e: PriceEventDB) -> dict:
    return {
        "id": e.id,
        "type": e.event_type,
        "peer_group_id": e.peer_group_id,
        "current": {
            "uid": e.current_uid,
            "supplier": e.current_supplier,
            "price_eur": e.current_price_eur,
            "offer_date": e.current_offer_date.isoformat() if e.current_offer_date else None,
            "source_filename": e.current_source_file,
        },
        "previous_best": {
            "uid": e.previous_uid,
            "supplier": e.previous_supplier,
            "price_eur": e.previous_price_eur,
            "offer_date": e.previous_offer_date.isoformat() if e.previous_offer_date else None,
            "source_filename": e.previous_source_file,
            "delta_pct": e.delta_pct,
            "delta_abs": e.delta_abs,
        },
        "message": e.message,
        "created_at": e.created_at.isoformat() if e.created_at else None,
    }


@router.get("/events")
def list_events(
    peer_group_id: Optional[str] = Query(None, description="Filter to one peer group"),
    limit: int = Query(200),
    db: Session = Depends(get_db),
):
    q = db.query(PriceEventDB)
    if peer_group_id:
        q = q.filter(PriceEventDB.peer_group_id == peer_group_id)
    q = q.order_by(PriceEventDB.created_at.desc()).limit(limit)
    events = [_event_to_dict(e) for e in q.all()]
    return {"count": len(events), "events": events}
