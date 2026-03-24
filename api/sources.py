import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.database import get_db
from models.offer_item import OfferItemDB
from models.source_file import SourceFileDB
from sqlalchemy import func


logger = logging.getLogger(__name__)

router = APIRouter()

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