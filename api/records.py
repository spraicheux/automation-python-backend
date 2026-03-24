import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from core.database import get_db
from models.offer_item import OfferItemDB
from models.source_file import SourceFileDB

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/records")
async def get_records(
    skip: int = Query(0),
    limit: int = Query(24),
    source_file_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    query = db.query(OfferItemDB)

    if source_file_id:
        query = query.filter(OfferItemDB.source_file_id == source_file_id)

    total = query.count()
    rows = query.offset(skip).limit(limit).all()

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "records": [row.to_dict() for row in rows],
    }