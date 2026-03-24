import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from core.database import get_db
from models.offer_item import OfferItemDB
from models.source_file import SourceFileDB

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/records")
async def get_records_by_filename(
    filename: str = Query(..., description="The source filename to look up"),
    db: Session = Depends(get_db),
):
    source_file = (
        db.query(SourceFileDB)
        .filter(SourceFileDB.source_filename == filename)
        .first()
    )

    if not source_file:
        raise HTTPException(
            status_code=404,
            detail=f"No source file found for filename '{filename}'.",
        )

    rows = (
        db.query(OfferItemDB)
        .filter(OfferItemDB.source_file_id == source_file.id)
        .all()
    )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No products found for filename '{filename}'.",
        )

    products = [row.to_dict() for row in rows]

    return {
        "filename": filename,
        "source_file_id": source_file.id,
        "job_id": source_file.job_id,
        "total": len(products),
        "products": products,
    }