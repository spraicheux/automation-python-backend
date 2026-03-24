"""
api/records.py
GET /api/records?filename=<filename>
Returns all extracted products saved to the database for a given source filename.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from core.database import get_db
from models.offer_item import OfferItemDB

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/records")
async def get_records_by_filename(
    filename: str = Query(..., description="The source filename to look up"),
    db: Session = Depends(get_db),
):
    """
    Return all offer items stored in the database for the given filename.

    - **filename**: exact match against `source_filename` column
    - Returns 404 if no records are found for that filename
    """
    rows = (
        db.query(OfferItemDB)
        .filter(OfferItemDB.source_filename == filename)
        .all()
    )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No records found for filename '{filename}'. "
                   "Either it was never processed or was rejected as a duplicate.",
        )

    products = [row.to_dict() for row in rows]

    return {
        "filename": filename,
        "job_id": rows[0].job_id if rows else None,
        "total": len(products),
        "products": products,
    }
