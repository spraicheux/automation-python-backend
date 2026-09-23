"""
Manual offer entry + edit endpoints (Milestone 4).

Two entrypoints:
- POST /api/offers        — create a new offer manually (phone / meeting)
- PATCH /api/offers/{uid} — edit any field on an existing offer

Every edit is also logged to `offer_corrections` so we can later feed systematic
patterns back into the extractor (learning loop — MVP scope for now).
"""
import json
import uuid
from datetime import datetime
from typing import Optional, Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.orm import Session

from core.database import Base, get_db, get_engine
from models.offer_item import OfferItemDB
from models.source_file import SourceFileDB

router = APIRouter()


class OfferCorrectionDB(Base):
    """Every manual edit / correction is logged here for later analysis."""
    __tablename__ = "offer_corrections"
    id = Column(String(64), primary_key=True)
    offer_uid = Column(String(64), nullable=False, index=True)
    field = Column(String(128), nullable=False)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    edited_by = Column(String(255), nullable=True)
    edited_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# Ensure the corrections table exists (idempotent — only creates if missing)
def _ensure_corrections_table():
    try:
        Base.metadata.create_all(bind=get_engine(), tables=[OfferCorrectionDB.__table__])
    except Exception:
        pass


class OfferCreate(BaseModel):
    """Fields accepted when creating a new offer manually."""
    product_name: str
    brand: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    packaging: Optional[str] = None
    bottle_or_can_type: Optional[str] = None
    unit_volume_ml: Optional[float] = None
    units_per_case: Optional[float] = None
    cases_per_pallet: Optional[float] = None
    quantity_case: Optional[float] = None
    gift_box: Optional[str] = None
    refillable_status: Optional[str] = None
    alcohol_percent: Optional[float] = None
    origin_country: Optional[str] = None
    vintage: Optional[str] = None
    ean_code: Optional[str] = None
    label_language: Optional[str] = None
    product_reference: Optional[str] = None

    currency: Optional[str] = "EUR"
    price_per_unit: Optional[float] = None
    price_per_case: Optional[float] = None
    fx_rate: Optional[float] = 1.0

    supplier_name: Optional[str] = None
    supplier_email: Optional[str] = None
    supplier_country: Optional[str] = None
    supplier_reference: Optional[str] = None
    sender_name: Optional[str] = None
    sender_email: Optional[str] = None

    incoterm: Optional[str] = None
    location: Optional[str] = None
    lead_time: Optional[str] = None
    moq_cases: Optional[float] = None
    valid_until: Optional[str] = None
    best_before_date: Optional[str] = None
    offer_date: Optional[str] = None
    custom_status: Optional[str] = None

    source_channel: Optional[str] = "manual"
    notes: Optional[str] = None
    edited_by: Optional[str] = "admin"


class OfferPatch(BaseModel):
    """Any subset of OfferItemDB fields can be patched."""
    fields: Dict[str, Any]
    edited_by: Optional[str] = "admin"


_EDITABLE_FIELDS = {
    "product_name", "product_key", "brand", "category", "sub_category",
    "packaging", "packaging_raw", "bottle_or_can_type", "unit_volume_ml",
    "units_per_case", "cases_per_pallet", "quantity_case", "gift_box",
    "refillable_status", "currency", "price_per_unit", "price_per_unit_eur",
    "price_per_case", "price_per_case_eur", "fx_rate", "fx_date",
    "alcohol_percent", "origin_country", "supplier_country", "incoterm",
    "location", "lead_time", "moq_cases", "valid_until", "best_before_date",
    "vintage", "ean_code", "label_language", "product_reference",
    "supplier_name", "supplier_email", "supplier_reference", "sender_name",
    "sender_email", "source_channel", "source_message_id", "source_filename",
    "custom_status", "confidence_score", "needs_manual_review", "offer_date",
}


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s[:19].replace("Z", ""))
    except Exception:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except Exception:
            return None


@router.post("/offers")
def create_offer(payload: OfferCreate, db: Session = Depends(get_db)):
    """Create a new offer manually (phone / meeting call). Materialises a
    SourceFileDB with source_channel='manual' so the row still has traceability."""
    _ensure_corrections_table()

    now = datetime.utcnow()
    src = SourceFileDB(
        id=str(uuid.uuid4()),
        job_id=f"manual-{uuid.uuid4().hex[:12]}",
        source_filename=f"manual-{now.strftime('%Y%m%d-%H%M%S')}",
        sender_name=payload.sender_name or payload.edited_by,
        sender_email=payload.sender_email,
        supplier_name=payload.supplier_name,
        supplier_email=payload.supplier_email,
        source_channel=payload.source_channel or "manual",
        source_message_id=None,
        product_count=1,
    )
    db.add(src)
    db.flush()

    # Compute EUR prices if only native price given (assume EUR by default)
    price_unit_eur = payload.price_per_unit
    price_case_eur = payload.price_per_case
    fx = payload.fx_rate or 1.0
    if payload.currency and payload.currency.upper() != "EUR":
        if payload.price_per_unit is not None:
            price_unit_eur = round(payload.price_per_unit * fx, 4)
        if payload.price_per_case is not None:
            price_case_eur = round(payload.price_per_case * fx, 4)

    row = OfferItemDB(
        uid=str(uuid.uuid4()),
        source_file_id=src.id,
        job_id=src.job_id,
        product_name=payload.product_name,
        product_key=(payload.brand or "") + "_" + payload.product_name,
        brand=payload.brand,
        category=payload.category,
        sub_category=payload.sub_category,
        packaging=payload.packaging or "Bottle",
        packaging_raw=payload.packaging or "bottle",
        bottle_or_can_type=payload.bottle_or_can_type,
        unit_volume_ml=payload.unit_volume_ml,
        units_per_case=payload.units_per_case,
        cases_per_pallet=payload.cases_per_pallet,
        quantity_case=payload.quantity_case,
        gift_box=payload.gift_box,
        refillable_status=payload.refillable_status,
        currency=(payload.currency or "EUR").upper(),
        price_per_unit=payload.price_per_unit,
        price_per_unit_eur=price_unit_eur,
        price_per_case=payload.price_per_case,
        price_per_case_eur=price_case_eur,
        fx_rate=fx,
        alcohol_percent=payload.alcohol_percent,
        origin_country=payload.origin_country,
        supplier_country=payload.supplier_country,
        incoterm=payload.incoterm,
        location=payload.location,
        lead_time=payload.lead_time,
        moq_cases=payload.moq_cases,
        valid_until=payload.valid_until,
        best_before_date=payload.best_before_date,
        vintage=payload.vintage,
        ean_code=payload.ean_code,
        label_language=payload.label_language,
        product_reference=payload.product_reference,
        supplier_name=payload.supplier_name,
        supplier_email=payload.supplier_email,
        supplier_reference=payload.supplier_reference,
        sender_name=payload.sender_name,
        sender_email=payload.sender_email,
        source_channel=payload.source_channel or "manual",
        source_filename=src.source_filename,
        source_message_id=payload.notes,
        custom_status=payload.custom_status,
        confidence_score=1.0,          # manually entered → full confidence
        needs_manual_review=False,
        error_flags=json.dumps([]),
        processing_version="manual-1.0",
        offer_date=_parse_iso(payload.offer_date) or datetime.utcnow(),
        date_received=datetime.utcnow(),
        attachment_filenames=json.dumps([]),
        attachment_count=0,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"status": "ok", "uid": row.uid, "source_file_id": src.id}


@router.patch("/offers/{uid}")
def edit_offer(uid: str, payload: OfferPatch, db: Session = Depends(get_db)):
    """Patch any subset of fields on an existing offer. Every changed field is
    logged to offer_corrections."""
    _ensure_corrections_table()

    row = db.query(OfferItemDB).filter(OfferItemDB.uid == uid).first()
    if not row:
        raise HTTPException(404, f"offer {uid} not found")

    changes = []
    for k, v in payload.fields.items():
        if k not in _EDITABLE_FIELDS:
            continue
        old = getattr(row, k, None)
        if old == v:
            continue
        # Cast numerics
        if k in ("unit_volume_ml", "units_per_case", "cases_per_pallet",
                 "quantity_case", "moq_cases", "alcohol_percent",
                 "price_per_unit", "price_per_case", "price_per_unit_eur",
                 "price_per_case_eur", "fx_rate", "confidence_score"):
            try: v = float(v) if v not in (None, "") else None
            except Exception: continue
        if k in ("needs_manual_review",):
            v = bool(v)
        if k == "offer_date":
            parsed = _parse_iso(str(v))
            if parsed: v = parsed
        setattr(row, k, v)
        changes.append((k, old, v))

    # Recompute EUR prices if currency + native price changed together
    cur = (row.currency or "EUR").upper()
    fx = row.fx_rate or 1.0
    if cur == "EUR":
        if row.price_per_unit is not None: row.price_per_unit_eur = round(float(row.price_per_unit), 4)
        if row.price_per_case is not None: row.price_per_case_eur = round(float(row.price_per_case), 4)
    else:
        if row.price_per_unit is not None: row.price_per_unit_eur = round(float(row.price_per_unit) * fx, 4)
        if row.price_per_case is not None: row.price_per_case_eur = round(float(row.price_per_case) * fx, 4)

    for k, old, new in changes:
        db.add(OfferCorrectionDB(
            id=str(uuid.uuid4()), offer_uid=uid, field=k,
            old_value=str(old) if old is not None else None,
            new_value=str(new) if new is not None else None,
            edited_by=payload.edited_by or "admin",
        ))

    db.commit()
    return {"status": "ok", "uid": uid, "changed": [c[0] for c in changes]}


@router.get("/offers/{uid}/history")
def offer_history(uid: str, db: Session = Depends(get_db)):
    """Return the correction history for one offer."""
    _ensure_corrections_table()
    rows = (db.query(OfferCorrectionDB)
              .filter(OfferCorrectionDB.offer_uid == uid)
              .order_by(OfferCorrectionDB.edited_at.desc())
              .limit(100).all())
    return {"uid": uid, "history": [{
        "field": r.field, "old": r.old_value, "new": r.new_value,
        "edited_by": r.edited_by, "edited_at": r.edited_at.isoformat(),
    } for r in rows]}
