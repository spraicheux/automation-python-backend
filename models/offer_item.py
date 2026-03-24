"""
models/offer_item.py
SQLAlchemy ORM model for extracted OfferItem rows.
Each row is tied to `job_id` and `source_filename`.
`source_filename` has a UNIQUE constraint so duplicate files are rejected at the DB level too.
"""
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Boolean, Integer,
    DateTime, Text, UniqueConstraint, Index
)
from core.database import Base


class OfferItemDB(Base):
    __tablename__ = "offer_items"

    # Primary key — same uid as OfferItem Pydantic model
    uid = Column(String(64), primary_key=True)

    # Job tracking
    job_id = Column(String(64), nullable=False, index=True)

    # Core product fields
    product_name = Column(String(512), nullable=True)
    product_key = Column(String(255), nullable=True)
    brand = Column(String(255), nullable=True)
    category = Column(String(255), nullable=True)
    sub_category = Column(String(255), nullable=True)
    packaging = Column(String(255), nullable=True)
    packaging_raw = Column(String(255), nullable=True)
    bottle_or_can_type = Column(String(255), nullable=True)
    unit_volume_ml = Column(Float, nullable=True)
    units_per_case = Column(Float, nullable=True)
    cases_per_pallet = Column(Float, nullable=True)
    quantity_case = Column(Float, nullable=True)
    gift_box = Column(String(255), nullable=True)
    refillable_status = Column(String(64), nullable=True)

    # Pricing
    currency = Column(String(16), nullable=True)
    price_per_unit = Column(Float, nullable=True)
    price_per_unit_eur = Column(Float, nullable=True)
    price_per_case = Column(Float, nullable=True)
    price_per_case_eur = Column(Float, nullable=True)
    fx_rate = Column(Float, nullable=True)
    fx_date = Column(String(64), nullable=True)

    # Product details
    alcohol_percent = Column(Float, nullable=True)
    origin_country = Column(String(255), nullable=True)
    supplier_country = Column(String(255), nullable=True)
    incoterm = Column(String(128), nullable=True)
    location = Column(String(255), nullable=True)
    lead_time = Column(String(255), nullable=True)
    moq_cases = Column(Float, nullable=True)
    valid_until = Column(String(64), nullable=True)
    best_before_date = Column(String(64), nullable=True)
    vintage = Column(String(64), nullable=True)
    ean_code = Column(String(64), nullable=True)
    label_language = Column(String(32), nullable=True)
    product_reference = Column(String(255), nullable=True)

    # Supplier / sender
    supplier_name = Column(String(255), nullable=True)
    supplier_email = Column(String(255), nullable=True)
    supplier_reference = Column(String(255), nullable=True)
    sender_name = Column(String(255), nullable=True)
    sender_email = Column(String(255), nullable=True)

    # Source / tracking
    source_channel = Column(String(128), nullable=True)
    source_message_id = Column(String(255), nullable=True)

    # ── UNIQUE on source_filename ────────────────────────────────────────────
    # A NULL filename is allowed (e.g. plain-text emails with no attachment),
    # but any non-NULL filename must be globally unique.
    source_filename = Column(String(512), nullable=True, unique=True, index=True)

    attachment_filenames = Column(Text, nullable=True)   # JSON array stored as text
    attachment_count = Column(Integer, nullable=True)

    # Quality / review
    confidence_score = Column(Float, nullable=True)
    needs_manual_review = Column(Boolean, nullable=True)
    error_flags = Column(Text, nullable=True)            # JSON array stored as text
    custom_status = Column(String(128), nullable=True)
    processing_version = Column(String(32), nullable=True)

    # Timestamps
    offer_date = Column(DateTime, nullable=True)
    date_received = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        # Additional composite index for job_id + filename queries
        Index("ix_offer_items_job_filename", "job_id", "source_filename"),
    )

    def to_dict(self) -> dict:
        """Return a plain dict representation (mirrors OfferItem schema)."""
        import json
        return {
            "uid": self.uid,
            "job_id": self.job_id,
            "product_name": self.product_name,
            "product_key": self.product_key,
            "brand": self.brand,
            "category": self.category,
            "sub_category": self.sub_category,
            "packaging": self.packaging,
            "packaging_raw": self.packaging_raw,
            "bottle_or_can_type": self.bottle_or_can_type,
            "unit_volume_ml": self.unit_volume_ml,
            "units_per_case": self.units_per_case,
            "cases_per_pallet": self.cases_per_pallet,
            "quantity_case": self.quantity_case,
            "gift_box": self.gift_box,
            "refillable_status": self.refillable_status,
            "currency": self.currency,
            "price_per_unit": self.price_per_unit,
            "price_per_unit_eur": self.price_per_unit_eur,
            "price_per_case": self.price_per_case,
            "price_per_case_eur": self.price_per_case_eur,
            "fx_rate": self.fx_rate,
            "fx_date": self.fx_date,
            "alcohol_percent": self.alcohol_percent,
            "origin_country": self.origin_country,
            "supplier_country": self.supplier_country,
            "incoterm": self.incoterm,
            "location": self.location,
            "lead_time": self.lead_time,
            "moq_cases": self.moq_cases,
            "valid_until": self.valid_until,
            "best_before_date": self.best_before_date,
            "vintage": self.vintage,
            "ean_code": self.ean_code,
            "label_language": self.label_language,
            "product_reference": self.product_reference,
            "supplier_name": self.supplier_name,
            "supplier_email": self.supplier_email,
            "supplier_reference": self.supplier_reference,
            "sender_name": self.sender_name,
            "sender_email": self.sender_email,
            "source_channel": self.source_channel,
            "source_message_id": self.source_message_id,
            "source_filename": self.source_filename,
            "attachment_filenames": json.loads(self.attachment_filenames) if self.attachment_filenames else [],
            "attachment_count": self.attachment_count,
            "confidence_score": self.confidence_score,
            "needs_manual_review": self.needs_manual_review,
            "error_flags": json.loads(self.error_flags) if self.error_flags else [],
            "custom_status": self.custom_status,
            "processing_version": self.processing_version,
            "offer_date": self.offer_date.isoformat() if self.offer_date else None,
            "date_received": self.date_received.isoformat() if self.date_received else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
