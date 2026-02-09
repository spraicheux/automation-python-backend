from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class OfferItem(BaseModel):
    alcohol_percent: Optional[str]
    best_before_date: Optional[str]
    bottle_or_can_type: Optional[str]
    brand: Optional[str]
    cases_per_pallet: Optional[int]
    category: Optional[str]
    confidence_score: float
    currency: Optional[str]
    custom_status: Optional[str]
    date_received: datetime
    ean_code: Optional[str]
    error_flags: List[str]
    fx_date: Optional[str]
    fx_rate: Optional[float]
    gift_box: Optional[bool]
    incoterm: Optional[str]
    label_language: Optional[str]
    lead_time: Optional[str]
    location: Optional[str]
    moq_cases: Optional[int]
    needs_manual_review: bool
    offer_date: datetime
    origin_country: Optional[str]
    packaging: Optional[str]
    packaging_raw: Optional[str]
    price_per_case: Optional[float]
    price_per_case_eur: Optional[float]
    price_per_unit: Optional[float]
    price_per_unit_eur: Optional[float]
    processing_version: str
    product_key: str
    product_name: str
    product_reference: Optional[str]
    quantity_case: Optional[int]
    refillable_status: Optional[str]
    source_channel: str
    source_filename: str
    source_message_id: str
    sub_category: Optional[str]
    supplier_country: Optional[str]
    supplier_email: Optional[str]
    supplier_name: Optional[str]
    supplier_reference: Optional[str]
    uid: str
    unit_volume_ml: Optional[int]
    units_per_case: Optional[int]
    valid_until: Optional[str]
    vintage: Optional[str]

class OfferResponse(BaseModel):
    data: List[OfferItem]
