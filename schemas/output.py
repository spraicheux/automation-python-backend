from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

class OfferItem(BaseModel):
    uid: str
    product_name: str
    product_key: str
    brand: str
    category: Optional[str] = None
    sub_category: Optional[str] = None
    packaging: str
    packaging_raw: str
    bottle_or_can_type: Optional[str] = None
    unit_volume_ml: int
    units_per_case: int
    cases_per_pallet: Optional[int] = None
    quantity_case: Optional[int] = None
    gift_box: Optional[bool] = None
    refillable_status: str
    currency: str
    price_per_unit: float
    price_per_unit_eur: float
    price_per_case: float
    price_per_case_eur: float
    fx_rate: float
    fx_date: str
    alcohol_percent: str
    origin_country: Optional[str] = None
    supplier_country: str
    incoterm: str
    location: str
    lead_time: str
    moq_cases: Optional[int] = None
    valid_until: Optional[str] = None
    offer_date: datetime
    date_received: datetime
    best_before_date: Optional[str] = None
    vintage: Optional[str] = None
    supplier_name: str
    supplier_email: str
    supplier_reference: Optional[str] = None
    source_channel: str
    source_message_id: str
    source_filename: str
    attachment_filenames: List[str] = []
    attachment_count: int = 0
    confidence_score: float
    needs_manual_review: bool
    error_flags: List[str]
    custom_status: Optional[str] = None
    processing_version: str
    ean_code: Optional[str] = None
    label_language: str
    product_reference: Optional[str] = None

class OfferResponse(BaseModel):
    data: List[OfferItem]
