from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

# schemas/output.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class OfferItem(BaseModel):
    uid: str
    product_name: str = Field(default="Not Found")
    product_key: str = Field(default="Not Found")
    brand: str = Field(default="Not Found")
    category: Optional[str] = None
    sub_category: Optional[str] = None
    packaging: str = Field(default="Bottle")
    packaging_raw: str = Field(default="bottle")
    bottle_or_can_type: Optional[str] = None
    unit_volume_ml: Optional[float] = Field(default=0)
    units_per_case: Optional[float] = Field(default=0)
    cases_per_pallet: Optional[float] = None
    quantity_case: Optional[float] = None
    gift_box: Optional[str] = None
    refillable_status: str = Field(default="NRF")
    currency: str = Field(default="EUR")
    price_per_unit: Optional[float] = Field(default=0)
    price_per_unit_eur: Optional[float] = Field(default=0)
    price_per_case: Optional[float] = Field(default=0)
    price_per_case_eur: Optional[float] = Field(default=0)
    fx_rate: Optional[float] = Field(default=1.0)
    fx_date: Optional[str] = None
    alcohol_percent: Optional[float] = None
    origin_country: Optional[str] = None
    supplier_country: str = Field(default="")
    incoterm: str = Field(default="Not Found")
    location: str = Field(default="Not Found")
    lead_time: str = Field(default="Not Found")
    moq_cases: Optional[float] = None
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
    attachment_filenames: List[str] = Field(default_factory=list)
    attachment_count: int = Field(default=0)
    confidence_score: float = Field(default=0.85)
    needs_manual_review: bool = Field(default=False)
    error_flags: List[str] = Field(default_factory=list)
    custom_status: Optional[str] = None
    processing_version: str = Field(default="1.0.0")
    ean_code: Optional[str] = None
    label_language: str = Field(default="EN")
    product_reference: Optional[str] = None

class OfferResponse(BaseModel):
    data: List[OfferItem]
