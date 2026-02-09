import uuid
from datetime import datetime
from schemas.output import OfferItem
from core.openai_client import extract_offer

async def process_offer(payload) -> OfferItem:
    uid = str(uuid.uuid4())

    # TEMP: text-only (attachments later)
    extracted = await extract_offer(payload.text_body or "")

    # Normally: json.loads + validation
    # Milestone 0: mocked merge

    return OfferItem(
        uid=uid,
        alcohol_percent="11%",
        best_before_date=None,
        bottle_or_can_type=None,
        brand="Freixenet",
        cases_per_pallet=None,
        category=None,
        confidence_score=0.85,
        currency="EUR",
        custom_status=None,
        date_received=datetime.utcnow(),
        ean_code=None,
        error_flags=[],
        fx_date="2025-12-19",
        fx_rate=1,
        gift_box=None,
        incoterm="DAP",
        label_language="EN",
        lead_time="2 weeks",
        location="Loendersloot",
        moq_cases=None,
        needs_manual_review=False,
        offer_date=datetime.utcnow(),
        origin_country=None,
        packaging="Bottle",
        packaging_raw="bottle",
        price_per_case=19.6,
        price_per_case_eur=19.6,
        price_per_unit=3.2667,
        price_per_unit_eur=3.2667,
        processing_version="1.0.0",
        product_key="freixenet_freixenet_carta_nevada_extra_dry_bottle",
        product_name="Freixenet Carta Nevada Extra Dry",
        product_reference=None,
        quantity_case=None,
        refillable_status="NRF",
        source_channel=payload.source_channel,
        source_filename=payload.source_filename,
        source_message_id=payload.source_message_id,
        sub_category=None,
        supplier_country="",
        supplier_email=payload.supplier_email,
        supplier_name=payload.supplier_name,
        supplier_reference=None,
        unit_volume_ml=750,
        units_per_case=6,
        valid_until=None,
        vintage=None,
    )
