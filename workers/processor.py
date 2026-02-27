import uuid
import traceback
from datetime import datetime

from core.file_download import resolve_attachment_bytes
from schemas.output import OfferItem
from core.openai_client import extract_offer, extract_from_file, parse_buffer_data
from core.redis_client import redis_manager
import tempfile
import os
import logging
from core.webhook_client import send_consolidated_webhook

logger = logging.getLogger(__name__)


def is_valid_offer(offer_dict: dict) -> bool:
    """Strict validation for extracted offers"""
    if not offer_dict:
        return False

    name = offer_dict.get('product_name')
    if not name or name in ["Not Found", "Unknown", "Row", ""]:
        return False

    if name.lower().startswith('row '):  # Skip generic placeholders
        return False

    # Check for minimal commercial data
    # FIX BUG 5: openai_client now returns None (not 0) for missing prices.
    # None == 0 is always False in Python, so the original check would silently
    # pass products with no price. Check for both None and 0.
    price_unit = offer_dict.get('price_per_unit')
    price_case = offer_dict.get('price_per_case')
    unit_is_empty = price_unit is None or price_unit == 0
    case_is_empty = price_case is None or price_case == 0
    if unit_is_empty and case_is_empty:
        # If no price is found, it's often a false positive or truncated data
        logger.debug(f"Offer {name} rejected: No price data found.")
        return False

    return True


def _safe_float(value, default=None):
    """Safely convert a value to float, returning default if conversion fails.

    FIX BUG 4: openai_client now returns None for missing numeric fields.
    bare float(None) raises TypeError causing the whole product to be silently
    skipped. This helper handles None, empty string, and invalid values safely.
    """
    if value is None or value == "" or value == "Not Found":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


async def process_offer(payload, job_id: str):
    try:
        uid = str(uuid.uuid4())
        extracted_data = {}
        all_products = []
        processed_keys = set()  # For deduplication
        valid_count = 0
        duplicate_count = 0

        redis_manager.set_job_status(job_id, "processing")

        logger.info(f"Processing job {job_id} for supplier: {payload.supplier_name}")
        logger.info(f"Text body length: {len(payload.text_body) if payload.text_body else 0}")
        logger.info(f"Attachment count: {len(payload.attachments) if payload.attachments else 0}")

        if payload.text_body:
            try:
                extracted_data = await extract_offer(payload.text_body or "")
                logger.info(f"Text extraction completed for job {job_id}")
                if isinstance(extracted_data, dict) and 'products' in extracted_data:
                    products_list = extracted_data['products']
                    logger.info(f"Found {len(products_list)} products in text")
                    all_products.extend(products_list)
                else:
                    logger.info("Using extracted_data as single product")
                    all_products.append(extracted_data)

            except Exception as e:
                logger.error(f"Error in text extraction for job {job_id}: {e}")

        if payload.attachments:
            for attachment in payload.attachments:
                try:
                    file_name = attachment.fileName
                    logger.info(f"Processing attachment: {file_name}")

                    file_ext = file_name.split('.')[-1].lower() if '.' in file_name else ''
                    file_bytes = await resolve_attachment_bytes(attachment)

                    if not file_bytes:
                        logger.warning(f"Could not parse file data for {file_name}")
                        continue

                    with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=f'.{file_ext}' if file_ext else '.bin'
                    ) as tmp_file:
                        tmp_file.write(file_bytes)
                        tmp_file_path = tmp_file.name

                    try:
                        file_extracted = await extract_from_file(
                            tmp_file_path,
                            attachment.contentType
                        )

                        if file_extracted:
                            logger.info(f"File extraction completed for {file_name}")

                            # Check if this is multiple products from Excel
                            if 'products' in file_extracted and isinstance(file_extracted['products'], list):
                                logger.info(f"Found {len(file_extracted['products'])} products in Excel file")
                                all_products.extend(file_extracted['products'])
                            else:
                                # Single product extraction (old format)
                                for key, value in file_extracted.items():
                                    if value not in [None, "", []]:
                                        extracted_data[key] = value

                    finally:
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass

                except Exception as e:
                    logger.error(f"Error processing attachment {attachment.fileName}: {e}")
                    continue

        # Create offers
        offers = []

        if all_products:
            # Process each product from Excel
            logger.info(f"Processing {len(all_products)} products from Excel for job {job_id}")

            for idx, product_data in enumerate(all_products):
                try:
                    merged_data = {**extracted_data, **product_data}
                    logger.debug(
                        f"Row {idx} raw data before cleaning: {merged_data.get('product_name')} | {merged_data.get('brand')}")

                    safe_data = {
                        'product_name': merged_data.get('product_name') or "Not Found",
                        'product_key': merged_data.get('product_key') or "Not Found",
                        'brand': merged_data.get('brand') or "Not Found",
                        'category': merged_data.get('category'),
                        'sub_category': merged_data.get('sub_category'),
                        'packaging': merged_data.get('packaging') or "Bottle",
                        'packaging_raw': merged_data.get('packaging_raw') or "bottle",
                        'bottle_or_can_type': merged_data.get('bottle_or_can_type'),
                        # FIX BUG 4: use _safe_float() instead of bare float() so that
                        # None values from openai_client don't raise TypeError and silently
                        # skip the entire product.
                        'unit_volume_ml': _safe_float(merged_data.get('unit_volume_ml'), 0),
                        'units_per_case': _safe_float(merged_data.get('units_per_case'), 0),
                        'cases_per_pallet': merged_data.get('cases_per_pallet'),
                        'quantity_case': merged_data.get('quantity_case'),
                        'gift_box': merged_data.get('gift_box'),
                        'refillable_status': merged_data.get('refillable_status') or "",
                        'currency': merged_data.get('currency') or "EUR",
                        'price_per_unit': _safe_float(merged_data.get('price_per_unit'), 0),
                        'price_per_unit_eur': _safe_float(merged_data.get('price_per_unit_eur'), 0),
                        'price_per_case': _safe_float(merged_data.get('price_per_case'), 0),
                        'price_per_case_eur': _safe_float(merged_data.get('price_per_case_eur'), 0),
                        'fx_rate': _safe_float(merged_data.get('fx_rate'), 1.0),
                        'fx_date': merged_data.get('fx_date'),
                        # FIX BUG 1: alcohol_percent is now a string like "40%" from
                        # openai_client. Keep it as a string here — do NOT put it in the
                        # numeric_fields conversion loop below.
                        'alcohol_percent': merged_data.get('alcohol_percent'),
                        'origin_country': merged_data.get('origin_country'),
                        'supplier_country': merged_data.get('supplier_country') or "",
                        'incoterm': merged_data.get('incoterm') or "Not Found",
                        'location': merged_data.get('location') or "Not Found",
                        'lead_time': merged_data.get('lead_time') or "Not Found",
                        'moq_cases': merged_data.get('moq_cases'),
                        'valid_until': merged_data.get('valid_until'),
                        'best_before_date': merged_data.get('best_before_date'),
                        'vintage': merged_data.get('vintage'),
                        'supplier_reference': merged_data.get('supplier_reference'),
                        'ean_code': merged_data.get('ean_code'),
                        'label_language': merged_data.get('label_language') or "EN",
                        'product_reference': merged_data.get('product_reference'),
                        # FIX BUG 2: custom_status was hardcoded as None. Extract it from
                        # the merged data so T1/T2 values extracted by the AI are passed through.
                        'custom_status': merged_data.get('custom_status'),
                    }

                    # Convert numeric fields
                    # FIX BUG 1: alcohol_percent removed from this list. It is now a string
                    # like "40%" returned by openai_client. float("40%") raises ValueError
                    # which caused it to always be set to None. Keep it as a string.
                    numeric_fields = ['cases_per_pallet', 'quantity_case', 'moq_cases']
                    for field in numeric_fields:
                        if safe_data[field] is not None:
                            try:
                                safe_data[field] = float(safe_data[field])
                            except (ValueError, TypeError):
                                safe_data[field] = None

                    offer = OfferItem(
                        uid=f"{uid}_{idx}",
                        product_name=safe_data['product_name'],
                        product_key=safe_data['product_key'],
                        brand=safe_data['brand'],
                        category=safe_data['category'],
                        sub_category=safe_data['sub_category'],
                        packaging=safe_data['packaging'],
                        packaging_raw=safe_data['packaging_raw'],
                        bottle_or_can_type=safe_data['bottle_or_can_type'],
                        unit_volume_ml=safe_data['unit_volume_ml'],
                        units_per_case=safe_data['units_per_case'],
                        cases_per_pallet=safe_data['cases_per_pallet'],
                        quantity_case=safe_data['quantity_case'],
                        gift_box=safe_data['gift_box'],
                        refillable_status=safe_data['refillable_status'],
                        currency=safe_data['currency'],
                        price_per_unit=safe_data['price_per_unit'],
                        price_per_unit_eur=safe_data['price_per_unit_eur'],
                        price_per_case=safe_data['price_per_case'],
                        price_per_case_eur=safe_data['price_per_case_eur'],
                        fx_rate=safe_data['fx_rate'],
                        fx_date=safe_data['fx_date'],
                        alcohol_percent=safe_data['alcohol_percent'],
                        origin_country=safe_data['origin_country'],
                        supplier_country=safe_data['supplier_country'],
                        incoterm=safe_data['incoterm'],
                        location=safe_data['location'],
                        lead_time=safe_data['lead_time'],
                        moq_cases=safe_data['moq_cases'],
                        valid_until=safe_data['valid_until'],
                        offer_date=datetime.utcnow(),
                        date_received=datetime.utcnow(),
                        best_before_date=safe_data['best_before_date'],
                        vintage=safe_data['vintage'],
                        supplier_name=payload.supplier_name,
                        supplier_email=payload.supplier_email,
                        supplier_reference=safe_data['supplier_reference'],
                        source_channel=payload.source_channel,
                        source_message_id=payload.source_message_id,
                        source_filename=payload.source_filename,
                        attachment_filenames=[att.fileName for att in
                                              payload.attachments] if payload.attachments else [],
                        attachment_count=len(payload.attachments) if payload.attachments else 0,
                        confidence_score=0.95,
                        needs_manual_review=False,
                        error_flags=[],
                        # FIX BUG 2: pass extracted custom_status instead of hardcoded None.
                        # The AI now correctly extracts T1/T2 from the STATUS column.
                        custom_status=safe_data['custom_status'],
                        processing_version="2.0.0",
                        ean_code=safe_data['ean_code'],
                        label_language=safe_data['label_language'],
                        product_reference=safe_data['product_reference'],
                    )

                    offer_dict = offer.model_dump(mode='json')

                    # Validation & Deduplication
                    if is_valid_offer(offer_dict):
                        p_key = offer_dict.get('product_key', '')
                        if p_key not in processed_keys:
                            processed_keys.add(p_key)
                            offers.append(offer_dict)
                            valid_count += 1
                            # Sequential Webhook Dispatch - Structured product collection for visibility
                            logger.info(f"Dispatching sequential webhook for product: {offer_dict['product_name']}")
                            send_consolidated_webhook(
                                job_id=job_id,
                                payload_type="single_row",
                                data={"product": offer_dict},
                                delivery_id=f"{job_id}_{valid_count}"
                            )
                        else:
                            duplicate_count += 1
                            brand = offer_dict.get('brand', 'Not Found')
                            p_name = offer_dict.get('product_name', 'Not Found')
                            logger.info(f"Skipping duplicate product in sheet: {brand} | {p_name} (Key: {p_key})")
                    else:
                        brand = offer_dict.get('brand', 'Not Found')
                        p_name = offer_dict.get('product_name', 'Not Found')
                        logger.warning(f"Skipping row {idx} - Invalid/Incomplete commercial data: {brand} | {p_name}")

                except Exception as e:
                    error_trace = traceback.format_exc()
                    logger.error(f"Error creating offer for product {idx}: {e}\n{error_trace}")
                    continue

        else:
            logger.info(f"No Excel products found, using single product extraction for job {job_id}")
            try:
                safe_data = {
                    'product_name': extracted_data.get('product_name') or "Not Found",
                    'product_key': extracted_data.get('product_key') or "Not Found",
                    'brand': extracted_data.get('brand') or "Not Found",
                    'category': extracted_data.get('category'),
                    'sub_category': extracted_data.get('sub_category'),
                    'packaging': extracted_data.get('packaging') or "Bottle",
                    'packaging_raw': extracted_data.get('packaging_raw') or "bottle",
                    'bottle_or_can_type': extracted_data.get('bottle_or_can_type'),
                    # FIX BUG 4: use _safe_float() for all numeric fields.
                    'unit_volume_ml': _safe_float(extracted_data.get('unit_volume_ml'), 0),
                    'units_per_case': _safe_float(extracted_data.get('units_per_case'), 0),
                    'cases_per_pallet': extracted_data.get('cases_per_pallet'),
                    'quantity_case': extracted_data.get('quantity_case'),
                    'gift_box': extracted_data.get('gift_box'),
                    # FIX BUG 3: the original defaulted to "NRF" when refillable_status
                    # was blank. openai_client now returns "" for absent fields.
                    # Defaulting to "NRF" violates the business rule: never assume NRF.
                    'refillable_status': extracted_data.get('refillable_status') or "",
                    'currency': extracted_data.get('currency') or "EUR",
                    'price_per_unit': _safe_float(extracted_data.get('price_per_unit'), 0),
                    'price_per_unit_eur': _safe_float(extracted_data.get('price_per_unit_eur'), 0),
                    'price_per_case': _safe_float(extracted_data.get('price_per_case'), 0),
                    'price_per_case_eur': _safe_float(extracted_data.get('price_per_case_eur'), 0),
                    'fx_rate': _safe_float(extracted_data.get('fx_rate'), 1.0),
                    'fx_date': extracted_data.get('fx_date'),
                    # FIX BUG 1: keep alcohol_percent as string — do not convert to float.
                    'alcohol_percent': extracted_data.get('alcohol_percent'),
                    'origin_country': extracted_data.get('origin_country'),
                    'supplier_country': extracted_data.get('supplier_country') or "",
                    'incoterm': extracted_data.get('incoterm') or "Not Found",
                    'location': extracted_data.get('location') or "Not Found",
                    'lead_time': extracted_data.get('lead_time') or "Not Found",
                    'moq_cases': extracted_data.get('moq_cases'),
                    'valid_until': extracted_data.get('valid_until'),
                    'best_before_date': extracted_data.get('best_before_date'),
                    'vintage': extracted_data.get('vintage'),
                    'supplier_reference': extracted_data.get('supplier_reference'),
                    'ean_code': extracted_data.get('ean_code'),
                    'label_language': extracted_data.get('label_language') or "EN",
                    'product_reference': extracted_data.get('product_reference'),
                    # FIX BUG 2: pass extracted custom_status instead of hardcoded None.
                    'custom_status': extracted_data.get('custom_status'),
                }

                # Convert numeric fields
                # FIX BUG 1: alcohol_percent removed — it is a string "40%" not a number.
                numeric_fields = ['cases_per_pallet', 'quantity_case', 'moq_cases']
                for field in numeric_fields:
                    if safe_data[field] is not None:
                        try:
                            safe_data[field] = float(safe_data[field])
                        except (ValueError, TypeError):
                            safe_data[field] = None

                offer = OfferItem(
                    uid=uid,
                    product_name=safe_data['product_name'],
                    product_key=safe_data['product_key'],
                    brand=safe_data['brand'],
                    category=safe_data['category'],
                    sub_category=safe_data['sub_category'],
                    packaging=safe_data['packaging'],
                    packaging_raw=safe_data['packaging_raw'],
                    bottle_or_can_type=safe_data['bottle_or_can_type'],
                    unit_volume_ml=safe_data['unit_volume_ml'],
                    units_per_case=safe_data['units_per_case'],
                    cases_per_pallet=safe_data['cases_per_pallet'],
                    quantity_case=safe_data['quantity_case'],
                    gift_box=safe_data['gift_box'],
                    refillable_status=safe_data['refillable_status'],
                    currency=safe_data['currency'],
                    price_per_unit=safe_data['price_per_unit'],
                    price_per_unit_eur=safe_data['price_per_unit_eur'],
                    price_per_case=safe_data['price_per_case'],
                    price_per_case_eur=safe_data['price_per_case_eur'],
                    fx_rate=safe_data['fx_rate'],
                    fx_date=safe_data['fx_date'],
                    alcohol_percent=safe_data['alcohol_percent'],
                    origin_country=safe_data['origin_country'],
                    supplier_country=safe_data['supplier_country'],
                    incoterm=safe_data['incoterm'],
                    location=safe_data['location'],
                    lead_time=safe_data['lead_time'],
                    moq_cases=safe_data['moq_cases'],
                    valid_until=safe_data['valid_until'],
                    offer_date=datetime.utcnow(),
                    date_received=datetime.utcnow(),
                    best_before_date=safe_data['best_before_date'],
                    vintage=safe_data['vintage'],
                    supplier_name=payload.supplier_name,
                    supplier_email=payload.supplier_email,
                    supplier_reference=safe_data['supplier_reference'],
                    source_channel=payload.source_channel,
                    source_message_id=payload.source_message_id,
                    source_filename=payload.source_filename,
                    attachment_filenames=[att.fileName for att in payload.attachments] if payload.attachments else [],
                    attachment_count=len(payload.attachments) if payload.attachments else 0,
                    confidence_score=0.95,
                    needs_manual_review=False,
                    error_flags=[],
                    # FIX BUG 2: pass extracted custom_status instead of hardcoded None.
                    custom_status=safe_data['custom_status'],
                    processing_version="2.0.0",
                    ean_code=safe_data['ean_code'],
                    label_language=safe_data['label_language'],
                    product_reference=safe_data['product_reference'],
                )

                offer_dict = offer.model_dump(mode='json')
                if is_valid_offer(offer_dict):
                    offers.append(offer_dict)
                    logger.info(f"Dispatching sequential webhook for single offer: {offer_dict['product_name']}")
                    send_consolidated_webhook(
                        job_id=job_id,
                        payload_type="single_row",
                        data={"product": offer_dict},
                        delivery_id=f"{job_id}_single"
                    )
                else:
                    logger.warning(f"Single offer extraction failed validation: {offer_dict.get('product_name')}")

            except Exception as e:
                error_trace = traceback.format_exc()
                logger.error(f"Error creating offer item for job {job_id}: {e}\n{error_trace}")

                result_data = {
                    "job_id": job_id,
                    "status": "partial_success",
                    "error": str(e),
                    "error_trace": error_trace,
                    "extracted_data": extracted_data,
                    "duplicate_count": duplicate_count,
                    "source_data": {
                        "supplier_name": payload.supplier_name,
                        "supplier_email": payload.supplier_email,
                        "source_channel": payload.source_channel,
                        "text_body_preview": payload.text_body[:200] if payload.text_body else None,
                        "attachments": [att.fileName for att in payload.attachments] if payload.attachments else []
                    }
                }

                redis_manager.set_job_result(job_id, result_data)
                redis_manager.set_job_status(job_id, "done")
                logger.info(f"Job {job_id} marked as done with partial success")
                return

        # Store results
        result_data = {
            "job_id": job_id,
            "status": "done",
            "total_products": len(offers),
            "duplicate_count": duplicate_count,
            "products": offers,
            "source_info": {
                "supplier": payload.supplier_name,
                "attachments": len(payload.attachments) if payload.attachments else 0,
                "file_processed": len(all_products) > 0
            }
        }

        redis_manager.set_job_result(job_id, result_data)
        redis_manager.set_job_status(job_id, "done")
        logger.info(f"Successfully processed {len(offers)} products for job {job_id}")

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Critical error processing job {job_id}: {e}\n{error_trace}")

        error_result = {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "error_trace": error_trace,
            "timestamp": datetime.utcnow().isoformat(),
            "payload_info": {
                "supplier_name": payload.supplier_name if hasattr(payload, 'supplier_name') else None,
                "has_text": bool(payload.text_body) if hasattr(payload, 'text_body') else None,
                "attachment_count": len(payload.attachments) if hasattr(payload, 'attachments') else None
            }
        }
        redis_manager.set_job_result(job_id, error_result)
        redis_manager.set_job_status(job_id, "failed")