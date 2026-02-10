import uuid
import traceback
from datetime import datetime
from schemas.output import OfferItem
from core.openai_client import extract_offer, extract_from_file, parse_buffer_data
from core.redis_client import redis_manager
import tempfile
import os
import logging

logger = logging.getLogger(__name__)


async def process_offer(payload, job_id: str):
    try:
        uid = str(uuid.uuid4())
        extracted_data = {}
        all_products = []

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
                    file_bytes = parse_buffer_data(attachment.data)

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

                    safe_data = {
                        'product_name': merged_data.get('product_name') or "Not Found",
                        'product_key': merged_data.get('product_key') or "Not Found",
                        'brand': merged_data.get('brand') or "Not Found",
                        'category': merged_data.get('category'),
                        'sub_category': merged_data.get('sub_category'),
                        'packaging': merged_data.get('packaging') or "Bottle",
                        'packaging_raw': merged_data.get('packaging_raw') or "bottle",
                        'bottle_or_can_type': merged_data.get('bottle_or_can_type'),
                        'unit_volume_ml': float(merged_data.get('unit_volume_ml', 0) or 0),
                        'units_per_case': float(merged_data.get('units_per_case', 0) or 0),
                        'cases_per_pallet': merged_data.get('cases_per_pallet'),
                        'quantity_case': merged_data.get('quantity_case'),
                        'gift_box': merged_data.get('gift_box'),
                        'refillable_status': merged_data.get('refillable_status') or "NRF",
                        'currency': merged_data.get('currency') or "EUR",
                        'price_per_unit': float(merged_data.get('price_per_unit', 0) or 0),
                        'price_per_unit_eur': float(merged_data.get('price_per_unit_eur', 0) or 0),
                        'price_per_case': float(merged_data.get('price_per_case', 0) or 0),
                        'price_per_case_eur': float(merged_data.get('price_per_case_eur', 0) or 0),
                        'fx_rate': float(merged_data.get('fx_rate', 1.0) or 1.0),
                        'fx_date': merged_data.get('fx_date'),
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
                    }

                    # Convert numeric fields
                    numeric_fields = ['cases_per_pallet', 'quantity_case', 'moq_cases', 'alcohol_percent']
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
                        confidence_score=0.85,
                        needs_manual_review=False,
                        error_flags=[],
                        custom_status=None,
                        processing_version="1.0.0",
                        ean_code=safe_data['ean_code'],
                        label_language=safe_data['label_language'],
                        product_reference=safe_data['product_reference'],
                    )

                    offers.append(offer.model_dump(mode='json'))
                    logger.info(f"Created offer {idx + 1}/{len(all_products)}: {safe_data['product_name']}")

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
                    'unit_volume_ml': float(extracted_data.get('unit_volume_ml', 0) or 0),
                    'units_per_case': float(extracted_data.get('units_per_case', 0) or 0),
                    'cases_per_pallet': extracted_data.get('cases_per_pallet'),
                    'quantity_case': extracted_data.get('quantity_case'),
                    'gift_box': extracted_data.get('gift_box'),
                    'refillable_status': extracted_data.get('refillable_status') or "NRF",
                    'currency': extracted_data.get('currency') or "EUR",
                    'price_per_unit': float(extracted_data.get('price_per_unit', 0) or 0),
                    'price_per_unit_eur': float(extracted_data.get('price_per_unit_eur', 0) or 0),
                    'price_per_case': float(extracted_data.get('price_per_case', 0) or 0),
                    'price_per_case_eur': float(extracted_data.get('price_per_case_eur', 0) or 0),
                    'fx_rate': float(extracted_data.get('fx_rate', 1.0) or 1.0),
                    'fx_date': extracted_data.get('fx_date'),
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
                }

                # Convert numeric fields
                numeric_fields = ['cases_per_pallet', 'quantity_case', 'moq_cases', 'alcohol_percent']
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
                    confidence_score=0.85,
                    needs_manual_review=False,
                    error_flags=[],
                    custom_status=None,
                    processing_version="1.0.0",
                    ean_code=safe_data['ean_code'],
                    label_language=safe_data['label_language'],
                    product_reference=safe_data['product_reference'],
                )

                offers.append(offer.model_dump(mode='json'))
                logger.info(f"Successfully created single offer for job {job_id}")

            except Exception as e:
                error_trace = traceback.format_exc()
                logger.error(f"Error creating offer item for job {job_id}: {e}\n{error_trace}")

                result_data = {
                    "job_id": job_id,
                    "status": "partial_success",
                    "error": str(e),
                    "error_trace": error_trace,
                    "extracted_data": extracted_data,
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