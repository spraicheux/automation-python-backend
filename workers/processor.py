import uuid
import re
import json
import os
import traceback
import tempfile
import logging
from datetime import datetime

from core.file_download import resolve_attachment_bytes
from schemas.output import OfferItem
from core.openai_client import extract_offer, extract_from_file, parse_buffer_data
from core.redis_client import redis_manager
from core.webhook_client import send_consolidated_webhook
from core.database import get_session_factory
from models.offer_item import OfferItemDB
from models.source_file import SourceFileDB

logger = logging.getLogger(__name__)


def save_offer_to_db(offer_dict: dict, job_id: str) -> None:
    """
    Persist a single extracted OfferItem to the PostgreSQL database.
    Non-fatal: logs errors without interrupting the main processing flow.

    SourceFileDB logic:
    - First product of a job → creates the SourceFileDB record.
    - Subsequent products of the same job → reuses the existing SourceFileDB.
    - Same filename in a DIFFERENT job → skipped (already processed).
    """
    try:
        db = get_session_factory()()
        try:
            source_filename = offer_dict.get("source_filename") or None
            if source_filename:
                duplicate = db.query(SourceFileDB).filter(
                    SourceFileDB.source_filename == source_filename,
                    SourceFileDB.job_id != job_id
                ).first()
                if duplicate:
                    logger.warning(
                        f"DB: skipping '{offer_dict.get('product_name')}' — "
                        f"file '{source_filename}' already processed in job {duplicate.job_id}"
                    )
                    return
            # ────────────────────────────────────────────────────────────────

            # ── Get or create SourceFileDB for this job ──────────────────────
            source_file = db.query(SourceFileDB).filter(
                SourceFileDB.job_id == job_id
            ).first()

            if not source_file:
                source_file = SourceFileDB(
                    id=str(uuid.uuid4()),
                    job_id=job_id,
                    source_filename=source_filename,
                    sender_name=offer_dict.get("sender_name"),
                    sender_email=offer_dict.get("sender_email"),
                    supplier_name=offer_dict.get("supplier_name"),
                    supplier_email=offer_dict.get("supplier_email"),
                    source_channel=offer_dict.get("source_channel"),
                    source_message_id=offer_dict.get("source_message_id"),
                    product_count=0,
                )
                db.add(source_file)
                db.flush()  # get the id without committing yet
                logger.info(f"DB: created SourceFileDB for job {job_id}, file='{source_filename}'")
            # ────────────────────────────────────────────────────────────────

            row = OfferItemDB(
                uid=offer_dict.get("uid"),
                source_file_id=source_file.id,
                job_id=job_id,
                product_name=offer_dict.get("product_name"),
                product_key=offer_dict.get("product_key"),
                brand=offer_dict.get("brand"),
                category=offer_dict.get("category"),
                sub_category=offer_dict.get("sub_category"),
                packaging=offer_dict.get("packaging"),
                packaging_raw=offer_dict.get("packaging_raw"),
                bottle_or_can_type=offer_dict.get("bottle_or_can_type"),
                unit_volume_ml=offer_dict.get("unit_volume_ml"),
                units_per_case=offer_dict.get("units_per_case"),
                cases_per_pallet=offer_dict.get("cases_per_pallet"),
                quantity_case=offer_dict.get("quantity_case"),
                gift_box=offer_dict.get("gift_box"),
                refillable_status=offer_dict.get("refillable_status"),
                currency=offer_dict.get("currency"),
                price_per_unit=offer_dict.get("price_per_unit"),
                price_per_unit_eur=offer_dict.get("price_per_unit_eur"),
                price_per_case=offer_dict.get("price_per_case"),
                price_per_case_eur=offer_dict.get("price_per_case_eur"),
                fx_rate=offer_dict.get("fx_rate"),
                fx_date=offer_dict.get("fx_date"),
                alcohol_percent=offer_dict.get("alcohol_percent"),
                origin_country=offer_dict.get("origin_country"),
                supplier_country=offer_dict.get("supplier_country"),
                incoterm=offer_dict.get("incoterm"),
                location=offer_dict.get("location"),
                lead_time=offer_dict.get("lead_time"),
                moq_cases=offer_dict.get("moq_cases"),
                valid_until=offer_dict.get("valid_until"),
                best_before_date=offer_dict.get("best_before_date"),
                vintage=offer_dict.get("vintage"),
                ean_code=offer_dict.get("ean_code"),
                label_language=offer_dict.get("label_language"),
                product_reference=offer_dict.get("product_reference"),
                supplier_name=offer_dict.get("supplier_name"),
                supplier_email=offer_dict.get("supplier_email"),
                supplier_reference=offer_dict.get("supplier_reference"),
                sender_name=offer_dict.get("sender_name"),
                sender_email=offer_dict.get("sender_email"),
                source_channel=offer_dict.get("source_channel"),
                source_message_id=offer_dict.get("source_message_id"),
                source_filename=source_filename,
                attachment_filenames=json.dumps(offer_dict.get("attachment_filenames", [])),
                attachment_count=offer_dict.get("attachment_count"),
                confidence_score=offer_dict.get("confidence_score"),
                needs_manual_review=offer_dict.get("needs_manual_review"),
                error_flags=json.dumps(offer_dict.get("error_flags", [])),
                custom_status=offer_dict.get("custom_status"),
                processing_version=offer_dict.get("processing_version"),
                offer_date=offer_dict.get("offer_date"),
                date_received=offer_dict.get("date_received"),
            )
            db.add(row)

            source_file.product_count += 1

            db.commit()
            logger.info(f"DB: saved product '{offer_dict.get('product_name')}' for job {job_id}")
        except Exception as db_err:
            db.rollback()
            logger.error(f"DB: failed to save product '{offer_dict.get('product_name')}' for job {job_id}: {db_err}")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"DB: save_offer_to_db error for job {job_id}: {e}")


def is_valid_offer(offer_dict: dict) -> bool:
    """Strict validation for extracted offers"""
    if not offer_dict:
        return False

    name = offer_dict.get('product_name')
    if not name or name in ["Not Found", "Unknown", "Row", ""]:
        return False

    if name.lower().startswith('row '):
        return False
    price_unit = offer_dict.get('price_per_unit')
    price_case = offer_dict.get('price_per_case')
    unit_is_empty = price_unit is None or price_unit == 0
    case_is_empty = price_case is None or price_case == 0
    if unit_is_empty and case_is_empty:
        logger.debug(f"Offer {name} rejected: No price data found.")
        return False

    return True


def _safe_float(value, default=None):
    if value is None or value == "" or value == "Not Found":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _resolve_supplier_name(ai_name, payload) -> str:
    """Return the best available supplier name.

    Priority:
      1. payload.supplier_name  — Make extracted this from email body / WhatsApp text
      2. ai_name                — AI extracted from attached file (PDF/Excel)
      3. "Not Found"            — sender info is NOT used (sender ≠ supplier)
    """
    _missing = [None, "", "Not Found"]
    pname = getattr(payload, 'supplier_name', None)
    if pname not in _missing:
        return str(pname)
    if ai_name not in _missing:
        return str(ai_name)
    return "Not Found"


def _resolve_supplier_email(payload, ai_email=None) -> str:
    """Return the best available supplier email.

    Priority:
      1. payload.supplier_email — Make extracted this from email/WA metadata
      2. ai_email                — AI extracted from attached file (PDF/Excel)
      3. "Not Found"
    """
    _missing = [None, "", "Not Found"]
    pemail = getattr(payload, 'supplier_email', None)
    if pemail not in _missing:
        return str(pemail)
    if ai_email not in _missing:
        return str(ai_email)
    return "Not Found"


# Approximate fixed fallback exchange rates (used only when fx_rate is absent)
_FALLBACK_FX = {
    "USD": 0.92,   # 1 USD ≈ 0.92 EUR
    "GBP": 1.17,   # 1 GBP ≈ 1.17 EUR
    "CHF": 1.05,   # 1 CHF ≈ 1.05 EUR
    "EUR": 1.0,
}

# Currency symbol → ISO code mapping
_CURRENCY_SYMBOLS = {
    "€": "EUR",
    "$": "USD",
    "£": "GBP",
    "₣": "CHF",
}


def _normalize_currency_and_prices(safe_data: dict) -> dict:
    """Detect currency from sign characters and populate EUR price fields.

    Logic:
    1. If currency is already a known non-EUR code (USD/GBP/…), convert to EUR.
    2. If currency still looks like a symbol (€/$), resolve to ISO code.
    3. Populate price_per_unit_eur and price_per_case_eur:
       - If currency == EUR: EUR fields = native price fields.
       - Otherwise: EUR = native * fx_rate (using AI-provided fx_rate or fallback)
    """
    currency = (safe_data.get('currency') or 'EUR').strip()

    # Resolve symbol to ISO code
    if currency in _CURRENCY_SYMBOLS:
        currency = _CURRENCY_SYMBOLS[currency]

    # Normalise to uppercase
    currency = currency.upper()
    if currency not in _FALLBACK_FX:
        currency = 'EUR'  # Unknown symbols → default EUR

    safe_data['currency'] = currency

    # Determine fx_rate: use AI-provided rate, or if it's 1.0/invalid, use fallback
    ai_fx = safe_data.get('fx_rate') or 1.0
    try:
        ai_fx = float(ai_fx)
    except (TypeError, ValueError):
        ai_fx = 1.0

    # If AI didn't provide a meaningful rate for non-EUR, use table fallback
    fx = ai_fx if (currency == 'EUR' or ai_fx != 1.0) else _FALLBACK_FX.get(currency, 1.0)
    safe_data['fx_rate'] = fx

    pu  = safe_data.get('price_per_unit')  or 0.0
    pc  = safe_data.get('price_per_case')  or 0.0

    if currency == 'EUR':
        safe_data['price_per_unit_eur'] = round(pu, 4)
        safe_data['price_per_case_eur'] = round(pc, 4)
    else:
        safe_data['price_per_unit_eur'] = round(pu * fx, 4)
        safe_data['price_per_case_eur'] = round(pc * fx, 4)

    return safe_data


def _apply_offer_defaults(data: dict) -> dict:
    """
    Replace AI's 'Not Found' with the proper default value according to the OfferItem schema.
    For optional fields with default None, 'Not Found' becomes None.
    For fields with a non‑None default (e.g. "Bottle", "EUR"), that default is applied.
    """
    defaults = {
        # fields with non‑None defaults (keep 'Not Found' only if it is the default)
        'product_name': "Not Found",
        'product_key': "Not Found",
        'brand': "Not Found",
        'packaging': "Bottle",
        'packaging_raw': "bottle",
        'refillable_status': "",
        'currency': "EUR",
        'incoterm': "Not Found",
        'location': "Not Found",
        'lead_time': "Not Found",
        'source_channel': "",
        'source_message_id': "",
        'source_filename': "",
        'label_language': "EN",
        'processing_version': "1.0.0",  # will be overridden in OfferItem

        # optional fields (should become None when missing)
        'category': None,
        'sub_category': None,
        'bottle_or_can_type': None,
        'cases_per_pallet': None,
        'quantity_case': None,
        'gift_box': None,
        'alcohol_percent': None,
        'origin_country': None,
        'supplier_country': "",
        'moq_cases': None,
        'valid_until': None,
        'best_before_date': None,
        'vintage': None,
        'supplier_name': None,
        'supplier_email': None,
        'sender_name': None,
        'sender_email': None,
        'supplier_reference': None,
        'custom_status': None,
        'ean_code': None,
        'product_reference': None,

        # fx and price fields (already handled by _safe_float, but we keep them here for completeness)
        'fx_date': None,
        'fx_rate': 1.0,
        'price_per_unit': 0,
        'price_per_unit_eur': 0,
        'price_per_case': 0,
        'price_per_case_eur': 0,
        'unit_volume_ml': 0,
        'units_per_case': 0,
    }

    for field, default in defaults.items():
        if field in data:
            if data[field] == "Not Found" or data[field] is None:
                data[field] = default
    return data


async def process_offer(payload, job_id: str):
    try:
        uid = str(uuid.uuid4())
        extracted_data = {}
        all_products = []
        valid_count = 0

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
            logger.info(f"Processing {len(all_products)} products from Excel for job {job_id}")

            for idx, product_data in enumerate(all_products):
                try:
                    merged_data = {**extracted_data, **product_data}

                    if merged_data.get('product_key') in [None, '', 'Not Found']:
                        logger.warning(
                            f"Row {idx} skipped — product_key not found: "
                            f"name={merged_data.get('product_name')!r}"
                        )
                        continue

                    logger.debug(
                        f"Row {idx} raw data: {merged_data.get('product_name')} | {merged_data.get('brand')}")

                    safe_data = {
                        'product_name': merged_data.get('product_name') or "Not Found",
                        'product_key': merged_data.get('product_key') or "Not Found",
                        'brand': merged_data.get('brand') or "Not Found",
                        'category': merged_data.get('category'),
                        'sub_category': merged_data.get('sub_category'),
                        'packaging': merged_data.get('packaging') or "Bottle",
                        'packaging_raw': merged_data.get('packaging_raw') or "bottle",
                        'bottle_or_can_type': merged_data.get('bottle_or_can_type'),
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
                        'alcohol_percent': merged_data.get('alcohol_percent'),
                        'origin_country': merged_data.get('origin_country'),
                        'supplier_country': merged_data.get('supplier_country') or "",
                        'incoterm': merged_data.get('incoterm') or "Not Found",
                        'location': merged_data.get('location') or "Not Found",
                        'lead_time': merged_data.get('lead_time') or "Not Found",
                        'moq_cases': merged_data.get('moq_cases'),
                        'min_order_quantity_case': merged_data.get('min_order_quantity_case'),
                        'port': merged_data.get('port') or "Not Found",
                        'valid_until': merged_data.get('valid_until'),
                        'best_before_date': merged_data.get('best_before_date'),
                        'vintage': merged_data.get('vintage'),
                        'supplier_reference': merged_data.get('supplier_reference'),
                        'ean_code': merged_data.get('ean_code'),
                        'label_language': merged_data.get('label_language') or "EN",
                        'product_reference': merged_data.get('product_reference'),
                        'custom_status': merged_data.get('custom_status') if merged_data.get('custom_status') not in [None, '', 'Not Found'] else None,
                        'supplier_name': merged_data.get('supplier_name'),
                        'supplier_email': merged_data.get('supplier_email'),  # AI extracted email
                        'error_flags': merged_data.get('error_flags', []),
                    }

                    # Convert numeric fields (except those already handled by _safe_float)
                    numeric_fields = ['cases_per_pallet', 'quantity_case', 'moq_cases', 'min_order_quantity_case']
                    for field in numeric_fields:
                        if safe_data[field] is not None:
                            try:
                                safe_data[field] = float(safe_data[field])
                            except (ValueError, TypeError):
                                safe_data[field] = None

                    safe_data['alcohol_percent'] = (
                        float(str(safe_data['alcohol_percent']).replace('%', '').strip())
                        if safe_data.get('alcohol_percent') not in [None, '', 'Not Found']
                        else None
                    )

                    # Apply OfferItem defaults (convert "Not Found" to None or the proper default)
                    safe_data = _apply_offer_defaults(safe_data)

                    # ── Currency normalisation & EUR conversion ──────────────
                    safe_data = _normalize_currency_and_prices(safe_data)

                    # ── Parse units_per_case & unit_volume_ml from packaging ──
                    _pkg_str = str(safe_data.get('packaging') or '')
                    _pkg_m = re.search(
                        r'(\d+)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(cl|ml|l)',
                        _pkg_str, re.IGNORECASE
                    )
                    if _pkg_m:
                        _pu = int(_pkg_m.group(1))
                        _pv = float(_pkg_m.group(2).replace(',', '.'))
                        _pu_unit = _pkg_m.group(3).lower()
                        safe_data['units_per_case'] = float(_pu)
                        safe_data['unit_volume_ml'] = (
                            _pv * 10 if _pu_unit == 'cl' else
                            _pv * 1000 if _pu_unit == 'l' else _pv
                        )
                        logger.debug(
                            f"Row {idx}: packaging '{_pkg_str}' → "
                            f"units_per_case={_pu}, unit_volume_ml={safe_data['unit_volume_ml']}"
                        )

                    _sup_name  = safe_data.get('supplier_name')
                    _sup_email = _resolve_supplier_email(payload, safe_data.get('supplier_email'))

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
                        supplier_name=_resolve_supplier_name(_sup_name, payload),
                        supplier_email=_sup_email,
                        sender_name=payload.sender_name,
                        sender_email=payload.sender_email,
                        supplier_reference=safe_data['supplier_reference'],
                        source_channel=payload.source_channel or "",
                        source_message_id=payload.source_message_id or "",
                        source_filename=payload.source_filename or "",
                        attachment_filenames=[att.fileName for att in
                                              payload.attachments] if payload.attachments else [],
                        attachment_count=len(payload.attachments) if payload.attachments else 0,
                        confidence_score=0.95,
                        needs_manual_review=False,
                        error_flags=safe_data['error_flags'] if isinstance(safe_data.get('error_flags'), list) else [],
                        custom_status=safe_data['custom_status'],
                        processing_version="2.0.0",
                        ean_code=safe_data['ean_code'],
                        label_language=safe_data['label_language'],
                        product_reference=safe_data['product_reference'],
                    )

                    offer_dict = offer.model_dump(mode='json')

                    offers.append(offer_dict)
                    valid_count += 1

                    # ── Persist to database ──────────────────────────────────
                    save_offer_to_db(offer_dict, job_id)

                    logger.info(f"Dispatching sequential webhook for product: {offer_dict['product_name']}")
                    send_consolidated_webhook(
                        job_id=job_id,
                        payload_type="single_row",
                        data={"product": offer_dict},
                        delivery_id=f"{job_id}_{valid_count}"
                    )

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
                    'unit_volume_ml': _safe_float(extracted_data.get('unit_volume_ml'), 0),
                    'units_per_case': _safe_float(extracted_data.get('units_per_case'), 0),
                    'cases_per_pallet': extracted_data.get('cases_per_pallet'),
                    'quantity_case': extracted_data.get('quantity_case'),
                    'gift_box': extracted_data.get('gift_box'),
                    'refillable_status': extracted_data.get('refillable_status') or "",
                    'currency': extracted_data.get('currency') or "EUR",
                    'price_per_unit': _safe_float(extracted_data.get('price_per_unit'), 0),
                    'price_per_unit_eur': _safe_float(extracted_data.get('price_per_unit_eur'), 0),
                    'price_per_case': _safe_float(extracted_data.get('price_per_case'), 0),
                    'price_per_case_eur': _safe_float(extracted_data.get('price_per_case_eur'), 0),
                    'fx_rate': _safe_float(extracted_data.get('fx_rate'), 1.0),
                    'fx_date': extracted_data.get('fx_date'),
                    'alcohol_percent': extracted_data.get('alcohol_percent'),
                    'origin_country': extracted_data.get('origin_country'),
                    'supplier_country': extracted_data.get('supplier_country') or "",
                    'incoterm': extracted_data.get('incoterm') or "Not Found",
                    'location': extracted_data.get('location') or "Not Found",
                    'lead_time': extracted_data.get('lead_time') or "Not Found",
                    'moq_cases': extracted_data.get('moq_cases'),
                    'min_order_quantity_case': extracted_data.get('min_order_quantity_case'),
                    'port': extracted_data.get('port') or "Not Found",
                    'valid_until': extracted_data.get('valid_until'),
                    'best_before_date': extracted_data.get('best_before_date'),
                    'vintage': extracted_data.get('vintage'),
                    'supplier_reference': extracted_data.get('supplier_reference'),
                    'ean_code': extracted_data.get('ean_code'),
                    'label_language': extracted_data.get('label_language') or "EN",
                    'product_reference': extracted_data.get('product_reference'),
                    'custom_status': extracted_data.get('custom_status') if extracted_data.get('custom_status') not in [None, '', 'Not Found'] else None,
                    'supplier_name': extracted_data.get('supplier_name'),
                    'supplier_email': extracted_data.get('supplier_email'),  # AI extracted email
                    'error_flags': extracted_data.get('error_flags', []),
                }

                # Convert numeric fields
                numeric_fields = ['cases_per_pallet', 'quantity_case', 'moq_cases', 'min_order_quantity_case']
                for field in numeric_fields:
                    if safe_data[field] is not None:
                        try:
                            safe_data[field] = float(safe_data[field])
                        except (ValueError, TypeError):
                            safe_data[field] = None

                safe_data['alcohol_percent'] = (
                    float(str(safe_data['alcohol_percent']).replace('%', '').strip())
                    if safe_data.get('alcohol_percent') not in [None, '', 'Not Found']
                    else None
                )

                safe_data = _apply_offer_defaults(safe_data)

                safe_data = _normalize_currency_and_prices(safe_data)

                _sup_name  = safe_data.get('supplier_name')
                _sup_email = _resolve_supplier_email(payload, safe_data.get('supplier_email'))

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
                    alcohol_percent=safe_data.get('alcohol_percent'),
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
                    supplier_name=_resolve_supplier_name(_sup_name, payload),
                    supplier_email=_sup_email,
                    sender_name=payload.sender_name,
                    sender_email=payload.sender_email,
                    supplier_reference=safe_data['supplier_reference'],
                    source_channel=payload.source_channel or "",
                    source_message_id=payload.source_message_id or "",
                    source_filename=payload.source_filename or "",
                    attachment_filenames=[att.fileName for att in payload.attachments] if payload.attachments else [],
                    attachment_count=len(payload.attachments) if payload.attachments else 0,
                    confidence_score=0.95,
                    needs_manual_review=False,
                    error_flags=safe_data['error_flags'] if isinstance(safe_data.get('error_flags'), list) else [],
                    custom_status=safe_data['custom_status'],
                    processing_version="2.0.0",
                    ean_code=safe_data['ean_code'],
                    label_language=safe_data['label_language'],
                    product_reference=safe_data['product_reference'],
                )

                offer_dict = offer.model_dump(mode='json')
                offers.append(offer_dict)

                # ── Persist to database ──────────────────────────────────────
                save_offer_to_db(offer_dict, job_id)

                logger.info(f"Dispatching sequential webhook for single offer: {offer_dict['product_name']}")
                send_consolidated_webhook(
                    job_id=job_id,
                    payload_type="single_row",
                    data={"product": offer_dict},
                    delivery_id=f"{job_id}_single"
                )

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