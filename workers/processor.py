import uuid
import base64
from datetime import datetime
from schemas.output import OfferItem
from core.openai_client import extract_offer, extract_from_file, parse_buffer_data
from workers.state import JOB_RESULTS, JOB_STATUS
import tempfile
import os


async def process_offer(payload, job_id: str):
    JOB_STATUS[job_id] = "processing"

    try:
        uid = str(uuid.uuid4())
        extracted_data = {}

        # Process text body if available
        if payload.text_body:
            extracted_data = await extract_offer(payload.text_body or "")

        # Process attachments if available
        if payload.attachments:
            for attachment in payload.attachments:
                try:
                    # Get file extension
                    file_name = attachment.fileName
                    file_ext = file_name.split('.')[-1].lower() if '.' in file_name else ''

                    # Convert buffer data to bytes
                    file_bytes = parse_buffer_data(attachment.data)

                    if not file_bytes:
                        print(f"Warning: Could not parse file data for {file_name}")
                        continue

                    # Create temp file
                    with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=f'.{file_ext}' if file_ext else '.bin'
                    ) as tmp_file:
                        tmp_file.write(file_bytes)
                        tmp_file_path = tmp_file.name

                    try:
                        # Extract data from file
                        file_extracted = await extract_from_file(
                            tmp_file_path,
                            attachment.contentType
                        )

                        # Merge extracted data (file data overrides text data)
                        if file_extracted:
                            # Update only non-null values from file extraction
                            for key, value in file_extracted.items():
                                if value not in [None, "", []]:
                                    extracted_data[key] = value

                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass

                except Exception as e:
                    print(f"Error processing attachment {attachment.fileName}: {e}")
                    continue

        offer = OfferItem(
            uid=uid,
            product_name=extracted_data.get('product_name', "Not Found"),
            product_key=extracted_data.get('product_key', "Not Found"),
            brand=extracted_data.get('brand', "Not Found"),
            category=extracted_data.get('category'),
            sub_category=extracted_data.get('sub_category'),
            packaging=extracted_data.get('packaging', "Bottle"),
            packaging_raw=extracted_data.get('packaging_raw', "bottle"),
            bottle_or_can_type=extracted_data.get('bottle_or_can_type'),
            unit_volume_ml=extracted_data.get('unit_volume_ml', "Not Found"),
            units_per_case=extracted_data.get('units_per_case', "Not Found"),
            cases_per_pallet=extracted_data.get('cases_per_pallet'),
            quantity_case=extracted_data.get('quantity_case'),
            gift_box=extracted_data.get('gift_box'),
            refillable_status=extracted_data.get('refillable_status', "NRF"),
            currency=extracted_data.get('currency', "EUR"),
            price_per_unit=float(extracted_data.get('price_per_unit', "Not Found")),
            price_per_unit_eur=float(extracted_data.get('price_per_unit_eur', "Not Found")),
            price_per_case=float(extracted_data.get('price_per_case', "Not Found")),
            price_per_case_eur=float(extracted_data.get('price_per_case_eur', "Not Found")),
            fx_rate=float(extracted_data.get('fx_rate', "Not Found")),
            fx_date=extracted_data.get('fx_date', "Not Found"),
            alcohol_percent=extracted_data.get('alcohol_percent', "Not Found"),
            origin_country=extracted_data.get('origin_country'),
            supplier_country=extracted_data.get('supplier_country', ""),
            incoterm=extracted_data.get('incoterm', "Not Found"),
            location=extracted_data.get('location', "Not Found"),
            lead_time=extracted_data.get('lead_time', "Not Found"),
            moq_cases=extracted_data.get('moq_cases'),
            valid_until=extracted_data.get('valid_until'),
            offer_date=datetime.utcnow(),
            date_received=datetime.utcnow(),
            best_before_date=extracted_data.get('best_before_date'),
            vintage=extracted_data.get('vintage'),
            supplier_name=payload.supplier_name,
            supplier_email=payload.supplier_email,
            supplier_reference=extracted_data.get('supplier_reference'),
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
            ean_code=extracted_data.get('ean_code'),
            label_language=extracted_data.get('label_language', "EN"),
            product_reference=extracted_data.get('product_reference'),
        )

        JOB_RESULTS[job_id] = offer.model_dump()
        JOB_STATUS[job_id] = "done"

    except Exception as e:
        JOB_STATUS[job_id] = "failed"
        JOB_RESULTS[job_id] = {
            "job_id": job_id,
            "error": str(e),
        }