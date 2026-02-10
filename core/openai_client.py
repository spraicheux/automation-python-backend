import os
import json
import base64
import pandas as pd
from typing import Dict, Any, List
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging
import traceback

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def extract_offer(text: str) -> dict:
    logger.info(f"extract_offer called with text length: {len(text)}")
    logger.debug(f"extract_offer text preview: {text[:200]}...")

    prompt = f"""
    You are extracting commercial alcohol offers from text.
    Return JSON ONLY, no explanation.

    Extract ALL products from the text. Return a JSON object with a 'products' array containing ALL products found.

    SCHEMA DEFINITION - Use EXACTLY these field names and rules:
    - uid: Unique internal ID for each row (DO NOT generate this - leave as empty string "")
    - product_key: Logical ID for deduplication (brand + name + volume + packaging) in UPPERCASE with underscores
    - processing_version: Leave as empty string "" (backend will fill)
    - brand: Brand or trademark
    - product_name: Commercial product name
    - product_reference: Supplier or internal reference/SKU
    - category: Main category (Wine, Spirits, Beer, Soft Drinks, Food...)
    - sub_category: Sub-category (e.g. Red Wine, Whisky, Lager...)
    - origin_country: Country of origin (ISO 2 code or full name)
    - vintage: Vintage year (for wine/champagne)
    - alcohol_percent: Alcohol percentage
    - packaging: Full packaging description (e.g. "6x750ml Bottles")
    - unit_volume_ml: Volume per unit in milliliters (convert from CL: 75CL = 750ml)
    - units_per_case: Number of units (bottles/cans) per case
    - cases_per_pallet: Number of cases per pallet
    - quantity_case: Number of cases offered
    - bottle_or_can_type: "bottle", "can", or other
    - price_per_unit: Unit price
    - price_per_case: Case price
    - currency: Currency (EUR, USD, GBP...)
    - price_per_unit_eur: Unit price in EUR (same as price_per_unit if EUR)
    - price_per_case_eur: Case price in EUR (same as price_per_case if EUR)
    - incoterm: Incoterm (FOB, CIF, EXW, DAP…)
    - location: Location/port associated with the incoterm
    - min_order_quantity_case: Minimum order quantity in cases
    - port: Port of loading/destination if applicable
    - lead_time: Lead time or availability (e.g., "5 weeks", "Available now")
    - supplier_name: Leave as empty string "" (will be filled by backend)
    - supplier_reference: Supplier offer reference
    - supplier_country: Supplier's country
    - offer_date: Leave as empty string "" (backend will fill)
    - valid_until: Offer validity date
    - date_received: Leave as empty string "" (backend will fill)
    - source_channel: Leave as empty string "" (backend will fill)
    - source_filename: Leave as empty string "" (backend will fill)
    - source_message_id: Leave as empty string "" (backend will fill)
    - confidence_score: Leave as 0.0 (backend will fill)
    - error_flags: Leave as empty array [] (backend will fill)
    - needs_manual_review: Leave as false (backend will fill)
    - best_before_date: Best before date (mainly for beers). Can be a date or 'fresh'
    - label_language: Languages printed on label (e.g. 'UK text', 'SA label')
    - ean_code: EAN product barcode
    - gift_box: Indicates if includes gift box ("GBX" if yes, else "")
    - refillable_status: "REF" for refillable, "NRF" for non-refillable
    - custom_status: Customs status: "T1" or "T2"
    - moq_cases: Minimum order quantity stated in the offer

    IMPORTANT RULES:
    1. Extract ALL products mentioned in the text
    2. For each product, extract ALL fields you can find in the text
    3. If a field is NOT FOUND in the text, return EMPTY STRING "" for string fields and 0 for numeric fields
    4. Do NOT use null or None - only empty strings "" or 0
    5. Do NOT invent data - only extract what you see
    6. Use AI to intelligently match values to fields - if something in email matches a field, extract it
    PRICE INTERPRETATION:
- "15.95eur" → price_per_case: 15.95 (when no /btl or /cs suffix, assume per case)
- "11,40eur/btl" → price_per_unit: 11.40
- "32,50eur/cs" → price_per_case: 32.50

QUANTITY EXTRACTION:
- "960 cs" → quantity_case: 960
- "256cs x 3" → quantity_case: 768 (calculate: 256 × 3)
- "1932cs" → quantity_case: 1932
- If quantity not specified, leave as 0

DATE FIELDS:
- "9/2026", "8/2026" → best_before_date: "2026-09-01", "2026-08-01"
- "BBD 03.06.2026" → best_before_date: "2026-06-03"
- "fresh" → best_before_date: "fresh"
- These are NOT lead_time

CUSTOM STATUS:
- "T1" → custom_status: "T1"
- "T2" → custom_status: "T2"

PACKAGING_RAW:
- "cans" → packaging_raw: "can"
- "btls" or "bottle" → packaging_raw: "bottle"

LABEL LANGUAGE:
- Only extract when explicitly mentioned: "UK text", "SA label", "multi text"
- "UK text" → label_language: "EN"
- "SA label" → label_language: "multiple" 
- "multi text" → label_language: "multiple"
- If not mentioned, leave as empty string ""

    COMMON PATTERNS IN EMAILS:
    - "Baileys Original 12/100/17/DF/T2" → 12 bottles per case, 100cl (1000ml), 17% alcohol, DF packaging, T2 status
    - "6x70" → units_per_case: 6, unit_volume_ml: 700
    - "24x50cl cans" → units_per_case: 24, unit_volume_ml: 500, bottle_or_can_type: "can"
    - "960 cs" → quantity_case: 960
    - "98,5€" → price_per_case: 98.5, currency: "EUR"
    - "11,40eur/btl" → price_per_unit: 11.40, currency: "EUR"
    - "EXW Loendersloot" → incoterm: "EXW", location: "Loendersloot bonded warehouse in Netherlands"
    - "DAP LOE" → incoterm: "DAP", location: "Loendersloot bonded warehouse in Netherlands"
    - "5 weeks LT" → lead_time: "5 weeks"
    - "BBD 03.06.2026" → best_before_date: "2026-06-03"
    - "fresh" → best_before_date: "fresh"
    - "UK text" → label_language: "EN"
    - "SA label" → label_language: "multiple"
    - "T1" or "T2" → custom_status: "T1" or "T2"
    - "REF" → refillable_status: "REF"
    - "NRF" → refillable_status: "NRF"

    CATEGORY DETECTION:
    - Whisky/Whiskey/Scotch/Bourbon → category: "Spirits", sub_category: "Whisky"
    - Champagne/Sparkling → category: "Wine", sub_category: "Champagne"
    - Wine/Red/White → category: "Wine", sub_category: (red/white/rose)
    - Beer/Lager/Ale → category: "Beer", sub_category: (lager/ale/stout)
    - Cognac/Brandy → category: "Spirits", sub_category: "Brandy"
    - Vodka/Gin/Rum → category: "Spirits", sub_category: (vodka/gin/rum)
    - Liqueur → category: "Spirits", sub_category: "Liqueur"
    - Soft Drinks/Energy Drinks → category: "Soft Drinks"
    - Food → category: "Food"

    RETURN FORMAT:
    {{
      "products": [
        {{
          "uid": "",
          "product_key": "BAILEYS_ORIGINAL_1000ML_BOTTLE",
          "processing_version": "",
          "brand": "Baileys",
          "product_name": "Baileys Original",
          "product_reference": "",
          "category": "Spirits",
          "sub_category": "Cream Liqueur",
          "origin_country": "",
          "vintage": "",
          "alcohol_percent": 17,
          "packaging": "12x1000ml Bottles",
          "unit_volume_ml": 1000,
          "units_per_case": 12,
          "cases_per_pallet": 0,
          "quantity_case": 960,
          "bottle_or_can_type": "bottle",
          "price_per_unit": 8.21,
          "price_per_case": 98.5,
          "currency": "EUR",
          "price_per_unit_eur": 8.21,
          "price_per_case_eur": 98.5,
          "incoterm": "EXW",
          "location": "Loendersloot bonded warehouse in Netherlands",
          "min_order_quantity_case": 0,
          "port": "",
          "lead_time": "Available now",
          "supplier_name": "",
          "supplier_reference": "",
          "supplier_country": "",
          "offer_date": "",
          "valid_until": "",
          "date_received": "",
          "source_channel": "",
          "source_filename": "",
          "source_message_id": "",
          "confidence_score": 0.0,
          "error_flags": [],
          "needs_manual_review": false,
          "best_before_date": "",
          "label_language": "EN",
          "ean_code": "",
          "gift_box": "",
          "refillable_status": "NRF",
          "custom_status": "T2",
          "moq_cases": 0
        }}
      ]
    }}

    Text:
    {text}
    """

    try:
        logger.info("Calling OpenAI API for text extraction...")
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        content = response.choices[0].message.content
        logger.info(f"OpenAI response received, length: {len(content)}")
        logger.debug(f"OpenAI response preview: {content[:200]}...")

        result = json.loads(content)
        logger.info(f"JSON parsed successfully, keys: {list(result.keys())}")

        null_count_before = sum(1 for v in result.values() if v is None)
        if null_count_before > 0:
            logger.warning(f"Found {null_count_before} null values in OpenAI response")

        for key in result:
            if result[key] is None:
                numeric_fields = [
                    'unit_volume_ml', 'units_per_case', 'cases_per_pallet',
                    'quantity_case', 'price_per_unit', 'price_per_unit_eur',
                    'price_per_case', 'price_per_case_eur', 'fx_rate',
                    'alcohol_percent', 'moq_cases'
                ]

                if key in numeric_fields:
                    result[key] = 0
                    logger.debug(f"Converted null to 0 for numeric field: {key}")
                else:
                    result[key] = ""
                    logger.debug(f"Converted null to empty string for field: {key}")

        logger.info(f"extract_offer completed successfully")
        if 'products' not in result:
            logger.warning("No 'products' key in response, creating empty array")
            result = {'products': []}

        products = result.get('products', [])
        logger.info(f"Extracted {len(products)} products from text")

        cleaned_products = []
        for product in products:
            cleaned_product = clean_product_data(product)
            cleaned_products.append(cleaned_product)

        result['products'] = cleaned_products

        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in extract_offer: {e}")
        logger.error(f"Raw content that failed to parse: {content[:500]}")
        return {}
    except Exception as e:
        logger.error(f"Error extracting from text: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {}


async def extract_from_file(file_path: str, content_type: str) -> Dict[str, Any]:
    logger.info(f"extract_from_file called with file_path: {file_path}, content_type: {content_type}")

    try:
        text_content = ""

        if content_type in [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel',
            'application/vnd.ms-excel.sheet.macroEnabled.12',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.macroEnabled.12'
        ] or file_path.lower().endswith(('.xlsx', '.xls', '.xlsm')):
            logger.info("Processing Excel file...")
            try:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    file_ext = os.path.splitext(file_path)[1].lower()
                    logger.info(f"File exists, size: {file_size} bytes, extension: {file_ext}")

                    if file_ext == '.xlsm':
                        logger.info("Processing XLSM (Macro-Enabled Excel) file...")
                else:
                    logger.error(f"File does not exist: {file_path}")
                    return {"error": f"File not found: {file_path}"}

                logger.info("Reading Excel file with pandas...")

                try:
                    if file_path.lower().endswith('.xls'):
                        df = pd.read_excel(file_path, engine='xlrd')
                    elif file_path.lower().endswith('.xlsm'):
                        df = pd.read_excel(file_path, engine='openpyxl')
                    else:
                        df = pd.read_excel(file_path, engine='openpyxl')
                except Exception as read_error:
                    logger.warning(f"Primary read method failed: {read_error}")
                    logger.info("Trying fallback read method...")
                    df = pd.read_excel(file_path)

                logger.info(f"Excel file loaded successfully. Shape: {df.shape}")
                logger.info(f"Columns: {list(df.columns)}")

                # Log first few rows for debugging
                if len(df) > 0:
                    logger.debug(f"First 3 rows:\n{df.head(3).to_string()}")
                else:
                    logger.warning("DataFrame is empty after loading")

                if df.empty:
                    logger.warning("Excel file is empty")
                    return {"error": "Excel file is empty"}

                total_rows = len(df)
                logger.info(f"Total rows in Excel: {total_rows}")

                logger.debug(f"DataFrame dtypes:\n{df.dtypes}")

                for col in df.columns:
                    sample_data = df[col].head(3).tolist()
                    logger.debug(f"Column '{col}' sample data: {sample_data}")

                batch_size = 6
                all_extracted_products = []
                processed_row_count = 0

                for batch_start in range(0, total_rows, batch_size):
                    batch_end = min(batch_start + batch_size, total_rows)
                    batch_df = df.iloc[batch_start:batch_end]

                    logger.info(
                        f"Processing batch {batch_start // batch_size + 1}: rows {batch_start} to {batch_end - 1} ({len(batch_df)} rows)")

                    data_rows = []
                    for idx, row in batch_df.iterrows():
                        row_dict = {}
                        for col in batch_df.columns:
                            value = row[col]
                            if pd.isna(value):
                                row_dict[col] = ""
                            else:
                                row_dict[col] = str(value)
                        data_rows.append(row_dict)

                    batch_text = f"""
                    EXCEL DATA BATCH ({batch_start + 1}-{batch_end} of {total_rows}):

                    Extract EXACTLY {len(batch_df)} products from this data:

                    {json.dumps(data_rows, indent=2)}

                    SCHEMA DEFINITION - Use EXACTLY these field names:
                    - uid: (leave empty)
                    - product_key: UPPERCASE(brand + name + volume + packaging)
                    - processing_version: (leave empty)
                    - brand: Brand name
                    - product_name: Full product name
                    - product_reference: Supplier reference
                    - category: Main category (Wine, Spirits, Beer...)
                    - sub_category: Sub-category
                    - origin_country: Country of origin
                    - vintage: Vintage year
                    - alcohol_percent: Alcohol percentage
                    - packaging: Packaging description
                    - unit_volume_ml: Convert CL to ml (75CL = 750ml)
                    - units_per_case: Number per case
                    - cases_per_pallet: Cases per pallet
                    - quantity_case: Quantity in cases
                    - bottle_or_can_type: "bottle" or "can"
                    - price_per_unit: Unit price
                    - price_per_case: Case price
                    - currency: Currency code
                    - price_per_unit_eur: Unit price in EUR
                    - price_per_case_eur: Case price in EUR
                    - incoterm: Incoterm (EXW, DAP, etc.)
                    - location: Warehouse location
                    - min_order_quantity_case: MOQ in cases
                    - port: Port if mentioned
                    - lead_time: Lead time
                    - supplier_name: (leave empty)
                    - supplier_reference: Supplier reference
                    - supplier_country: Supplier country
                    - offer_date: (leave empty)
                    - valid_until: Validity date
                    - date_received: (leave empty)
                    - source_channel: (leave empty)
                    - source_filename: (leave empty)
                    - source_message_id: (leave empty)
                    - confidence_score: (leave 0.0)
                    - error_flags: (leave empty array [])
                    - needs_manual_review: (leave false)
                    - best_before_date: BBD
                    - label_language: Label language
                    - ean_code: EAN barcode
                    - gift_box: "GBX" or empty
                    - refillable_status: "REF" or "NRF"
                    - custom_status: "T1" or "T2"
                    - moq_cases: MOQ in cases

                    MAPPING FROM EXCEL COLUMNS:
                    - Description → product_name, brand
                    - REF/NRF → refillable_status
                    - GB/WGB → origin_country: "United Kingdom"
                    - CL → unit_volume_ml (×10)
                    - BTL → units_per_case
                    - Alc,% → alcohol_percent
                    - D → supplier_reference
                    - Qty CS → quantity_case
                    - Price EUR/CS → price_per_case
                    - Date → valid_until (if not 'STOCK')
                    - Incoterms/Warehouse → location and incoterm
                    - GCP → location: 'Grupo Corporativo Pérez bonded warehouse in Panama', incoterm: 'DAP'
                    - LOE → location: 'Loendersloot bonded warehouse in Netherlands', incoterm: 'DAP'
                    
                    PRICE INTERPRETATION:
                    - "15.95eur" → price_per_case: 15.95 (when no /btl or /cs suffix, assume per case)
                    - "11,40eur/btl" → price_per_unit: 11.40
                    - "32,50eur/cs" → price_per_case: 32.50

                    QUANTITY EXTRACTION:
                    - "960 cs" → quantity_case: 960
                    - "256cs x 3" → quantity_case: 768 (calculate: 256 × 3)
                    - "1932cs" → quantity_case: 1932
                    - If quantity not specified, leave as 0

                    DATE FIELDS:
                    - "9/2026", "8/2026" → best_before_date: "2026-09-01", "2026-08-01"
                    - "BBD 03.06.2026" → best_before_date: "2026-06-03"
                    - "fresh" → best_before_date: "fresh"
                    - These are NOT lead_time

                    CUSTOM STATUS:
                    - "T1" → custom_status: "T1"
                    - "T2" → custom_status: "T2"
                    
                    PACKAGING_RAW:
                    - "cans" → packaging_raw: "can"
                    - "btls" or "bottle" → packaging_raw: "bottle"
                    
                    LABEL LANGUAGE:
                    - Only extract when explicitly mentioned: "UK text", "SA label", "multi text"
                    - "UK text" → label_language: "EN"
                    - "SA label" → label_language: "multiple" 
                    - "multi text" → label_language: "multiple"
                    - If not mentioned, leave as empty string ""

                    RETURN FORMAT:
                    {{
                      "products": [
                        {{"product_name": "...", "product_key": "...", ...}}
                      ]
                    }}
                    """

                    logger.debug(f"Batch text length: {len(batch_text)}")

                    try:
                        logger.info(f"Calling OpenAI API for batch {batch_start // batch_size + 1}...")
                        response = await client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {
                                    "role": "system",
                                    "content": f"You extract commercial alcohol product data from Excel. Return COMPLETE JSON with 'products' array containing EXACTLY {len(batch_df)} products. NEVER skip rows. Create a product for every row even if some data is missing."
                                },
                                {"role": "user", "content": batch_text}
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.1,
                            max_tokens=10000,
                            top_p=0.95,
                            frequency_penalty=0.0,
                            presence_penalty=0.0
                        )

                        content = response.choices[0].message.content
                        logger.info(f"Batch {batch_start // batch_size + 1} OpenAI response length: {len(content)}")
                        logger.debug(f"Batch {batch_start // batch_size + 1} response preview: {content[:200]}...")

                        if not content.strip().endswith('}'):
                            logger.warning(
                                f"Batch {batch_start // batch_size + 1}: JSON appears incomplete, attempting repair")
                            json_start = content.find('{')
                            if json_start != -1:
                                open_braces = 0
                                close_braces = 0
                                for i, char in enumerate(content[json_start:]):
                                    if char == '{':
                                        open_braces += 1
                                    elif char == '}':
                                        close_braces += 1
                                        if close_braces == open_braces:
                                            content = content[json_start:json_start + i + 1]
                                            logger.info(
                                                f"Batch {batch_start // batch_size + 1}: Repaired JSON, new length: {len(content)}")
                                            break

                        try:
                            result = json.loads(content)
                            logger.info(f"Batch {batch_start // batch_size + 1} JSON parsed successfully")

                            if isinstance(result, dict) and 'products' in result:
                                batch_products = result['products']
                                logger.info(
                                    f"Batch {batch_start // batch_size + 1} extracted {len(batch_products)} products from 'products' key")

                                if len(batch_products) != len(batch_df):
                                    logger.warning(
                                        f"Batch {batch_start // batch_size + 1}: Expected {len(batch_df)} products but got {len(batch_products)}. Attempting to salvage...")
                                    if len(batch_products) < len(batch_df):
                                        missing_count = len(batch_df) - len(batch_products)
                                        for i in range(missing_count):
                                            default_product = clean_product_data({})
                                            default_product[
                                                'product_name'] = f"Row {batch_start + len(batch_products) + i + 1}"
                                            batch_products.append(default_product)

                                cleaned_batch_products = []
                                for product in batch_products:
                                    cleaned_product = clean_product_data(product)
                                    cleaned_batch_products.append(cleaned_product)

                                all_extracted_products.extend(cleaned_batch_products)
                                processed_row_count += len(batch_df)

                                if cleaned_batch_products:
                                    logger.debug(
                                        f"First product in batch: {json.dumps(cleaned_batch_products[0], indent=2)[:300]}...")
                            elif isinstance(result, list):
                                logger.info(
                                    f"Batch {batch_start // batch_size + 1}: Found direct list with {len(result)} items")
                                if len(result) != len(batch_df):
                                    logger.warning(
                                        f"Batch {batch_start // batch_size + 1}: List count mismatch. Expected {len(batch_df)}, got {len(result)}")
                                    if len(result) < len(batch_df):
                                        missing_count = len(batch_df) - len(result)
                                        for i in range(missing_count):
                                            default_product = clean_product_data({})
                                            default_product['product_name'] = f"Row {batch_start + len(result) + i + 1}"
                                            result.append(default_product)

                                cleaned_batch_products = []
                                for product in result:
                                    cleaned_product = clean_product_data(product)
                                    cleaned_batch_products.append(cleaned_product)
                                all_extracted_products.extend(cleaned_batch_products)
                                processed_row_count += len(batch_df)
                            else:
                                logger.warning(
                                    f"Batch {batch_start // batch_size + 1}: Unexpected format, creating default products")
                                for i in range(len(batch_df)):
                                    default_product = clean_product_data({})
                                    default_product['product_name'] = f"Row {batch_start + i + 1}"
                                    all_extracted_products.append(default_product)
                                processed_row_count += len(batch_df)

                        except json.JSONDecodeError as e:
                            logger.error(f"JSON parse error in batch {batch_start // batch_size + 1}: {e}")
                            logger.error(
                                f"Batch {batch_start // batch_size + 1} raw response (first 500 chars): {content[:500]}")
                            logger.error(
                                f"Batch {batch_start // batch_size + 1} raw response (last 500 chars): {content[-500:]}")

                            try:
                                import re
                                json_pattern = r'\{.*\}'
                                matches = re.findall(json_pattern, content, re.DOTALL)
                                if matches:
                                    for match in matches:
                                        try:
                                            result = json.loads(match)
                                            if isinstance(result, dict) and 'products' in result:
                                                batch_products = result['products']
                                                if isinstance(batch_products, list):
                                                    if len(batch_products) < len(batch_df):
                                                        missing_count = len(batch_df) - len(batch_products)
                                                        for i in range(missing_count):
                                                            default_product = clean_product_data({})
                                                            default_product[
                                                                'product_name'] = f"Row {batch_start + len(batch_products) + i + 1}"
                                                            batch_products.append(default_product)

                                                    cleaned_batch_products = []
                                                    for product in batch_products:
                                                        cleaned_product = clean_product_data(product)
                                                        cleaned_batch_products.append(cleaned_product)
                                                    all_extracted_products.extend(cleaned_batch_products)
                                                    processed_row_count += len(batch_df)
                                                    logger.warning(
                                                        f"Batch {batch_start // batch_size + 1}: Salvaged {len(cleaned_batch_products)} products from regex")
                                                    break
                                        except:
                                            continue
                            except Exception as salvage_error:
                                logger.error(f"Failed to salvage JSON: {salvage_error}")
                                for i in range(len(batch_df)):
                                    default_product = clean_product_data({})
                                    default_product['product_name'] = f"Row {batch_start + i + 1}"
                                    all_extracted_products.append(default_product)
                                processed_row_count += len(batch_df)

                    except Exception as e:
                        logger.error(f"Error extracting batch {batch_start // batch_size + 1}: {e}")
                        for i in range(len(batch_df)):
                            default_product = clean_product_data({})
                            default_product['product_name'] = f"Row {batch_start + i + 1}"
                            all_extracted_products.append(default_product)
                        processed_row_count += len(batch_df)

                logger.info(f"Total products extracted from all batches: {len(all_extracted_products)}")
                logger.info(f"Total rows processed: {processed_row_count} of {total_rows}")

                if len(all_extracted_products) != total_rows:
                    logger.warning(
                        f"Product count mismatch! Excel has {total_rows} rows but extracted {len(all_extracted_products)} products")
                    if len(all_extracted_products) < total_rows:
                        missing_count = total_rows - len(all_extracted_products)
                        logger.warning(f"Adding {missing_count} default products to match row count")
                        for i in range(missing_count):
                            default_product = clean_product_data({})
                            default_product['product_name'] = f"Missing Row {len(all_extracted_products) + i + 1}"
                            all_extracted_products.append(default_product)

                if all_extracted_products:
                    logger.debug(
                        f"Sample extracted products (first 2): {json.dumps(all_extracted_products[:2], indent=2)}")
                    logger.info(f"Total extracted products after fixes: {len(all_extracted_products)}")
                else:
                    logger.warning("No products extracted from any batch, trying fallback...")
                    simplified_rows = []
                    for i in range(min(10, len(df))):
                        row = df.iloc[i]
                        row_dict = {}
                        for col in df.columns:
                            value = row[col]
                            row_dict[col] = str(value) if not pd.isna(value) else ""
                        simplified_rows.append(row_dict)

                    text_content = f"Excel with {total_rows} rows. Sample data:\n{json.dumps(simplified_rows, indent=2)}"
                    logger.info(f"Fallback text content length: {len(text_content)}")
                    fallback_result = await extract_offer(text_content)
                    logger.info(f"Fallback extraction result type: {type(fallback_result)}")
                    return fallback_result

                if all_extracted_products:
                    result = {
                        'products': all_extracted_products,
                        'total_products': len(all_extracted_products),
                        'file_type': 'excel',
                        'processed_in_batches': True,
                        'batches_processed': (total_rows + batch_size - 1) // batch_size,
                        'original_rows': total_rows
                    }
                    logger.info(f"extract_from_file completed successfully with {len(all_extracted_products)} products")
                    return result
                else:
                    return {"error": "No products could be extracted from the Excel file"}

            except Exception as e:
                logger.error(f"Error reading Excel file: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                text_content = f"Excel file - error reading: {str(e)}"
                return {"error": f"Excel read error: {str(e)}"}

        elif content_type == 'application/pdf':
            logger.info("Processing PDF file...")
            try:
                import PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_content = ""
                    for page in pdf_reader.pages:
                        text_content += page.extract_text()
                logger.info(f"PDF extracted, text length: {len(text_content)}")
            except ImportError:
                logger.error("PyPDF2 not installed for PDF processing")
                text_content = "PDF processing requires PyPDF2 library"
                return {"error": "PyPDF2 not installed"}
            except Exception as e:
                logger.error(f"Error reading PDF file: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                text_content = f"PDF file - error reading: {str(e)}"
                return {"error": f"PDF read error: {str(e)}"}

        elif 'image' in content_type:
            logger.info("Processing image file...")
            try:
                with open(file_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": """You extract commercial alcohol product data from Excel 
                                using the exact schema provided. Return COMPLETE JSON with 'products' array containing 
                                EXACTLY {len(batch_df)} products. Use AI to best match values from Excel to schema fields. 
                                If a field is not found, use empty string or 0. NEVER skip rows."""},
                                {
                                    "type": "image_url",
                                    "image_url": f"data:{content_type};base64,{base64_image}",
                                },
                            ],
                        }
                    ],
                    max_tokens=1000,
                )
                logger.info("Image processed with OpenAI")
                return await extract_offer(response.choices[0].message.content)

            except Exception as e:
                logger.error(f"Error extracting from image: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return {"error": f"Image processing error: {str(e)}"}

        else:
            logger.info(f"Processing text file with content_type: {content_type}")
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_content = file.read()
                logger.info(f"Text file read, length: {len(text_content)}")
            except:
                try:
                    with open(file_path, 'r', encoding='latin-1') as file:
                        text_content = file.read()
                    logger.info(f"Text file read with latin-1 encoding, length: {len(text_content)}")
                except Exception as e:
                    logger.error(f"Error reading text file: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return {"error": f"Text file read error: {str(e)}"}

        if text_content:
            logger.info(f"Calling extract_offer with text content length: {len(text_content)}")
            result = await extract_offer(text_content)
            logger.info(f"extract_offer returned result type: {type(result)}")
            return result
        else:
            logger.warning("No text content extracted from file")
            return {"error": "No content extracted from file"}

    except Exception as e:
        logger.error(f"Error extracting from file: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": f"General extraction error: {str(e)}"}


def clean_product_data(product: dict) -> dict:
    """Clean up product data to ensure it matches schema exactly"""
    schema_fields = {
        'uid': '',
        'product_key': '',
        'processing_version': '',
        'brand': '',
        'product_name': '',
        'product_reference': '',
        'category': '',
        'sub_category': '',
        'origin_country': '',
        'vintage': '',
        'alcohol_percent': 0,
        'packaging': '',
        'unit_volume_ml': 0,
        'units_per_case': 0,
        'cases_per_pallet': 0,
        'quantity_case': 0,
        'bottle_or_can_type': '',
        'price_per_unit': 0,
        'price_per_case': 0,
        'currency': 'EUR',
        'price_per_unit_eur': 0,
        'price_per_case_eur': 0,
        'incoterm': '',
        'location': '',
        'min_order_quantity_case': 0,
        'port': '',
        'lead_time': '',
        'supplier_name': '',
        'supplier_reference': '',
        'supplier_country': '',
        'offer_date': '',
        'valid_until': '',
        'date_received': '',
        'source_channel': '',
        'source_filename': '',
        'source_message_id': '',
        'confidence_score': 0.0,
        'error_flags': [],
        'needs_manual_review': False,
        'best_before_date': '',
        'label_language': 'EN',
        'ean_code': '',
        'gift_box': '',
        'refillable_status': 'NRF',
        'custom_status': '',
        'moq_cases': 0
    }

    cleaned_product = {}

    for field, default_value in schema_fields.items():
        if field in product:
            value = product[field]

            if value is None:
                if isinstance(default_value, (int, float)):
                    cleaned_product[field] = 0
                else:
                    cleaned_product[field] = ""
            else:
                if isinstance(default_value, (int, float)):
                    try:
                        cleaned_product[field] = float(value)
                    except (ValueError, TypeError):
                        cleaned_product[field] = 0
                else:
                    cleaned_product[field] = str(value) if value is not None else ""
        else:
            cleaned_product[field] = default_value

    if cleaned_product['product_key'] == 'Not Found' and cleaned_product['product_name'] != 'Not Found':
        product_key = cleaned_product['product_name'].replace(' ', '_').replace('/', '_').replace('&', '_').replace('.',
                                                                                                                    '').upper()
        cleaned_product['product_key'] = product_key

    if (cleaned_product['price_per_unit'] == 0 and
            cleaned_product['price_per_case'] > 0 and
            cleaned_product['units_per_case'] > 0):
        cleaned_product['price_per_unit'] = cleaned_product['price_per_case'] / cleaned_product['units_per_case']
        cleaned_product['price_per_unit_eur'] = cleaned_product['price_per_unit']

    if cleaned_product['price_per_unit_eur'] == 0 and cleaned_product['price_per_unit'] > 0:
        cleaned_product['price_per_unit_eur'] = cleaned_product['price_per_unit']

    if cleaned_product['price_per_case_eur'] == 0 and cleaned_product['price_per_case'] > 0:
        cleaned_product['price_per_case_eur'] = cleaned_product['price_per_case']

    return cleaned_product


def parse_buffer_data(buffer_data: dict) -> bytes:
    logger.debug(f"parse_buffer_data called with buffer_data type: {type(buffer_data)}")

    if isinstance(buffer_data, dict) and buffer_data.get('type') == 'Buffer':
        try:
            data_bytes = bytes(buffer_data['data'])
            logger.debug(f"Parsed Buffer type, length: {len(data_bytes)} bytes")
            return data_bytes
        except Exception as e:
            logger.error(f"Error parsing Buffer type: {e}")
            return b''
    elif isinstance(buffer_data, dict) and 'data' in buffer_data:
        try:
            if isinstance(buffer_data['data'], str):
                data_bytes = base64.b64decode(buffer_data['data'])
                logger.debug(f"Parsed base64 string, length: {len(data_bytes)} bytes")
                return data_bytes
            elif isinstance(buffer_data['data'], list):
                data_bytes = bytes(buffer_data['data'])
                logger.debug(f"Parsed list data, length: {len(data_bytes)} bytes")
                return data_bytes
            else:
                logger.warning(f"Unexpected data type in buffer_data['data']: {type(buffer_data['data'])}")
                return b''
        except Exception as e:
            logger.error(f"Error parsing buffer_data with 'data' key: {e}")
            return b''
    elif isinstance(buffer_data, str):
        try:
            data_bytes = base64.b64decode(buffer_data)
            logger.debug(f"Parsed base64 string directly, length: {len(data_bytes)} bytes")
            return data_bytes
        except Exception as e:
            logger.error(f"Error parsing base64 string: {e}")
            return b''
    else:
        logger.warning(f"Unexpected buffer_data type: {type(buffer_data)}")
        logger.debug(f"buffer_data value: {buffer_data}")
        return b''