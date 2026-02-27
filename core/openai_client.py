import os
import json
import base64
import pandas as pd
from typing import Dict, Any, List
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging
import traceback
import re
from datetime import datetime

# Root logging configured in entry points
logger = logging.getLogger(__name__)

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def get_exchange_rate_to_eur(currency: str) -> float:
    """Get exchange rate from any currency to EUR using OpenAI"""
    if currency in ["Not Found", "", None, "EUR"]:
        return 1.0

    try:
        logger.info(f"Getting exchange rate for {currency} to EUR")

        prompt = f"""
        What is the current exchange rate from {currency} to EUR (Euro)?
        Return ONLY a JSON object with the exchange rate as a float number.
        Example: {{"rate": 0.85}} for USD to EUR
        Use the most recent reliable exchange rate.
        """

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=100
        )

        content = response.choices[0].message.content
        result = json.loads(content)
        rate = float(result.get('rate', 1.0))

        logger.info(f"Exchange rate for {currency} to EUR: {rate}")
        return rate

    except Exception as e:
        logger.error(f"Error getting exchange rate for {currency}: {e}")
        return 1.0


def convert_price_to_eur(price, currency, exchange_rate):
    """Convert price to EUR using exchange rate"""
    if price in [None, "Not Found", "", 0, "0"]:
        return None
    try:
        price_float = float(price)
        if price_float == 0:
            return None
        return round(price_float * exchange_rate, 2)
    except (ValueError, TypeError):
        return None


async def extract_offer(text: str) -> dict:
    logger.info(f"extract_offer called with text length: {len(text)}")
    logger.debug(f"extract_offer text preview: {text[:200]}...")

    # Maximum characters per chunk to prevent context overflow (roughly 5000 tokens)
    # Using characters vs tokens as a rough generic heuristic
    CHUNK_SIZE = 25000

    # Split text into manageable chunks
    text_chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)] if len(text) > CHUNK_SIZE else [
        text]
    logger.info(f"Split input text into {len(text_chunks)} chunk(s).")

    all_products = []

    for idx, chunk in enumerate(text_chunks):
        logger.info(f"Processing chunk {idx + 1} of {len(text_chunks)}...")

        prompt = f"""
        🔥 MASTER EXTRACTION POLICY - CRITICAL RULES TO FOLLOW:
        The system must never fabricate, assume, or auto-fill missing information.
        If a value is not explicitly present in the source file, leave the field blank.
        When multiple incoterms or pricing structures are present, create separate rows.
        All normalization rules (volume, percentage, currency, incoterm conversion) must follow strict formatting standards defined below.

        You are extracting commercial alcohol offers from text.
        Return JSON ONLY, no explanation.

        Extract ALL products from the text. Return a JSON object with a 'products' array containing ALL products found.

        SCHEMA DEFINITION - Use EXACTLY these field names and rules:
        - uid: Unique internal ID for each row (DO NOT generate - leave as "Not Found")
        - product_key: Logical ID for deduplication (brand + name + volume + packaging). UPPERCASE with underscores.
        - processing_version: Backend version used (leave as "Not Found")
        - brand: Brand or trademark of the product.
        - product_name: Commercial product name.
        - product_reference: Supplier or internal reference/SKU.
        - category: Main category (Wine, Spirits, Beer, Soft Drinks, Food...).
        - sub_category: Sub-category (e.g. Red Wine, Whisky, Lager...).
        - origin_country: Country of origin (ISO 2 code or full name).
        - vintage: Vintage year (for wine/champagne).
        - alcohol_percent: Alcohol percentage if applicable.
        - packaging: Full packaging description (e.g. 6x750ml).
        - unit_volume_ml: Volume per unit in milliliters (convert CL: 75CL = 750ml).
        - units_per_case: Number of units (bottles/cans) per case.
        - cases_per_pallet: Number of cases per pallet.
        - quantity_case: Number of cases offered or ordered.
        - bottle_or_can_type: Packaging type (bottle/can/other).
        - price_per_unit: Unit price.
        - price_per_case: Case price.
        - currency: Currency (EUR, USD, GBP...). Extract this from the text - look for currency symbols (€, $, £) or currency codes. If EURO is written, convert to EUR.
        - price_per_unit_eur: Unit price converted into EUR. (Leave as "Not Found" - will be calculated later)
        - price_per_case_eur: Case price converted into EUR. (Leave as "Not Found" - will be calculated later)
        - incoterm: Incoterm (FOB, CIF, EXW, DAP…).
        - location: Location/port associated with the incoterm.
        - min_order_quantity_case: Minimum order quantity in cases.
        - port: Port of loading/destination if applicable.
        - lead_time: Lead time or availability.
        - supplier_name: Name of the supplier. Must be extracted from email signature, company footer, header, or official company name. NOT person names or sales desk names.
        - supplier_reference: Supplier offer reference.
        - supplier_country: Supplier's country.
        - offer_date: Date of the offer (leave as "Not Found").
        - valid_until: Offer validity date.
        - date_received: Actual timestamp when received (leave as "Not Found").
        - source_channel: Source of data (leave as "Not Found").
        - source_filename: Name of the received file (leave as "Not Found").
        - source_message_id: Message ID (leave as "Not Found").
        - confidence_score: Confidence indicator AI (leave as 0.0).
        - error_flags: List of errors detected (leave as empty array []).
        - needs_manual_review: Boolean indicating if review needed (leave as false).
        - best_before_date: Best before date (date or 'fresh').
        - label_language: Languages on label (e.g. 'UK text', 'SA label').
        - ean_code: EAN product barcode.
        - gift_box: Indicates if product includes gift box (GBX).
        - refillable_status: REF/NRF: refillable or non‑refillable. Only extract if explicitly written. Never assume NRF by default.
        - custom_status: Customs status: T1 or T2. Extract if present in offer.
        - moq_cases: Minimum order quantity stated in the offer.

        🚨 1️⃣ EMPTY VALUES RULE (VERY IMPORTANT):
        If a value is NOT explicitly present in the original offer, the system MUST:
        - Leave the field EMPTY
        - NEVER insert: 0, "Not Found", "NRF", default placeholders
        - Blank means blank.

        🚨 2️⃣ RF / NRF RULE:
        Column: refillable_status
        - Only extract RF or NRF if explicitly written in the original offer
        - If RF/NRF is NOT present → leave blank
        - Never assume NRF by default

        🚨 3️⃣ CATEGORY & SUB-CATEGORY LOGIC:
        If category = Spirits:
        System must intelligently classify sub-category based on brand/product name:
        Examples:
        - Martell → Cognac
        - Hennessy → Cognac
        - Bacardi → Rum
        - Absolut → Vodka
        - Jack Daniel's → Whiskey
        - Jameson → Irish Whiskey
        - Captain Morgan → Rum
        If sub-category cannot be confidently determined → leave blank
        Do NOT use "Not Found".

        🚨 4️⃣ ALCOHOL PERCENT FORMAT RULE:
        Column: alcohol_percent
        Rules:
        - Must always be formatted as percentage string
        Example:
        - 37.5% (correct)
        - NOT 0.375
        - NOT 37.5
        - If not present → leave blank
        - Never default to 0

        🚨 5️⃣ INCOTERM EXTRACTION RULE:
        If multiple incoterms exist (e.g., "EXW Riga / DAP Loendersloot"):
        System must:
        - Split into separate rows
        - Duplicate all product data
        - Only change incoterm + location
        If incoterm appears in sentence like "EX Warehouse Dublin":
        - Convert to standardized format: EXW Dublin

        🚨 6️⃣ SUPPLIER NAME EXTRACTION RULE:
        Supplier name must be extracted from:
        - Email signature
        - Company footer
        - Header
        - Official company name
        NOT:
        - The sender person name
        - Sales desk name
        - Email username
        Person names must not be used as supplier_name.

        🚨 7️⃣ CUSTOM STATUS RULE (T1 / T2):
        Column: custom_status
        If offer contains:
        If offer contains T1 or T2 anywhere in the offer send it to custom status
        - T1
        - T2
        Extract it.
        If not present → leave blank.

        🚨 8️⃣ PRICE RULES:
        - If price per case is not provided: Leave blank
        - Do NOT calculate automatically unless clearly instructed
        - If currency is written as: EURO → convert to EUR
        - If EUR price already provided: price_per_unit_eur = price_per_unit (if same currency)

        🚨 9️⃣ UNIT VOLUME EXTRACTION RULE:
        Examples:
        - 0.5L → 500 ml
        - 70cl → 700 ml
        - 0.375L → 375 ml
        Always normalize to: unit_volume_ml
        Never use: 0.38 or rounded incorrect values

        🚨 🔟 MULTI-PRICE OFFERS RULE:
        If offer contains multiple products with different prices:
        System must create:
        - Separate row per product
        - No merging

        🚨 1️⃣1️⃣ LEAD TIME RULE:
        Extract natural language:
        Example: "approximately 3 weeks"
        Store exactly as written.
        Do not rewrite.

        🚨 1️⃣2️⃣ MOQ RULE:
        If MOQ not explicitly mentioned:
        - Leave blank
        - Do NOT assume 0.

        🚨 1️⃣3️⃣ CONFIDENCE SCORE RULE:
        Confidence should decrease when:
        - Sub-category inferred
        - Incoterm converted
        - Volume converted
        - Supplier inferred from signature
        High confidence only when directly written.

        🚨 1️⃣4️⃣ ERROR FLAGS:
        If:
        - Multiple incoterms detected
        - Missing currency
        - Volume ambiguous
        - Supplier unclear
        Add clear internal flag in error_flags

        🚨 1️⃣5️⃣ SOURCE DATA RULE:
        Always preserve:
        - source_channel
        - source_filename
        - source_message_id
        Never modify.

        🚨 1️⃣6️⃣ STRICT EXTRACTION PRINCIPLE:
        AI must follow:
        "Extract only what is explicitly written or confidently inferred by well-defined classification logic.
        Never assume missing data."

        IMPORTANT RULES:
        1. Extract ALL products mentioned in the text. Return one object per product in the 'products' array.
        2. CAPTURE FULL NAMES: 'Baileys Original' is the product_name, not just 'Original'.
        3. If you find multiple quantities/prices for one product, create separate entries if they look like distinct offers.
        4. If a field is NOT FOUND or doesn't exist, leave it blank (NOT null and NOT 0).
        5. DO NOT leave string fields as empty string "" - if missing, leave blank.
        6. DO NOT use null for any field - always leave blank for missing values.
        7. Use AI to intelligently match values to fields - if something in email matches a field, extract it

        PRICE INTERPRETATION:
        - "15.95eur" → price_per_case: 15.95, currency: "EUR" (when no /btl or /cs suffix, assume per case)
        - "11,40eur/btl" → price_per_unit: 11.40, currency: "EUR"
        - "32,50eur/cs" → price_per_case: 32.50, currency: "EUR"
        - "$15.95" → price_per_case: 15.95, currency: "USD" (when no /btl or /cs suffix, assume per case)
        - "£11.40/btl" → price_per_unit: 11.40, currency: "GBP"
        - Always extract the currency from the price notation (€, $, £, EUR, USD, GBP, etc.)

        QUANTITY EXTRACTION:
        - "960 cs" → quantity_case: 960
        - "256cs x 3" → quantity_case: 768 (only calculate when multiplication is explicitly shown like "x 3")
        - "1932cs" → quantity_case: 1932
        - If quantity not specified, leave blank. Do NOT default to 0.
        - IMPORTANT: For packaging like "12x750ml", this defines units_per_case (12) and unit_volume_ml (750), NOT quantity_case.
        - quantity_case is the total number of cases offered, not the packaging configuration.
        - If quantity explicitly relates to "FTL" or "Full Truck Load", do NOT assign it to cases_per_pallet. Only assign cases_per_pallet if explicitly stated as a pallet quantity.

        ALCOHOL PERCENT - CRITICAL INSTRUCTION:
        - Extract the alcohol percentage exactly as it appears in the source text.
        - If the text shows "40%" → output alcohol_percent: 40
        - If the text shows "5%" → output alcohol_percent: 5  
        - If the text shows "17%" → output alcohol_percent: 17
        - If the text shows "40" (without % sign) → output alcohol_percent: 40
        - If the text shows "0.4" or "0,4" → output alcohol_percent: 0.4 (DO NOT multiply by 100)
        - If the text shows "40.0" → output alcohol_percent: 40.0
        - NEVER perform any mathematical conversion or multiplication on the alcohol value.
        - NEVER change 0.4 to 40 - keep it exactly as 0.4.
        - If alcohol percentage is not found in the text, return "Not Found".
        - Do NOT default to 0 when alcohol percentage is missing.

        CASES_PER_PALLET - CRITICAL RULE:
        - Only populate cases_per_pallet if pallet quantity is EXPLICITLY stated.
        - Examples of explicit pallet quantity: "60 cases per pallet", "60 cs/pallet", "palletizes 60 cases"
        - If you see "FTL", "Full Truck Load", or truck-related quantities, do NOT populate cases_per_pallet.
        - If not explicitly stated, cases_per_pallet must be blank.
        - If cases_per_pallet = 0, that means missing - treat as blank.

        QUANTITY_CASE - CRITICAL RULE:
        - Only populate quantity_case if the total number of cases is EXPLICITLY stated.
        - Examples: "960 cs", "quantity: 500 cases", "order: 250 cs"
        - Do NOT derive quantity_case from packaging information like "12x750ml".
        - "12x750ml" describes the packaging format (12 bottles of 750ml per case), not how many cases are being offered.
        - If quantity_case is not explicitly stated, it must be blank.
        - If quantity_case = 0, that means missing - treat as blank.

        SUPPLIER NAME EXTRACTION:
        - Attempt to extract in this exact priority:
          1. Extract from file content itself.
          2. Extract from email body (e.g., "Offer from MILANAKO company").
          3. Extract from forwarded email signature block.
        - If none found in those places, leave blank. DO NOT default to the email sender.

        INCOTERM & LOCATION:
        - Example: "FCA Prague" → incoterm: "FCA", location: "Prague"
        - If no incoterm or location found, leave blank.

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
        - If not mentioned, leave blank.

        COMMON PATTERNS IN EMAILS:
        - "Baileys Original 12/100/17/DF/T2" → 12 bottles per case, 100cl (1000ml), 17% alcohol, DF packaging, T2 status
        - "6x70" → units_per_case: 6, unit_volume_ml: 700
        - "24x50cl cans" → units_per_case: 24, unit_volume_ml: 500, bottle_or_can_type: "can"
        - "960 cs" → quantity_case: 960
        - "98,5€" → price_per_case: 98.5, currency: "EUR"
        - "11,40eur/btl" → price_per_unit: 11.40, currency: "EUR"
        - "$15.95/btl" → price_per_unit: 15.95, currency: "USD"
        - "£32.50/cs" → price_per_case: 32.50, currency: "GBP"
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

        REMEMBER: 
        - When in doubt, leave blank. 
        - Never invent numbers. 
        - Only extract what is explicitly stated. 
        - If a value is 0, that means missing - treat as blank.
        - Leave ALL missing fields blank - both strings AND numbers.
        - IMPORTANT: Extract the currency from price notations (€, $, £, EUR, USD, GBP, etc.)
        - If "EURO" is written, convert to "EUR"

        Text Chunk ({idx + 1}/{len(text_chunks)}):
        {chunk}
        """

        try:
            logger.info("Calling OpenAI API for text extraction chunk...")
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=4096
            )

            content = response.choices[0].message.content
            logger.info(f"OpenAI response received for chunk {idx + 1}, length: {len(content)}")

            result = json.loads(content)
            products = result.get('products', [])

            cleaned_products = []
            for product in products:
                # Clean product nulls
                for key in product:
                    if product[key] is None:
                        numeric_fields = [
                            'unit_volume_ml', 'units_per_case', 'cases_per_pallet',
                            'quantity_case', 'price_per_unit', 'price_per_unit_eur',
                            'price_per_case', 'price_per_case_eur', 'fx_rate',
                            'alcohol_percent', 'moq_cases'
                        ]
                        if key in numeric_fields:
                            product[key] = "Not Found"
                        else:
                            product[key] = "Not Found"

                cleaned_product = clean_product_data(product)
                cleaned_products.append(cleaned_product)

            all_products.extend(cleaned_products)
            logger.info(f"Chunk {idx + 1} yielded {len(cleaned_products)} products.")

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in extract_offer for chunk {idx + 1}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error extracting from text chunk {idx + 1}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue

    # After extracting all products, convert prices to EUR
    logger.info("Converting prices to EUR for all products...")
    for product in all_products:
        currency = product.get('currency', "Not Found")

        # Skip if currency is not found or already EUR
        if currency in ["Not Found", "", None, "EUR"]:
            product['price_per_unit_eur'] = product.get('price_per_unit')
            product['price_per_case_eur'] = product.get('price_per_case')
            continue

        # Get exchange rate for this currency
        exchange_rate = await get_exchange_rate_to_eur(currency)

        # Convert unit price if exists
        if product.get('price_per_unit') not in [None, "Not Found", "", 0, "0"]:
            product['price_per_unit_eur'] = convert_price_to_eur(
                product['price_per_unit'],
                currency,
                exchange_rate
            )

        # Convert case price if exists
        if product.get('price_per_case') not in [None, "Not Found", "", 0, "0"]:
            product['price_per_case_eur'] = convert_price_to_eur(
                product['price_per_case'],
                currency,
                exchange_rate
            )

        logger.debug(f"Converted {currency} to EUR for product {product.get('product_name')}: "
                     f"rate={exchange_rate}, unit_eur={product.get('price_per_unit_eur')}, "
                     f"case_eur={product.get('price_per_case_eur')}")

    logger.info(f"extract_offer completed successfully. Total products aggregated: {len(all_products)}")
    return {"products": all_products}


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
                    🔥 MASTER EXTRACTION POLICY - CRITICAL RULES TO FOLLOW:
                    The system must never fabricate, assume, or auto-fill missing information.
                    If a value is not explicitly present in the source file, leave the field blank.
                    When multiple incoterms or pricing structures are present, create separate rows.
                    All normalization rules (volume, percentage, currency, incoterm conversion) must follow strict formatting standards defined below.

                    EXCEL DATA BATCH ({batch_start + 1}-{batch_end} of {total_rows}):

                    Extract EXACTLY {len(batch_df)} products from this data:

                    {json.dumps(data_rows, indent=2)}

                    SCHEMA DEFINITION - Use EXACTLY these field names:
                    - uid: (leave as "Not Found")
                    - product_key: UPPERCASE(brand + name + volume + packaging)
                    - processing_version: (leave as "Not Found")
                    - brand: Brand or trademark of the product.
                    - product_name: Commercial product name.
                    - product_reference: Supplier or internal reference/SKU.
                    - category: Main category (Wine, Spirits, Beer, Soft Drinks, Food...).
                    - sub_category: Sub-category (e.g. Red Wine, Whisky, Lager...).
                    - origin_country: Country of origin (ISO 2 code or full name).
                    - vintage: Vintage year (for wine/champagne).
                    - alcohol_percent: Alcohol percentage if applicable.
                    - packaging: Full packaging description (e.g. 6x750ml).
                    - unit_volume_ml: Volume per unit in milliliters (convert CL).
                    - units_per_case: Number of units per case.
                    - cases_per_pallet: Number of cases per pallet.
                    - quantity_case: Number of cases offered or ordered.
                    - bottle_or_can_type: Packaging type (bottle/can/other).
                    - price_per_unit: Unit price.
                    - price_per_case: Case price.
                    - currency: Currency (EUR, USD, GBP...). Extract from price columns if present.
                    - price_per_unit_eur: Unit price in EUR. (Leave as "Not Found" - will be calculated later)
                    - price_per_case_eur: Case price in EUR. (Leave as "Not Found" - will be calculated later)
                    - incoterm: Incoterm (FOB, CIF, EXW, DAP…).
                    - location: Location/port.
                    - min_order_quantity_case: Minimum order quantity in cases.
                    - port: Port of loading/destination if applicable.
                    - lead_time: Lead time or availability.
                    - supplier_name: (leave as "Not Found").
                    - supplier_reference: Supplier offer reference.
                    - supplier_country: Supplier's country.
                    - offer_date: (leave as "Not Found").
                    - valid_until: Offer validity date.
                    - date_received: (leave as "Not Found").
                    - source_channel: (leave as "Not Found").
                    - source_filename: (leave as "Not Found").
                    - source_message_id: (leave as "Not Found").
                    - confidence_score: (leave 0.0)
                    - error_flags: (leave empty array [])
                    - needs_manual_review: (leave false)
                    - best_before_date: Best before date (BBD)
                    - label_language: Label language
                    - ean_code: EAN product barcode
                    - gift_box: Indicates if includes gift box (GBX)
                    - refillable_status: REF/NRF
                    - custom_status: Customs status: T1 or T2
                    - moq_cases: Minimum order quantity stated in the offer.

                    🚨 1️⃣ EMPTY VALUES RULE (VERY IMPORTANT):
                    If a value is NOT explicitly present in the original offer, the system MUST:
                    - Leave the field EMPTY
                    - NEVER insert: 0, "Not Found", "NRF", default placeholders
                    - Blank means blank.

                    🚨 2️⃣ RF / NRF RULE:
                    Column: refillable_status
                    - Only extract RF or NRF if explicitly written in the original offer
                    - If RF/NRF is NOT present → leave blank
                    - Never assume NRF by default

                    🚨 3️⃣ CATEGORY & SUB-CATEGORY LOGIC:
                    If category = Spirits:
                    System must intelligently classify sub-category based on brand/product name:
                    Examples:
                    - Martell → Cognac
                    - Hennessy → Cognac
                    - Bacardi → Rum
                    - Absolut → Vodka
                    - Jack Daniel's → Whiskey
                    - Jameson → Irish Whiskey
                    - Captain Morgan → Rum
                    If sub-category cannot be confidently determined → leave blank
                    Do NOT use "Not Found".

                    🚨 4️⃣ ALCOHOL PERCENT FORMAT RULE:
                    Column: alcohol_percent
                    Rules:
                    - Must always be formatted as percentage string
                    Example:
                    - 37.5% (correct)
                    - NOT 0.375
                    - NOT 37.5
                    - If not present → leave blank
                    - Never default to 0

                    🚨 5️⃣ INCOTERM EXTRACTION RULE:
                    If multiple incoterms exist (e.g., "EXW Riga / DAP Loendersloot"):
                    System must:
                    - Split into separate rows
                    - Duplicate all product data
                    - Only change incoterm + location
                    If incoterm appears in sentence like "EX Warehouse Dublin":
                    - Convert to standardized format: EXW Dublin

                    🚨 6️⃣ SUPPLIER NAME EXTRACTION RULE:
                    Supplier name must be extracted from:
                    - Email signature
                    - Company footer
                    - Header
                    - Official company name
                    NOT:
                    - The sender person name
                    - Sales desk name
                    - Email username
                    Person names must not be used as supplier_name.

                    🚨 7️⃣ CUSTOM STATUS RULE (T1 / T2):
                    Column: custom_status
                    If offer contains:
                    - T1
                    - T2
                    Extract it.
                    If not present → leave blank.

                    🚨 8️⃣ PRICE RULES:
                    - If price per case is not provided: Leave blank
                    - Do NOT calculate automatically unless clearly instructed
                    - If currency is written as: EURO → convert to EUR
                    - If EUR price already provided: price_per_unit_eur = price_per_unit (if same currency)

                    🚨 9️⃣ UNIT VOLUME EXTRACTION RULE:
                    Examples:
                    - 0.5L → 500 ml
                    - 70cl → 700 ml
                    - 0.375L → 375 ml
                    Always normalize to: unit_volume_ml
                    Never use: 0.38 or rounded incorrect values

                    🚨 🔟 MULTI-PRICE OFFERS RULE:
                    If offer contains multiple products with different prices:
                    System must create:
                    - Separate row per product
                    - No merging

                    🚨 1️⃣1️⃣ LEAD TIME RULE:
                    Extract natural language:
                    Example: "approximately 3 weeks"
                    Store exactly as written.
                    Do not rewrite.

                    🚨 1️⃣2️⃣ MOQ RULE:
                    If MOQ not explicitly mentioned:
                    - Leave blank
                    - Do NOT assume 0.

                    🚨 1️⃣3️⃣ CONFIDENCE SCORE RULE:
                    Confidence should decrease when:
                    - Sub-category inferred
                    - Incoterm converted
                    - Volume converted
                    - Supplier inferred from signature
                    High confidence only when directly written.

                    🚨 1️⃣4️⃣ ERROR FLAGS:
                    If:
                    - Multiple incoterms detected
                    - Missing currency
                    - Volume ambiguous
                    - Supplier unclear
                    Add clear internal flag in error_flags

                    🚨 1️⃣5️⃣ SOURCE DATA RULE:
                    Always preserve:
                    - source_channel
                    - source_filename
                    - source_message_id
                    Never modify.

                    🚨 1️⃣6️⃣ STRICT EXTRACTION PRINCIPLE:
                    AI must follow:
                    "Extract only what is explicitly written or confidently inferred by well-defined classification logic.
                    Never assume missing data."

                    MAPPING FROM EXCEL DATA:
                    - Ensure you capture the full product name and brand.
                    - If a row is clearly a product offer, extract it.
                    - If a row is just a subtotal or header, skip it.
                    - If a field is NOT FOUND or doesn't exist, leave it blank (NOT null and NOT 0).
                    - NEVER return empty string "" for missing values, always leave blank.
                    - DO NOT use null for any field - always leave blank for missing values.

                    PRICE INTERPRETATION:
                    - "15.95eur" → price_per_case: 15.95, currency: "EUR" (when no /btl or /cs suffix, assume per case)
                    - "11,40eur/btl" → price_per_unit: 11.40, currency: "EUR"
                    - "32,50eur/cs" → price_per_case: 32.50, currency: "EUR"
                    - "$15.95" → price_per_case: 15.95, currency: "USD" (when no /btl or /cs suffix, assume per case)
                    - Always extract currency from price notations

                    QUANTITY EXTRACTION:
                    - "960 cs" → quantity_case: 960
                    - "256cs x 3" → quantity_case: 768 (only calculate when multiplication is explicitly shown like "x 3")
                    - "1932cs" → quantity_case: 1932
                    - If quantity not specified, leave blank. Do NOT default to 0.
                    - IMPORTANT: For packaging like "12x750ml", this defines units_per_case (12) and unit_volume_ml (750), NOT quantity_case.
                    - quantity_case is the total number of cases offered, not the packaging configuration.
                    - If quantity explicitly relates to "FTL" or "Full Truck Load", do NOT assign it to cases_per_pallet. Only assign cases_per_pallet if explicitly stated as a pallet quantity.

                    ALCOHOL PERCENT - CRITICAL INSTRUCTION:
                    - Extract the alcohol percentage exactly as it appears in the source text.
                    - If the text shows "40%" → output alcohol_percent: 40
                    - If the text shows "5%" → output alcohol_percent: 5
                    - If the text shows "17%" → output alcohol_percent: 17
                    - If the text shows "40" (without % sign) → output alcohol_percent: 40
                    - If the text shows "0.4" or "0,4" → output alcohol_percent: 0.4 (DO NOT multiply by 100)
                    - If the text shows "40.0" → output alcohol_percent: 40.0
                    - NEVER perform any mathematical conversion or multiplication on the alcohol value.
                    - NEVER change 0.4 to 40 - keep it exactly as 0.4.
                    - If alcohol percentage is not found in the text, return "Not Found".
                    - Do NOT default to 0 when alcohol percentage is missing.

                    CASES_PER_PALLET - CRITICAL RULE:
                    - Only populate cases_per_pallet if pallet quantity is EXPLICITLY stated.
                    - Examples of explicit pallet quantity: "60 cases per pallet", "60 cs/pallet", "palletizes 60 cases"
                    - If you see "FTL", "Full Truck Load", or truck-related quantities, do NOT populate cases_per_pallet.
                    - If not explicitly stated, cases_per_pallet must be blank.
                    - If cases_per_pallet = 0, that means missing - treat as blank.

                    QUANTITY_CASE - CRITICAL RULE:
                    - Only populate quantity_case if the total number of cases is EXPLICITLY stated.
                    - Examples: "960 cs", "quantity: 500 cases", "order: 250 cs"
                    - Do NOT derive quantity_case from packaging information like "12x750ml".
                    - "12x750ml" describes the packaging format (12 bottles of 750ml per case), not how many cases are being offered.
                    - If quantity_case is not explicitly stated, it must be blank.
                    - If quantity_case = 0, that means missing - treat as blank.

                    SUPPLIER NAME EXTRACTION:
                    - Attempt to extract in this exact priority:
                      1. Extract from file content itself.
                      2. Extract from email body (e.g., "Offer from MILANAKO company").
                      3. Extract from forwarded email signature block.
                    - If none found in those places, leave blank. DO NOT default to the email sender.

                    INCOTERM & LOCATION:
                    - Example: "FCA Prague" → incoterm: "FCA", location: "Prague"
                    - If no incoterm or location found, leave blank.

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
                    - If not mentioned, leave blank.

                    REMEMBER: 
                    - When in doubt, leave blank. 
                    - Never invent numbers. 
                    - Only extract what is explicitly stated. 
                    - If a value is 0, that means missing - treat as blank.
                    - Leave ALL missing fields blank - both strings AND numbers.
                    - IMPORTANT: Extract currency from price columns where present.
                    - If "EURO" is written, convert to "EUR"

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
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system",
                                    "content": f"You are a professional data extraction expert. You extract commercial alcohol product data from Excel. Return COMPLETE JSON with 'products' array containing EXACTLY {len(batch_df)} products. NEVER skip rows. Create a product for every row even if data is missing, using logical defaults."
                                },
                                {"role": "user", "content": batch_text}
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.0,
                            max_tokens=4096,
                            top_p=1.0,
                            frequency_penalty=0.0,
                            presence_penalty=0.0
                        )

                        content = response.choices[0].message.content
                        logger.info(f"Batch {batch_start // batch_size + 1} OpenAI response length: {len(content)}")

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

                    # Convert prices to EUR for all extracted products
                    logger.info("Converting prices to EUR for all Excel products...")
                    for product in all_extracted_products:
                        currency = product.get('currency', "")

                        # Skip if currency is not found or already EUR
                        if currency in ["", None, "EUR"]:
                            product['price_per_unit_eur'] = product.get('price_per_unit')
                            product['price_per_case_eur'] = product.get('price_per_case')
                            continue

                        exchange_rate = await get_exchange_rate_to_eur(currency)

                        # Convert unit price if exists
                        if product.get('price_per_unit') not in [None, "", 0, "0"]:
                            product['price_per_unit_eur'] = convert_price_to_eur(
                                product['price_per_unit'],
                                currency,
                                exchange_rate
                            )

                        # Convert case price if exists
                        if product.get('price_per_case') not in [None, "", 0, "0"]:
                            product['price_per_case_eur'] = convert_price_to_eur(
                                product['price_per_case'],
                                currency,
                                exchange_rate
                            )
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

                    if isinstance(fallback_result, dict) and 'products' in fallback_result:
                        for product in fallback_result['products']:
                            if product.get('currency') not in ["", None, "EUR"]:
                                exchange_rate = await get_exchange_rate_to_eur(product.get('currency'))
                                if product.get('price_per_unit') not in [None, "", 0, "0"]:
                                    product['price_per_unit_eur'] = convert_price_to_eur(
                                        product['price_per_unit'],
                                        product.get('currency'),
                                        exchange_rate
                                    )
                                if product.get('price_per_case') not in [None, "", 0, "0"]:
                                    product['price_per_case_eur'] = convert_price_to_eur(
                                        product['price_per_case'],
                                        product.get('currency'),
                                        exchange_rate
                                    )

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
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text",
                                 "text": "Extract all commercial alcohol offers from this image. Return a JSON object with a 'products' array following the standard schema. Make sure to extract the currency from any price notations (€, $, £, EUR, USD, GBP, etc.)"},
                                {
                                    "type": "image_url",
                                    "image_url": f"data:{content_type};base64,{base64_image}",
                                },
                            ],
                        }
                    ],
                    max_tokens=4096,
                )
                logger.info("Image processed with OpenAI")
                result = await extract_offer(response.choices[0].message.content)

                if isinstance(result, dict) and 'products' in result:
                    for product in result['products']:
                        if product.get('currency') not in ["", None, "EUR"]:
                            exchange_rate = await get_exchange_rate_to_eur(product.get('currency'))
                            if product.get('price_per_unit') not in [None, "", 0, "0"]:
                                product['price_per_unit_eur'] = convert_price_to_eur(
                                    product['price_per_unit'],
                                    product.get('currency'),
                                    exchange_rate
                                )
                            if product.get('price_per_case') not in [None, "", 0, "0"]:
                                product['price_per_case_eur'] = convert_price_to_eur(
                                    product['price_per_case'],
                                    product.get('currency'),
                                    exchange_rate
                                )

                return result

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
        'uid': "Not Found",
        'product_key': "Not Found",
        'processing_version': "Not Found",
        'brand': "Not Found",
        'product_name': "Not Found",
        'product_reference': "Not Found",
        'category': "Not Found",
        'sub_category': "Not Found",
        'origin_country': "Not Found",
        'vintage': "Not Found",
        'alcohol_percent': None,  # Changed from "Not Found" to None for numeric fields
        'packaging': "Not Found",
        'unit_volume_ml': None,  # Changed from "Not Found" to None for numeric fields
        'units_per_case': None,  # Changed from "Not Found" to None for numeric fields
        'cases_per_pallet': None,  # Changed from "Not Found" to None for numeric fields
        'quantity_case': None,  # Changed from "Not Found" to None for numeric fields
        'bottle_or_can_type': "Not Found",
        'price_per_unit': None,  # Changed from "Not Found" to None for numeric fields
        'price_per_case': None,  # Changed from "Not Found" to None for numeric fields
        'currency': "EUR",
        'price_per_unit_eur': None,  # Changed from "Not Found" to None for numeric fields
        'price_per_case_eur': None,  # Changed from "Not Found" to None for numeric fields
        'incoterm': "Not Found",
        'location': "Not Found",
        'min_order_quantity_case': None,  # Changed from "Not Found" to None for numeric fields
        'port': "Not Found",
        'lead_time': "Not Found",
        'supplier_name': "Not Found",
        'supplier_reference': "Not Found",
        'supplier_country': "Not Found",
        'offer_date': "Not Found",
        'valid_until': "Not Found",
        'date_received': "Not Found",
        'source_channel': "Not Found",
        'source_filename': "Not Found",
        'source_message_id': "Not Found",
        'confidence_score': 0.0,
        'error_flags': [],
        'needs_manual_review': False,
        'best_before_date': "Not Found",
        'label_language': "Not Found",
        'ean_code': "Not Found",
        'gift_box': "Not Found",
        'refillable_status': "NRF",
        'custom_status': "Not Found",
        'moq_cases': None  # Changed from "Not Found" to None for numeric fields
    }

    cleaned_product = {}

    for field, default_value in schema_fields.items():
        if field in product:
            value = product[field]

            if value in [None, "Not Found", "", "null", 0, "0"]:
                cleaned_product[field] = default_value
            else:
                numeric_keys = [
                    'unit_volume_ml', 'units_per_case', 'cases_per_pallet',
                    'quantity_case', 'price_per_unit', 'price_per_unit_eur',
                    'price_per_case', 'price_per_case_eur', 'alcohol_percent', 'moq_cases'
                ]

                if field in numeric_keys:
                    try:
                        if field == 'alcohol_percent':
                            cleaned_product[field] = float(value)
                        else:
                            cleaned_product[field] = float(value)
                    except (ValueError, TypeError):
                        cleaned_product[field] = None  # Use None for failed conversions
                elif isinstance(default_value, list):
                    cleaned_product[field] = value if isinstance(value, list) else []
                elif isinstance(default_value, bool):
                    cleaned_product[field] = bool(value)
                else:
                    cleaned_product[field] = str(value) if value is not None else default_value
        else:
            cleaned_product[field] = default_value

    if cleaned_product['product_key'] in ["", None] and cleaned_product['product_name'] not in ["", None]:
        product_key = str(cleaned_product['product_name']).replace(' ', '_').replace('/', '_').replace('&',
                                                                                                       '_').replace('.',
                                                                                                                    '').upper()
        cleaned_product['product_key'] = product_key

    # Convert any 0 values to None for numeric fields that shouldn't have 0 as a valid value
    numeric_fields_never_zero = [
        'cases_per_pallet', 'quantity_case', 'moq_cases',
        'alcohol_percent', 'unit_volume_ml', 'units_per_case'
    ]

    for field in numeric_fields_never_zero:
        if field in cleaned_product and cleaned_product[field] == 0:
            cleaned_product[field] = None

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