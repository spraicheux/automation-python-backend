import os
import json
import base64
import pandas as pd
from typing import Dict, Any, List
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging
import traceback

# Root logging configured in entry points
logger = logging.getLogger(__name__)

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ──────────────────────────────────────────────────────────────────────────────
# SHARED EXTRACTION RULES — used in BOTH extract_offer and Excel batch prompts
# ──────────────────────────────────────────────────────────────────────────────
SHARED_EXTRACTION_RULES = """
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
- currency: Currency (EUR, USD, GBP...).
- price_per_unit_eur: Unit price converted into EUR.
- price_per_case_eur: Case price converted into EUR.
- incoterm: Incoterm (FOB, CIF, EXW, DAP…).
- location: Location/port associated with the incoterm.
- min_order_quantity_case: Minimum order quantity in cases.
- port: Port of loading/destination if applicable.
- lead_time: Lead time or availability.
- supplier_name: Name of the supplier company (leave as "Not Found" only if truly absent).
- supplier_reference: Supplier offer reference.
- supplier_country: Supplier's country.
- offer_date: Date of the offer (leave as "Not Found").
- valid_until: Offer validity date.
- date_received: Actual timestamp when received (leave as "Not Found").
- source_channel: Source of data (leave as "Not Found").
- source_filename: Name of the received file (leave as "Not Found").
- source_message_id: Message ID (leave as "Not Found").
- confidence_score: Confidence indicator AI (leave as 0.0).
- error_flags: List of extraction warnings — see ERROR FLAGS rules below.
- needs_manual_review: Boolean indicating if review needed (leave as false).
- best_before_date: Best before date (date or 'fresh').
- label_language: Languages on label (e.g. 'UK text', 'SA label').
- ean_code: EAN product barcode.
- gift_box: Indicates if product includes gift box (GBX).
- refillable_status: REF or NRF. Use "Not Found" if not stated.
- custom_status: T1 or T2 customs status. Use "Not Found" if not stated.
- moq_cases: Minimum order quantity stated in the offer.

══════════════════════════════════════════════════════════════════════
RULE 1 — CUSTOM STATUS (T1 / T2)  ⚠️ HIGHEST PRIORITY
══════════════════════════════════════════════════════════════════════
Scan the ENTIRE text character by character for the tokens "T1" or "T2".
- They may appear in a product code, a column, a note, anywhere — e.g.
  "Baileys 12/100/17/DF/T2", "Status: T1", "T2 goods", "T1", a standalone cell.
- DO NOT require a header or label to be present.
- If "T1" appears ANYWHERE → custom_status: "T1"
- If "T2" appears ANYWHERE → custom_status: "T2"
- If NEITHER T1 nor T2 is found anywhere → custom_status: "Not Found"
- Never default to "Not Found" when T1 or T2 is present in any form.

══════════════════════════════════════════════════════════════════════
RULE 2 — SUB_CATEGORY — INTELLIGENT INFERENCE  ⚠️
══════════════════════════════════════════════════════════════════════
Always try to infer sub_category from brand name, product name, or context.
Use the mapping below (not exhaustive — apply common sense for unlisted brands):

SPIRITS:
  Whisky/Whiskey/Scotch/Bourbon/Malt  → sub_category: "Whisky"
  Vodka                                → sub_category: "Vodka"
  Gin                                  → sub_category: "Gin"
  Rum / BACARDI / Captain Morgan / Havana Club / Diplomatico / Appleton
                                       → sub_category: "Rum"
  Tequila / Mezcal                     → sub_category: "Tequila"
  Cognac / HENNESSY / REMY MARTIN / MARTELL / COURVOISIER / CAMUS / HINE / DELAMAIN / HARDY
                                       → sub_category: "Cognac"
  Brandy / Armagnac (non-Cognac)       → sub_category: "Brandy"
  NOTE: If the word "Cognac" appears in the product name or category label, ALWAYS use
        sub_category: "Cognac" — NEVER "Brandy". Cognac is a protected designation of
        origin and must not be grouped under the generic "Brandy" sub_category.
  Liqueur / BAILEYS / KAHLUA / COINTREAU / AMARETTO / CAMPARI / APEROL
                                       → sub_category: "Liqueur"
  Absinthe                             → sub_category: "Absinthe"
  Grappa                               → sub_category: "Grappa"

WINE:
  Red Wine / Cabernet / Merlot / Shiraz / Malbec / Pinot Noir
                                       → sub_category: "Red Wine"
  White Wine / Chardonnay / Sauvignon / Riesling / Pinot Grigio
                                       → sub_category: "White Wine"
  Rosé / Rose                          → sub_category: "Rosé Wine"
  Champagne / Prosecco / Cava / Sparkling
                                       → sub_category: "Sparkling Wine"
  Port / Sherry / Vermouth / Fortified → sub_category: "Fortified Wine"

BEER:
  Lager / Pilsner / HEINEKEN / STELLA / BUDWEISER / CORONA / PERONI
                                       → sub_category: "Lager"
  Ale / IPA / Pale Ale / GUINNESS (Stout)
                                       → sub_category: "Ale"
  Stout / Porter                       → sub_category: "Stout"
  Wheat Beer / Weiss                   → sub_category: "Wheat Beer"

SOFT DRINKS:
  Energy Drink / RED BULL / Monster    → sub_category: "Energy Drink"
  Cola / Juice / Water / Mixer         → sub_category: (cola/juice/water/mixer)

If you can confidently infer the sub_category from the brand or product name,
DO SO even if it is not explicitly written.
If truly impossible to determine → sub_category: "Not Found"

══════════════════════════════════════════════════════════════════════
RULE 3 — REFILLABLE STATUS (REF / NRF)
══════════════════════════════════════════════════════════════════════
- "REF" anywhere → refillable_status: "REF"
- "NRF" anywhere → refillable_status: "NRF"
- If neither found → refillable_status: "Not Found"
- DO NOT default to "NRF" when not stated.

══════════════════════════════════════════════════════════════════════
RULE 4 — SUPPLIER NAME — COMPANY NAME ONLY  ⚠️ OVERRIDE RULE
══════════════════════════════════════════════════════════════════════
Extract the COMPANY name (not a person's name, not an email username).
Priority order:
  1. Official company name in file header, title, or footer
     (e.g. "KOLLARAS & CO", "DIAGEO PLC", "MILANAKO LTD")
  2. Company name in email signature block
     (look for lines like "John Smith | KOLLARAS Trading Co.")
  3. "Offer from <Company>" or "On behalf of <Company>" in body text
  4. Sheet/tab name if it contains a company name
  5. Letterhead, logo caption, or "From:" company line
Look carefully — the company name is often in:
- The top of a PDF or Excel file
- The footer of an email
- An email signature after the person's name and title
- A "Supplier:" or "Company:" field in the document
NEVER use: person names, email addresses, sales desk names, or email usernames.
If truly not found after exhaustive search → supplier_name: "Not Found"

══════════════════════════════════════════════════════════════════════
RULE 5 — ERROR FLAGS
══════════════════════════════════════════════════════════════════════
Populate error_flags as a list of strings describing extraction issues.
Add a flag for each of the following situations (use clear English):
- "sub_category inferred from brand name" — when you inferred sub_category
- "supplier_name not found" — when supplier_name could not be extracted
- "custom_status not found" — when T1/T2 was not present
- "refillable_status not found" — when REF/NRF was not present
- "price ambiguous — assumed per case" — when price suffix was unclear
- "quantity_case not explicitly stated" — when quantity was missing
- "incoterm not found" — when no incoterm was present
- "multiple incoterms detected — row duplicated" — when rows were split
- Any other notable extraction issue or ambiguity
If no issues → error_flags: []

══════════════════════════════════════════════════════════════════════
RULE 6 — MULTIPLE INCOTERMS → SEPARATE ROWS  ⚠️
══════════════════════════════════════════════════════════════════════
If a product has multiple incoterms (e.g. "EXW RIGA / DAP LOENDERSLOOT",
"FOB Rotterdam or CIF London"), create ONE separate product row per incoterm.
- All other fields (brand, product_name, price, etc.) are IDENTICAL across rows.
- Only incoterm and location differ between the duplicate rows.
- Add "multiple incoterms detected — row duplicated" to error_flags on each row.
- If only one incoterm found → single row as normal.
- If no incoterm found → incoterm: "Not Found", location: "Not Found".

══════════════════════════════════════════════════════════════════════
SUPPLIER REFERENCE — OVERRIDE RULE  ⚠️
══════════════════════════════════════════════════════════════════════
Scan EVERY column and piece of text. If found, MUST write to supplier_reference.
Column names to scan:
  "P.Code", "P Code", "Ref", "Reference", "Ref No", "Supplier Ref",
  "Offer Ref", "Offer No", "SKU", "Item Code", "Product Code",
  "Stock Code", "Art No", "Article", "Code", "Barcode"

══════════════════════════════════════════════════════════════════════
CRITICAL RULES FOR MISSING VALUES
══════════════════════════════════════════════════════════════════════
1. If a field is NOT explicitly stated in the text, return "Not Found". Do NOT invent or hallucinate values.
2. For numeric fields, if the value is missing, return "Not Found" - NOT 0.
3. 0 should NEVER be used as a default for missing numeric values. 0 is ONLY used when "0" appears explicitly in the source.
4. If a numeric field has value 0, that means "Not Found" - treat it as "Not Found".
5. Do NOT calculate or derive values that aren't directly stated. Only extract what is explicitly written.
6. For "12x750ml" -> units_per_case = 12, unit_volume_ml = 750. Do NOT create a quantity_case value from this.
7. If you see "12x750ml" and no other quantity information, quantity_case must be "Not Found", NOT 233 or any other number.
8. cases_per_pallet must be "Not Found" unless pallet quantity is explicitly written (e.g., "60 cases per pallet", "60 cs/pallet").
9. If you see "FTL", "Full Truck Load", or similar, do NOT assign any value to cases_per_pallet.
10. Never hallucinate quantities. If you're unsure, use "Not Found".

IMPORTANT RULES:
1. Extract ALL products mentioned in the text. Return one object per product in the 'products' array.
2. CAPTURE FULL NAMES: 'Baileys Original' is the product_name, not just 'Original'.
3. If you find multiple quantities/prices for one product, create separate entries if they look like distinct offers.
4. If a field is NOT FOUND or doesn't exist, use "Not Found" (NOT null and NOT 0).
5. DO NOT leave string fields as empty string "" - if missing, use "Not Found".
6. DO NOT use null for any field - always use "Not Found" for missing values.
7. Use AI to intelligently match values to fields - if something in email matches a field, extract it.

UNITS_PER_CASE — CRITICAL PARSING RULE:
- The format NxVOLUME (e.g. "6x70cl", "12x100cl", "24x50cl") means N units per case of the given volume.
- units_per_case is ALWAYS the number BEFORE the "x". NEVER use the price or any other number.
- unit_volume_ml is ALWAYS derived from the volume AFTER the "x".
- Examples:
  "6x70cl"   → units_per_case: 6,  unit_volume_ml: 700
  "12x100cl" → units_per_case: 12, unit_volume_ml: 1000
  "6x75cl"   → units_per_case: 6,  unit_volume_ml: 750
  "24x50cl"  → units_per_case: 24, unit_volume_ml: 500
- The number after "at" is ALWAYS the price — NEVER units_per_case.
- Example: "4180 cs Absolut 6x70cl at 29 euro"
    → quantity_case: 4180, units_per_case: 6, unit_volume_ml: 700, price_per_case: 29, currency: "EUR"
  NEVER set units_per_case to 29 or 69 — those are prices.

UNIT_VOLUME_ML — CRITICAL PARSING RULE:
- Always convert volume to milliliters (ml).
- The suffix after a number is ALWAYS a unit letter, NEVER a digit:
  "l" means LITRES, not the digit 1.
- "0,7l" → 0.7 litres → unit_volume_ml: 700   ← NOT 0.71
- "0.7l" → 0.7 litres → unit_volume_ml: 700   ← NOT 0.71
- "1l"   → 1 litre   → unit_volume_ml: 1000
- "1,5l" → 1.5 litres → unit_volume_ml: 1500
- "70cl" → 700ml, "75cl" → 750ml, "100cl" → 1000ml
- "700ml" → 700ml (no conversion needed)
- NEVER append the unit letter to the number. Always convert to ml as a pure integer.


- "15.95eur" → price_per_case: 15.95 (when no /btl or /cs suffix, assume per case)
- "11,40eur/btl" → price_per_unit: 11.40
- "32,50eur/cs" → price_per_case: 32.50

QUANTITY EXTRACTION:
- "960 cs" → quantity_case: 960
- "256cs x 3" → quantity_case: 768 (only calculate when multiplication is explicitly shown like "x 3")
- "1932cs" → quantity_case: 1932
- If quantity not specified, return "Not Found". Do NOT default to 0.
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
- If not explicitly stated, cases_per_pallet must be "Not Found".
- If cases_per_pallet = 0, that means "Not Found" - treat as "Not Found".

QUANTITY_CASE - CRITICAL RULE:
- Only populate quantity_case if the total number of cases is EXPLICITLY stated.
- Examples: "960 cs", "quantity: 500 cases", "order: 250 cs"
- Do NOT derive quantity_case from packaging information like "12x750ml".
- "12x750ml" describes the packaging format (12 bottles of 750ml per case), not how many cases are being offered.
- If quantity_case is not explicitly stated, it must be "Not Found".
- If quantity_case = 0, that means "Not Found" - treat as "Not Found".

INCOTERM & LOCATION:
- Example: "FCA Prague" → incoterm: "FCA", location: "Prague"
- If no incoterm or location found, return "Not Found".
- MULTIPLE INCOTERMS: See RULE 6 above — create separate rows.

DATE FIELDS:
- "9/2026", "8/2026" → best_before_date: "2026-09-01", "2026-08-01"
- "BBD 03.06.2026" → best_before_date: "2026-06-03"
- "fresh" → best_before_date: "fresh"
- These are NOT lead_time

PACKAGING_RAW:
- "cans" → packaging_raw: "can"
- "btls" or "bottle" → packaging_raw: "bottle"

LABEL LANGUAGE:
- Only extract when explicitly mentioned: "UK text", "SA label", "multi text"
- "UK text" → label_language: "EN"
- "SA label" → label_language: "multiple"
- "multi text" → label_language: "multiple"
- If not mentioned, return "Not Found".

COMMON PATTERNS IN OFFERS:
- "Baileys Original 12/100/17/DF/T2" → 12 bottles per case, 100cl (1000ml), 17% alcohol, DF packaging, T2 status
- "6x70" → units_per_case: 6, unit_volume_ml: 700
- "24x50cl cans" → units_per_case: 24, unit_volume_ml: 500, bottle_or_can_type: "can"
- "960 cs" → quantity_case: 960
- "98,5€" → price_per_case: 98.5, currency: "EUR"
- "11,40eur/btl" → price_per_unit: 11.40, currency: "EUR"
- "EXW Loendersloot" → incoterm: "EXW", location: "Loendersloot bonded warehouse in Netherlands"
- "DAP LOE" → incoterm: "DAP", location: "Loendersloot bonded warehouse in Netherlands"
- "EXW RIGA / DAP LOENDERSLOOT" → TWO rows: one EXW/RIGA, one DAP/LOENDERSLOOT
- "5 weeks LT" → lead_time: "5 weeks"
- "BBD 03.06.2026" → best_before_date: "2026-06-03"
- "fresh" → best_before_date: "fresh"
- "UK text" → label_language: "EN"
- "SA label" → label_language: "multiple"
- "T1" or "T2" (anywhere) → custom_status: "T1" or "T2"
- "REF" → refillable_status: "REF"
- "NRF" → refillable_status: "NRF"
- "4180 cs Absolut 6x70cl at 29 euro" → quantity_case: 4180, units_per_case: 6, unit_volume_ml: 700, price_per_case: 29, currency: "EUR"
- "2007 cs Absolut 12x100cl at 69 euro" → quantity_case: 2007, units_per_case: 12, unit_volume_ml: 1000, price_per_case: 69, currency: "EUR"

REMEMBER:
- When in doubt, use "Not Found".
- Never invent numbers.
- Only extract what is explicitly stated.
- If a value is 0, that means "Not Found" - treat as "Not Found".
- Use "Not Found" for ALL missing fields - both strings AND numbers.
"""


async def extract_offer(text: str) -> dict:
    logger.info(f"extract_offer called with text length: {len(text)}")
    logger.debug(f"extract_offer text preview: {text[:200]}...")

    CHUNK_SIZE = 25000

    text_chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)] if len(text) > CHUNK_SIZE else [
        text]
    logger.info(f"Split input text into {len(text_chunks)} chunk(s).")

    all_products = []

    for idx, chunk in enumerate(text_chunks):
        logger.info(f"Processing chunk {idx + 1} of {len(text_chunks)}...")

        prompt = f"""
        You are extracting commercial alcohol offers from text.
        Return JSON ONLY, no explanation.

        Extract ALL products from the text. Return a JSON object with a 'products' array containing ALL products found.
        If a product has MULTIPLE INCOTERMS, create one row per incoterm (all other fields identical).

        {SHARED_EXTRACTION_RULES}

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
                max_tokens=16000
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
                    You are extracting commercial alcohol product data from Excel rows.
                    Return JSON ONLY, no explanation.

                    EXCEL DATA BATCH ({batch_start + 1}-{batch_end} of {total_rows}):
                    Extract EXACTLY {len(batch_df)} products from this data.
                    If a product has MULTIPLE INCOTERMS, create one row per incoterm (duplicate all other fields).

                    {json.dumps(data_rows, indent=2)}

                    {SHARED_EXTRACTION_RULES}

                    MAPPING FROM EXCEL DATA:
                    - Ensure you capture the full product name and brand.
                    - If a row is clearly a product offer, extract it.
                    - If a row is just a subtotal or header, skip it.
                    - If a field is NOT FOUND or doesn't exist, use "Not Found" (NOT null and NOT 0).
                    - NEVER return empty string "" for missing values, always use "Not Found".
                    - DO NOT use null for any field - always use "Not Found" for missing values.

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
                            max_tokens=16000,
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
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text",
                                 "text": "Extract all commercial alcohol offers from this image. Return a JSON object with a 'products' array following the standard schema."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{content_type};base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=16000,
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
        'alcohol_percent': None,
        'packaging': "Not Found",
        'packaging_raw': "bottle",
        'unit_volume_ml': None,
        'units_per_case': None,
        'cases_per_pallet': None,
        'quantity_case': None,
        'bottle_or_can_type': "Not Found",
        'price_per_unit': None,
        'price_per_case': None,
        'currency': "EUR",
        'price_per_unit_eur': None,
        'price_per_case_eur': None,
        'incoterm': "Not Found",
        'location': "Not Found",
        'min_order_quantity_case': None,
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
        'refillable_status': "Not Found",
        'custom_status': "Not Found",
        'moq_cases': None
    }

    cleaned_product = {}

    for field, default_value in schema_fields.items():
        if field in product:
            value = product[field]

            # Convert None, empty strings, null, and 0 to appropriate default
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
                        # For alcohol_percent, preserve the exact value without any conversion
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

    if cleaned_product['product_key'] in ["Not Found"] and cleaned_product['product_name'] not in ["Not Found"]:
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