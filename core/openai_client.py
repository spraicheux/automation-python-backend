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
- price_per_unit: Unit (bottle) price.
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
- supplier_email: Email address of the sender/supplier extracted from "De:" or "From:" header.
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
- gift_box: Indicates if product includes gift box (GBX) or not (NGB). See RULE 9.
- refillable_status: REF or NRF. Use "Not Found" if not stated.
- custom_status: T1 or T2 customs status. Use "Not Found" if not stated.
- moq_cases: Minimum order quantity stated in the offer (in cases).

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
RULE 4b — SUPPLIER EMAIL — FROM / DE HEADER  ⚠️
══════════════════════════════════════════════════════════════════════
Extract the sender's email address into supplier_email.
Look for:
  - "De: Name <email@domain.com>" → extract email@domain.com
  - "From: Name <email@domain.com>" → extract email@domain.com
  - "De: Name email@domain.com" → extract email@domain.com
  - Email in signature block (e.g. "sales@elixemarket.com")
The supplier_email is the SENDER's email, not any recipient.
Apply the SAME email to ALL products from the same offer.
If not found → supplier_email: "Not Found"

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
- "price_per_case calculated from price_per_unit x units_per_case" — when calculated
- "price_per_unit calculated from price_per_case / units_per_case" — when calculated
- "MOQ in bottles, not cases" — when MOQ is given in bottles instead of cases
- "brand name corrected" — when a brand name spelling was corrected to its official form
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
⚠️ DO NOT create extra rows for any other reason.

══════════════════════════════════════════════════════════════════════
RULE 7 — COMPOUND LOCATION — PRESERVE FULL STRING  ⚠️ CRITICAL
══════════════════════════════════════════════════════════════════════
When the location references multiple warehouses or places joined by "/" or "and" or "&",
you MUST preserve the FULL compound string EXACTLY for ALL products from that offer.

Examples:
  "DAP LOENDERSLOOT / NEWCORP" → location: "LOENDERSLOOT / NEWCORP" for EVERY product
  "EXW AMSTERDAM / ROTTERDAM"  → location: "AMSTERDAM / ROTTERDAM" for EVERY product

NEVER split the compound location or assign different parts to different products.
ALL products from the same offer get the SAME compound location string.
This rule applies even if one product seems to fit one location better.

══════════════════════════════════════════════════════════════════════
RULE 8 — PRICE IDENTIFICATION AND CALCULATION  ⚠️ CRITICAL
══════════════════════════════════════════════════════════════════════

STEP 1 — DETECT CURRENCY from symbol (scan the full text):
  - "€" or "EUR" or "eur" or "euro" or "Euros" → currency: "EUR"
  - "$" or "USD" or "usd" or "US$"             → currency: "USD"
  - "£" or "GBP" or "gbp"                      → currency: "GBP"
  Apply the detected currency to ALL products in the offer.

STEP 2 — IDENTIFY PRICE TYPE using explicit suffixes:
  - "/btl", "/bottle", "per bottle", "EUR/btl", "per btl", "€/btl" → price is PRICE PER UNIT (bottle)
    → put value in price_per_unit, leave price_per_case for calculation
  - "/cs", "/case", "per case", "EUR/cs", "USD/cs", "per cs"       → price is PRICE PER CASE
    → put value in price_per_case, leave price_per_unit for calculation
  - No suffix or ambiguous → default to price_per_case

STEP 3 — CALCULATE the missing price (ALWAYS do this when units_per_case is known):
  - If price_per_unit is known AND units_per_case is known AND price_per_case is missing:
    → price_per_case = ROUND(price_per_unit × units_per_case, 2)
    → add "price_per_case calculated from price_per_unit x units_per_case" to error_flags
  - If price_per_case is known AND units_per_case is known AND price_per_unit is missing:
    → price_per_unit = ROUND(price_per_case / units_per_case, 2)
    → add "price_per_unit calculated from price_per_case / units_per_case" to error_flags

EXAMPLES:
  "ABERFELDY 18YO 6/700/43/REF/GBX/T2 €90.30" — context is EUR/btl (per bottle list):
    → price_per_unit: 90.30, currency: "EUR", units_per_case: 6
    → price_per_case = 90.30 × 6 = 541.80

  "USD194/cs Exwork Loendersloot" — explicitly per case:
    → price_per_case: 194.00, currency: "USD"
    → price_per_unit = 194.00 / 12 = 16.17  (if 12 bottles per case)

  "11,40eur/btl" → price_per_unit: 11.40, currency: "EUR"
  "32,50eur/cs"  → price_per_case: 32.50, currency: "EUR"

══════════════════════════════════════════════════════════════════════
RULE 9 — GIFT BOX / NON GIFT BOX  ⚠️
══════════════════════════════════════════════════════════════════════
Scan the product name and product code for gift box indicators:
  - "GBX" or "GB" or "Gift Box" or "gift box" → gift_box: "GBX"
  - "NGB" or "NGBX" or "NGB/T" or "non gift box" or "no gift box" → gift_box: "NGB"
  - "IBC" (individual box carton) → gift_box: "IBC"
  - "STD" (standard, no gift box implied) → gift_box: "NGB"
  - Not stated → gift_box: "Not Found"
NEVER default to "Not Found" when GBX, NGB, NGBX, IBC, or STD is present.
NOTE: "NGB" is NOT the same as gift_box — it means NO gift box.

══════════════════════════════════════════════════════════════════════
RULE 10 — LEAD TIME AND AVAILABILITY  ⚠️
══════════════════════════════════════════════════════════════════════
Extract lead time and availability status into the lead_time field:
  - "ON FLOOR"                          → lead_time: "ON FLOOR"
  - "ON FLOOR (No Escrow)"              → lead_time: "ON FLOOR (No Escrow)"
  - "on floor" (any case)               → lead_time: "ON FLOOR"
  - "6 weeks" or "6 weeks LT"           → lead_time: "6 weeks"
  - "Lead time ~ 3 weeks"               → lead_time: "3 weeks"
  - "Lead time ~3 weeks/One time offer" → lead_time: "3 weeks / One time offer"
  - If not stated → lead_time: "Not Found"
IMPORTANT: If the same lead time applies to multiple products (e.g. stated in header),
apply it to ALL products in the offer.

══════════════════════════════════════════════════════════════════════
RULE 11 — MOQ (MINIMUM ORDER QUANTITY)  ⚠️
══════════════════════════════════════════════════════════════════════
Extract MOQ (minimum order quantity) into moq_cases.
Look for:
  - "MOQ X cases", "MOQ: X cs", "MOQ X"
  - "MOQ X bottles", "MOQ: X btls", "MOQ X btls"
  - "minimum order quantity X", "min order X cases", "min order X bottles"
  - "min X cs", "min. X cases", "min X bottles"
  - "MOQ:" followed by a number
Extract the numeric value, ignoring thousand separators (e.g., "14,352" → 14352).
Store the numeric value in moq_cases.
If the unit is explicitly "bottles" or "btls" (not cases), store the numeric value AND add the error flag:
  "MOQ in bottles, not cases" to the error_flags list.
If MOQ is stated in pallets (e.g. "~ 2 pll of each SKU"), do NOT store in moq_cases
  — this is pallet info, not a case MOQ. Leave moq_cases as "Not Found".
If not found → moq_cases: "Not Found"
Also populate min_order_quantity_case with the same value as moq_cases.

══════════════════════════════════════════════════════════════════════
RULE 12 — STRICT PRODUCT COUNT  ⚠️ CRITICAL — NO HALLUCINATION
══════════════════════════════════════════════════════════════════════
Extract EXACTLY the number of distinct product lines in the offer. NO MORE, NO LESS.
DO NOT create extra rows for:
  - Section headers (e.g. "Whisky", "Rum", "Gin", "Beer")
  - Footer lines, brand lists, signature blocks
  - Subtotals, totals, or summary rows
  - Repeated or duplicate descriptions of the same product
  - General promotional text ("we work with all major brands...")

The ONLY valid reason to create MORE rows than product lines is Rule 6
(multiple incoterms per product → one row per incoterm).

If an offer lists exactly 3 products → output EXACTLY 3 rows (no more, no less).
If an offer lists exactly 7 products → output EXACTLY 7 rows.
Count carefully before outputting.

══════════════════════════════════════════════════════════════════════
RULE 13 — BRAND AND PRODUCT NAME CORRECTION  ⚠️ CRITICAL
══════════════════════════════════════════════════════════════════════
You MUST correct misspelled or truncated brand names to their OFFICIAL commercial spelling.
This is essential for product deduplication and price comparison across offers.

Apply corrections ALWAYS, regardless of how the brand appears in the source text.
Add "brand name corrected" to error_flags whenever you make a correction.

MANDATORY CORRECTIONS (non-exhaustive — apply your knowledge for all spirits/wine/beer brands):

WHISKY / WHISKEY:
  "Ballantine"         → brand: "Ballantine's",   correct the apostrophe
  "Ballantines"        → brand: "Ballantine's"
  "Jack Daniel"        → brand: "Jack Daniel's",  correct the apostrophe
  "Jack Daniels"       → brand: "Jack Daniel's"
  "Johnnie Walker"     → brand: "Johnnie Walker"  (correct — no change)
  "Johnny Walker"      → brand: "Johnnie Walker"  (common misspelling)
  "Chivas"             → brand: "Chivas Regal"    (if no other qualifier present)
  "Grants"             → brand: "Grant's"
  "Dewar"              → brand: "Dewar's"
  "Dewars"             → brand: "Dewar's"
  "Teachers"           → brand: "Teacher's"
  "Famous Grouse"      → brand: "The Famous Grouse"
  "Laphroig"           → brand: "Laphroaig"
  "Glenfidich"         → brand: "Glenfiddich"
  "Oban"               → brand: "Oban"            (correct — no change)
  "Makers Mark"        → brand: "Maker's Mark"
  "Knob Creek"         → brand: "Knob Creek"      (correct — no change)
  "Woodford"           → brand: "Woodford Reserve" (if no other qualifier)
  "Jim Beam"           → brand: "Jim Beam"         (correct — no change)

COGNAC / BRANDY:
  "Hennesy"            → brand: "Hennessy"
  "Hennessey"          → brand: "Hennessy"
  "Remy"               → brand: "Rémy Martin"
  "Remy Martin"        → brand: "Rémy Martin"     (add accent)
  "Remy Martin"        → brand: "Rémy Martin"
  "Courvoisier"        → brand: "Courvoisier"     (correct — no change)
  "Martell"            → brand: "Martell"         (correct — no change)

VODKA:
  "Absolut"            → brand: "Absolut"         (correct — no change; NOT "Absolute")
  "Absolute"           → brand: "Absolut"
  "Smirnof"            → brand: "Smirnoff"
  "Grey Goose"         → brand: "Grey Goose"      (correct — no change)
  "Belvedeer"          → brand: "Belvedere"
  "Ciroc"              → brand: "Cîroc"

GIN:
  "Tanquerey"          → brand: "Tanqueray"
  "Hendricks"          → brand: "Hendrick's"
  "Beefeaters"         → brand: "Beefeater"
  "Gordons"            → brand: "Gordon's"
  "Bombay"             → brand: "Bombay Sapphire" (if no other qualifier)

RUM:
  "Bacardi"            → brand: "Bacardí"         (add accent if source has none)
  "Captain Morgan"     → brand: "Captain Morgan"  (correct — no change)
  "Havana"             → brand: "Havana Club"     (if no other qualifier)
  "Diplomatico"        → brand: "Diplomático"

LIQUEUR:
  "Baileys"            → brand: "Baileys"         (correct — no change)
  "Bailey's"           → brand: "Baileys"         (remove apostrophe — official spelling)
  "Kahlua"             → brand: "Kahlúa"
  "Cointreau"          → brand: "Cointreau"       (correct — no change)
  "Malibu"             → brand: "Malibu"          (correct — no change)
  "Tia Maria"          → brand: "Tia Maria"       (correct — no change)
  "Disarono"           → brand: "Disaronno"
  "Jagermeister"       → brand: "Jägermeister"
  "Sambuca"            → brand: "Sambuca"         (correct — no change; check for brand name e.g. Molinari)

BEER:
  "Heineken"           → brand: "Heineken"        (correct — no change)
  "Stella"             → brand: "Stella Artois"   (if no other qualifier)
  "Budweiser"          → brand: "Budweiser"       (correct — no change)
  "Corona"             → brand: "Corona"          (correct — no change)
  "Guiness"            → brand: "Guinness"
  "Peroni"             → brand: "Peroni"          (correct — no change)

CHAMPAGNE / SPARKLING:
  "Moet"               → brand: "Moët & Chandon"  (if no other qualifier)
  "Moët"               → brand: "Moët & Chandon"
  "Veuve Clicquot"     → brand: "Veuve Clicquot"  (correct — no change)
  "Dom Perignon"       → brand: "Dom Pérignon"    (add accent)
  "Cristal"            → brand: "Louis Roederer Cristal" (if context is Champagne)
  "Bollinger"          → brand: "Bollinger"       (correct — no change)
  "Laurent Perrier"    → brand: "Laurent-Perrier"

GENERAL RULE FOR BRAND CORRECTIONS:
- If a brand is missing its possessive apostrophe (e.g. "Ballantine" → "Ballantine's"), ADD it.
- If a brand has a common accent that was omitted (e.g. "Remy Martin" → "Rémy Martin"), ADD it.
- If a brand name is a well-known partial (e.g. "Chivas" without "Regal"), complete it ONLY if
  no additional qualifier is present in the product name that would indicate a specific sub-brand.
- When in doubt about the correct official spelling, use your knowledge of the alcohol industry
  to apply the most widely recognised commercial brand name.
- ALWAYS add "brand name corrected" to error_flags when you make any correction.

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
5. Do NOT calculate or derive values that aren't directly stated — EXCEPTION: Rule 8 price calculations are REQUIRED.
6. For "12x750ml" -> units_per_case = 12, unit_volume_ml = 750. Do NOT create a quantity_case value from this.
7. If you see "12x750ml" and no other quantity information, quantity_case must be "Not Found", NOT 233 or any other number.
8. cases_per_pallet must be "Not Found" unless pallet quantity is explicitly written (e.g., "60 cases per pallet", "60 cs/pallet").
9. If you see "FTL", "Full Truck Load", or similar, do NOT assign any value to cases_per_pallet.
10. Never hallucinate quantities. If you're unsure, use "Not Found".

IMPORTANT RULES:
1. Extract ALL products mentioned in the text. Return one object per product in the 'products' array.
2. CAPTURE FULL NAMES: 'Baileys Original' is the product_name, not just 'Original'.
3. If a product has MULTIPLE INCOTERMS, create separate entries — one per incoterm (Rule 6).
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

PRICE IDENTIFICATION — REFER TO RULE 8 ABOVE FOR FULL DETAILS:
- "15.95eur" (no suffix) → price_per_case: 15.95 (default: no suffix = per case)
- "11,40eur/btl" → price_per_unit: 11.40
- "32,50eur/cs" → price_per_case: 32.50
- "EUR/btl" context: all prices in that column are per bottle
- "Price/Btle" or "EUR/btl" column header → ALL values in that column are price_per_unit
- ALWAYS calculate the counterpart (case↔unit) using Rule 8.

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
- COMPOUND LOCATIONS: See RULE 7 above — NEVER split compound locations.

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
- "11,40eur/btl" → price_per_unit: 11.40, currency: "EUR" → price_per_case = 11.40 × units_per_case
- "EXW Loendersloot" → incoterm: "EXW", location: "Loendersloot bonded warehouse in Netherlands"
- "DAP LOE" → incoterm: "DAP", location: "Loendersloot bonded warehouse in Netherlands"
- "DAP LOENDERSLOOT / NEWCORP" → incoterm: "DAP", location: "LOENDERSLOOT / NEWCORP" (ALL products)
- "EXW RIGA / DAP LOENDERSLOOT" → TWO rows: one EXW/RIGA, one DAP/LOENDERSLOOT (Rule 6)
- "5 weeks LT" → lead_time: "5 weeks"
- "ON FLOOR" → lead_time: "ON FLOOR"
- "ON FLOOR (No Escrow)" → lead_time: "ON FLOOR (No Escrow)"
- "Lead time ~ 3 weeks" → lead_time: "3 weeks"
- "BBD 03.06.2026" → best_before_date: "2026-06-03"
- "fresh" → best_before_date: "fresh"
- "UK text" → label_language: "EN"
- "SA label" → label_language: "multiple"
- "T1" or "T2" (anywhere) → custom_status: "T1" or "T2"
- "REF" → refillable_status: "REF"
- "NRF" → refillable_status: "NRF"
- "GBX" or "GB" → gift_box: "GBX"
- "NGB" or "NGBX" → gift_box: "NGB"
- "IBC" → gift_box: "IBC"
- "STD" → gift_box: "NGB"
- "4180 cs Absolut 6x70cl at 29 euro" → quantity_case: 4180, units_per_case: 6, unit_volume_ml: 700, price_per_case: 29, currency: "EUR"
- "2007 cs Absolut 12x100cl at 69 euro" → quantity_case: 2007, units_per_case: 12, unit_volume_ml: 1000, price_per_case: 69, currency: "EUR"
- Column header "Price/Btle" or "EUR/btl" → ALL prices in that column are price_per_unit → calculate price_per_case = price × units_per_case
- "MOQ 50 cs" → moq_cases: 50
- "Ballantine" or "Ballantines" → brand: "Ballantine's" + add "brand name corrected" to error_flags
- "Jack Daniel" or "Jack Daniels" → brand: "Jack Daniel's" + add "brand name corrected" to error_flags

FINAL CHECKS BEFORE OUTPUTTING:
1. Count the products in your output — it must match the number of product lines in the offer.
   (Exception: Rule 6 — one extra row per additional incoterm only.)
2. Every product with both price_per_unit and units_per_case must also have price_per_case.
3. Every product with both price_per_case and units_per_case must also have price_per_unit.
4. All products from the same offer share the same: supplier_name, supplier_email,
   incoterm, location, currency, custom_status (unless per-product differences are explicit).
5. Compound locations (X / Y) must appear IDENTICALLY on ALL rows.
6. Review every brand name against Rule 13 — correct any misspellings before outputting.

REMEMBER:
- When in doubt, use "Not Found".
- Never invent numbers.
- Only extract what is explicitly stated (Rule 8 price calculations are the only permitted derivation).
- If a value is 0, that means "Not Found" - treat as "Not Found".
- Use "Not Found" for ALL missing fields - both strings AND numbers.
- ALWAYS correct brand names to their official spelling per Rule 13.
"""


async def extract_offer(text: str) -> dict:
    logger.info(f"[extract_offer] ===== START =====")
    logger.info(f"[extract_offer] Input text length: {len(text)} chars")
    logger.info(f"[extract_offer] Text preview (first 300 chars): {text[:300]!r}")

    CHUNK_SIZE = 25000

    text_chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)] if len(text) > CHUNK_SIZE else [text]
    logger.info(f"[extract_offer] Split into {len(text_chunks)} chunk(s) — CHUNK_SIZE={CHUNK_SIZE}")

    all_products = []

    for idx, chunk in enumerate(text_chunks):
        logger.info(f"[extract_offer] ----- Chunk {idx + 1}/{len(text_chunks)} START — length: {len(chunk)} chars -----")
        logger.info(f"[extract_offer] Chunk {idx + 1} content preview (first 200 chars): {chunk[:200]!r}")

        prompt = f"""
        You are extracting commercial alcohol offers from text.
        Return JSON ONLY, no explanation.

        Extract ALL products from the text. Return a JSON object with a 'products' array containing ALL products found.
        If a product has MULTIPLE INCOTERMS, create one row per incoterm (all other fields identical).

        CRITICAL: Extract ONLY the actual product lines. Do NOT create rows for section headers
        (like "Whisky", "Rum", "Gin"), brand lists, footers, or promotional text.
        Count the product lines carefully — your output must contain exactly that many rows
        (plus duplicates only for multiple incoterms per Rule 6).

        CRITICAL: Apply Rule 13 to correct all brand names to their official commercial spelling
        before outputting. E.g. "Ballantine" → "Ballantine's", "Jack Daniel" → "Jack Daniel's".

        {SHARED_EXTRACTION_RULES}

        Text Chunk ({idx + 1}/{len(text_chunks)}):
        {chunk}
        """

        try:
            logger.info(f"[extract_offer] Chunk {idx + 1}: Calling OpenAI API — model=gpt-4o, max_tokens=16000, temperature=0.0")
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=16000
            )

            content = response.choices[0].message.content
            logger.info(f"[extract_offer] Chunk {idx + 1}: OpenAI response received — response length: {len(content)} chars")
            logger.info(f"[extract_offer] Chunk {idx + 1}: Raw AI response preview (first 500 chars): {content[:500]!r}")

            result = json.loads(content)
            products = result.get('products', [])
            logger.info(f"[extract_offer] Chunk {idx + 1}: AI returned {len(products)} product(s) in JSON")

            cleaned_products = []
            for p_idx, product in enumerate(products):
                logger.info(f"[extract_offer] Chunk {idx + 1}, Product {p_idx + 1}/{len(products)}: === AI RAW VALUES ===")
                logger.info(f"[extract_offer]   product_name    = {product.get('product_name')!r}")
                logger.info(f"[extract_offer]   brand           = {product.get('brand')!r}")
                logger.info(f"[extract_offer]   packaging       = {product.get('packaging')!r}")
                logger.info(f"[extract_offer]   units_per_case  = {product.get('units_per_case')!r}  <-- watch this")
                logger.info(f"[extract_offer]   unit_volume_ml  = {product.get('unit_volume_ml')!r}")
                logger.info(f"[extract_offer]   quantity_case   = {product.get('quantity_case')!r}")
                logger.info(f"[extract_offer]   price_per_case  = {product.get('price_per_case')!r}")
                logger.info(f"[extract_offer]   price_per_unit  = {product.get('price_per_unit')!r}")
                logger.info(f"[extract_offer]   currency        = {product.get('currency')!r}")
                logger.info(f"[extract_offer]   incoterm        = {product.get('incoterm')!r}")
                logger.info(f"[extract_offer]   location        = {product.get('location')!r}")
                logger.info(f"[extract_offer]   custom_status   = {product.get('custom_status')!r}")
                logger.info(f"[extract_offer]   supplier_name   = {product.get('supplier_name')!r}")
                logger.info(f"[extract_offer]   supplier_email  = {product.get('supplier_email')!r}")
                logger.info(f"[extract_offer]   alcohol_percent = {product.get('alcohol_percent')!r}")
                logger.info(f"[extract_offer]   origin_country  = {product.get('origin_country')!r}")
                logger.info(f"[extract_offer]   gift_box        = {product.get('gift_box')!r}")
                logger.info(f"[extract_offer]   moq_cases       = {product.get('moq_cases')!r}")
                logger.info(f"[extract_offer]   error_flags     = {product.get('error_flags')!r}")

                # Clean product nulls
                for key in product:
                    if product[key] is None:
                        logger.info(f"[extract_offer] Chunk {idx + 1}, Product {p_idx + 1}: field '{key}' is None — replacing with 'Not Found'")
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

                logger.info(f"[extract_offer] Chunk {idx + 1}, Product {p_idx + 1}: passing to clean_product_data()")
                cleaned_product = clean_product_data(product)

                logger.info(f"[extract_offer] Chunk {idx + 1}, Product {p_idx + 1}: === AFTER clean_product_data ===")
                logger.info(f"[extract_offer]   product_name    = {cleaned_product.get('product_name')!r}")
                logger.info(f"[extract_offer]   packaging       = {cleaned_product.get('packaging')!r}")
                logger.info(f"[extract_offer]   units_per_case  = {cleaned_product.get('units_per_case')}  <-- should match packaging N")
                logger.info(f"[extract_offer]   unit_volume_ml  = {cleaned_product.get('unit_volume_ml')}")
                logger.info(f"[extract_offer]   quantity_case   = {cleaned_product.get('quantity_case')}")
                logger.info(f"[extract_offer]   price_per_case  = {cleaned_product.get('price_per_case')}")
                logger.info(f"[extract_offer]   price_per_unit  = {cleaned_product.get('price_per_unit')}")
                logger.info(f"[extract_offer]   incoterm        = {cleaned_product.get('incoterm')!r}")
                logger.info(f"[extract_offer]   custom_status   = {cleaned_product.get('custom_status')!r}")
                logger.info(f"[extract_offer]   supplier_name   = {cleaned_product.get('supplier_name')!r}")
                logger.info(f"[extract_offer]   supplier_email  = {cleaned_product.get('supplier_email')!r}")
                logger.info(f"[extract_offer]   gift_box        = {cleaned_product.get('gift_box')!r}")
                logger.info(f"[extract_offer]   moq_cases       = {cleaned_product.get('moq_cases')}")

                cleaned_products.append(cleaned_product)

            all_products.extend(cleaned_products)
            logger.info(f"[extract_offer] Chunk {idx + 1}: done — yielded {len(cleaned_products)} product(s). Running total: {len(all_products)}")

        except json.JSONDecodeError as e:
            logger.error(f"[extract_offer] Chunk {idx + 1}: JSON decode error: {e}")
            continue
        except Exception as e:
            logger.error(f"[extract_offer] Chunk {idx + 1}: Unexpected error: {e}")
            logger.error(f"[extract_offer] Traceback: {traceback.format_exc()}")
            continue

    logger.info(f"[extract_offer] ===== END — total products aggregated: {len(all_products)} =====")
    return {"products": all_products}


async def extract_from_file(file_path: str, content_type: str) -> Dict[str, Any]:
    logger.info(f"[extract_from_file] ===== START =====")
    logger.info(f"[extract_from_file] file_path: {file_path!r}")
    logger.info(f"[extract_from_file] content_type: {content_type!r}")

    try:
        text_content = ""

        if content_type in [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel',
            'application/vnd.ms-excel.sheet.macroEnabled.12',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.macroEnabled.12'
        ] or file_path.lower().endswith(('.xlsx', '.xls', '.xlsm')):
            logger.info(f"[extract_from_file] Detected file type: EXCEL")
            try:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    file_ext = os.path.splitext(file_path)[1].lower()
                    logger.info(f"[extract_from_file] File exists — size: {file_size} bytes, extension: {file_ext!r}")

                    if file_ext == '.xlsm':
                        logger.info(f"[extract_from_file] Subtype: XLSM (Macro-Enabled Excel)")
                else:
                    logger.error(f"[extract_from_file] File does NOT exist at path: {file_path!r}")
                    return {"error": f"File not found: {file_path}"}

                logger.info(f"[extract_from_file] Reading Excel with pandas...")

                try:
                    if file_path.lower().endswith('.xls'):
                        logger.info(f"[extract_from_file] Using engine: xlrd")
                        df = pd.read_excel(file_path, engine='xlrd')
                    elif file_path.lower().endswith('.xlsm'):
                        logger.info(f"[extract_from_file] Using engine: openpyxl (xlsm)")
                        df = pd.read_excel(file_path, engine='openpyxl')
                    else:
                        logger.info(f"[extract_from_file] Using engine: openpyxl (xlsx)")
                        df = pd.read_excel(file_path, engine='openpyxl')
                except Exception as read_error:
                    logger.warning(f"[extract_from_file] Primary read failed: {read_error}")
                    logger.info(f"[extract_from_file] Trying fallback pd.read_excel() without engine...")
                    df = pd.read_excel(file_path)

                logger.info(f"[extract_from_file] Excel loaded — shape: {df.shape}, columns: {list(df.columns)}")

                if len(df) > 0:
                    logger.debug(f"[extract_from_file] First 3 rows:\n{df.head(3).to_string()}")
                else:
                    logger.warning(f"[extract_from_file] DataFrame is empty after loading")

                if df.empty:
                    logger.warning(f"[extract_from_file] Excel file is empty — returning error")
                    return {"error": "Excel file is empty"}

                total_rows = len(df)
                logger.info(f"[extract_from_file] Total rows to process: {total_rows}")

                logger.debug(f"[extract_from_file] DataFrame dtypes:\n{df.dtypes}")

                for col in df.columns:
                    sample_data = df[col].head(3).tolist()
                    logger.debug(f"[extract_from_file] Column '{col}' sample: {sample_data}")

                batch_size = 6
                all_extracted_products = []
                processed_row_count = 0
                total_batches = (total_rows + batch_size - 1) // batch_size
                logger.info(f"[extract_from_file] Will process {total_rows} rows in {total_batches} batch(es) of up to {batch_size} rows each")

                for batch_start in range(0, total_rows, batch_size):
                    batch_end = min(batch_start + batch_size, total_rows)
                    batch_df = df.iloc[batch_start:batch_end]
                    batch_num = batch_start // batch_size + 1

                    logger.info(f"[extract_from_file] ----- Batch {batch_num}/{total_batches}: rows {batch_start}–{batch_end - 1} ({len(batch_df)} rows) -----")

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

                    logger.info(f"[extract_from_file] Batch {batch_num}: built {len(data_rows)} row dict(s) for AI")
                    logger.info(f"[extract_from_file] Batch {batch_num}: raw rows sent to AI: {json.dumps(data_rows)[:600]!r}")

                    batch_text = f"""
                    You are extracting commercial alcohol product data from Excel rows.
                    Return JSON ONLY, no explanation.

                    EXCEL DATA BATCH ({batch_start + 1}-{batch_end} of {total_rows}):
                    Extract EXACTLY {len(batch_df)} products from this data.
                    If a product has MULTIPLE INCOTERMS, create one row per incoterm (duplicate all other fields).

                    CRITICAL: Apply Rule 13 to correct all brand names to their official commercial
                    spelling before outputting. E.g. "Ballantine" → "Ballantine's",
                    "Jack Daniel" → "Jack Daniel's", "Hennesy" → "Hennessy", etc.

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

                    logger.info(f"[extract_from_file] Batch {batch_num}: prompt length: {len(batch_text)} chars")

                    try:
                        logger.info(f"[extract_from_file] Batch {batch_num}: Calling OpenAI API — model=gpt-4o, max_tokens=16000")
                        response = await client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system",
                                    "content": f"You are a professional data extraction expert. You extract commercial alcohol product data from Excel. Return COMPLETE JSON with 'products' array containing EXACTLY {len(batch_df)} products. NEVER skip rows. Create a product for every row even if data is missing, using logical defaults. ALWAYS correct brand names to their official spelling per Rule 13 (e.g. Ballantine → Ballantine's, Jack Daniel → Jack Daniel's)."
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
                        logger.info(f"[extract_from_file] Batch {batch_num}: OpenAI response received — length: {len(content)} chars")
                        logger.info(f"[extract_from_file] Batch {batch_num}: Raw AI response preview (first 500 chars): {content[:500]!r}")

                        if not content.strip().endswith('}'):
                            logger.warning(f"[extract_from_file] Batch {batch_num}: JSON looks incomplete — attempting bracket repair")
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
                                            logger.info(f"[extract_from_file] Batch {batch_num}: JSON repaired — new length: {len(content)} chars")
                                            break

                        try:
                            result = json.loads(content)
                            logger.info(f"[extract_from_file] Batch {batch_num}: JSON parsed successfully")

                            if isinstance(result, dict) and 'products' in result:
                                batch_products = result['products']
                                logger.info(f"[extract_from_file] Batch {batch_num}: AI returned {len(batch_products)} product(s) (expected {len(batch_df)})")

                                if len(batch_products) != len(batch_df):
                                    logger.warning(f"[extract_from_file] Batch {batch_num}: COUNT MISMATCH — expected {len(batch_df)}, got {len(batch_products)}")
                                    if len(batch_products) < len(batch_df):
                                        missing_count = len(batch_df) - len(batch_products)
                                        logger.warning(f"[extract_from_file] Batch {batch_num}: Padding {missing_count} missing product(s) with defaults")
                                        for i in range(missing_count):
                                            default_product = clean_product_data({})
                                            default_product['product_name'] = f"Row {batch_start + len(batch_products) + i + 1}"
                                            batch_products.append(default_product)
                                            logger.warning(f"[extract_from_file] Batch {batch_num}: Added placeholder for row {batch_start + len(batch_products) + i + 1}")

                                cleaned_batch_products = []
                                for p_idx, product in enumerate(batch_products):
                                    logger.info(f"[extract_from_file] Batch {batch_num}, Product {p_idx + 1}/{len(batch_products)}: === AI RAW VALUES ===")
                                    logger.info(f"[extract_from_file]   product_name    = {product.get('product_name')!r}")
                                    logger.info(f"[extract_from_file]   brand           = {product.get('brand')!r}")
                                    logger.info(f"[extract_from_file]   packaging       = {product.get('packaging')!r}")
                                    logger.info(f"[extract_from_file]   units_per_case  = {product.get('units_per_case')!r}  <-- watch this")
                                    logger.info(f"[extract_from_file]   unit_volume_ml  = {product.get('unit_volume_ml')!r}")
                                    logger.info(f"[extract_from_file]   quantity_case   = {product.get('quantity_case')!r}")
                                    logger.info(f"[extract_from_file]   price_per_case  = {product.get('price_per_case')!r}")
                                    logger.info(f"[extract_from_file]   price_per_unit  = {product.get('price_per_unit')!r}")
                                    logger.info(f"[extract_from_file]   currency        = {product.get('currency')!r}")
                                    logger.info(f"[extract_from_file]   incoterm        = {product.get('incoterm')!r}")
                                    logger.info(f"[extract_from_file]   custom_status   = {product.get('custom_status')!r}")
                                    logger.info(f"[extract_from_file]   supplier_name   = {product.get('supplier_name')!r}")
                                    logger.info(f"[extract_from_file]   supplier_email  = {product.get('supplier_email')!r}")
                                    logger.info(f"[extract_from_file]   alcohol_percent = {product.get('alcohol_percent')!r}")
                                    logger.info(f"[extract_from_file]   gift_box        = {product.get('gift_box')!r}")
                                    logger.info(f"[extract_from_file] Batch {batch_num}, Product {p_idx + 1}: passing to clean_product_data()")
                                    cleaned_product = clean_product_data(product)
                                    logger.info(f"[extract_from_file] Batch {batch_num}, Product {p_idx + 1}: === AFTER clean_product_data ===")
                                    logger.info(f"[extract_from_file]   product_name    = {cleaned_product.get('product_name')!r}")
                                    logger.info(f"[extract_from_file]   packaging       = {cleaned_product.get('packaging')!r}")
                                    logger.info(f"[extract_from_file]   units_per_case  = {cleaned_product.get('units_per_case')}  <-- should match packaging N")
                                    logger.info(f"[extract_from_file]   unit_volume_ml  = {cleaned_product.get('unit_volume_ml')}")
                                    logger.info(f"[extract_from_file]   quantity_case   = {cleaned_product.get('quantity_case')}")
                                    logger.info(f"[extract_from_file]   price_per_case  = {cleaned_product.get('price_per_case')}")
                                    logger.info(f"[extract_from_file]   price_per_unit  = {cleaned_product.get('price_per_unit')}")
                                    logger.info(f"[extract_from_file]   gift_box        = {cleaned_product.get('gift_box')!r}")
                                    cleaned_batch_products.append(cleaned_product)

                                all_extracted_products.extend(cleaned_batch_products)
                                processed_row_count += len(batch_df)
                                logger.info(f"[extract_from_file] Batch {batch_num}: complete — added {len(cleaned_batch_products)} product(s). Running total: {len(all_extracted_products)}")

                                if cleaned_batch_products:
                                    logger.debug(f"[extract_from_file] Batch {batch_num}: first product sample: {json.dumps(cleaned_batch_products[0], indent=2)[:300]}...")

                            elif isinstance(result, list):
                                logger.info(f"[extract_from_file] Batch {batch_num}: AI returned a direct LIST with {len(result)} item(s) (expected {len(batch_df)})")
                                if len(result) != len(batch_df):
                                    logger.warning(f"[extract_from_file] Batch {batch_num}: List count mismatch — expected {len(batch_df)}, got {len(result)}")
                                    if len(result) < len(batch_df):
                                        missing_count = len(batch_df) - len(result)
                                        logger.warning(f"[extract_from_file] Batch {batch_num}: Padding {missing_count} missing product(s)")
                                        for i in range(missing_count):
                                            default_product = clean_product_data({})
                                            default_product['product_name'] = f"Row {batch_start + len(result) + i + 1}"
                                            result.append(default_product)

                                cleaned_batch_products = []
                                for p_idx, product in enumerate(result):
                                    logger.info(f"[extract_from_file] Batch {batch_num} (list), Product {p_idx + 1}: product_name={product.get('product_name')!r}, packaging={product.get('packaging')!r}, units_per_case={product.get('units_per_case')!r}, price_per_case={product.get('price_per_case')!r}")
                                    cleaned_product = clean_product_data(product)
                                    logger.info(f"[extract_from_file] Batch {batch_num} (list), Product {p_idx + 1} CLEANED: units_per_case={cleaned_product.get('units_per_case')}, price_per_case={cleaned_product.get('price_per_case')}")
                                    cleaned_batch_products.append(cleaned_product)
                                all_extracted_products.extend(cleaned_batch_products)
                                processed_row_count += len(batch_df)
                                logger.info(f"[extract_from_file] Batch {batch_num} (list): complete — running total: {len(all_extracted_products)}")
                            else:
                                logger.warning(f"[extract_from_file] Batch {batch_num}: Unexpected JSON format (type={type(result)}) — creating {len(batch_df)} default product(s)")
                                for i in range(len(batch_df)):
                                    default_product = clean_product_data({})
                                    default_product['product_name'] = f"Row {batch_start + i + 1}"
                                    all_extracted_products.append(default_product)
                                processed_row_count += len(batch_df)

                        except json.JSONDecodeError as e:
                            logger.error(f"[extract_from_file] Batch {batch_num}: JSON parse error: {e}")
                            logger.error(f"[extract_from_file] Batch {batch_num}: raw response first 500 chars: {content[:500]!r}")
                            logger.error(f"[extract_from_file] Batch {batch_num}: raw response last 500 chars: {content[-500:]!r}")

                            try:
                                import re
                                json_pattern = r'\{.*\}'
                                matches = re.findall(json_pattern, content, re.DOTALL)
                                logger.info(f"[extract_from_file] Batch {batch_num}: regex salvage found {len(matches)} match(es)")
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
                                                            default_product['product_name'] = f"Row {batch_start + len(batch_products) + i + 1}"
                                                            batch_products.append(default_product)

                                                    cleaned_batch_products = []
                                                    for product in batch_products:
                                                        cleaned_product = clean_product_data(product)
                                                        cleaned_batch_products.append(cleaned_product)
                                                    all_extracted_products.extend(cleaned_batch_products)
                                                    processed_row_count += len(batch_df)
                                                    logger.warning(f"[extract_from_file] Batch {batch_num}: Salvaged {len(cleaned_batch_products)} product(s) via regex")
                                                    break
                                        except:
                                            continue
                            except Exception as salvage_error:
                                logger.error(f"[extract_from_file] Batch {batch_num}: Salvage FAILED: {salvage_error}")
                                for i in range(len(batch_df)):
                                    default_product = clean_product_data({})
                                    default_product['product_name'] = f"Row {batch_start + i + 1}"
                                    all_extracted_products.append(default_product)
                                processed_row_count += len(batch_df)

                    except Exception as e:
                        logger.error(f"[extract_from_file] Batch {batch_num}: OpenAI call FAILED: {e}")
                        logger.error(f"[extract_from_file] Batch {batch_num}: {traceback.format_exc()}")
                        for i in range(len(batch_df)):
                            default_product = clean_product_data({})
                            default_product['product_name'] = f"Row {batch_start + i + 1}"
                            all_extracted_products.append(default_product)
                        processed_row_count += len(batch_df)

                logger.info(f"[extract_from_file] All batches complete — total products: {len(all_extracted_products)}, rows processed: {processed_row_count}/{total_rows}")

                if len(all_extracted_products) != total_rows:
                    logger.warning(f"[extract_from_file] PRODUCT/ROW MISMATCH — Excel rows: {total_rows}, products extracted: {len(all_extracted_products)}")
                    if len(all_extracted_products) < total_rows:
                        missing_count = total_rows - len(all_extracted_products)
                        logger.warning(f"[extract_from_file] Padding {missing_count} missing product(s) with defaults")
                        for i in range(missing_count):
                            default_product = clean_product_data({})
                            default_product['product_name'] = f"Missing Row {len(all_extracted_products) + i + 1}"
                            all_extracted_products.append(default_product)

                if all_extracted_products:
                    logger.debug(f"[extract_from_file] Sample extracted products (first 2): {json.dumps(all_extracted_products[:2], indent=2)}")
                    logger.info(f"[extract_from_file] Total extracted products after all fixes: {len(all_extracted_products)}")
                else:
                    logger.warning(f"[extract_from_file] No products extracted from any batch — trying fallback text extraction")
                    simplified_rows = []
                    for i in range(min(10, len(df))):
                        row = df.iloc[i]
                        row_dict = {}
                        for col in df.columns:
                            value = row[col]
                            row_dict[col] = str(value) if not pd.isna(value) else ""
                        simplified_rows.append(row_dict)

                    text_content = f"Excel with {total_rows} rows. Sample data:\n{json.dumps(simplified_rows, indent=2)}"
                    logger.info(f"[extract_from_file] Fallback text length: {len(text_content)} chars — calling extract_offer()")
                    fallback_result = await extract_offer(text_content)
                    logger.info(f"[extract_from_file] Fallback extraction result type: {type(fallback_result)}")
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
                    logger.info(f"[extract_from_file] ===== END EXCEL — returning {len(all_extracted_products)} products =====")
                    return result
                else:
                    logger.error(f"[extract_from_file] No products could be extracted from the Excel file")
                    return {"error": "No products could be extracted from the Excel file"}

            except Exception as e:
                logger.error(f"[extract_from_file] Excel processing FAILED: {e}")
                logger.error(f"[extract_from_file] Traceback: {traceback.format_exc()}")
                text_content = f"Excel file - error reading: {str(e)}"
                return {"error": f"Excel read error: {str(e)}"}

        elif content_type == 'application/pdf':
            # ── PDF: process in page batches for accuracy ──────────────────
            logger.info(f"[extract_from_file] Detected file type: PDF — will process in page batches")
            PAGES_PER_BATCH = 5  # pages per AI call (tunable)
            try:
                import PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    num_pages = len(pdf_reader.pages)
                    logger.info(f"[extract_from_file] PDF has {num_pages} page(s)")

                    # Collect (page_num, text) pairs
                    pages_text = []
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text() or ""
                        logger.info(f"[extract_from_file] PDF page {page_num + 1}/{num_pages}: extracted {len(page_text)} chars")
                        pages_text.append((page_num + 1, page_text))

                total_pages = len(pages_text)
                total_pdf_batches = (total_pages + PAGES_PER_BATCH - 1) // PAGES_PER_BATCH
                logger.info(f"[extract_from_file] PDF: {total_pages} page(s) → {total_pdf_batches} batch(es) of up to {PAGES_PER_BATCH} pages")

                all_pdf_products = []

                for batch_idx in range(total_pdf_batches):
                    start_page = batch_idx * PAGES_PER_BATCH
                    end_page = min(start_page + PAGES_PER_BATCH, total_pages)
                    batch_pages = pages_text[start_page:end_page]
                    batch_num = batch_idx + 1

                    combined_text = "\n\n".join(
                        f"--- PAGE {pn} ---\n{pt}" for pn, pt in batch_pages
                    )
                    logger.info(f"[extract_from_file] PDF Batch {batch_num}/{total_pdf_batches}: pages {start_page + 1}–{end_page}, text length {len(combined_text)} chars")
                    logger.info(f"[extract_from_file] PDF Batch {batch_num} preview (first 300 chars): {combined_text[:300]!r}")

                    prompt = f"""
You are extracting commercial alcohol offers from a PDF document (pages {start_page + 1} to {end_page} of {total_pages}).
Return JSON ONLY, no explanation.

Extract ALL product lines from the text below.
Return a JSON object with a 'products' array.

CRITICAL: Extract ONLY actual product lines. Do NOT create rows for:
- Section headers (e.g. "Whisky", "Rum", "Gin")
- Brand marketing lists (pages listing brand names the company works with)
- Footer / signature blocks / unsubscribe links
- Repeated items that are the same product

If a product has MULTIPLE INCOTERMS, create one row per incoterm (all other fields identical).

CRITICAL: Apply Rule 13 to correct all brand names to their official commercial spelling
before outputting. E.g. "Ballantine" → "Ballantine's", "Jack Daniel" → "Jack Daniel's".

{SHARED_EXTRACTION_RULES}

PDF TEXT (pages {start_page + 1}–{end_page} of {total_pages}):
{combined_text}
"""

                    try:
                        logger.info(f"[extract_from_file] PDF Batch {batch_num}: calling OpenAI — model=gpt-4o, max_tokens=16000")
                        response = await client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a professional data extraction expert. "
                                        "Extract commercial alcohol product offers from PDF text. "
                                        "Return ONLY valid JSON with a 'products' array. "
                                        "Do NOT include section headers, brand lists, or footer text as products. "
                                        "Only extract actual product offer lines. "
                                        "ALWAYS correct brand names to their official spelling per Rule 13 "
                                        "(e.g. Ballantine → Ballantine's, Jack Daniel → Jack Daniel's)."
                                    )
                                },
                                {"role": "user", "content": prompt}
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.0,
                            max_tokens=16000,
                            top_p=1.0,
                            frequency_penalty=0.0,
                            presence_penalty=0.0
                        )

                        content = response.choices[0].message.content
                        logger.info(f"[extract_from_file] PDF Batch {batch_num}: response received — {len(content)} chars")
                        logger.info(f"[extract_from_file] PDF Batch {batch_num}: preview: {content[:400]!r}")

                        result = json.loads(content)
                        batch_products = result.get('products', [])
                        logger.info(f"[extract_from_file] PDF Batch {batch_num}: AI returned {len(batch_products)} product(s)")

                        cleaned_batch = []
                        for p_idx, product in enumerate(batch_products):
                            logger.info(f"[extract_from_file] PDF Batch {batch_num}, Product {p_idx + 1}: product_name={product.get('product_name')!r}, price_per_unit={product.get('price_per_unit')!r}, price_per_case={product.get('price_per_case')!r}, units_per_case={product.get('units_per_case')!r}, gift_box={product.get('gift_box')!r}")

                            # Null cleanup
                            for key in list(product.keys()):
                                if product[key] is None:
                                    product[key] = "Not Found"

                            cleaned_product = clean_product_data(product)
                            logger.info(f"[extract_from_file] PDF Batch {batch_num}, Product {p_idx + 1} CLEANED: price_per_unit={cleaned_product.get('price_per_unit')}, price_per_case={cleaned_product.get('price_per_case')}, gift_box={cleaned_product.get('gift_box')!r}")
                            cleaned_batch.append(cleaned_product)

                        all_pdf_products.extend(cleaned_batch)
                        logger.info(f"[extract_from_file] PDF Batch {batch_num}: done — added {len(cleaned_batch)}. Running total: {len(all_pdf_products)}")

                    except json.JSONDecodeError as e:
                        logger.error(f"[extract_from_file] PDF Batch {batch_num}: JSON decode error: {e}")
                        # Fallback: send the batch text through extract_offer
                        logger.info(f"[extract_from_file] PDF Batch {batch_num}: falling back to extract_offer()")
                        fallback = await extract_offer(combined_text)
                        all_pdf_products.extend(fallback.get('products', []))
                    except Exception as e:
                        logger.error(f"[extract_from_file] PDF Batch {batch_num}: error: {e}")
                        logger.error(f"[extract_from_file] Traceback: {traceback.format_exc()}")

                logger.info(f"[extract_from_file] PDF processing complete — total products: {len(all_pdf_products)}")

                if all_pdf_products:
                    return {
                        'products': all_pdf_products,
                        'total_products': len(all_pdf_products),
                        'file_type': 'pdf',
                        'processed_in_batches': True,
                        'batches_processed': total_pdf_batches,
                        'original_pages': total_pages
                    }
                else:
                    logger.warning(f"[extract_from_file] No products extracted from PDF — returning empty result")
                    return {"products": [], "error": "No products extracted from PDF"}

            except ImportError:
                logger.error(f"[extract_from_file] PyPDF2 is NOT installed — cannot process PDF")
                return {"error": "PyPDF2 not installed"}
            except Exception as e:
                logger.error(f"[extract_from_file] PDF read FAILED: {e}")
                logger.error(f"[extract_from_file] Traceback: {traceback.format_exc()}")
                return {"error": f"PDF read error: {str(e)}"}

        elif 'image' in content_type:
            logger.info(f"[extract_from_file] Detected file type: IMAGE (content_type={content_type!r})")
            try:
                with open(file_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                logger.info(f"[extract_from_file] Image base64 encoded — length: {len(base64_image)} chars")
                logger.info(f"[extract_from_file] Calling OpenAI vision API for image extraction...")

                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text",
                                 "text": "Extract all commercial alcohol offers from this image. Return a JSON object with a 'products' array following the standard schema. ALWAYS correct brand names to their official spelling (e.g. Ballantine → Ballantine's, Jack Daniel → Jack Daniel's)."},
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
                image_response_text = response.choices[0].message.content
                logger.info(f"[extract_from_file] Image OpenAI response length: {len(image_response_text)} chars")
                logger.info(f"[extract_from_file] Image response preview: {image_response_text[:400]!r}")
                logger.info(f"[extract_from_file] Passing image response to extract_offer() for structured extraction")
                return await extract_offer(image_response_text)

            except Exception as e:
                logger.error(f"[extract_from_file] Image processing FAILED: {e}")
                logger.error(f"[extract_from_file] Traceback: {traceback.format_exc()}")
                return {"error": f"Image processing error: {str(e)}"}

        else:
            logger.info(f"[extract_from_file] Detected file type: TEXT (content_type={content_type!r})")
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_content = file.read()
                logger.info(f"[extract_from_file] Text file read with utf-8 — length: {len(text_content)} chars")
            except:
                try:
                    with open(file_path, 'r', encoding='latin-1') as file:
                        text_content = file.read()
                    logger.info(f"[extract_from_file] Text file read with latin-1 fallback — length: {len(text_content)} chars")
                except Exception as e:
                    logger.error(f"[extract_from_file] Text file read FAILED: {e}")
                    logger.error(f"[extract_from_file] Traceback: {traceback.format_exc()}")
                    return {"error": f"Text file read error: {str(e)}"}

        if text_content:
            logger.info(f"[extract_from_file] Passing text to extract_offer() — length: {len(text_content)} chars")
            logger.info(f"[extract_from_file] Text preview (first 300 chars): {text_content[:300]!r}")
            result = await extract_offer(text_content)
            logger.info(f"[extract_from_file] extract_offer() returned — result type: {type(result)}")
            return result
        else:
            logger.warning(f"[extract_from_file] No text content was extracted from file — aborting")
            return {"error": "No content extracted from file"}

    except Exception as e:
        logger.error(f"[extract_from_file] CRITICAL error: {e}")
        logger.error(f"[extract_from_file] Traceback: {traceback.format_exc()}")
        return {"error": f"General extraction error: {str(e)}"}


def clean_product_data(product: dict) -> dict:
    """Clean up product data to ensure it matches schema exactly"""
    logger.info(f"[clean_product_data] ===== START =====")
    logger.info(f"[clean_product_data] Input product keys: {list(product.keys())}")
    logger.info(f"[clean_product_data] Input: product_name={product.get('product_name')!r}, packaging={product.get('packaging')!r}, units_per_case={product.get('units_per_case')!r}, price_per_case={product.get('price_per_case')!r}, price_per_unit={product.get('price_per_unit')!r}")

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
        'supplier_email': "Not Found",
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
                logger.info(f"[clean_product_data] Field '{field}': value={value!r} is empty/zero — using default: {default_value!r}")
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
                        logger.info(f"[clean_product_data] Field '{field}': {value!r} → {cleaned_product[field]} (converted to float)")
                    except (ValueError, TypeError) as conv_err:
                        logger.warning(f"[clean_product_data] Field '{field}': could NOT convert {value!r} to float ({conv_err}) — setting to None")
                        cleaned_product[field] = None  # Use None for failed conversions
                elif isinstance(default_value, list):
                    cleaned_product[field] = value if isinstance(value, list) else []
                elif isinstance(default_value, bool):
                    cleaned_product[field] = bool(value)
                else:
                    cleaned_product[field] = str(value) if value is not None else default_value
        else:
            logger.info(f"[clean_product_data] Field '{field}': NOT in product — using default: {default_value!r}")
            cleaned_product[field] = default_value

    if cleaned_product['product_key'] in ["Not Found"] and cleaned_product['product_name'] not in ["Not Found"]:
        product_key = str(cleaned_product['product_name']).replace(' ', '_').replace('/', '_').replace('&', '_').replace('.', '').upper()
        logger.info(f"[clean_product_data] product_key was 'Not Found' — auto-generated: {product_key!r}")
        cleaned_product['product_key'] = product_key

    # Convert any 0 values to None for numeric fields that shouldn't have 0 as a valid value
    numeric_fields_never_zero = [
        'cases_per_pallet', 'quantity_case', 'moq_cases',
        'alcohol_percent', 'unit_volume_ml', 'units_per_case'
    ]

    for field in numeric_fields_never_zero:
        if field in cleaned_product and cleaned_product[field] == 0:
            logger.warning(f"[clean_product_data] Field '{field}' is 0 — treating as Not Found (None)")
            cleaned_product[field] = None


    import re as _re
    packaging_str = cleaned_product.get('packaging') or ''
    logger.info(f"[clean_product_data] Packaging correction check: packaging='{packaging_str}', current units_per_case={cleaned_product.get('units_per_case')}")
    pkg_match = _re.search(r'(\d+)\s*[xX]\s*\d+', packaging_str)
    if pkg_match:
        correct_units = float(pkg_match.group(1))
        current_units = cleaned_product.get('units_per_case')
        if current_units != correct_units:
            logger.warning(f"[clean_product_data] PACKAGING CORRECTION TRIGGERED!")
            logger.warning(f"[clean_product_data]   packaging='{packaging_str}'")
            logger.warning(f"[clean_product_data]   AI gave units_per_case={current_units} (WRONG — likely the price)")
            logger.warning(f"[clean_product_data]   Correct units_per_case from packaging N = {correct_units}")
            logger.warning(f"[clean_product_data]   Overriding units_per_case: {current_units} → {correct_units}")
            cleaned_product['units_per_case'] = correct_units
        else:
            logger.info(f"[clean_product_data] Packaging check OK: units_per_case={current_units} already matches packaging N={correct_units}")
    else:
        if packaging_str and packaging_str not in ("Not Found", "Bottle", "bottle"):
            logger.info(f"[clean_product_data] No NxVOL pattern found in packaging='{packaging_str}' — no correction applied")
    # ─────────────────────────────────────────────────────────────────────────

    units = cleaned_product.get('units_per_case')
    ppu = cleaned_product.get('price_per_unit')
    ppc = cleaned_product.get('price_per_case')

    if units and isinstance(units, (int, float)) and units > 0:

        if ppu and isinstance(ppu, (int, float)) and ppu > 0:
            if not ppc or not isinstance(ppc, (int, float)) or ppc <= 0:
                calculated_ppc = round(ppu * units, 2)
                logger.info(
                    f"[clean_product_data] PRICE CALC (unit→case): "
                    f"{ppu} × {units} = {calculated_ppc}"
                )
                cleaned_product['price_per_case'] = calculated_ppc
                # Add flag if not already present
                flags = cleaned_product.get('error_flags') or []
                flag_text = "price_per_case calculated from price_per_unit x units_per_case"
                if flag_text not in flags:
                    flags.append(flag_text)
                    cleaned_product['error_flags'] = flags

        if ppc and isinstance(ppc, (int, float)) and ppc > 0:
            if not ppu or not isinstance(ppu, (int, float)) or ppu <= 0:
                calculated_ppu = round(ppc / units, 2)
                logger.info(
                    f"[clean_product_data] PRICE CALC (case→unit): "
                    f"{ppc} / {units} = {calculated_ppu}"
                )
                cleaned_product['price_per_unit'] = calculated_ppu
                flags = cleaned_product.get('error_flags') or []
                flag_text = "price_per_unit calculated from price_per_case / units_per_case"
                if flag_text not in flags:
                    flags.append(flag_text)
                    cleaned_product['error_flags'] = flags

    ppu = cleaned_product.get('price_per_unit')
    ppc = cleaned_product.get('price_per_case')
    logger.info(f"[clean_product_data] After price calc: price_per_unit={ppu}, price_per_case={ppc}")
    # ─────────────────────────────────────────────────────────────────────────

    moq = cleaned_product.get('moq_cases')
    moq_min = cleaned_product.get('min_order_quantity_case')
    if moq and isinstance(moq, (int, float)) and moq > 0:
        if not moq_min or not isinstance(moq_min, (int, float)) or moq_min <= 0:
            cleaned_product['min_order_quantity_case'] = moq
            logger.info(f"[clean_product_data] Synced min_order_quantity_case from moq_cases: {moq}")
    elif moq_min and isinstance(moq_min, (int, float)) and moq_min > 0:
        if not moq or not isinstance(moq, (int, float)) or moq <= 0:
            cleaned_product['moq_cases'] = moq_min
            logger.info(f"[clean_product_data] Synced moq_cases from min_order_quantity_case: {moq_min}")
    # ─────────────────────────────────────────────────────────────────────────

    logger.info(f"[clean_product_data] ===== END =====")
    logger.info(f"[clean_product_data] Final: product_name={cleaned_product.get('product_name')!r}, packaging={cleaned_product.get('packaging')!r}, units_per_case={cleaned_product.get('units_per_case')}, unit_volume_ml={cleaned_product.get('unit_volume_ml')}, price_per_case={cleaned_product.get('price_per_case')}, price_per_unit={cleaned_product.get('price_per_unit')}, quantity_case={cleaned_product.get('quantity_case')}, incoterm={cleaned_product.get('incoterm')!r}, custom_status={cleaned_product.get('custom_status')!r}, supplier_name={cleaned_product.get('supplier_name')!r}, supplier_email={cleaned_product.get('supplier_email')!r}, gift_box={cleaned_product.get('gift_box')!r}, moq_cases={cleaned_product.get('moq_cases')}")

    return cleaned_product


def parse_buffer_data(buffer_data: dict) -> bytes:
    logger.info(f"[parse_buffer_data] ===== START — input type: {type(buffer_data)} =====")

    if isinstance(buffer_data, dict) and buffer_data.get('type') == 'Buffer':
        try:
            data_bytes = bytes(buffer_data['data'])
            logger.info(f"[parse_buffer_data] Parsed Buffer type — {len(data_bytes)} bytes")
            return data_bytes
        except Exception as e:
            logger.error(f"[parse_buffer_data] Failed to parse Buffer type: {e}")
            return b''
    elif isinstance(buffer_data, dict) and 'data' in buffer_data:
        try:
            if isinstance(buffer_data['data'], str):
                data_bytes = base64.b64decode(buffer_data['data'])
                logger.info(f"[parse_buffer_data] Parsed base64 string — {len(data_bytes)} bytes")
                return data_bytes
            elif isinstance(buffer_data['data'], list):
                data_bytes = bytes(buffer_data['data'])
                logger.info(f"[parse_buffer_data] Parsed list data — {len(data_bytes)} bytes")
                return data_bytes
            else:
                logger.warning(f"[parse_buffer_data] Unexpected data type in buffer_data['data']: {type(buffer_data['data'])}")
                return b''
        except Exception as e:
            logger.error(f"[parse_buffer_data] Failed to parse buffer_data with 'data' key: {e}")
            return b''
    elif isinstance(buffer_data, str):
        try:
            data_bytes = base64.b64decode(buffer_data)
            logger.info(f"[parse_buffer_data] Parsed base64 string directly — {len(data_bytes)} bytes")
            return data_bytes
        except Exception as e:
            logger.error(f"[parse_buffer_data] Failed to parse base64 string: {e}")
            return b''
    else:
        logger.warning(f"[parse_buffer_data] Unexpected buffer_data type: {type(buffer_data)}")
        logger.debug(f"[parse_buffer_data] buffer_data value: {buffer_data}")
        return b''