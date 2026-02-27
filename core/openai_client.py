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


# ─────────────────────────────────────────────────────────────────────────────
# MASTER SYSTEM PROMPT — embedded once so both extract_offer() and the Excel
# batch path share identical rules.
# ─────────────────────────────────────────────────────────────────────────────
MASTER_SYSTEM_PROMPT = """You are a professional commercial alcohol offer extraction engine.

═══════════════════════════════════════════════════════════
🔥 MASTER EXTRACTION POLICY — ALL RULES ARE MANDATORY
═══════════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1 — EMPTY VALUES (HIGHEST PRIORITY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If a value is NOT explicitly present in the source, the field MUST be left BLANK.
NEVER insert: 0, "Not Found", "NRF", "Unknown", null, or any placeholder.
Blank means blank — no exceptions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 2 — refillable_status (RF / NRF)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Only populate if "RF" or "NRF" is EXPLICITLY written in the offer.
• If absent → leave blank.
• NEVER default to "NRF".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 3 — category & sub_category
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use these mappings for Spirits sub-categories:
  Martell / Hennessy / Rémy Martin / Courvoisier → Cognac
  Bacardi / Captain Morgan / Havana Club          → Rum
  Absolut / Grey Goose / Smirnoff / Belvedere    → Vodka
  Jack Daniel's / Jim Beam / Maker's Mark        → Whiskey (American)
  Jameson / Bushmills / Tullamore                → Irish Whiskey
  Johnnie Walker / Chivas / Glenfiddich / Macallan / Famous Grouse → Whisky (Scotch)
  Gordon's / Tanqueray / Bombay / Hendrick's     → Gin
  Baileys / Kahlúa / Amaretto / Malibu           → Liqueur
If sub-category CANNOT be confidently determined → leave blank.
Do NOT use "Not Found".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 4 — alcohol_percent EXTRACTION & FORMAT  ⚠️ CRITICAL ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
alcohol_percent MUST be extracted from ANY column or field that contains
alcohol / ABV / volume strength data. Scan EVERY column for this value.

Common column names that contain alcohol percentage — check ALL of them:
  "ABV", "Abv", "abv"
  "Alc%", "Alc %", "ALC", "Alcohol", "Alcohol %", "Alcohol%"
  "Vol%", "Vol %", "VOL", "Volume", "Volume %"
  "Strength", "Strength%", "STRENGTH"
  "Proof" (divide by 2 to get %)
  "Degree", "Degrees", "°"
  Any column whose values look like: 40, 43.2, 37.5, 40.0, 17, 38

Extraction examples from Excel columns:
  Column "ABV" = 43      → alcohol_percent: "43%"
  Column "Alc%" = 40.0   → alcohol_percent: "40%"
  Column "Vol%" = 37.5   → alcohol_percent: "37.5%"
  Column "Strength" = 38 → alcohol_percent: "38%"

Extraction examples from free text / product codes:
  "40%"              → alcohol_percent: "40%"
  "43.2%"            → alcohol_percent: "43.2%"
  "37.5% vol"        → alcohol_percent: "37.5%"
  "40 ABV"           → alcohol_percent: "40%"
  "12/100/17/DF/T2"  → alcohol_percent: "17%"  (third segment)
  "6x700ml 43%"      → alcohol_percent: "43%"
  "Baileys 17% 12x1L"→ alcohol_percent: "17%"

Format rules (MANDATORY):
• Always output as a string with "%" sign: "40%", "43.2%", "37.5%"
• NEVER output as a decimal (0.375) or without % sign (40).
• NEVER multiply or divide the raw value.
• If "0.4" is in the source → output "0.4%" (do NOT convert to 40%).
• If truly absent from ALL columns and text → leave blank. NEVER default to 0.
• This field is HIGH PRIORITY — always check before leaving blank.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 5 — INCOTERM SPLITTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When multiple incoterms appear (e.g. "EXW Riga / DAP Loendersloot"):
• Create SEPARATE rows — one per incoterm.
• Duplicate ALL other product fields exactly.
• Only incoterm and location differ.
Standardise informal incoterms:
  "EX Warehouse Dublin" → incoterm: "EXW", location: "Dublin"
  "DAP LOE"            → incoterm: "DAP", location: "Loendersloot bonded warehouse in Netherlands"
  "EXW LOE"            → incoterm: "EXW", location: "Loendersloot bonded warehouse in Netherlands"
Add "multiple_incoterms_detected" to error_flags when splitting.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 6 — supplier_name EXTRACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract from (in priority order):
  1. Official company name in file header / footer
  2. Email signature company name
  3. "Offer from <Company>" in the body
If none found → leave blank.
NEVER use: person names, sales desk names, email usernames.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 6b — supplier_reference EXTRACTION  ⚠️ OVERRIDE RULE ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
supplier_reference must be actively searched across the ENTIRE file/text.
If ANY supplier reference / offer reference is found, it MUST override the
supplier_reference field — even if a value was already set.

Scan every part of the source for these patterns (column names, labels, inline text):
  Column names: "Ref", "Reference", "Ref No", "Ref #", "Supplier Ref",
                "Offer Ref", "Offer Reference", "Offer No", "Offer Number",
                "Order Ref", "PO Ref", "SKU", "Item Code", "Product Code",
                "Supplier Code", "Supplier SKU", "Article", "Art No",
                "Art#", "Code", "Item No", "Stock Code"
  Inline patterns:
    "Ref: ABC123"            → supplier_reference: "ABC123"
    "Offer No: OFF-2024-001" → supplier_reference: "OFF-2024-001"
    "PO Ref: XYZ789"        → supplier_reference: "XYZ789"
    "Our ref: 45892"         → supplier_reference: "45892"
    "Your ref: SUP-007"      → supplier_reference: "SUP-007"
    "Reference: OM-NOV25"    → supplier_reference: "OM-NOV25"

Priority order when multiple candidates exist:
  1. Explicit "Supplier Ref" / "Offer Ref" column or label → highest priority
  2. "Ref No" / "Reference" / "Offer No" column or label
  3. "SKU" / "Item Code" / "Product Code" / "Stock Code" column
  4. Any other reference-like alphanumeric code associated with the product

If found → always write to supplier_reference (override any previous value).
If truly absent from ALL sources → leave blank.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 7 — custom_status (T1 / T2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• If "T1" or "T2" appears anywhere in the offer → extract it.
• If absent → leave blank.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 8 — PRICE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• If price_per_case is not in the source → leave blank. Do NOT calculate.
• "EURO" → convert to "EUR".
• If currency is already EUR: price_per_unit_eur = price_per_unit; price_per_case_eur = price_per_case.
Price notation parsing:
  "15.95eur"       → price_per_case: 15.95, currency: "EUR"
  "11,40eur/btl"   → price_per_unit: 11.40, currency: "EUR"
  "32,50eur/cs"    → price_per_case: 32.50, currency: "EUR"
  "$15.95"         → price_per_case: 15.95, currency: "USD"
  "£11.40/btl"     → price_per_unit: 11.40, currency: "GBP"
  "98,5€"          → price_per_case: 98.5,  currency: "EUR"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 9 — unit_volume_ml NORMALISATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Always convert to millilitres (integer):
  0.5L  → 500    |  70cl  → 700   |  75cl  → 750
  1L    → 1000   |  0.375L → 375  |  100cl → 1000
NEVER use decimals like 0.38. Always use exact values.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 10 — MULTI-PRODUCT / MULTI-PRICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Create a SEPARATE row for each distinct product or price point. NEVER merge.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 11 — lead_time
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Store exactly as written ("approximately 3 weeks", "5 weeks LT"). Do NOT rewrite.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 12 — moq_cases / min_order_quantity_case
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If MOQ is not explicitly stated → leave blank. NEVER default to 0.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 13 — confidence_score
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Start at 1.0. Reduce by 0.1 for each of:
  • sub_category inferred (not written)
  • incoterm converted / standardised
  • unit_volume_ml converted
  • supplier_name inferred from signature
  • any ambiguous field

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 14 — error_flags
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Add a descriptive string to error_flags[] when:
  • Multiple incoterms detected → "multiple_incoterms_detected"
  • Currency missing           → "missing_currency"
  • Volume ambiguous           → "ambiguous_volume"
  • Supplier unclear           → "supplier_unclear"
  • Sub-category inferred      → "sub_category_inferred"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 15 — SOURCE FIELDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEVER modify: source_channel, source_filename, source_message_id.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 16 — STRICT EXTRACTION PRINCIPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract ONLY what is explicitly written OR confidently inferred by the
classification logic in Rule 3. NEVER assume or fabricate missing data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 17 — FINAL DECISION TABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Situation               → Action
  ─────────────────────────────────────────────────────
  Not present             → Leave blank
  Present clearly         → Extract as-is
  Multiple values/incos   → Split into separate rows
  Needs standardisation   → Normalise (volume, incoterm)
  Ambiguous               → Leave blank + add error_flag

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUANTITY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
quantity_case:
  • Only populate if total case count is EXPLICITLY stated.
  • "12x750ml" describes PACKAGING, NOT quantity_case.
  • "256cs x 3" → quantity_case: 768 (only when "x N" multiplier is explicit).
  • If absent → leave blank.

cases_per_pallet:
  • Only populate if pallet quantity is EXPLICITLY stated
    (e.g. "60 cases per pallet", "60 cs/pallet").
  • "FTL" / "Full Truck Load" → do NOT populate cases_per_pallet.
  • If absent → leave blank.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATE FIELDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  "9/2026"        → best_before_date: "2026-09-01"
  "BBD 03.06.2026"→ best_before_date: "2026-06-03"
  "fresh"         → best_before_date: "fresh"
These are NOT lead_time.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LABEL LANGUAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Only extract when explicitly mentioned:
  "UK text"   → "EN"
  "SA label"  → "multiple"
  "multi text"→ "multiple"
If absent → leave blank.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PACKAGING PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  "6x70"           → units_per_case: 6, unit_volume_ml: 700
  "12x750ml"       → units_per_case: 12, unit_volume_ml: 750
  "24x50cl cans"   → units_per_case: 24, unit_volume_ml: 500, bottle_or_can_type: "can"
  "12/100/17/DF/T2"→ units_per_case: 12, unit_volume_ml: 1000, alcohol_percent: "17%", custom_status: "T2"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORY DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Whisky/Whiskey/Scotch/Bourbon  → Spirits / Whisky
  Champagne/Sparkling            → Wine   / Champagne
  Red/White/Rosé wine            → Wine   / (Red Wine / White Wine / Rosé)
  Beer/Lager/Ale/Stout           → Beer   / (Lager / Ale / Stout)
  Cognac/Brandy                  → Spirits/ Cognac
  Vodka/Gin/Rum/Tequila          → Spirits/ (Vodka / Gin / Rum / Tequila)
  Liqueur                        → Spirits/ Liqueur
  Soft Drinks/Energy Drinks      → Soft Drinks
  Food                           → Food

══════════════════════════════════════════════════════════
SCHEMA — output a JSON object {"products": [...]} ONLY
══════════════════════════════════════════════════════════
⚠️  alcohol_percent is a REQUIRED field. Always scan all columns/text for it.
    Output format: "43%" / "40%" / "37.5%" — always with % sign.

Fields (all missing values MUST be left blank — not null, not 0, not "Not Found"):
uid, product_key, processing_version, brand, product_name, product_reference,
category, sub_category, origin_country, vintage, alcohol_percent, packaging,
unit_volume_ml, units_per_case, cases_per_pallet, quantity_case,
bottle_or_can_type, price_per_unit, price_per_case, currency,
price_per_unit_eur, price_per_case_eur, incoterm, location,
min_order_quantity_case, port, lead_time, supplier_name, supplier_reference,
supplier_country, offer_date, valid_until, date_received, source_channel,
source_filename, source_message_id, confidence_score, error_flags[],
needs_manual_review, best_before_date, label_language, ean_code,
gift_box, refillable_status, custom_status, moq_cases

product_key  → UPPERCASE, underscores: BRAND_NAME_VOLUME_PACKAGING
uid          → leave blank (populated by backend)
processing_version → leave blank
confidence_score   → float 0.0–1.0 (start 1.0, reduce per Rule 13)
error_flags        → array of strings
needs_manual_review→ boolean
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helper: user-turn extraction prompt for free-text / email content
# ─────────────────────────────────────────────────────────────────────────────
def _build_text_user_prompt(chunk: str, idx: int, total: int) -> str:
    return f"""Extract ALL commercial alcohol products from the text below.
Return ONLY a JSON object with a 'products' array. No explanations.

Follow the Master Extraction Policy in the system prompt without exception.
Key reminders:
• Leave fields BLANK (not "Not Found", not 0) when data is absent.
• refillable_status: only "RF" or "NRF" if explicitly written — otherwise blank.
• Split rows for multiple incoterms.
• alcohol_percent: MANDATORY — scan the entire text for ABV, Alc%, Vol%, strength,
  or any numeric value that represents alcohol content. Format as "43%" not 43.
  Check product codes too (e.g. "12/100/17%" → alcohol_percent: "17%").
• supplier_reference: MANDATORY — scan for any Ref, Reference, Offer Ref, Offer No,
  SKU, Item Code, Product Code, Stock Code, or similar. If found, override the field.
• quantity_case ≠ packaging config (12x750ml is packaging, not quantity).
• Supplier name = company name only (never a person name).

Text Chunk ({idx + 1}/{total}):
{chunk}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper: user-turn extraction prompt for Excel batch rows
# ─────────────────────────────────────────────────────────────────────────────
def _build_excel_user_prompt(data_rows: list, batch_start: int, batch_end: int, total_rows: int) -> str:
    return f"""Extract products from these Excel rows ({batch_start + 1}–{batch_end} of {total_rows}).
Return ONLY a JSON object with a 'products' array. No explanations.

Follow the Master Extraction Policy in the system prompt without exception.
Key reminders:
• Leave fields BLANK (not "Not Found", not 0) when data is absent.
• refillable_status: only "RF" or "NRF" if explicitly written — otherwise blank.
• Split rows for multiple incoterms (creates extra rows in output).
• alcohol_percent: MANDATORY — scan EVERY column for alcohol/ABV data.
  Column names to check: "ABV", "Alc%", "Alc", "Alcohol", "Alcohol %", "Vol%",
  "Vol", "Volume", "Strength", "Degree", or any column with values like 40, 43.2, 37.5.
  Output format MUST include % sign: "43%" not 43, "40.0%" not 40.0.
  If a column exists with a numeric value between 1 and 99 that looks like an
  alcohol percentage, extract it as alcohol_percent.
• supplier_reference: MANDATORY — scan EVERY column for Ref, Reference, Offer Ref,
  Offer No, SKU, Item Code, Product Code, Stock Code, Art No, Supplier Code, or
  any alphanumeric code associated with the product. If found, override the field.
• quantity_case ≠ packaging config (12x750ml is packaging, not quantity).
• Supplier name = company name only (never a person name).
• Skip header/subtotal rows; extract only product offer rows.

Excel rows (JSON):
{json.dumps(data_rows, indent=2)}
"""


async def extract_offer(text: str) -> dict:
    logger.info(f"extract_offer called with text length: {len(text)}")
    logger.debug(f"extract_offer text preview: {text[:200]}...")

    # Maximum characters per chunk to prevent context overflow (roughly 5000 tokens)
    CHUNK_SIZE = 25000

    text_chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)] if len(text) > CHUNK_SIZE else [text]
    logger.info(f"Split input text into {len(text_chunks)} chunk(s).")

    all_products = []

    for idx, chunk in enumerate(text_chunks):
        logger.info(f"Processing chunk {idx + 1} of {len(text_chunks)}...")

        try:
            logger.info("Calling OpenAI API for text extraction chunk...")
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": MASTER_SYSTEM_PROMPT},
                    {"role": "user", "content": _build_text_user_prompt(chunk, idx, len(text_chunks))}
                ],
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
                            product[key] = None
                        else:
                            product[key] = ""

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
        currency = product.get('currency', "")

        if currency in ["", None, "EUR"]:
            product['price_per_unit_eur'] = product.get('price_per_unit')
            product['price_per_case_eur'] = product.get('price_per_case')
            continue

        exchange_rate = await get_exchange_rate_to_eur(currency)

        if product.get('price_per_unit') not in [None, "", 0, "0"]:
            product['price_per_unit_eur'] = convert_price_to_eur(
                product['price_per_unit'],
                currency,
                exchange_rate
            )

        if product.get('price_per_case') not in [None, "", 0, "0"]:
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

                    try:
                        logger.info(f"Calling OpenAI API for batch {batch_start // batch_size + 1}...")
                        response = await client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system",
                                    "content": MASTER_SYSTEM_PROMPT + f"\n\nIMPORTANT: You must return EXACTLY {len(batch_df)} products (one per Excel row). If a row lacks data, still create an entry with whatever can be extracted. Never skip rows."
                                },
                                {
                                    "role": "user",
                                    "content": _build_excel_user_prompt(data_rows, batch_start, batch_end, total_rows)
                                }
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

                        if currency in ["", None, "EUR"]:
                            product['price_per_unit_eur'] = product.get('price_per_unit')
                            product['price_per_case_eur'] = product.get('price_per_case')
                            continue

                        exchange_rate = await get_exchange_rate_to_eur(currency)

                        if product.get('price_per_unit') not in [None, "", 0, "0"]:
                            product['price_per_unit_eur'] = convert_price_to_eur(
                                product['price_per_unit'],
                                currency,
                                exchange_rate
                            )

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
                            "role": "system",
                            "content": MASTER_SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text",
                                 "text": "Extract all commercial alcohol offers from this image. Return a JSON object with a 'products' array following the schema in the system prompt. Apply all Master Extraction Policy rules without exception. Leave all absent fields BLANK — never use 'Not Found', 0, or placeholders."},
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
        'uid': "",
        'product_key': "",
        'processing_version': "",
        'brand': "",
        'product_name': "",
        'product_reference': "",
        'category': "",
        'sub_category': "",
        'origin_country': "",
        'vintage': "",
        'alcohol_percent': "",  # String with % sign
        'packaging': "",
        'unit_volume_ml': None,
        'units_per_case': None,
        'cases_per_pallet': None,
        'quantity_case': None,
        'bottle_or_can_type': "",
        'price_per_unit': None,
        'price_per_case': None,
        'currency': "",          # Blank by default — not "EUR" — so missing currency is detectable
        'price_per_unit_eur': None,
        'price_per_case_eur': None,
        'incoterm': "",
        'location': "",
        'min_order_quantity_case': None,
        'port': "",
        'lead_time': "",
        'supplier_name': "",
        'supplier_reference': "",
        'supplier_country': "",
        'offer_date': "",
        'valid_until': "",
        'date_received': "",
        'source_channel': "",
        'source_filename': "",
        'source_message_id': "",
        'confidence_score': 0.0,
        'error_flags': [],
        'needs_manual_review': False,
        'best_before_date': "",
        'label_language': "",
        'ean_code': "",
        'gift_box': "",
        'refillable_status': "",
        'custom_status': "",
        'moq_cases': None
    }

    # Values that should always be treated as "missing / blank"
    BLANK_SENTINELS = {None, "Not Found", "not found", "NOT FOUND", "null", "NULL", "N/A", "n/a", ""}

    cleaned_product = {}

    numeric_keys = [
        'unit_volume_ml', 'units_per_case', 'cases_per_pallet',
        'quantity_case', 'price_per_unit', 'price_per_unit_eur',
        'price_per_case', 'price_per_case_eur', 'moq_cases',
        'min_order_quantity_case'
    ]

    for field, default_value in schema_fields.items():
        if field in product:
            value = product[field]

            # ── Numeric fields ──────────────────────────────────────────────
            if field in numeric_keys:
                if value in BLANK_SENTINELS or value == 0 or value == "0":
                    cleaned_product[field] = None
                else:
                    try:
                        float_val = float(value)
                        cleaned_product[field] = None if float_val == 0 else float_val
                    except (ValueError, TypeError):
                        cleaned_product[field] = None

            # ── alcohol_percent — always "XX%" string ───────────────────────
            elif field == 'alcohol_percent':
                if value in BLANK_SENTINELS or value == 0 or value == "0":
                    cleaned_product[field] = ""
                elif isinstance(value, str) and '%' in value:
                    cleaned_product[field] = value.strip()
                else:
                    try:
                        float_val = float(str(value).replace(',', '.'))
                        if float_val == 0:
                            cleaned_product[field] = ""
                        elif float_val.is_integer():
                            cleaned_product[field] = f"{int(float_val)}%"
                        else:
                            cleaned_product[field] = f"{float_val}%"
                    except (ValueError, TypeError):
                        cleaned_product[field] = ""

            # ── refillable_status — only "RF" or "NRF" ──────────────────────
            elif field == 'refillable_status':
                if value in BLANK_SENTINELS:
                    cleaned_product[field] = ""
                else:
                    val_upper = str(value).strip().upper()
                    if val_upper in ("RF", "REF", "REFILLABLE"):
                        cleaned_product[field] = "RF"
                    elif val_upper in ("NRF", "NON-REFILLABLE", "NON REFILLABLE"):
                        cleaned_product[field] = "NRF"
                    else:
                        # Not explicitly RF or NRF — leave blank per Rule 2
                        cleaned_product[field] = ""

            # ── currency — normalise EURO → EUR ─────────────────────────────
            elif field == 'currency':
                if value in BLANK_SENTINELS:
                    cleaned_product[field] = ""
                else:
                    val = str(value).strip().upper()
                    if val in ("EURO", "EUROS", "€"):
                        cleaned_product[field] = "EUR"
                    else:
                        cleaned_product[field] = val

            # ── List fields (error_flags) ────────────────────────────────────
            elif isinstance(default_value, list):
                cleaned_product[field] = value if isinstance(value, list) else []

            # ── Boolean fields ───────────────────────────────────────────────
            elif isinstance(default_value, bool):
                cleaned_product[field] = bool(value)

            # ── Float (confidence_score) ─────────────────────────────────────
            elif isinstance(default_value, float):
                try:
                    cleaned_product[field] = float(value)
                except (ValueError, TypeError):
                    cleaned_product[field] = default_value

            # ── All other string fields ──────────────────────────────────────
            else:
                if value in BLANK_SENTINELS:
                    cleaned_product[field] = ""
                else:
                    cleaned_product[field] = str(value)
        else:
            cleaned_product[field] = default_value

    # ── Auto-generate product_key if missing ────────────────────────────────
    if cleaned_product.get('product_key') in ("", None) and cleaned_product.get('product_name') not in ("", None):
        product_key = (
            str(cleaned_product['product_name'])
            .replace(' ', '_').replace('/', '_').replace('&', '_').replace('.', '')
            .upper()
        )
        cleaned_product['product_key'] = product_key

    # ── Enforce zero = blank for specific numeric fields ────────────────────
    numeric_fields_never_zero = [
        'cases_per_pallet', 'quantity_case', 'moq_cases',
        'unit_volume_ml', 'units_per_case', 'min_order_quantity_case'
    ]
    for field in numeric_fields_never_zero:
        if cleaned_product.get(field) == 0:
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