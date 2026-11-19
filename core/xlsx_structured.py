"""core/xlsx_structured.py

Deterministic extraction for well-structured supplier spreadsheets.

Why this exists
---------------
The XLSX pipeline handed each row to the LLM and let it decide which column was
which. On a cleanly-labelled sheet that mis-mapped commercial data (Samuel's QA
file, 2026-11-05):
  * a "Price per unit" value (€8) landed in price_per_case, so price_per_unit
    was then computed as 8 / units_per_case = 1.33;
  * an "ABV" cell formatted "40.0%" was read as the raw fraction 0.4;
  * the per-row "Offer date" / "Valid until" columns were ignored (offer_date
    fell back to the filename and mis-parsed), and the two date columns shifted;
  * only the first worksheet was read, dropping a whole second sheet.

When a sheet has a recognisable header row we instead map columns → fields
DETERMINISTICALLY — correct types, percentages scaled, dates kept, NO division,
every sheet, every row — and hand the per-product dicts to the SAME
clean_product_data / normalisation the LLM path uses (which then correctly
derives price_per_case = price_per_unit × units_per_case, etc.).

A sheet whose header we can't confidently recognise returns nothing, so the
caller falls back to the LLM path for messy / free-form files.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _norm(h: Any) -> str:
    """Normalise a header cell to lowercase words for synonym matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(h if h is not None else "").lower()).strip()


# Normalised-header → canonical field. Order doesn't matter; longest/explicit
# matches are preferred in _match_header.
_FIELD_SYNONYMS: Dict[str, str] = {
    "price per unit": "price_per_unit", "unit price": "price_per_unit",
    "price per bottle": "price_per_unit", "price bottle": "price_per_unit",
    "price per piece": "price_per_unit", "unit price eur": "price_per_unit",
    "price per case": "price_per_case", "case price": "price_per_case",
    "price per carton": "price_per_case",
    # A bare "Price" column paired with a "Price basis" column (per bottle /
    # per case / per piece) — resolved to price_per_unit vs price_per_case
    # AFTER the row is read (Samuel QA_EMAIL E07, 2026-11-07).
    "price": "_price_value", "net price": "_price_value", "price eur": "_price_value",
    "price basis": "_price_basis", "basis": "_price_basis",
    "price type": "_price_basis", "per": "_price_basis",
    "units per case": "units_per_case", "unit per case": "units_per_case",
    "bottles per case": "units_per_case", "pieces per case": "units_per_case",
    "pack size": "units_per_case", "pack": "units_per_case",
    "abv": "alcohol_percent", "alcohol": "alcohol_percent",
    "alcohol percent": "alcohol_percent", "alc vol": "alcohol_percent",
    "alcohol by volume": "alcohol_percent", "abv vol": "alcohol_percent",
    "available quantity": "_quantity", "quantity available": "_quantity",
    "quantity": "_quantity", "qty": "_quantity", "stock": "_quantity",
    "available": "_quantity", "stock quantity": "_quantity",
    "availability": "_quantity", "availability quantity": "_quantity",
    "quantity as stated": "_quantity", "availability quantity as stated": "_quantity",
    "quantity unit": "quantity_unit", "qty unit": "quantity_unit",
    "offer date": "offer_date", "date of offer": "offer_date",
    "valid until": "valid_until", "validity": "valid_until",
    "valid till": "valid_until", "expiry": "valid_until",
    "expiry date": "valid_until", "valid to": "valid_until",
    "best before": "best_before_date", "best before date": "best_before_date",
    "bbd": "best_before_date",
    "incoterm": "incoterm", "incoterms": "incoterm", "delivery terms": "incoterm",
    "location": "location", "warehouse": "location", "stock location": "location",
    "supplier": "supplier_name", "supplier name": "supplier_name",
    "vendor": "supplier_name", "seller": "supplier_name",
    "currency": "currency", "ccy": "currency", "cur": "currency",
    "customs status": "custom_status", "customs": "custom_status",
    "custom status": "custom_status", "customs t1 t2": "custom_status",
    "lead time": "lead_time", "leadtime": "lead_time", "delivery time": "lead_time",
    "moq cases": "moq_cases", "moq": "moq_cases", "min order": "moq_cases",
    "minimum order": "moq_cases", "min order quantity": "moq_cases",
    "ean": "ean_code", "ean code": "ean_code", "barcode": "ean_code",
    "gtin": "ean_code", "ean13": "ean_code",
    "vintage": "vintage", "year": "vintage",
    "brand": "brand", "maker": "brand", "marque": "brand",
    "product": "product_name", "product name": "product_name",
    "description": "product_name", "item": "product_name", "designation": "product_name",
    # "Line" is AMBIGUOUS: in the Dutch/EU wholesale template it is the product-
    # description column (EXW · ID · BarCode · Type · Brand · Line · QTY · EUR ·
    # ORDER — Samuel's W23/W39/W46/Cosmetics/LOREAL files), but in other sheets
    # it is a genuine line-NUMBER column sitting next to a real "Product
    # description" column (Samuel AQT 2026-11-17). Map it to a WEAK alias and let
    # _detect_header_row resolve it: it becomes product_name only when no strong
    # product/description column is present, otherwise it is dropped as a number.
    "line": "_line_name", "product line": "_line_name",
    "line description": "product_name", "article": "product_name",
    "size": "_size", "volume": "_size", "format": "_size", "capacity": "_size",
    # A column whose header is a bare currency ("EUR", "USD") holds the unit
    # price in that currency (the price column in the EU wholesale template and
    # hot-offer sheets). A header literally named "Currency"/"Ccy" stays the
    # currency field above.
    "eur": "price_per_unit", "euro": "price_per_unit", "usd": "price_per_unit",
    "gbp": "price_per_unit", "eur btl": "price_per_unit",
    "eur bottle": "price_per_unit", "eur per bottle": "price_per_unit",
    # "Type" in the wholesale template carries the retail/customs state
    # (Tester / Sample / Clean EU / CLN), not a product category — routed in the
    # row loop; real category words still fall through to category.
    "category": "_category", "type": "_type_state", "state": "_type_state",
    "notes": "notes", "note": "notes", "comment": "notes", "comments": "notes",
    "remarks": "notes",
    "supplier reference": "supplier_reference", "supplier ref": "supplier_reference",
    "reference": "supplier_reference", "product reference": "product_reference",
    "sku": "product_reference",
}

# Headers we recognise but intentionally ignore (keeps the match score honest).
# "Line" is NO LONGER ignored (it is the product column in the EU wholesale
# template); genuine line-NUMBER columns are ignored by their explicit names.
_IGNORE_HEADERS = {"test id", "id", "no", "line no", "line number", "line id",
                   "row", "row no", "index"}

# Incoterm literals that appear AS a column header ("EXW" over location codes
# like "NL"). The header IS the incoterm for every row; the cell is the stock
# location. Distinct from an "Incoterm" column whose CELLS hold EXW/DAP/…
_INCOTERM_LITERALS = {"exw", "fca", "fob", "cif", "cfr", "cpt", "cip",
                      "dap", "dpu", "ddp", "dat", "fas", "exworks"}

# Fields that make a row a real product line (need at least one). "_price_value"
# is the bare "Price" column resolved later against its "Price basis".
_PRICE_FIELDS = {"price_per_unit", "price_per_case", "_price_value"}


def _match_header(norm: str) -> Optional[str]:
    if not norm:
        return None
    if norm in _IGNORE_HEADERS:
        return "_ignore"
    if norm in _INCOTERM_LITERALS:
        return "_incoterm_header"
    if norm in _FIELD_SYNONYMS:
        return _FIELD_SYNONYMS[norm]
    # token-subset fallback: "offer date (utc)" → "offer date"
    for syn, field in _FIELD_SYNONYMS.items():
        if norm == syn or norm.startswith(syn + " ") or norm.endswith(" " + syn):
            return field
    return None


def _detect_header_row(ws, scan: int = 12) -> Optional[Tuple[int, Dict[int, str]]]:
    """Find the header row (most recognised columns) and its column→field map.
    Returns None when no row recognises enough labelled columns."""
    best: Optional[Tuple[int, Dict[int, str]]] = None
    best_score = 0
    for r in range(1, min(scan, ws.max_row) + 1):
        colmap: Dict[int, str] = {}
        seen_fields = set()
        for c in range(1, ws.max_column + 1):
            field = _match_header(_norm(ws.cell(row=r, column=c).value))
            if field and field not in ("_ignore",):
                # First column wins a given field (avoid a stray later dup).
                if field not in seen_fields:
                    colmap[c] = field
                    seen_fields.add(field)
        # Resolve the ambiguous "Line" column: it is the product name only when
        # no strong product/description column exists (EU wholesale template);
        # when a real product column is present it is a line NUMBER -> drop it
        # (Samuel AQT 2026-11-17).
        if "_line_name" in seen_fields:
            if "product_name" in seen_fields:
                colmap = {c: f for c, f in colmap.items() if f != "_line_name"}
                seen_fields.discard("_line_name")
            else:
                colmap = {c: ("product_name" if f == "_line_name" else f)
                          for c, f in colmap.items()}
                seen_fields.discard("_line_name")
                seen_fields.add("product_name")
        score = len(seen_fields)
        has_product = "product_name" in seen_fields or "brand" in seen_fields
        has_price = bool(_PRICE_FIELDS & seen_fields)
        # Require a product column, a price column, and enough total structure
        # so we never hijack a messy/free-form sheet from the LLM path.
        if has_product and has_price and score >= 5 and score > best_score:
            best = (r, colmap)
            best_score = score
    return best


_SIZE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(l|cl|ml|litre|liter)\b", re.IGNORECASE)


def _parse_size_ml(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    m = _SIZE_RE.search(str(raw))
    if not m:
        return None
    val = float(m.group(1).replace(",", "."))
    unit = m.group(2).lower()
    if unit in ("l", "litre", "liter"):
        return val * 1000
    if unit == "cl":
        return val * 10
    return val  # ml


# When the Size cell is a bare number ("700") the unit lives in the HEADER
# ("Size (ml)", "Volume (cl)"). Read the unit from the header and apply it so
# unit_volume_ml is populated instead of left blank (Samuel E07 2026-11-10 —
# the "6x?ml" in the export came from a missing unit_volume_ml).
_HEADER_UNIT_RE = re.compile(r"\b(ml|cl|l|litre|liter)\b", re.IGNORECASE)


_WEIGHT_RE = re.compile(r"\d\s*(?:g|gr|gram|grams|kg)\b", re.IGNORECASE)
_VOLUME_UNIT_RE = re.compile(r"\d\s*(?:ml|cl|l|litre|liter)\b", re.IGNORECASE)


def _parse_size_with_header_unit(raw: Any, header: Any) -> Optional[float]:
    # A weight ("210 g", "210gr", "1 kg") is NOT a volume. The old code
    # defaulted a bare number to ml even when the cell said grams, so a 210 g
    # candle was stored as 210 ml (Samuel 2026-11-18, Nishane). Grams are
    # captured as size_weight_g by the caller; return no ml here.
    _s = str(raw if raw is not None else "")
    if _WEIGHT_RE.search(_s) and not _VOLUME_UNIT_RE.search(_s):
        return None
    v = _to_float(raw)
    if v is None:
        return None
    m = _HEADER_UNIT_RE.search(str(header if header is not None else ""))
    unit = (m.group(1).lower() if m else "ml")
    if unit in ("l", "litre", "liter"):
        return v * 1000
    if unit == "cl":
        return v * 10
    return v  # ml (default when the header gives no unit)


# The EU wholesale template carries the bottle size INSIDE the product
# description ("… EDP 100ml"), with no dedicated Size column, so unit_volume_ml
# was left blank for the whole file and later coerced to 0 (Samuel W46
# 2026-11-13 — "volume_ml is 0 for all 367 rows"). Derive it from the name:
# the bounded form ("100ml", "0,7l", "70cl") OR a size glued to a format code
# ("100mlEDP"). A solid measured in grams ("210gr" candle) has no ml and stays
# blank — never 0. A bare "l" must be word-bounded so "12 Limited" is not read
# as 12 litres.
_NAME_VOL_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(ml|cl|l)\b", re.IGNORECASE)
_NAME_VOL_GLUED_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(ml|cl)(?=[A-Za-z])", re.IGNORECASE)


def _volume_from_name(name: Any) -> Optional[float]:
    s = str(name if name is not None else "")
    best: Optional[float] = None
    for m in _NAME_VOL_RE.finditer(s):
        v = float(m.group(1).replace(",", "."))
        u = m.group(2).lower()
        best = v * 1000 if u == "l" else v * 10 if u == "cl" else v
    if best is None:
        m = _NAME_VOL_GLUED_RE.search(s)
        if m:
            v = float(m.group(1).replace(",", "."))
            best = v * 10 if m.group(2).lower() == "cl" else v
    return best


# Customs tier (T1 / T2) is often embedded in the product-description column
# ("Dry Gin T2") with no dedicated Customs column. Capture it into
# custom_status at extraction time and strip it from the name BEFORE anything
# downstream treats the token as noise and drops it (Samuel E07 2026-11-10 —
# the nine T2 rows stored custom_status = None).
_CUSTOMS_TOKEN_RE = re.compile(r"\bT([12])\b")


def _to_float(raw: Any) -> Optional[float]:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = re.sub(r"[^\d.,\-]", "", str(raw)).replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _iso_date(raw: Any) -> Optional[str]:
    if raw is None or raw == "":
        return None
    if hasattr(raw, "isoformat"):
        try:
            return raw.date().isoformat() if hasattr(raw, "date") else raw.isoformat()
        except Exception:
            return str(raw)[:10]
    return str(raw).strip()


def _category_slug(raw: Any) -> Optional[str]:
    s = _norm(raw)
    if not s:
        return None
    if any(k in s for k in ("spirit", "wine", "whisky", "whiskey", "vodka",
                            "gin", "rum", "cognac", "liqueur", "champagne", "beer")):
        return "wines_spirits"
    if "perfume" in s or "fragrance" in s:
        return "perfumes"
    if "cosmetic" in s or "beauty" in s or "skincare" in s:
        return "cosmetics"
    return None


# quantity unit → the stock field it belongs in (NEVER divide/convert).
_QTY_CASE = {"case", "cases", "cs", "carton", "cartons", "ctn"}
_QTY_PALLET = {"pallet", "pallets", "pll", "plt"}
_QTY_FTL = {"ftl", "truck", "trucks", "load"}
_QTY_UNIT = {"bottle", "bottles", "btl", "unit", "units", "piece", "pieces",
             "pcs", "pc", "each"}

# A quantity/MOQ cell often carries its unit inline with no separate unit
# column ("100 cases", "3 pallets", "1 FTL", "9600 bottles", "24 pieces").
# Parse the leading number AND the trailing unit word so the unit routing
# (_assign_quantity) is correct — a bottle count must never be stored as cases.
_NUM_UNIT_RE = re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*([A-Za-z]+)?')
_ON_REQUEST_RE = re.compile(
    r'on\s*request|price\s*on\s*req|\bp\.?o\.?a\.?\b|\bpoa\b|\btbc\b|'
    r'to\s*be\s*(?:confirmed|advised)|\bask\b|'
    r'call\s*for\s*(?:quot\w*|price|offer)|\bquotation\b', re.IGNORECASE)


def _split_num_unit(raw: Any) -> Tuple[Optional[float], Optional[str]]:
    """('9600 bottles') -> (9600.0, 'bottles'); ('100') -> (100.0, None).
    Commas are treated as thousands separators (English QA files)."""
    if raw is None:
        return (None, None)
    if isinstance(raw, (int, float)):
        return (float(raw), None)
    m = _NUM_UNIT_RE.search(str(raw).strip())
    if not m:
        return (None, None)
    try:
        num = float(m.group(1).replace(",", ""))
    except ValueError:
        num = None
    unit = (m.group(2) or "").strip().lower() or None
    return (num, unit)


def _is_on_request(val: Any) -> bool:
    """A Price / Price-basis cell that means 'price on request' — never 0."""
    return bool(_ON_REQUEST_RE.search(str(val if val is not None else "")))


def _num_thousands(raw: Any) -> Optional[float]:
    """Parse a number whose comma is a THOUSANDS separator ('1,440' -> 1440.0,
    '765.00' -> 765.0). Distinct from _to_float, which treats a comma as a
    decimal point (European sheets)."""
    try:
        return float(str(raw).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


# ── Composite free-text price cells ────────────────────────────────────────
# A single "Price" column often carries the currency AND one or more basis-
# specific prices: "EUR 534.00/case (EUR 89.00/bottle)", "USD 16.25/bottle;
# USD 195.00/case", "HKD 765.00/case". _to_float stripped the letters/slashes
# and concatenated the two numbers into an invalid float -> None, so the whole
# row was silently dropped (Samuel AQT 2026-11-17: Glenfiddich / Chivas 12 /
# JD Tennessee lost; HKD/USD/GBP all collapsed to EUR). Parse them properly.
_CUR_WORDS = {
    "eur": "EUR", "euro": "EUR", "euros": "EUR", "€": "EUR",
    "usd": "USD", "us$": "USD", "$": "USD", "dollar": "USD", "dollars": "USD",
    "gbp": "GBP", "£": "GBP", "pound": "GBP", "pounds": "GBP",
    "hkd": "HKD", "hk$": "HKD", "chf": "CHF", "sgd": "SGD", "aed": "AED",
    "jpy": "JPY", "¥": "JPY", "cny": "CNY", "rmb": "CNY", "cad": "CAD",
    "aud": "AUD", "nzd": "NZD", "sek": "SEK", "dkk": "DKK", "nok": "NOK",
    "pln": "PLN", "zar": "ZAR",
}
_CUR_RE = re.compile(
    r'(hk\$|us\$|€|£|\$|¥|\b(?:eur|euro|usd|gbp|hkd|chf|sgd|aed|jpy|cny|rmb|'
    r'cad|aud|nzd|sek|dkk|nok|pln|zar)\b)', re.IGNORECASE)
# A number tied to an explicit per-unit basis: "534.00/case", "89 per bottle".
_BASIS_PRICE_RE = re.compile(
    r'(\d[\d,]*(?:\.\d+)?)\s*(?:/|per)\s*'
    r'(case|cases|carton|cartons|ctn|bottle|bottles|btl|unit|units|'
    r'piece|pieces|pcs|pc|each|can|cans)\b', re.IGNORECASE)
_BARE_NUM_RE = re.compile(r'(\d[\d,]*(?:\.\d+)?)')
# A cell that deliberately withholds the basis ("basis not stated", "per bottle
# or per case?") — never guess a basis; retain the amount as-quoted + review.
_AMBIG_BASIS_RE = re.compile(
    r'basis\s*(?:not\s*stated|unclear|unknown|unspecified|\?)|'
    r'per\s+\w+\s+or\s+per\s+\w+|not\s*stated[^.]*basis', re.IGNORECASE)
_CASE_WORDS = {"case", "cases", "carton", "cartons", "ctn"}


def _detect_currency(s: str) -> Optional[str]:
    m = _CUR_RE.search(s or "")
    return _CUR_WORDS.get(m.group(1).lower()) if m else None


def _parse_price_cell(val: Any) -> Dict[str, Any]:
    """Parse a free-text price cell -> {currency, price_per_case,
    price_per_unit, price_value, basis_ambiguous, on_request, raw}. price_value
    is a basisless number for the caller to resolve against a Price-basis column
    (or default). Basis-tagged numbers go straight to case/unit."""
    out: Dict[str, Any] = {"currency": None, "price_per_case": None,
                           "price_per_unit": None, "price_value": None,
                           "basis_ambiguous": False, "on_request": False,
                           "raw": "" if val is None else str(val).strip()}
    s = out["raw"]
    if not s:
        return out
    out["currency"] = _detect_currency(s)
    found = False
    for m in _BASIS_PRICE_RE.finditer(s):
        num = _num_thousands(m.group(1))
        if num is None:
            continue
        if m.group(2).lower() in _CASE_WORDS:
            out["price_per_case"] = out["price_per_case"] or num
        else:
            out["price_per_unit"] = out["price_per_unit"] or num
        found = True
    if found:
        return out
    bm = _BARE_NUM_RE.search(s)
    if bm:
        out["price_value"] = _to_float(bm.group(1))
        if _AMBIG_BASIS_RE.search(s):
            out["basis_ambiguous"] = True
    elif _is_on_request(s):
        out["on_request"] = True
    return out


# ── Pack size ("6 bottles/case", "4 inner boxes × 12 pieces = 48 pieces") ──
def _parse_pack(val: Any) -> Optional[float]:
    """Units-per-case from a free-text pack cell. Prefers the total after '='
    ("… = 48 pieces" -> 48), else the count before '/case' or 'per case'
    ("6 bottles/case" -> 6), else the first integer. 'Individual pieces',
    'Single candle/piece' and other non-case packs -> None (never 0 and never a
    concatenation of every digit — Samuel AQT 2026-11-17 Montblanc = 41248)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    low = s.lower()
    if any(k in low for k in ("individual", "single", "not case", "loose", "n/a")):
        return None
    # Inner-box pack ("4 inner boxes × 12 pieces = 48 pieces"): the pack size is
    # the pieces-per-inner-box (12), NOT the 48-piece grand total. The item is
    # still priced per piece — the per-piece guard in clean_product_data keeps
    # price_per_case empty — and the export labels the pack an "inner box" via
    # moq_unit (Samuel 2026-11-19: Montblanc was shown "per piece", MOQ "cases").
    m_ib = (re.search(r'inner\s*box(?:es)?\s*[x×]\s*(\d[\d,]*)\s*(?:pieces?|pcs?)',
                      s, re.IGNORECASE)
            or re.search(r'(\d[\d,]*)\s*(?:pieces?|pcs?)\s*(?:/|per)\s*inner\s*box',
                         s, re.IGNORECASE))
    if m_ib:
        return _num_thousands(m_ib.group(1))
    # A pack expressed purely in PIECES with no case/carton (an individual-piece
    # item sold per piece, e.g. a single candle) is not a case — returning the
    # piece total here let a fabricated case price form downstream (Samuel
    # 2026-11-18). A real case names a case / carton.
    if ("piece" in low or "pcs" in low) and not any(
            w in low for w in ("case", "carton", "/cs", "per cs", "ctn")):
        return None
    m = re.search(r'=\s*(\d[\d,]*)', s)
    if m:
        return _num_thousands(m.group(1))
    m = re.search(r'(\d[\d,]*)\s*[A-Za-z ]*?(?:/|per)\s*case', s, re.IGNORECASE)
    if m:
        return _num_thousands(m.group(1))
    m = re.search(r'(\d[\d,]*)', s)
    return _num_thousands(m.group(1)) if m else None


# ── Quantity cell ("12 pallets × 48 cases/pallet = 576 cases",
#    "1,440 bottles available (= 120 cases)") ─────────────────────────────────
_PAREN_RE = re.compile(r'\([^)]*\)')


def _parse_quantity_cell(val: Any) -> Tuple[Optional[float], Optional[str]]:
    """Stated availability -> (number, unit). Prefers a '= N unit' TOTAL that is
    NOT inside parentheses ("… = 576 cases"), else the first "N unit" outside any
    parentheses (so "1,440 bottles available (= 120 cases)" keeps the stated
    1,440 bottles, not the parenthetical case equivalent)."""
    if val is None:
        return (None, None)
    if isinstance(val, (int, float)):
        return (float(val), None)
    outside = _PAREN_RE.sub(" ", str(val).strip())  # drop parenthetical equivalents
    m = re.search(r'=\s*(\d[\d,]*(?:\.\d+)?)\s*([A-Za-z]+)', outside)
    if m:
        return (_num_thousands(m.group(1)), (m.group(2) or "").strip().lower() or None)
    num, unit = _split_num_unit(outside)
    return (num, unit)


def _assign_quantity(prod: Dict[str, Any], qty: Optional[float], unit: Optional[str]) -> None:
    if qty is None:
        return
    u = _norm(unit)
    prod["quantity_unit"] = unit if unit else prod.get("quantity_unit")
    if u in _QTY_PALLET:
        prod["quantity_pallets"] = qty
    elif u in _QTY_FTL:
        prod["quantity_ftl"] = qty
    elif u in _QTY_UNIT:
        prod["quantity_bottles"] = qty
    else:
        # default: cases (the common supplier unit) — matches the "cases" column
        prod["quantity_case"] = qty


def _scale_percent(cell) -> Optional[float]:
    """ABV cells are usually stored as a fraction with a % number format
    (0.4 shown as "40.0%"). openpyxl gives the raw 0.4 — scale by 100. A cell
    already holding 40 (no % format) is kept as-is."""
    v = _to_float(cell.value)
    if v is None:
        return None
    fmt = (cell.number_format or "")
    if "%" in fmt:
        return round(v * 100, 4)
    # No % format but a plausible fraction (0 < v <= 1) → treat as fraction.
    if 0 < v <= 1:
        return round(v * 100, 4)
    return v


def extract_structured_products(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """Deterministically extract products from every well-structured sheet.
    Returns a list of raw product dicts (pre-clean), or None when no sheet has a
    recognisable header (caller then uses the LLM path)."""
    try:
        import openpyxl
    except Exception as e:  # pragma: no cover
        logger.warning(f"[xlsx_structured] openpyxl unavailable: {e}")
        return None
    try:
        wb = openpyxl.load_workbook(file_path, data_only=True, read_only=False)
    except Exception as e:
        logger.warning(f"[xlsx_structured] could not open {file_path}: {e}")
        return None

    products: List[Dict[str, Any]] = []
    recognised_sheets = 0

    for ws in wb.worksheets:
        if ws.max_row is None or ws.max_row < 2:
            continue
        detected = _detect_header_row(ws)
        if not detected:
            logger.info(f"[xlsx_structured] sheet {ws.title!r}: no recognisable header — skipped")
            continue
        header_row, colmap = detected
        recognised_sheets += 1
        logger.info(f"[xlsx_structured] sheet {ws.title!r}: header at row {header_row}, "
                    f"{len(colmap)} mapped columns: {sorted(set(colmap.values()))}")

        for r in range(header_row + 1, ws.max_row + 1):
            prod: Dict[str, Any] = {}
            qty_val: Optional[float] = None
            qty_unit: Optional[str] = None
            qty_raw: str = ""
            moq_val: Optional[float] = None
            moq_unit_inline: Optional[str] = None
            moq_raw_cell: str = ""
            price_value: Optional[float] = None
            price_basis: str = ""
            price_on_request = False
            basis_ambiguous = False
            price_cell_raw = ""
            size_raw: Any = None
            any_value = False
            nobox = False

            for col, field in colmap.items():
                cell = ws.cell(row=r, column=col)
                val = cell.value
                if val is None or (isinstance(val, str) and not val.strip()):
                    continue
                any_value = True

                if field == "alcohol_percent":
                    prod["alcohol_percent"] = _scale_percent(cell)
                elif field == "_price_value":
                    # Free-text "Price" cell: may carry the currency and one or
                    # more basis-tagged prices ("EUR 534/case (EUR 89/bottle)",
                    # "USD 16.25/bottle; USD 195/case"). Parse rather than letting
                    # _to_float concatenate two numbers into None (= a dropped row).
                    price_cell_raw = str(val).strip()
                    pc = _parse_price_cell(val)
                    if pc["currency"] and not prod.get("currency"):
                        prod["currency"] = pc["currency"]
                    if pc["on_request"]:
                        price_on_request = True
                    elif pc["price_per_case"] is not None or pc["price_per_unit"] is not None:
                        if pc["price_per_case"] is not None:
                            prod["price_per_case"] = pc["price_per_case"]
                        if pc["price_per_unit"] is not None:
                            prod["price_per_unit"] = pc["price_per_unit"]
                    elif pc["price_value"] is not None:
                        price_value = pc["price_value"]
                        basis_ambiguous = pc["basis_ambiguous"]
                elif field == "_price_basis":
                    price_basis = str(val).strip().lower()
                    if _is_on_request(val):
                        price_on_request = True
                elif field in ("price_per_unit", "price_per_case"):
                    pc = _parse_price_cell(val)
                    if pc["currency"] and not prod.get("currency"):
                        prod["currency"] = pc["currency"]
                    prod[field] = (pc["price_per_case"] if field == "price_per_case"
                                   else pc["price_per_unit"])
                    if prod[field] is None:
                        prod[field] = pc["price_value"]
                    if prod[field] is None:
                        prod[field] = _to_float(val)
                elif field == "units_per_case":
                    prod[field] = _parse_pack(val)
                elif field == "moq_cases":
                    # MOQ may carry its own inline unit ("1 pallet", "1 FTL").
                    moq_val, moq_unit_inline = _split_num_unit(val)
                    # Keep the verbatim source wording ("2 inner boxes", "24
                    # bottles") so the export shows the MOQ exactly as stated and
                    # matches the WhatsApp copy of the same product.
                    moq_raw_cell = str(val).strip()
                elif field == "_quantity":
                    qty_raw = str(val)
                    qty_val, _qu = _parse_quantity_cell(val)
                    if _qu:
                        qty_unit = _qu
                elif field == "quantity_unit":
                    qty_unit = str(val).strip()
                elif field in ("offer_date", "valid_until", "best_before_date"):
                    prod[field] = _iso_date(val)
                elif field == "_size":
                    size_raw = val
                    prod["packaging_raw"] = str(val).strip()
                    ml = _parse_size_ml(val)
                    if ml is None:
                        # Bare number ("700") whose unit is only in the header
                        # ("Size (ml)"); _parse_size_ml needs an inline unit.
                        ml = _parse_size_with_header_unit(
                            val, ws.cell(row=header_row, column=col).value)
                    if ml is not None:
                        prod["unit_volume_ml"] = ml
                    else:
                        # A weight ("210 g") is a solid's size — store grams,
                        # never coerce to ml (Samuel 2026-11-18, Nishane candle).
                        _gm = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:g|gr|gram|grams)\b",
                                        str(val), re.IGNORECASE)
                        if _gm:
                            prod["size_weight_g"] = float(_gm.group(1).replace(",", "."))
                elif field == "_category":
                    slug = _category_slug(val)
                    if slug:
                        prod["category_slug"] = slug
                    prod["category"] = str(val).strip()
                elif field == "_incoterm_header":
                    # Header is the incoterm (EXW/FCA/…); the cell is the stock
                    # location (e.g. "NL", "NL-1w", "Rotterdam"). A location cell
                    # starts with letters; a purely-numeric cell would be a
                    # per-incoterm PRICE, not a location — leave location unset.
                    hdr = str(ws.cell(row=header_row, column=col).value or "").strip().upper()
                    if hdr:
                        prod.setdefault("incoterm", hdr)
                    sv = str(val).strip()
                    # Take the leading location token ("NL" from "NL-1w"); a
                    # trailing "<n>w" / "<n> week(s)" / "<n>d" is a lead time,
                    # lifted into lead_time. Previously "NL-1w" read as numeric
                    # (_to_float stripped it to -1) so NL was dropped entirely and
                    # the two Kajal rows showed location "Not Found" (Samuel W46
                    # 2026-11-13).
                    _lm = re.match(r"^([A-Za-z][A-Za-z .]*)", sv)
                    if _lm:
                        prod.setdefault("location", _lm.group(1).strip(" .-"))
                        _ltm = re.search(
                            r"(\d+)\s*(weeks?|wks?|w|days?|d|months?|mo?)\b",
                            sv, re.IGNORECASE)
                        if _ltm and not prod.get("lead_time"):
                            _n, _u = _ltm.group(1), _ltm.group(2).lower()
                            _uw = ("week" if _u.startswith("w")
                                   else "day" if _u.startswith("d") else "month")
                            prod["lead_time"] = f"{_n} {_uw}" + ("s" if _n != "1" else "")
                elif field == "_type_state":
                    # "Type" column: route retail vs customs state; fall through
                    # to category only for a genuine category word.
                    sv = str(val).strip()
                    lv = sv.lower()
                    # A "no box" / unboxed tester is a DISTINCT line from the
                    # boxed one (same EAN/price, different condition). Flag it so
                    # it survives the ingest-time _canon_key dedup, which keys on
                    # product_name + retail_state (Samuel W46 2026-11-13: two
                    # "Givenchy Enflamme" rows — "Tester*" and "Tester no box*").
                    if ("no box" in lv or "no-box" in lv or "nobox" in lv
                            or "without box" in lv or "unbox" in lv):
                        nobox = True
                        prod["gift_box"] = False
                    if any(k in lv for k in ("clean", "cln", "duty", "bond",
                                             "t1", "t2", "customs", "unpaid", "paid")):
                        prod.setdefault("custom_status", sv)
                    elif any(k in lv for k in ("tester", "sample", "miniature",
                                               "gift", "display", "retail",
                                               "unbox", "coffret", "promo")):
                        if "tester" in lv:
                            prod.setdefault("retail_state", "tester")
                        elif "sample" in lv:
                            prod.setdefault("retail_state", "sample")
                        elif "miniature" in lv:
                            prod.setdefault("retail_state", "miniature")
                        else:
                            prod.setdefault("retail_state", lv)
                    else:
                        slug = _category_slug(sv)
                        if slug:
                            prod["category_slug"] = slug
                        prod["category"] = sv
                elif field == "vintage":
                    prod["vintage"] = str(int(_to_float(val))) if _to_float(val) else str(val).strip()
                elif field == "ean_code":
                    # preserve verbatim (leading zeros matter); a dash / blank
                    # placeholder ("—", "-", "n/a") is NOT a barcode — skip it so
                    # it neither stores junk nor keeps an otherwise-empty row alive.
                    _ev = (str(int(val)) if isinstance(val, float) and val.is_integer()
                           else str(val).strip())
                    if any(ch.isdigit() for ch in _ev):
                        prod["ean_code"] = _ev
                elif field in ("_ignore",):
                    continue
                else:
                    prod[field] = str(val).strip()

            if not any_value:
                continue
            # A product line needs a name or a brand.
            if not (prod.get("product_name") or prod.get("brand")):
                continue

            # ── "No box" tester → make it a distinct line ───────────────────
            # Reflect the unboxed condition in the name (visible in the drawer /
            # export) so the ingest dedup keeps it separate from the boxed line.
            if nobox:
                _nm = prod.get("product_name") or ""
                if "no box" not in _nm.lower():
                    if not prod.get("product_name_raw"):
                        prod["product_name_raw"] = _nm
                    prod["product_name"] = (f"{_nm} (No Box)".strip()
                                            if _nm else "No Box")

            # ── Customs tier embedded in the description ("Dry Gin T2") ──────
            # No dedicated Customs column: lift the T1/T2 token into
            # custom_status and strip it from the name so the drawer, the
            # export and the peer-group key (which keys on custom_status) all
            # see it. Done here, at the source, before any downstream cleaner
            # can drop the token (Samuel E07, 2026-11-10).
            if not prod.get("custom_status"):
                _nm = prod.get("product_name") or ""
                _cm = _CUSTOMS_TOKEN_RE.search(_nm)
                if _cm:
                    prod["custom_status"] = "T" + _cm.group(1)
                    if not prod.get("product_name_raw"):
                        prod["product_name_raw"] = _nm
                    prod["product_name"] = re.sub(
                        r"\s{2,}", " ", _CUSTOMS_TOKEN_RE.sub("", _nm)).strip(" -,")

            # ── Volume from the product name (no dedicated Size column) ─────
            # The EU wholesale template puts the size in the Line ("… 100ml").
            # Fill unit_volume_ml from the name so it is populated, not blank /
            # zero (Samuel W46 2026-11-13). Candles / solids in grams → blank.
            if not prod.get("unit_volume_ml"):
                _vml = _volume_from_name(prod.get("product_name"))
                if _vml is not None:
                    prod["unit_volume_ml"] = _vml

            # ── Resolve the Price / Price-basis pair ───────────────────────
            if price_value is not None:
                if basis_ambiguous:
                    # An amount was quoted but the basis was deliberately
                    # withheld ("USD 75.00 — price basis not stated (per bottle
                    # or per case?)"). Never guess it into a per-unit/per-case
                    # field; retain it as-quoted and flag for review (Samuel AQT
                    # 2026-11-17 Beefeater, mirroring the NE-410 rule).
                    cur = (prod.get("currency") or "").strip()
                    prod["commercial_conditions"] = (
                        f"As quoted: {cur} {price_value:g}".strip()
                        + " — price basis not stated; excluded from price comparison.")
                    flags = prod.setdefault("error_flags", [])
                    flags.append("price basis not stated — amount retained "
                                 "as-quoted, excluded from comparison")
                    prod["needs_manual_review"] = True
                elif "case" in price_basis or "carton" in price_basis:
                    prod["price_per_case"] = price_value
                elif any(w in price_basis for w in
                         ("bottle", "btl", "unit", "piece", "each")):
                    prod["price_per_unit"] = price_value
                elif prod.get("price_per_unit") is None and prod.get("price_per_case") is None:
                    # basis unstated → the common bottle price
                    prod["price_per_unit"] = price_value
            # A priced line that still carries a TBC / on-request clause for a
            # second delivery option keeps its price but is flagged for review
            # ("EUR 171.00/case applies to DAP Antwerp only; EXW Breda TBC").
            if (not price_on_request and not basis_ambiguous and price_cell_raw
                    and _is_on_request(price_cell_raw)
                    and (prod.get("price_per_unit") is not None
                         or prod.get("price_per_case") is not None)):
                flags = prod.setdefault("error_flags", [])
                flags.append("partial pricing — some delivery terms quoted "
                             "TBC / on request; review")
                prod["needs_manual_review"] = True
            if price_on_request:
                # Samuel 2026-11-07 (E07-08): a "price on request" line is NEVER
                # €0 and is NEVER dropped — keep it with null prices and flag it
                # so it is reviewed, not treated as a zero-price offer.
                prod["price_per_unit"] = None
                prod["price_per_case"] = None
                flags = prod.setdefault("error_flags", [])
                if "Price on request retained" not in flags:
                    flags.append("Price on request retained")
                prod["needs_manual_review"] = True

            # ── Multi-option delivery where the PRICE applies to ONE option ──
            # "EUR 171/case applies to DAP Antwerp only; EXW Breda price TBC" →
            # resolve the stored incoterm/location to the PRICED option (DAP
            # Antwerp) and keep the whole clause as a note, so the price is not
            # shown as applying to both DAP and EXW (Samuel 2026-11-19, Moët).
            if price_cell_raw:
                _mo = re.search(r'applies to\s+([A-Z]{2,4})\s+([A-Z][A-Za-z]+)\s+only',
                                price_cell_raw, re.IGNORECASE)
                if _mo:
                    prod["incoterm"] = _mo.group(1).upper()
                    prod["location"] = _mo.group(2)
                    _mnote = re.sub(r'\s+', ' ', price_cell_raw).strip()
                    prod["commercial_conditions"] = (
                        (prod.get("commercial_conditions") + " | " + _mnote)
                        if prod.get("commercial_conditions") else _mnote)
                    prod["needs_manual_review"] = True

            # ── Quantity (unit parsed from the cell) ───────────────────────
            _assign_quantity(prod, qty_val, qty_unit)
            # One quantity lives in ONE field. Stock stated in bottles with a
            # known pack ("1,440 bottles available (= 120 cases)", "24 bottles
            # available") is NOT also written into quantity_case: that derived
            # case count is the SAME stock, and a consumer that sums cases×pack
            # + bottles then double-counts it (Samuel 2026-11-19: Dom Pérignon
            # "4 cases + 24 units", and the self-check caught Chivas 120/1,440
            # and Jack Daniel's 800/9,600 the same way). The case equivalent is
            # a presentation concern — the export derives "(= N cases)" from the
            # pack at display time, so nothing is lost.

            # ── Mixed quantity ("10 cases + 2 loose bottles") ──────────────
            # _parse_quantity_cell returns only the primary count, dropping the
            # loose remainder (Samuel 2026-11-18: Perrier-Jouët's 2 loose
            # bottles were lost). Capture a "+ N <unit>" secondary and keep
            # BOTH the cases and the loose units. Done after the bottles→cases
            # derivation above so it does not get overwritten.
            if qty_raw and "+" in qty_raw:
                _sec = re.search(
                    r"\+\s*(\d[\d,]*)\s*(?:loose\s+)?"
                    r"(bottle|bottles|btl|piece|pieces|pcs|can|cans|unit|units|case|cases)",
                    qty_raw, re.IGNORECASE)
                if _sec:
                    _sn = _num_thousands(_sec.group(1))
                    _su = _sec.group(2).lower()
                    if _sn:
                        if _su.startswith("case"):
                            prod["quantity_case"] = (prod.get("quantity_case") or 0) + _sn
                        elif not prod.get("quantity_bottles"):
                            prod["quantity_bottles"] = _sn

            # ── MOQ (unit follows the source — client §5) ──────────────────
            if moq_raw_cell and moq_raw_cell not in ("—", "-", "Not Found",
                                                     "Not stated", "n/a", "N/A"):
                # Verbatim MOQ wording for the export (kept even when the inline
                # unit is non-standard, e.g. "inner boxes").
                prod["moq_text"] = re.sub(r"(?i)^\s*moq[:\s]*", "",
                                          moq_raw_cell).strip() or None
            if moq_val is not None:
                _mu = _norm(moq_unit_inline)
                if _mu in _QTY_PALLET:
                    # Keep the native pallet count ("MOQ 1 pallet") so the
                    # drawer shows "1 plt", not blank (Samuel E07 2026-11-10).
                    prod["moq_pallets"] = moq_val
                    prod["moq_unit"] = "pallets"
                elif _mu in _QTY_FTL:
                    prod["moq_ftl"] = moq_val
                    prod["moq_unit"] = "ftl"
                elif _mu in ("piece", "pieces", "pcs", "pc"):
                    prod["moq_bottles"] = moq_val
                    prod["moq_unit"] = "pieces"
                elif _mu in ("bottle", "bottles", "btl", "unit", "units"):
                    prod["moq_bottles"] = moq_val
                    prod["moq_unit"] = "bottles"
                elif "inner" in _mu:
                    # Inner box(es) — a pack unit the source states explicitly;
                    # keep the native unit so the export reads "2 inner boxes",
                    # not "2 cases" (Samuel 2026-11-19, Montblanc).
                    prod["moq_cases"] = moq_val
                    prod["moq_unit"] = "inner boxes"
                else:
                    prod["moq_cases"] = moq_val
                    prod["moq_unit"] = "cases"

            # Keep the row when it carries ANY commercial signal — a price, a
            # price-on-request marker, a quantity, or an EAN. A no-price,
            # no-quantity line falls through to the LLM/flagging path.
            _has_price = (prod.get("price_per_unit") is not None
                          or prod.get("price_per_case") is not None)
            _has_qty = any(prod.get(k) for k in
                           ("quantity_case", "quantity_bottles",
                            "quantity_pallets", "quantity_ftl"))
            if not (_has_price or price_on_request or _has_qty or prod.get("ean_code")):
                continue

            prod.setdefault("category_slug", _category_slug(prod.get("category")))
            prod["_structured_source"] = f"{ws.title}:row{r}"
            products.append(prod)

    if recognised_sheets == 0 or not products:
        return None
    logger.info(f"[xlsx_structured] extracted {len(products)} product(s) from "
                f"{recognised_sheets} structured sheet(s)")
    return products
