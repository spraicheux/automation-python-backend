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
    # Abbreviated per-bottle / per-case headers ("Price / btl", "Price / cs")
    # carry the BASIS in the header — map them straight to the basis'd field so
    # the amount is a real per-bottle price, not a basis-unknown "_price_value"
    # (Samuel 2026-11-22: Spirits Online LAST_CHANCE "Price / btl" list).
    "price btl": "price_per_unit", "price per btl": "price_per_unit",
    "price btls": "price_per_unit",
    "price per case": "price_per_case", "case price": "price_per_case",
    "price per carton": "price_per_case",
    "price cs": "price_per_case", "price per cs": "price_per_case",
    # A bare "Price" column paired with a "Price basis" column (per bottle /
    # per case / per piece) — resolved to price_per_unit vs price_per_case
    # AFTER the row is read (Samuel QA_EMAIL E07, 2026-11-07).
    "price": "_price_value", "net price": "_price_value", "price eur": "_price_value",
    "price basis": "_price_basis", "basis": "_price_basis",
    "price type": "_price_basis", "per": "_price_basis",
    "units per case": "units_per_case", "unit per case": "units_per_case",
    "bottles per case": "units_per_case", "pieces per case": "units_per_case",
    "pack size": "units_per_case", "pack": "units_per_case",
    # "Packing" cell holds the pack format ("24 × 500 ml cans", "4 wrapped sixes
    # × 330 ml") — parsed for units-per-case AND kept verbatim as packaging_raw so
    # two formats of one product stay distinct (Alder A03 vs A04, 2026-11-24).
    "packing": "units_per_case", "packing format": "units_per_case",
    # Abbreviated bottles-per-case ("Btls / cs", "Btls per cs").
    "btls cs": "units_per_case", "btls per cs": "units_per_case",
    "btl cs": "units_per_case", "bottles cs": "units_per_case",
    "abv": "alcohol_percent", "alcohol": "alcohol_percent", "alc": "alcohol_percent",
    "alcohol percent": "alcohol_percent", "alc vol": "alcohol_percent",
    "alcohol by volume": "alcohol_percent", "abv vol": "alcohol_percent",
    "available quantity": "_quantity", "quantity available": "_quantity",
    "quantity": "_quantity", "qty": "_quantity", "stock": "_quantity",
    "available": "_quantity", "stock quantity": "_quantity",
    # Freeform stock-circular labels (Samuel 2026-11-24, Alder Quay 4-tab workbook):
    # "Warehouse position" holds the stock count, "Quotation" the price, "Ref./lot"
    # the reference, "Delivery" the incoterm+location. These let a multi-tab circular
    # route to the structured (all-sheet) path instead of the first-sheet-only LLM path.
    "warehouse position": "_quantity", "position": "_quantity",
    "quotation": "_price_value", "quote": "_price_value",
    "desk quote": "_price_value", "ref lot": "product_reference", "ref": "product_reference",
    "delivery": "incoterm", "desk notes": "notes",
    # Grey Inlet dock/cellar list (Samuel 2026-11-24): different column names for
    # the same fields. "Balance" = stock; "Article" = product; "Presentation" =
    # pack; "Net EUR per case" states currency AND basis in the heading itself.
    "balance": "_quantity", "article": "product_name", "presentation": "units_per_case",
    "net eur per case": "price_per_case", "net price per case": "price_per_case",
    "eur per case": "price_per_case", "eur/case": "price_per_case",
    "net eur per bottle": "price_per_unit", "eur per bottle": "price_per_unit",
    "supplier ref": "product_reference", "desk / warehouse remarks": "notes",
    "warehouse remarks": "notes",
    # Ashmere Quay "Quayside slips" + "Late circular" dialect (Samuel 2026-11-27):
    # clean columnar tables with prose headers — Slip · Article on the quay · How
    # it is assembled · What the warehouse says · Desk figure · Handover ·
    # Annotations  /  Ticket · Description · Presentation · Balance · Quotation ·
    # Place and terms · Commercial notes. Recognising these routes ALL Ashmere
    # sheets to the deterministic structured path (no LLM), so the lot-conflict,
    # expiry, co-equal-split and reference rules apply uniformly. "Slip"/"Ticket"
    # are the row's own reference.
    "slip": "product_reference", "ticket": "product_reference",
    "article on the quay": "product_name",
    "how it is assembled": "units_per_case",
    "what the warehouse says": "_quantity", "desk figure": "_price_value",
    "handover": "incoterm", "annotations": "notes",
    "place and terms": "incoterm", "commercial notes": "notes",
    # Stock stated in BOTTLES ("Q in btls" = available bottles, not cases).
    "q in btls": "_qty_bottles", "qty in btls": "_qty_bottles",
    "quantity in btls": "_qty_bottles", "q btls": "_qty_bottles",
    "qty btls": "_qty_bottles", "q in bottles": "_qty_bottles",
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
    "upc": "ean_code", "upc code": "ean_code", "upc number": "ean_code",
    # A bottle-vs-case EAN pair: the bottle EAN is the item barcode, the case
    # EAN is a separate carton barcode (Samuel 2026-11-22: LAST_CHANCE).
    "ean bottle": "ean_code", "bottle ean": "ean_code",
    "ean case": "ean_case", "ean carton": "ean_case", "case ean": "ean_case",
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

# Known multi-word delivery places. A location read out of a flattened row is ONE
# token UNLESS it is one of these pairs, so a capitalised NOTE word that follows
# the place ("EXW Antwerp Superseded price…", Ashmere B04; "EXW Ghent Entire
# parcel…", B05; "EXW Ghent Stock: 7 pallets…", B08) is never swallowed into the
# location (Samuel 2026-11-27).
_MULTIWORD_PLACES = {
    ("hong", "kong"), ("new", "york"), ("le", "havre"), ("los", "angeles"),
    ("san", "francisco"), ("cape", "town"), ("abu", "dhabi"), ("kuala", "lumpur"),
    ("port", "said"), ("ho", "chi"), ("rio", "de"),
}


def _trim_location(value: str) -> str:
    """Clean a location: the first place token, plus a second only when the pair is
    a known multi-word city. Stops a capitalised NOTE word that follows the place in
    a flattened row from reading as part of the location (Samuel 2026-11-27 Ashmere
    B04/B05/B08/B09/B10 — "Antwerp Superseded", "Ghent Entire", "Ghent Stock",
    "Breda This", "Antwerp Price")."""
    toks = str(value or "").strip(" .,-").split()
    if not toks:
        return ""
    if len(toks) >= 2 and (toks[0].lower(), toks[1].lower()) in _MULTIWORD_PLACES:
        return f"{toks[0]} {toks[1]}"
    return toks[0]

# Fields that make a row a real product line (need at least one). "_price_value"
# is the bare "Price" column resolved later against its "Price basis".
_PRICE_FIELDS = {"price_per_unit", "price_per_case", "_price_value"}


# A free-text "stock circular" dialect (Samuel 2026-11-26 Estuary EL-925 "Desk
# parcels": Ticket · What is on the floor · Presentation · Warehouse tally ·
# Commercial figure · Handover · Desk remarks). These names are distinctive enough
# not to collide with a normal columnar sheet, so they are OFF by default and only
# enabled as a deterministic COMPLETENESS BACKSTOP for a sheet the LLM read (used
# to recover a row the LLM dropped — never to replace a good LLM row).
_FREETEXT_SYNONYMS = {
    "ticket": "product_reference",
    "what is on the floor": "product_name",
    "presentation": "_size",
    "warehouse tally": "_quantity",
    "commercial figure": "_price_value",
    "handover": "incoterm",
    "desk remarks": "notes",
}


def _match_header(norm: str, extra: Optional[Dict[str, str]] = None) -> Optional[str]:
    if not norm:
        return None
    if extra and norm in extra:
        return extra[norm]
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


def _detect_header_row(ws, scan: int = 12,
                       extra_synonyms: Optional[Dict[str, str]] = None
                       ) -> Optional[Tuple[int, Dict[int, str]]]:
    """Find the header row (most recognised columns) and its column→field map.
    Returns None when no row recognises enough labelled columns."""
    best: Optional[Tuple[int, Dict[int, str]]] = None
    best_score = 0
    for r in range(1, min(scan, ws.max_row) + 1):
        colmap: Dict[int, str] = {}
        seen_fields = set()
        for c in range(1, ws.max_column + 1):
            field = _match_header(_norm(ws.cell(row=r, column=c).value), extra_synonyms)
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


# Month names (English + French) for a circular whose title row writes the date
# in words ("24 November 2026", "24 novembre 2026").
_MONTHS = {
    "jan": 1, "january": 1, "janv": 1, "janvier": 1,
    "feb": 2, "february": 2, "fev": 2, "fevr": 2, "fevrier": 2, "février": 2, "févr": 2,
    "mar": 3, "march": 3, "mars": 3,
    "apr": 4, "april": 4, "avr": 4, "avril": 4,
    "may": 5, "mai": 5,
    "jun": 6, "june": 6, "juin": 6,
    "jul": 7, "july": 7, "juil": 7, "juillet": 7,
    "aug": 8, "august": 8, "aout": 8, "août": 8,
    "sep": 9, "sept": 9, "september": 9, "septembre": 9,
    "oct": 10, "october": 10, "octobre": 10,
    "nov": 11, "november": 11, "novembre": 11,
    "dec": 12, "december": 12, "decembre": 12, "décembre": 12,
}
_HD_ISO_RE = re.compile(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b")
_HD_DMY_RE = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](20\d{2}|\d{2})\b")
_HD_WORD_DMY_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-zûéèçîôàâäïüö]+)\.?,?\s+(20\d{2})\b", re.I)
_HD_WORD_MDY_RE = re.compile(
    r"\b([A-Za-zûéèçîôàâäïüö]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d{2})\b", re.I)


def _header_date(raw: Any) -> Optional[str]:
    """Recognise an offer/circular date in a title-row cell → ISO string, or None.

    Conservative on purpose: a real date object, or a string that fully states
    day + month + year (a 4-digit year, or a spelled-out month). A bare number,
    a pack spec like "12/70/40", or a lone year is NOT a date — those must never
    be mistaken for the offer date (Samuel 2026-11-24, Grey Inlet dated header)."""
    if raw is None or raw == "":
        return None
    # A date / datetime / Timestamp cell (openpyxl data_only) — but not a plain number.
    if (not isinstance(raw, (int, float, bool))
            and hasattr(raw, "year") and hasattr(raw, "month") and hasattr(raw, "day")):
        try:
            return f"{int(raw.year):04d}-{int(raw.month):02d}-{int(raw.day):02d}"
        except Exception:
            return None
    s = str(raw).strip()
    if not s:
        return None
    m = _HD_ISO_RE.search(s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    m = _HD_WORD_DMY_RE.search(s)
    if m:
        mo = _MONTHS.get(m.group(2).lower().rstrip("."))
        if mo and 1 <= int(m.group(1)) <= 31:
            return f"{int(m.group(3)):04d}-{mo:02d}-{int(m.group(1)):02d}"
    m = _HD_WORD_MDY_RE.search(s)
    if m:
        mo = _MONTHS.get(m.group(1).lower().rstrip("."))
        if mo and 1 <= int(m.group(2)) <= 31:
            return f"{int(m.group(3)):04d}-{mo:02d}-{int(m.group(2)):02d}"
    m = _HD_DMY_RE.search(s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        # European day/month order; a pack spec "12/70/40" fails the month range.
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    return None


def _scan_header_offer_date(ws, header_row: int) -> Optional[str]:
    """Find the offer date stated in the rows ABOVE the table header — a stock
    circular states its date once, in a title row ("Northbank Trading BV ·
    Stock circular · 24 November 2026"), not in a per-row column. Prefer a cell
    sitting next to a date/offer label; never take a "valid until" / expiry date
    as the offer date. Returns an ISO string, or None."""
    labelled = fallback = None
    maxc = ws.max_column or 1
    for pr in range(1, header_row):
        for pc in range(1, maxc + 1):
            iso = _header_date(ws.cell(row=pr, column=pc).value)
            if not iso:
                continue
            # Context = this cell + the one to its left (a "Date:" label often
            # sits in the adjacent column).
            ctx = " ".join(
                str(ws.cell(row=pr, column=cc).value or "")
                for cc in range(max(1, pc - 1), pc + 1)).lower()
            if re.search(r"valid|expir|\buntil\b|\btill\b|best\s*before", ctx):
                continue  # an expiry / validity date is not the offer date
            if re.search(r"\bdate\b|\boffer\b|\bdated\b|\bas\s+of\b|circular|"
                         r"price\s*list|stock\s*list", ctx):
                labelled = labelled or iso
            else:
                fallback = fallback or iso
    return labelled or fallback


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


def _parse_amount(raw: Any) -> Tuple[Optional[float], bool]:
    """Parse a PRICE/amount under either number convention, resolving the comma
    so a European decimal comma is not read as a thousands separator. Returns
    (value, ambiguous). The two conventions are told apart by the digits AFTER
    the separator, never by guessing a magnitude (Samuel 2026-11-22: check the
    amount against the wording, and flag — don't guess — when it is ambiguous):

      '11,40'  -> (11.40, False)   comma + 2 trailing digits  = decimal comma
      '7,05'   -> (7.05,  False)
      '2,882'  -> (2882.0, False)  comma + groups of 3 digits = thousands
      '1,140'  -> (1140.0, False)
      '1.234,56' -> (1234.56, False)  mixed: last separator is the decimal
      '1,234.56' -> (1234.56, False)
      '1,2345' -> (12345.0, True)   a comma that is neither pattern -> ambiguous
    """
    if raw is None or raw == "":
        return (None, False)
    if isinstance(raw, (int, float)):
        return (float(raw), False)
    m = re.search(r"\d[\d.,]*", str(raw))
    if not m:
        return (None, False)
    t = m.group(0)
    has_dot, has_comma = "." in t, "," in t
    if has_dot and has_comma:
        # The LAST-occurring separator is the decimal point.
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif has_comma:
        parts = t.split(",")
        if len(parts) == 2 and 1 <= len(parts[1]) <= 2:
            t = parts[0] + "." + parts[1]            # 11,40 -> 11.40
        elif all(len(p) == 3 for p in parts[1:]):
            t = t.replace(",", "")                    # 2,882 / 1,140 -> thousands
        else:
            try:
                return (float(t.replace(",", "")), True)   # neither -> ambiguous
            except ValueError:
                return (None, True)
    try:
        return (float(t), False)
    except ValueError:
        return (None, False)


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
    r'(case|cases|cs|carton|cartons|ctn|bottle|bottles|btl|unit|units|'
    r'piece|pieces|pcs|pc|each|can|cans|keg|kegs|set|sets)\b', re.IGNORECASE)
_BARE_NUM_RE = re.compile(r'(\d[\d,]*(?:\.\d+)?)')
# A cell that deliberately withholds the basis ("basis not stated", "per bottle
# or per case?") — never guess a basis; retain the amount as-quoted + review.
_AMBIG_BASIS_RE = re.compile(
    r'basis\s*(?:not\s*stated|unclear|unknown|unspecified|\?)|'
    r'per\s+\w+\s+or\s+per\s+\w+|not\s*stated[^.]*basis', re.IGNORECASE)
_CASE_WORDS = {"case", "cases", "cs", "carton", "cartons", "ctn"}


def _detect_currency(s: str) -> Optional[str]:
    m = _CUR_RE.search(s or "")
    return _CUR_WORDS.get(m.group(1).lower()) if m else None


def _fmt_currency(number_format: Optional[str]) -> Optional[str]:
    """ISO currency UNAMBIGUOUSLY identified by a cell's NUMBER FORMAT, else None.
    Only two signals confirm a currency: an explicit `[$XXX-...]` ISO code in the
    mask, or the `€` symbol (used by exactly one currency). A bare `$`, `£` or `¥`
    glyph is shared by many currencies ($ = USD/CAD/AUD/HKD/SGD/…, ¥ = JPY/CNY,
    £ = GBP/EGP/LBP/…), so it does NOT identify a currency and returns None — the
    symbol is still available for display via _fmt_symbol, but the currency stays
    unconfirmed and the row is kept out of price comparisons. (Samuel 2026-11-21:
    HAZ's PRICE column is a bare `"$"` accounting mask with no locale/ISO code —
    `$` alone is not evidence of USD.)"""
    f = number_format or ""
    if not f or f.lower() in ("general", "@"):
        return None
    m = re.search(r"\[\$([A-Za-z]{3})", f)   # explicit [$USD-409] currency code
    if m:
        return m.group(1).upper()
    if "€" in f:                              # euro sign maps to exactly one currency
        return "EUR"
    return None


def _fmt_symbol(number_format: Optional[str]) -> Optional[str]:
    """The bare currency SYMBOL in a number-format mask ($/€/£/¥), for showing an
    amount as-quoted, independent of whether the currency itself is confirmed. A
    `$`-formatted cell with no ISO/locale code yields `$` here (so the record can
    read "$6, as quoted") while _fmt_currency returns None (currency unconfirmed)."""
    f = number_format or ""
    if not f or f.lower() in ("general", "@"):
        return None
    for sym in ("$", "€", "£", "¥"):
        if sym in f:
            return sym
    return None


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
        num, amb = _parse_amount(m.group(1))
        if num is None:
            continue
        if amb:
            out["basis_ambiguous"] = True
        if m.group(2).lower() in _CASE_WORDS:
            out["price_per_case"] = out["price_per_case"] or num
        else:
            out["price_per_unit"] = out["price_per_unit"] or num
        found = True
    if found:
        return out
    bm = _BARE_NUM_RE.search(s)
    if bm:
        _bv, _bamb = _parse_amount(bm.group(1))
        out["price_value"] = _bv
        if _bamb or _AMBIG_BASIS_RE.search(s):
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
    # An explicit per-case count IS a real case pack even when the units are
    # described as "loose" — "24 loose cans / case" is a 24-per-case pack whose
    # cans are loose (vs wrapped sixes, cf. A02), not an item sold singly (Samuel
    # 2026-11-28: Ashmere A01/A03/A04 read 0 units/case and lost their per-unit
    # price). Only a cell with NO per-case count is a non-case / per-piece pack.
    _per_case = bool(re.search(r'(?:/|\bper\b)\s*cases?\b', low))
    _nopack = ["individual", "single", "not case", "n/a"]
    if not _per_case:
        _nopack.append("loose")
    if any(k in low for k in _nopack):
        return None
    # A gift SET / pack sold as one set is the sale unit, not a case: never derive a
    # case price from "3 × 40 g" inside it (Samuel 2026-11-24: Alder S02 "Gift set of
    # 3 × 40 g" at EUR 21/set became EUR 2,520/case = 21 × 120). The price is per set.
    if re.search(r"\b(?:gift\s*set|set\s+of)\b", low):
        return None
    # A keg / drum / cask sold as a single vessel is NOT a multi-unit case: its
    # litre size must never be read as a pack count ("20 L keg" is one 20-litre
    # keg, not a 20-pack — Samuel 2026-11-22 QA Blue Fjord).
    if re.search(r"\bkegs?\b|\bdrums?\b|\bcasks?\b", low) and not re.search(r"[x×*]", low):
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
    # Strip bottle VOLUME measurements ("700 ml", "33cl", "1.75L") so a bottle
    # SIZE is never counted as a pack unit — the old first-integer fallback read
    # "6 x 700 ml bottles/case" as 700 and "4 x 6 x 330ml" as 4 (Samuel
    # 2026-11-22 QA Raven Creek 700/case, Iron Gate 4×6 format).
    counts = re.sub(r'\d+(?:[.,]\d+)?\s*(?:ml|cl|l|lt|litre|liter|cc|kg|g|gr|gram)s?\b',
                    ' ', s, flags=re.IGNORECASE)
    # "N (wrapped) sixes / six-packs" is N × 6 cans — so a case of "4 wrapped sixes"
    # is 24 cans, and the per-unit price is per CAN, not per six-pack (Samuel
    # 2026-11-24: Alder A04 EUR 2.16/unit must read EUR 0.36/can).
    counts = re.sub(r'(\d+)\s*(?:wrapped\s*)?(?:sixe?s|six[\s-]?packs?|6[\s-]?packs?)',
                    lambda m: f"{m.group(1)} x 6", counts, flags=re.IGNORECASE)
    # Dimensional pack "N x M [x K]" → multiply the count factors (4 × 6 = 24),
    # once the per-bottle volume has been removed.
    if re.search(r'\d+\s*[x×*]\s*\d+', counts):
        factors = [int(n) for n in re.findall(r'(?<![\d.])\d+(?![\d.])', counts)]
        if factors:
            p = 1
            for f in factors:
                p *= f
            return float(p)
    m = re.search(r'=\s*(\d[\d,]*)', counts)
    if m:
        return _num_thousands(m.group(1))
    m = re.search(r'(\d[\d,]*)\s*[A-Za-z ]*?(?:/|per)\s*case', counts, re.IGNORECASE)
    if m:
        return _num_thousands(m.group(1))
    m = re.search(r'(\d[\d,]*)', counts)
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
    # Drop a lot CODE so its trailing digits are never read as the quantity
    # (Samuel 2026-11-24: Alder C01 "Lot SN71: 19 cs + 5 loose" was read as 71).
    outside = re.sub(r'\blot\s+[A-Za-z0-9/\-]+\s*:?', ' ', outside, flags=re.IGNORECASE)
    outside = re.sub(r'\b[A-Za-z]{2,}\d+[A-Za-z0-9]*\b', ' ', outside)  # bare codes (SN71, VN39)
    # An explicit "Total N cases" wins over a split line ("72 cs …; 28 cs …";
    # Alder A03 "Total 100 cases on floor").
    mt = re.search(r'total\s+(\d[\d,]*)\s*cases?', outside, re.IGNORECASE)
    if mt and _num_thousands(mt.group(1)):
        return (_num_thousands(mt.group(1)), "cases")
    # "N pallets of/×/x M cases" is a real case count (N×M), not N pallets — the
    # pallet figure alone loses the per-pallet case size (Samuel 2026-11-20:
    # Bunnahabhain "7 pallets of 32 cases" = 224; 2026-11-24 Alder C02
    # "8 pallets × 42 cases" = 336).
    m = re.search(r'(\d[\d,]*)\s*pallets?\s*(?:of|[x×*])\s*(\d[\d,]*)\s*cases?',
                  outside, re.IGNORECASE)
    if m:
        _n = _num_thousands(m.group(1)); _m = _num_thousands(m.group(2))
        if _n and _m:
            return (_n * _m, "cases")
    # A date-split availability ("72 cs BBD …; 28 cs BBD …") with no explicit total
    # is the SUM of its parts (Alder A03 = 100). Only when several "N cs/cases"
    # counts appear and no single total is given.
    _parts = re.findall(r'(\d[\d,]*)\s*(?:cs|cases?)\b', outside, re.IGNORECASE)
    if len(_parts) >= 2:
        _tot = sum(_num_thousands(p) or 0 for p in _parts)
        if _tot:
            return (float(_tot), "cases")
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


def scan_workbook_offer_date(file_path: str) -> Optional[str]:
    """Offer date from a circular's TITLE rows, for the LLM-fallback path (which
    never runs the structured column reader). Opens the workbook and scans the
    top rows of the first sheet with the SAME date recogniser and valid/expiry
    exclusion the structured reader uses, so both readers read the dated header
    the same way (Samuel 2026-11-25, parity). Returns an ISO string, or None."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(file_path, data_only=True, read_only=False)
    except Exception:
        return None
    try:
        ws = wb.worksheets[0] if wb.worksheets else None
        if ws is None or not ws.max_row:
            return None
        upto = min(ws.max_row + 1, 13)   # the first ~12 rows (the title block)
        return _scan_header_offer_date(ws, upto)
    except Exception:
        return None
    finally:
        try:
            wb.close()
        except Exception:
            pass


_SHEET_PRICE_RE = re.compile(
    r"(?:EUR|USD|GBP|CHF|HKD|SGD|CAD|AUD|\$|€|£)\s*\d"
    r"|\d[\d.,]*\s*(?:per\s+(?:case|bottle|btl|can|piece|pc|set|pack|gift\s*set)"
    r"|/\s*(?:case|btl|bottle|cs))",
    re.IGNORECASE)


def _sheet_product_row_count(ws, limit: int = 80) -> int:
    """Count data rows that carry a price token. Used to tell a dropped PRODUCT
    sheet (which must be rescued, not skipped) from a notes/legend/empty tab
    (safe to skip): the former has several priced rows, the latter has none
    (Samuel 2026-11-25: the Estuary 'Desk parcels' sheet, 24 priced rows with an
    unfamiliar header, was silently dropped and the checker never noticed)."""
    n = 0
    try:
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i >= limit:
                break
            line = " ".join(str(c) for c in row if c is not None)
            if _SHEET_PRICE_RE.search(line):
                n += 1
    except Exception:
        return 0
    return n


def extract_structured_products(file_path: str,
                                extra_synonyms: Optional[Dict[str, str]] = None,
                                only_sheets: Optional[set] = None
                                ) -> Optional[List[Dict[str, Any]]]:
    """Deterministically extract products from every well-structured sheet.
    Returns a list of raw product dicts (pre-clean), or None when no sheet has a
    recognisable header, OR when a sheet whose header we cannot read still holds
    several priced rows — in that case we escalate the WHOLE workbook to the LLM
    fallback (which reads every sheet) rather than return a partial result that
    silently loses the unreadable sheet (Samuel 2026-11-25).

    extra_synonyms enables an extra header dialect (e.g. the free-text stock
    circular) for the completeness backstop; only_sheets restricts to named sheets.
    Both default off, so the normal structured path is unchanged."""
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
        if only_sheets is not None and ws.title not in only_sheets:
            continue
        detected = _detect_header_row(ws, extra_synonyms=extra_synonyms)
        if not detected:
            # An unreadable PRODUCT sheet is NOT dropped silently here: the caller
            # checks unreadable_product_sheet_names() and routes those sheets to the
            # LLM fallback, then merges — so a recognised sheet keeps its clean
            # deterministic extraction (Samuel 2026-11-25: Estuary N01 Copper Fen
            # EUR 78 was LOST when the whole workbook was force-escalated).
            logger.info(f"[xlsx_structured] sheet {ws.title!r}: no recognisable header — "
                        f"left for the fallback (see unreadable_product_sheet_names)")
            continue
        header_row, colmap = detected
        recognised_sheets += 1
        logger.info(f"[xlsx_structured] sheet {ws.title!r}: header at row {header_row}, "
                    f"{len(colmap)} mapped columns: {sorted(set(colmap.values()))}")

        # A price column header that NAMES a currency ("Net EUR per case") states
        # the currency explicitly — the "basis in a column heading" Samuel wants
        # honoured (2026-11-24). Apply it to a priced row that has no currency of
        # its own. A bare "$" in a heading is ambiguous and is NOT taken as USD.
        sheet_price_currency = None
        for _hc, _hf in colmap.items():
            if _hf in ("price_per_case", "price_per_unit", "_price_value"):
                _hcm = re.search(r"\b(EUR|USD|GBP|CHF|HKD|SGD)\b",
                                 str(ws.cell(row=header_row, column=_hc).value or ""),
                                 re.IGNORECASE)
                if _hcm:
                    sheet_price_currency = _hcm.group(1).upper()
                    break

        # Supplier from a title row ABOVE the header ("Larch Pier Trading BV |
        # Stock circular LP-207") — the structured sheet has no supplier column
        # and the sender is not trusted as the supplier (Samuel 2026-11-20).
        sheet_supplier = None
        for _pr in range(1, header_row):
            for _pc in range(1, (ws.max_column or 1) + 1):
                _v = ws.cell(row=_pr, column=_pc).value
                if isinstance(_v, str) and _v.strip():
                    _cand = _v.split("|")[0].strip()
                    if (re.search(r"[A-Za-z]", _cand) and len(_cand) >= 3
                            and not re.match(r"^\d", _cand)
                            and not re.match(r"^(stock|price|availability|offer|date|contact)\b",
                                              _cand, re.IGNORECASE)):
                        sheet_supplier = _cand
                        break
            if sheet_supplier:
                break

        # Offer date from the circular's title row (no per-row date column) — a
        # stock/price list states its date once, above the table (Grey Inlet
        # "… 24 November 2026"; Samuel 2026-11-24 "offer date unknown despite a
        # dated header"). Applied as each row's offer_date below when the row has
        # none of its own.
        sheet_offer_date = _scan_header_offer_date(ws, header_row)

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
            price_fmt_currency: Optional[str] = None   # CONFIRMED currency from the cell mask (€ / explicit ISO code)
            price_fmt_symbol: Optional[str] = None      # bare display symbol from the cell mask ($/£/¥) when unconfirmed
            size_raw: Any = None
            any_value = False
            nobox = False
            innerbox_pieces: Optional[float] = None   # pieces per inner box (Pack cell)
            qty_innerbox_count: Optional[float] = None  # "N inner boxes" (Available cell)
            # The full text of the row across ALL columns — including ones we do
            # not map to a field (e.g. a free "Delivery / notes" cell). Used to
            # catch a price-basis ambiguity note that lives outside the price cell
            # (Samuel 2026-11-20: Larch Pier Plymouth).
            row_text = " ".join(
                str(ws.cell(row=r, column=c).value).strip()
                for c in range(1, ws.max_column + 1)
                if ws.cell(row=r, column=c).value not in (None, ""))

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
                    price_fmt_currency = price_fmt_currency or _fmt_currency(cell.number_format)
                    price_fmt_symbol = price_fmt_symbol or _fmt_symbol(cell.number_format)
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
                        # Basis is known (the cell said /can, /case, /btl) but the
                        # cell named NO currency — read it from the number-format
                        # mask, else remember to flag it, never guess EUR (Samuel
                        # 2026-11-24: Alder A01 "0.03 / can", currency awaiting).
                        if not prod.get("currency"):
                            _fc = price_fmt_currency or _fmt_currency(cell.number_format)
                            if _fc:
                                prod["currency"] = _fc
                            else:
                                _fs = price_fmt_symbol or _fmt_symbol(cell.number_format)
                                if _fs:
                                    prod.setdefault("_quote_symbol", _fs)
                                prod["_price_no_currency"] = True
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
                    # The BASIS is known (the header says /btl or /cs) but the cell
                    # may name no currency. Read it from the cell's number-format
                    # mask (€ / explicit ISO code → confirmed; a bare $/£/¥ → symbol
                    # only), and if still none, remember to flag it — never invent
                    # one (Samuel 2026-11-21 HAZ rule; 2026-11-22 Spirits Online
                    # "Price / btl" list states no currency anywhere).
                    if prod[field] is not None and not prod.get("currency"):
                        _fc = _fmt_currency(cell.number_format)
                        if _fc:
                            prod["currency"] = _fc
                        else:
                            _fs = _fmt_symbol(cell.number_format)
                            if _fs:
                                prod.setdefault("_quote_symbol", _fs)
                            prod["_price_no_currency"] = True
                elif field == "units_per_case":
                    prod[field] = _parse_pack(val)
                    # Keep the verbatim pack/format cell ("24 x 330ml cans",
                    # "4 x 6 x 330ml cans", "20L keg") so two distinct formats of
                    # one product stay distinct in the table and are not collapsed
                    # as duplicates (Samuel 2026-11-22 Iron Gate). A bare count
                    # ("6") is not a format string — skip it.
                    if not prod.get("packaging_raw"):
                        _praw = str(val).strip()
                        if re.search(r"[A-Za-z×x]", _praw):
                            prod["packaging_raw"] = _praw
                    # The pack cell also states the volume ("24 × 500 ml cans",
                    # "6 × 1 L") — fill unit_volume_ml so the peer-group key carries
                    # the size and the SKU can match another supplier's (Alder, 2026-11-24).
                    if not prod.get("unit_volume_ml"):
                        _pv = _parse_size_ml(val)
                        if _pv:
                            prod["unit_volume_ml"] = _pv
                    # An inner-box pack ("8 pieces per inner box") means the item
                    # is sold PER PIECE, not per case: keep the pieces-per-box as
                    # the pack, remember it so the quantity ("7 inner boxes") can
                    # be expanded to pieces, and surface "inner box" in
                    # packaging_raw so the per-piece guard downstream fires and no
                    # case price is fabricated (Samuel 2026-11-20: Acqua di Parma).
                    # "N pieces per inner [box/carton]" — read the count directly
                    # (_parse_pack returns None for a pieces-only pack). Also the
                    # "inner box × N pieces" form (Alder S01 "8 pieces per inner").
                    # "N pieces per inner box" AND the reversed "inner box(es) of N
                    # candles/pieces" (Ashmere A07 "Inner boxes of 9 candles"), which
                    # the per/slash form missed — the 9 is the box size.
                    _ibm = re.search(
                        r"(\d[\d,]*)\s*(?:pieces?|pcs?|candles?|units?)\s*(?:/|per)\s*inner"
                        r"|inner\s*box(?:es)?\s*of\s*(\d[\d,]*)",
                        str(val), re.IGNORECASE)
                    if _ibm:
                        innerbox_pieces = _num_thousands(_ibm.group(1) or _ibm.group(2))
                        if not prod.get("packaging_raw"):
                            prod["packaging_raw"] = str(val).strip()
                    elif re.search(r"per\s*inner\s*box|inner\s*box(?:es)?\s*[x×]",
                                   str(val), re.IGNORECASE):
                        innerbox_pieces = prod[field]
                        if not prod.get("packaging_raw"):
                            prod["packaging_raw"] = str(val).strip()
                elif field == "moq_cases":
                    # MOQ may carry its own inline unit ("1 pallet", "1 FTL").
                    moq_val, moq_unit_inline = _split_num_unit(val)
                    # Keep the verbatim source wording ("2 inner boxes", "24
                    # bottles") so the export shows the MOQ exactly as stated and
                    # matches the WhatsApp copy of the same product.
                    moq_raw_cell = str(val).strip()
                elif field == "_quantity":
                    qty_raw = str(val)
                    # "N inner boxes", "N inner cartons", or just "N inners"
                    # (Alder S01 "3 inners + 2 single candles").
                    _ibq = re.search(r"(\d[\d,]*)\s*inner(?:\s*box(?:es)?|\s*cartons?|s)?\b",
                                     qty_raw, re.IGNORECASE)
                    if _ibq:
                        qty_innerbox_count = _num_thousands(_ibq.group(1))
                    _qv, _qu = _parse_quantity_cell(val)
                    # A zero "Quantity" cell is a placeholder, not a real stock of
                    # 0 — don't let it overwrite a bottle count already read from a
                    # "Q in btls" column (Samuel 2026-11-22: LAST_CHANCE).
                    if _qv not in (None, 0, 0.0) and qty_val in (None, 0, 0.0):
                        qty_val = _qv
                        if _qu:
                            qty_unit = _qu
                    elif _qu and not qty_unit:
                        qty_unit = _qu
                elif field == "_qty_bottles":
                    # Stock stated in BOTTLES ("Q in btls" = 432) — keep it in the
                    # bottle unit, never as cases (comma = thousands: "1,440").
                    _qb, _ = _split_num_unit(val)
                    if _qb not in (None, 0, 0.0):
                        qty_val = _qb
                        qty_unit = "bottles"
                elif field == "ean_case":
                    _ec = (str(int(val)) if isinstance(val, float) and val.is_integer()
                           else str(val).strip())
                    if any(ch.isdigit() for ch in _ec):
                        prod["ean_case"] = _ec
                elif field == "quantity_unit":
                    qty_unit = str(val).strip()
                elif field in ("offer_date", "valid_until", "best_before_date"):
                    prod[field] = _iso_date(val)
                    if field == "offer_date" and prod[field]:
                        # Read deterministically from a labelled column — a real
                        # document read, not an LLM guess, so the processor trusts
                        # it (see _used_structured / _offer_date_trusted) instead
                        # of dropping it as an uncorroborated per-row invention.
                        prod["_offer_date_trusted"] = True
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
            # A bare price with no basis token is NOT assumed per-bottle when the
            # row says the basis is unconfirmed somewhere (a "Delivery / notes"
            # cell: "Awaiting confirmation whether the quote is per bottle or per
            # case"). Treat it exactly like an in-cell ambiguity: keep the amount,
            # leave bottle AND case prices blank, keep the quantity, flag for
            # review — the same rule text/WhatsApp offers follow (Samuel
            # 2026-11-20: Larch Pier Plymouth USD 64.50).
            # A currency named in the price COLUMN HEADING fills a priced row that
            # states none of its own (Grey Inlet "Net EUR per case" → EUR).
            if (sheet_price_currency and not prod.get("currency")
                    and (prod.get("price_per_case") is not None
                         or prod.get("price_per_unit") is not None)):
                prod["currency"] = sheet_price_currency

            # An explicit basis + currency stated in a NOTE on the SAME row resolves
            # a bare price cell — read the supplier's note before flagging (Alder
            # W03: quotation "5.64", desk note "EUR 0.47 per can × 12 cans … Excel
            # quotation cell is in EUR/case" → EUR 5.64/case, green).
            if (price_value is not None
                    and prod.get("price_per_case") is None
                    and prod.get("price_per_unit") is None):
                _mnc = re.search(r"\b(EUR|USD|GBP|CHF|HKD|SGD)\s*/\s*(?:case|cases|cs)\b"
                                 r"|in\s+(EUR|USD|GBP|CHF|HKD|SGD)\s*/\s*(?:case|cs)",
                                 row_text, re.IGNORECASE)
                _mnu = re.search(r"\b(EUR|USD|GBP|CHF|HKD|SGD)\s*/\s*(?:bottle|btl|can|unit|piece|pc)\b",
                                 row_text, re.IGNORECASE)
                if _mnc:
                    prod["price_per_case"] = price_value
                    if not prod.get("currency"):
                        prod["currency"] = (_mnc.group(1) or _mnc.group(2)).upper()
                    price_value = None
                elif _mnu:
                    prod["price_per_unit"] = price_value
                    if not prod.get("currency"):
                        prod["currency"] = _mnu.group(1).upper()
                    price_value = None
            if price_value is not None and not basis_ambiguous and _AMBIG_BASIS_RE.search(row_text):
                basis_ambiguous = True
            # A bare price with NO currency AND no basis token is doubly
            # unsupported: a `$`-formatted "PRICE" cell carrying only a number,
            # no "per case/bottle" and no currency word. Never guess it into EUR
            # per-bottle (Samuel 2026-11-21: HAZ's 735-row list showed "€6 per
            # bottle, 360 cases" with no source evidence for EUR, the basis, or
            # the stock unit). Read the currency from the cell's $/€/£ number
            # format if present, retain the figure as-quoted, blank the
            # comparable, and flag for review — the same treatment an explicit
            # "basis not stated" cell gets.
            if (price_value is not None and not basis_ambiguous and not price_basis
                    and prod.get("price_per_unit") is None
                    and prod.get("price_per_case") is None
                    and not prod.get("currency")):
                basis_ambiguous = True
                if price_fmt_currency:
                    # An unambiguous mask (€ or an explicit [$XXX] ISO code).
                    prod["currency"] = price_fmt_currency
                    prod["_basis_cur_note"] = (
                        f"currency read from the {price_fmt_currency} cell format "
                        f"(none stated in words); stock unit not stated")
                elif price_fmt_symbol:
                    # A shared symbol ($/£/¥) with no ISO/locale code — NOT a
                    # currency. Keep the symbol for display, leave the currency
                    # field empty (unconfirmed), and stay out of comparisons
                    # (Samuel 2026-11-21: `$` alone is not evidence of USD).
                    prod["_quote_symbol"] = price_fmt_symbol
                    prod["_basis_cur_note"] = (
                        f"the PRICE cell uses a {price_fmt_symbol} accounting "
                        f"format but the workbook states no currency; "
                        f"{price_fmt_symbol} alone does not identify a currency "
                        f"— currency unconfirmed; stock unit not stated")
                else:
                    prod["_basis_cur_note"] = ("currency not stated in the "
                                               "source; stock unit not stated")
            if price_value is not None:
                if basis_ambiguous:
                    # An amount was quoted but the basis was deliberately
                    # withheld ("USD 75.00 — price basis not stated (per bottle
                    # or per case?)"). Never guess it into a per-unit/per-case
                    # field; retain it as-quoted and flag for review (Samuel AQT
                    # 2026-11-17 Beefeater, mirroring the NE-410 rule).
                    cur = (prod.get("currency") or "").strip()
                    _sym = prod.pop("_quote_symbol", "")
                    _extra = prod.pop("_basis_cur_note", None)
                    _extra = f" ({_extra})" if _extra else ""
                    # Render "$6" when only a shared symbol is known, "EUR 6" when
                    # the currency is confirmed, "6" when nothing is known.
                    _amt = (f"{_sym}{price_value:g}" if _sym
                            else f"{cur} {price_value:g}".strip())
                    prod["commercial_conditions"] = (
                        f"As quoted: {_amt}".strip()
                        + " — price basis not stated" + _extra
                        + "; excluded from price comparison.")
                    flags = prod.setdefault("error_flags", [])
                    flags.append("price basis not stated — amount retained "
                                 "as-quoted, excluded from comparison")
                    if not cur:
                        flags.append("currency unconfirmed — cell shows a symbol "
                                     "only, no currency stated; excluded from "
                                     "comparison")
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

            # ── Priced line whose file names NO currency ─────────────────────
            # The basis is known (header /btl or /cs) and the amount is shown as
            # quoted, but the file states no currency (no €, ISO code or currency
            # column). Keep the per-bottle/per-case amount visible, mark the
            # currency unconfirmed, and keep it out of comparison — never guess EUR
            # (Samuel 2026-11-21 HAZ; 2026-11-22 Spirits Online "Price / btl").
            if prod.pop("_price_no_currency", False) and not prod.get("currency"):
                prod.pop("_quote_symbol", None)
                # Set the key explicitly (not merely absent) so downstream cleaners
                # see "currency: None" and keep it, rather than filling a default.
                prod["currency"] = None
                flags = prod.setdefault("error_flags", [])
                _msg = ("currency unconfirmed — the file states no currency; "
                        "amount shown as quoted, excluded from comparison")
                if _msg not in flags:
                    flags.append(_msg)
                _note = ("price shown as quoted; the file states no currency "
                         "(no €, ISO code or currency column) — currency unconfirmed")
                prod["commercial_conditions"] = (
                    (prod.get("commercial_conditions") + " | " + _note)
                    if prod.get("commercial_conditions") else _note)
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

            # ── Supplier from the sheet title row (no supplier column) ─────
            if sheet_supplier and not prod.get("supplier_name"):
                prod["supplier_name"] = sheet_supplier

            # ── Offer date from the circular's title row (no date column) ──
            # Deterministic read of the document header, so flag it trusted: the
            # processor keeps it instead of dropping it as an uncorroborated
            # per-row date (Samuel 2026-11-24 Arden Crest "offer date unknown").
            if sheet_offer_date and not prod.get("offer_date"):
                prod["offer_date"] = sheet_offer_date
                prod["_offer_date_trusted"] = True

            # ── Split a combined "EXW NL" delivery cell into incoterm + location ─
            # A "Delivery" column mapped to incoterm holds the term AND the place
            # ("EXW NL", "FCA Ghent", "DAP Lille" — Alder Quay 2026-11-24). Parse
            # the term off the front and keep the rest as the location (first two
            # words), so the peer-group key matches a supplier that states the same
            # place, instead of carrying "EXW NL" as the whole incoterm.
            _inc = str(prod.get("incoterm") or "").strip()
            _ism = re.match(r"^(EXW|FCA|FOB|CIF|CFR|CPT|CIP|DAP|DDP|DAT|DPU|FAS)\b[\s\-]*(.*)$",
                            _inc, re.IGNORECASE)
            if _ism:
                prod["incoterm"] = _ism.group(1).upper()
                _loc = _ism.group(2).strip(" .-")
                if _loc and not prod.get("location"):
                    prod["location"] = _trim_location(_loc)

            # ── An incoterm cell that is a bare PLACE, not a term ─────────────
            # "Antwerp" with no EXW/FCA/… literal (Ashmere C06 "Place and terms"
            # column) is the warehouse LOCATION, not the incoterm. Move it; when the
            # row says the delivery term is unsettled, leave the incoterm blank and
            # flag (Samuel 2026-11-27 C06 Sable "Warehouse location only. Incoterm
            # has not been agreed").
            _inc_bare = str(prod.get("incoterm") or "").strip()
            if _inc_bare and not _ism and _inc_bare.lower() not in _INCOTERM_LITERALS:
                if not prod.get("location"):
                    prod["location"] = _trim_location(_inc_bare)
                prod["incoterm"] = None
                if re.search(r"incoterm[^.]*?(?:not|has\s*n.?t|have\s*n.?t)\s+"
                             r"(?:been\s+)?agreed|incoterm\s+to\s+be\s+agreed"
                             r"|location\s+only|warehouse\s+location",
                             row_text, re.IGNORECASE):
                    _flb = prod.setdefault("error_flags", [])
                    _mlb = ("delivery term not agreed in the source — incoterm "
                            "unconfirmed; the stored place is the warehouse location only")
                    if _mlb not in _flb:
                        _flb.append(_mlb)
                    prod["needs_manual_review"] = True

            # ── Currency explicitly left UNCONFIRMED in the row text ─────────
            # A note like "Settlement currency awaiting treasury confirmation"
            # (Alder A01, 2026-11-24) means the amount is real but the currency is
            # not: keep it as quoted, drop any guessed currency, blank the EUR
            # comparable, and flag — never show it as EUR.
            # Also covers an ambiguous dollar ("$24 … Dollar denomination to be
            # confirmed: Canada or US" — Grey Inlet P-16 Arden Crest, 2026-11-24)
            # and "settlement currency still with accounts" (P-15). A bare $ is
            # never silently taken as USD and converted to EUR.
            if (re.search(r"currency[^.]*?(awaiting|to\s+be\s+confirmed|not\s+confirmed"
                          r"|unconfirmed|t\.?b\.?c\.?|to\s+follow|with\s+accounts|still\s+with)"
                          r"|awaiting\s+treasury"
                          r"|denomination[^.]*?(?:to\s+be\s+confirmed|confirmed|unclear)"
                          r"|\b(?:canada\s+or\s+us|us\s+or\s+canada|cad\s+or\s+usd|usd\s+or\s+cad)\b"
                          r"|dollar[^.]*?(?:to\s+be\s+confirmed|canada|unclear)",
                          row_text, re.IGNORECASE)
                    and (prod.get("price_per_unit") is not None
                         or prod.get("price_per_case") is not None)):
                prod["currency"] = None
                prod["price_per_unit_eur"] = None
                prod["price_per_case_eur"] = None
                flags = prod.setdefault("error_flags", [])
                _m = ("currency unconfirmed — the source states the currency is "
                      "awaiting confirmation; amount kept as quoted, excluded from comparison")
                if _m not in flags:
                    flags.append(_m)
                _n = "price shown as quoted; currency awaiting confirmation — unconfirmed"
                prod["commercial_conditions"] = (
                    (prod.get("commercial_conditions") + " | " + _n)
                    if prod.get("commercial_conditions") else _n)
                prod["needs_manual_review"] = True

            # ── Location / incoterm from a free "Delivery / notes" cell ─────
            # That column is often unmapped, so parse the first incoterm +
            # location out of the whole-row text (Samuel 2026-11-20: Larch Pier
            # "EXW Breda…", "FOB Hong Kong; minimum…", "EXW Rotterdam. Awaiting…").
            if not prod.get("location") or not prod.get("incoterm"):
                _il = re.search(
                    r"\b(EXW|FCA|FOB|CIF|CFR|CPT|DAP|DDP|DAT|DPU)\b[ ]+"
                    r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)", row_text)
                if _il:
                    prod.setdefault("incoterm", _il.group(1).upper())
                    prod.setdefault("location", _trim_location(_il.group(2)))

            # ── Inner-box, per-piece item: "7 inner boxes" of "8 pieces per
            # inner box" = 56 pieces. Store the piece total, NOT 7 cases, and let
            # the per-piece guard keep the case price empty (Samuel 2026-11-20).
            if innerbox_pieces:
                # The pieces-per-inner-box is NOT a case count — a per-piece item
                # must show EMPTY case fields, with the inner-box size kept in
                # packaging_raw (Samuel 2026-11-23: Lumière/Nishane read "6/case").
                prod["units_per_case"] = None
                if qty_innerbox_count:
                    _pieces = qty_innerbox_count * innerbox_pieces
                    # Add a "+ N single/loose [candles/pieces]" remainder (Alder S01
                    # "3 inners + 2 single candles" = 24 + 2 = 26; Ashmere A07 "4
                    # inners plus 7 single candles" = 36 + 7 = 43 — "plus" as well as
                    # "+", Samuel 2026-11-27).
                    _lm = re.search(r"(?:\+|plus)\s*(\d[\d,]*)\s*(?:single|loose|individual)",
                                    qty_raw or "", re.IGNORECASE)
                    if _lm and _num_thousands(_lm.group(1)):
                        _pieces += _num_thousands(_lm.group(1))
                    prod["quantity_bottles"] = _pieces
                    prod["quantity_unit"] = "pieces"
                    prod.pop("quantity_case", None)
                    qty_val = None

            # ── Same physical lot / alternative delivery option ────────────
            # A row the source marks as the SAME stock shown again under another
            # delivery option ("18 cs (same physical lot) … two delivery options;
            # do not count quantity twice", Samuel 2026-11-23 Belle Rive) must not
            # add its quantity again: keep the price so both options stay visible,
            # but count the stock once.
            if qty_val is not None and re.search(
                    r"same\s+(?:physical\s+)?lot|same\s+\S+\s+parcel|same\s+parcel"
                    r"|do\s+not\s+count\b.*\btwice|do\s+not\s+add\b.*\bstock"
                    r"|two\s+delivery\s+options|price\s+on\s+(?:the\s+)?line\s+above",
                    row_text, re.IGNORECASE):
                _cc_sl = prod.get("commercial_conditions") or ""
                if ("same physical lot" not in _cc_sl.lower()
                        and "counted once" not in _cc_sl.lower()):
                    prod["commercial_conditions"] = (_cc_sl + " | " if _cc_sl else "") + (
                        "same physical lot as another line — stock counted once "
                        "(alternative delivery option)")
                qty_val = None

            # ── Quantity (unit parsed from the cell) ───────────────────────
            _assign_quantity(prod, qty_val, qty_unit)
            # ── "N cases per load" is a truckload CAPACITY, not available stock ──
            # Clear the figure the parser took as stock, keep it as a capacity
            # note, and flag for review — unless the row ALSO states a real
            # "N cases available" (Samuel 2026-11-22 QA Iron Gate; mirrors
            # offer_reconcile.separate_load_capacity for the structured path). The
            # field self-check surfaces the "per load" marker in the note.
            if (re.search(r"per\s+(?:full\s+)?load|load\s+capacity|per\s+(?:full\s+)?truck",
                          row_text, re.IGNORECASE)
                    and not re.search(r"\d[\d,]*\s*(?:cases?|cs)\s*available",
                                      row_text, re.IGNORECASE)):
                _cap = prod.get("quantity_case")
                prod["quantity_case"] = None
                _capn = f"{int(_cap)} cases per load " if _cap else ""
                _lnote = (f"{_capn}(truckload capacity; available stock not stated)").strip()
                _cc0 = prod.get("commercial_conditions") or ""
                if "per load" not in _cc0.lower():
                    prod["commercial_conditions"] = (_cc0 + " | " if _cc0 else "") + _lnote
                prod["needs_manual_review"] = True
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

            # A gift-box trade token in the name ("… GBX" / "… NGBX") is a
            # distinct presentation of the SKU. Record it in gift_box BEFORE
            # name-normalisation strips the token, so a plain vs gift-box pair is
            # labelled, not shown as two identical names (Samuel 2026-11-22:
            # Diplomatico plain vs GBX, Herradura/Pampero GBX).
            _nm_g = prod.get("product_name") or ""
            if re.search(r"(?<![A-Za-z0-9])N?GBX(?![A-Za-z0-9])", _nm_g, re.IGNORECASE):
                prod.setdefault(
                    "gift_box",
                    not re.search(r"(?<![A-Za-z0-9])NGBX(?![A-Za-z0-9])",
                                  _nm_g, re.IGNORECASE))

            # A row the source itself labels a re-statement, not a fresh offer
            # ("…shown again in source footer as stock summary, not another
            # offer") is dropped so it is not stored as a duplicate line (Samuel
            # 2026-11-22 QA Mountain Abbey footer; mirrors drop_summary_rows).
            if re.search(r"not\s+another\s+offer|\brecap\b|no\s+further\s+allocation"
                         r"|stock\s+summary", row_text, re.IGNORECASE):
                continue

            # A buy-side ENQUIRY row inside a sell workbook ("Wanted: 50 kegs …
            # Buying enquiry only. Can anyone supply?" — Alder A07, 2026-11-24) is
            # a request, not a sellable offer: drop it so it never enters stock,
            # Best Prices or the benchmark. Mirrors the text-offer RFQ drop.
            try:
                from core.inquiry import is_strong_buy_line
                if is_strong_buy_line(row_text):
                    logger.info(f"[xlsx_structured] sheet {ws.title!r} row {r}: "
                                f"buy-side enquiry — not stored as an offer")
                    continue
            except Exception:
                pass

            # ── GREEN-TRUST GUARDRAIL ───────────────────────────────────────
            # A buyer must be able to quote a GREEN row without reopening the
            # source. So any row whose price/quantity we cannot confidently
            # reconcile is flagged YELLOW, never left green (Samuel 2026-11-24:
            # he is happy with yellow when something is unclear; a wrong green is
            # the failure). These guards are deterministic and err toward review.
            def _flag(msg: str):
                prod["needs_manual_review"] = True
                _f = prod.setdefault("error_flags", [])
                if msg not in _f:
                    _f.append(msg)

            # (a0) Stock unit NOT labelled by the source → the count stands but the
            #      unit is unknown; never silently assume "cases" (Alder S03
            #      "Balance: 360 … Warehouse did not label quantity unit").
            if re.search(r"did\s+not\s+label|unit\s+not\s+(?:labelled|labeled|stated|given)"
                         r"|quantity\s+unit\s+(?:unknown|unclear|not\s+stated|missing)"
                         r"|no\s+quantity\s+unit|unlabel"
                         # "Count supplied without a unit. Warehouse cannot confirm
                         # whether 480 refers to bars or cartons" (Ashmere A09).
                         r"|without\s+a\s+unit|cannot\s+confirm\s+whether\s+\d",
                         row_text, re.IGNORECASE):
                _n = (prod.get("quantity_case") or prod.get("quantity_bottles")
                      or prod.get("quantity_pallets"))
                if _n:
                    prod["quantity_unit"] = "units (not stated)"
                    _flag(f"stock unit not stated by the source — {int(_n)} units, "
                          f"unit unconfirmed (price is per piece)")

            # (a1) A multi-pack-of-packs: say which unit the price is for (Alder A04
            #      "4 wrapped sixes" — EUR 2.16 is per 6-pack, EUR 0.36 per can).
            # The per-unit price now reads per CAN (units_per_case is total cans).
            # Surface the six-pack price too, so both are explicit (Samuel's A04).
            _praw = str(prod.get("packaging_raw") or "") + " " + row_text
            _mp = re.search(r"(\d+)\s*(?:wrapped\s*)?(?:six(?:es)?|six[\s-]?packs?|6[\s-]?packs?)",
                            _praw, re.IGNORECASE)
            # Per-six-pack amount quoted, no per-case price derived yet: a case holds
            # N six-packs, so the case price is amount × N (Ashmere A02 "4 wrapped
            # six-packs / case … EUR 2.82 per six-pack" → EUR 11.28/case, per-can
            # 0.47). Keep the row green with the real case price.
            if (_mp and prod.get("price_per_case") is None
                    and prod.get("price_per_unit")
                    and re.search(r"per\s+six[\s-]?pack|per\s+6[\s-]?pack", row_text, re.IGNORECASE)):
                try:
                    _n6 = int(_mp.group(1))
                    _ppc = round(float(prod["price_per_unit"]) * _n6, 4)
                    prod["price_per_case"] = _ppc
                    _upc = prod.get("units_per_case")
                    if _upc and float(_upc) > 0:
                        prod["price_per_unit"] = round(_ppc / float(_upc), 4)
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            if _mp and prod.get("price_per_case") and "per 6-pack" not in str(prod.get("commercial_conditions") or "").lower():
                _sp = int(_mp.group(1))
                try:
                    _six = round(float(prod["price_per_case"]) / _sp, 4)
                    _cur = prod.get("currency") or ""
                    prod["bottle_or_can_type"] = prod.get("bottle_or_can_type") or "can"
                    _nt = (f"wrapped {_sp} × 6 cans; "
                           f"{(_cur + ' ') if _cur else ''}{_six:g} per 6-pack")
                    _cc = prod.get("commercial_conditions") or ""
                    prod["commercial_conditions"] = (_cc + " | " if _cc else "") + _nt
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

            # (a2) Retail SET sold intact with no outer case (Ashmere A08 Mica "16
            #      intact sets … EUR 24 per set … Sell intact only. No outer case
            #      offered"; mirrors Northwick W07). The SET is the sellable unit:
            #      price per set, stock in sets, and NO case is invented from the
            #      set's component sprays (units_per_case "2" came from "2 × 50 ml").
            if (re.search(r"\bper\s+set\b|\bsets?\s+at\b|\bintact\s+sets?\b",
                          row_text, re.IGNORECASE)
                    and re.search(r"no\s+outer\s+cases?|sell\s+intact|sold\s+intact"
                                  r"|intact\s+only", row_text, re.IGNORECASE)):
                _setp = prod.get("price_per_unit")
                if _setp is None:
                    _setp = prod.get("price_per_case")
                prod["units_per_case"] = None
                prod["price_per_case"] = None
                prod["price_per_case_eur"] = None
                if _setp is not None:
                    prod["price_per_unit"] = _setp
                if prod.get("quantity_case") is not None and prod.get("quantity_bottles") is None:
                    prod["quantity_bottles"] = prod.pop("quantity_case")
                prod["quantity_unit"] = "sets"
                _ccs = prod.get("commercial_conditions") or ""
                if "per set" not in _ccs.lower():
                    prod["commercial_conditions"] = (_ccs + " | " if _ccs else "") + (
                        "sold as intact sets — priced per set, no outer case")

            # (a3) The row states its OWN units, overriding the case-based column
            #      headers: the tally is INDIVIDUAL pieces and the price is PER piece
            #      (Ashmere B03 Silk "tally is individual candles; price is EUR per
            #      candle. These units replace both column headings"; mirrors
            #      Northwick W17). Keep the piece count and the per-piece price —
            #      never read the count as cases or blank the confirmed price.
            if re.search(r"individual\s+candles?|individual\s+pieces?|per\s+candle\b"
                         r"|units\s+replace\s+both", row_text, re.IGNORECASE):
                _icq = prod.get("quantity_case") or prod.get("quantity_bottles")
                _icp = prod.get("price_per_unit")
                if _icp is None:
                    _icp = prod.get("price_per_case")
                if _icq is not None:
                    prod["quantity_bottles"] = _icq
                    prod.pop("quantity_case", None)
                    prod["quantity_unit"] = "pieces"
                prod["units_per_case"] = None
                if _icp is not None:
                    prod["price_per_unit"] = _icp
                prod["price_per_case"] = None
                prod["price_per_case_eur"] = None

            # (a) Sold out / reserved / closed parcel → not sellable stock. Catch
            #     the circular dialects too (Samuel 2026-11-27: Ashmere A14 "All 62
            #     cases SOLD … 0 available … no stock offered"; B05 Raven "parcel
            #     RESERVED … zero available for new orders").
            _hist_cleared = False
            if re.search(r"\bsold\s*out\b|closed\s+parcel|no\s+longer\s+available"
                         r"|out\s+of\s+stock|discontinued"
                         r"|all\s+\d[\d,]*\s+(?:cases?|units?|bottles?)\s+sold"
                         r"|\b0\s+available\b|\bzero\s+available\b|nothing\s+available"
                         r"|no\s+stock\s+offered|reserved\s+for\s+another"
                         r"|entire\s+parcel\s+reserved", row_text, re.IGNORECASE):
                for _qk in ("quantity_case", "quantity_bottles",
                            "quantity_pallets", "quantity_ftl"):
                    prod.pop(_qk, None)
                # A price the source marks as a FORMER / HISTORICAL / PREVIOUS quote is
                # a past figure, not a live offer, so it must not read as quotable — and
                # one extraction run captured it while another did not (Samuel 2026-11-28:
                # A14 "Former quote EUR 42/case … Historical price only"; mirrors Northwick
                # W12 "previous EUR 36/case"). A merely RESERVED parcel keeps its live
                # price (B05 "parcel reserved … zero available", price 102 is current).
                _hist_cleared = bool(re.search(
                    r"former\s+quote|historical\s+price|\bprevious\b|\bformer\b"
                    r"|price\s+only|old\s+price|was\s+(?:eur|gbp|usd|chf|€|£|\$)?\s*\d",
                    row_text, re.IGNORECASE))
                if _hist_cleared:
                    for _pk in ("price_per_case", "price_per_unit", "price_per_case_eur"):
                        prod[_pk] = None
                _reserved = bool(re.search(r"reserved", row_text, re.IGNORECASE))
                _flag(("reserved for another buyer — zero available for new orders; "
                       "price retained for reference") if _reserved else
                      ("sold out — historical price only, not a live offer" if _hist_cleared
                       else "sold out — not available; retained for reference"))
            # (b) A currency but no price = the amount wasn't captured. Never show
            #     it green (Alder A03 lost its EUR 8.64).
            if (prod.get("currency") and prod.get("currency") not in ("Not Found",)
                    and prod.get("price_per_unit") is None
                    and prod.get("price_per_case") is None
                    and not price_on_request
                    and not _hist_cleared
                    and "as quoted" not in str(prod.get("commercial_conditions") or "").lower()):
                _flag("currency present but no price captured — confirm the price against the source")
            # (c) Case price and unit price disagree (case ≠ unit × pack) → a
            #     conflicting or fabricated price (Alder W04 "EUR 36/case; EUR 8/btl").
            _pc = prod.get("price_per_case"); _pu = prod.get("price_per_unit")
            _uc = prod.get("units_per_case")
            try:
                if (_pc and _pu and _uc and float(_uc) > 1
                        and abs(float(_pc) - float(_pu) * float(_uc))
                        > max(0.02, float(_pu) * float(_uc) * 0.02)):
                    _flag("prices conflict (case price ≠ unit × pack) — confirm which applies")
            except (TypeError, ValueError):
                pass
            # (c2) Price basis UNCONFIRMED: the source states an amount but says it
            #      does not identify the pricing unit (Ashmere A06 Slate "GBP 72 …
            #      Packing is descriptive only. The quotation does not identify its
            #      pricing unit"). Keep the amount visible as-quoted, blank the
            #      per-case/per-unit prices (they would be a guess) and flag.
            if re.search(r"does\s+not\s+(?:identify|state|specify)[^.]{0,30}pricing\s+unit"
                         r"|pricing\s+unit[^.]{0,20}(?:not|unconfirmed)"
                         r"|price\s+basis\s+(?:unconfirmed|not\s+stated)",
                         row_text, re.IGNORECASE):
                _amt = prod.get("price_per_case")
                if _amt is None:
                    _amt = prod.get("price_per_unit")
                _cur = prod.get("currency") or ""
                if _amt is not None and "as quoted" not in str(prod.get("commercial_conditions") or "").lower():
                    _cc = prod.get("commercial_conditions") or ""
                    prod["commercial_conditions"] = (_cc + " | " if _cc else "") + (
                        f"as quoted: {(_cur + ' ') if _cur else ''}{_amt:g} "
                        f"(price basis unconfirmed — per bottle or per case not stated)")
                    for _pk in ("price_per_case", "price_per_case_eur",
                                "price_per_unit", "price_per_unit_eur"):
                        prod[_pk] = None
                    _flag("price basis unconfirmed — amount kept as quoted, per-unit "
                          "left blank; confirm the basis against the source")
            # (c3) Customs status NOT supplied for the row (Ashmere A16 Willow
            #      "Customs status not supplied. The T2 on Kestrel does not apply").
            if re.search(r"customs\s+(?:status\s+)?not\s+(?:supplied|stated|confirmed|given)",
                         row_text, re.IGNORECASE):
                _flag("customs status not supplied — confirm before quoting")
            # (d) A declared total that disagrees with its own lot/BBD breakdown.
            #     Two source dialects, BOTH must be caught (Samuel 2026-11-27):
            #       • "102 cases declared … 80 cs + 24 cs" (Alder W02)
            #       • "Declared total: 118 cases … Lot J1: 49 cases. Lot J2: 71
            #         cases. Please reconcile" (Ashmere A12) / "Lot N1=38 cases,
            #         N2=61 cases" (Northwick W10).
            #     The declared total is the quantity we already parsed for the row;
            #     sum the enumerated lots and flag any disagreement. A stated total
            #     that disagrees with its own breakdown is NOT a confirmed quantity.
            _lots = [int(x.replace(",", "")) for x in re.findall(
                r"(?:lot\s+)?[A-Za-z]{1,3}\d+\s*[:=]\s*(\d[\d,]*)\s*cases?\b",
                row_text, re.IGNORECASE)]
            _lots += [int(x.replace(",", "")) for x in
                      re.findall(r"(\d[\d,]*)\s*cs\b", row_text, re.IGNORECASE)]
            _declq = prod.get("quantity_case")
            try:
                _declq = int(round(float(_declq))) if _declq is not None else None
            except (TypeError, ValueError):
                _declq = None
            if _declq and len(_lots) >= 2 and sum(_lots) != _declq:
                _flag(f"declared total {_declq} cases ≠ lot breakdown "
                      f"{' + '.join(str(x) for x in _lots)} = {sum(_lots)} — "
                      f"quantity conflict, confirm with supplier")

            # (e) A row-level EXPIRY in the past overrides the circular's general
            #     validity: an offer that "expired <date>" / "has not been renewed"
            #     is stale and must not read as live (Samuel 2026-11-27: Ashmere
            #     B09 Hearth, Northwick W22 Moor both expired 25 Nov for a 26 Nov
            #     circular, yet showed green). Keep the row visible for reference,
            #     flag it, exclude it from active comparisons.
            _mexp = re.search(
                r"\bexpired\b[^.]*?(\d{1,2}\s+[A-Za-z]+\s+20\d\d|\d{4}-\d\d-\d\d)"
                r"|has\s+not\s+been\s+renewed|no\s+renewal", row_text, re.IGNORECASE)
            if _mexp:
                _edate = (_mexp.group(1) or "").strip() if _mexp.lastindex else ""
                _flag(f"offer expired{(' ' + _edate) if _edate else ''} and not "
                      f"renewed — stale; excluded from active comparisons, "
                      f"retained for reference")
                # The ACTUAL expiry date overrides the circular's general validity —
                # valid_until must read the stale date, not the generic 4 Dec (Samuel
                # 2026-11-28: B09 flagged expired yet valid_until still showed 4 Dec).
                _eiso = _header_date(_edate) if _edate else None
                if _eiso:
                    prod["valid_until"] = _eiso

            # (e2) Per-row commercial conditions that live in the Terms / Notes /
            #      Quotation column and must be carried onto THIS row, not globalised
            #      (Samuel 2026-11-28). A refundable deposit quoted separately from the
            #      goods price (Ashmere C01 "plus EUR 2.40 refundable deposit/case")
            #      and a whole-lot / all-or-nothing purchase condition (C04 "Whole lot
            #      only, minimum order all 18 cases") were being lost.
            def _add_cond(_n: str):
                _cc = prod.get("commercial_conditions") or ""
                if _n.lower() not in _cc.lower():
                    prod["commercial_conditions"] = (_cc + " | " + _n) if _cc else _n

            _mdep = re.search(
                r"(?:plus|\+|and)\s+(?:a\s+)?([€$£]|eur|usd|gbp|chf)?\s*(\d[\d.,]*)\s*"
                r"(?:[€$£]|eur|usd|gbp|chf)?\s*(?:refundable\s+)?deposit"
                r"(?:\s*(?:/|per)\s*(case|cs|bottle|btl|unit))?", row_text, re.IGNORECASE)
            if _mdep:
                _dcur = (_mdep.group(1) or prod.get("currency") or "").upper()
                _dcur = {"€": "EUR", "$": "USD", "£": "GBP"}.get(_dcur, _dcur)
                _dper = _mdep.group(3) or "case"
                _add_cond(f"refundable deposit {(_dcur + ' ') if _dcur else ''}"
                          f"{_mdep.group(2)} per {_dper} — separate from the goods price")

            if re.search(r"whole\s+lot\s+only|all\s+\d[\d,]*\s+cases?\s+must\s+be\s+taken"
                         r"|minimum\s+order\s+all\s+\d|entire\s+(?:lot|parcel)\s+only",
                         row_text, re.IGNORECASE):
                _mwl = re.search(r"(?:all\s+|minimum\s+order\s+all\s+)(\d[\d,]*)\s*cases?",
                                 row_text, re.IGNORECASE)
                _wln = _mwl.group(1) if _mwl else None
                _add_cond("whole lot only — "
                          + (f"all {_wln} cases must be taken" if _wln
                             else "the entire lot must be taken as one order"))

            # (f) Superseded / withdrawn price: the row states a CURRENT price and
            #     marks an earlier one withdrawn (Ashmere B04 Birch "CURRENT price
            #     EUR 63/case; EUR 57/case withdrawn"); the current figure wins
            #     (mirrors Northwick W18 and offer_reconcile.apply_superseded_price).
            #     NOT fired when the row says both prices stand (A13 Flint "neither
            #     supersedes the other").
            if (re.search(r"\bwithdrawn\b|\bsuperseded\b|no\s+longer\s+valid"
                          r"|previous\s+quote|earlier[^.]*\bquote", row_text, re.IGNORECASE)
                    and not re.search(r"neither\s+supersedes|both\s+prices?\s+"
                                      r"(?:remain|stand|current|are\s+current)",
                                      row_text, re.IGNORECASE)):
                _cm = re.search(r"(?:current|new|correct|revised)\s+price\s*(?:is\s*)?"
                                r"(?:USD|EUR|GBP|CHF|HKD|SGD|\$|€|£)?\s*(\d[\d.,]*)",
                                row_text, re.IGNORECASE)
                if _cm:
                    try:
                        _newp = round(float(_cm.group(1).replace(",", "")), 2)
                        _oldp = prod.get("price_per_case")
                        if _oldp and abs(float(_oldp) - _newp) > 0.01:
                            _ratio = _newp / float(_oldp)
                            prod["price_per_case"] = _newp
                            for _pf in ("price_per_unit", "price_per_unit_eur",
                                        "price_per_case_eur"):
                                _v = prod.get(_pf)
                                if _v is not None:
                                    prod[_pf] = round(float(_v) * _ratio, 4)
                            _ff = prod.setdefault("error_flags", [])
                            _fm = (f"used the stated current price {_newp:g} per case "
                                   f"(the earlier {float(_oldp):g} was withdrawn)")
                            if _fm not in _ff:
                                _ff.append(_fm)
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass

            # (b2) Available stock stated ONLY in the note as "N pallets × M cases"
            #      with an empty quantity column (Ashmere B08 Thistle "Stock: 7
            #      pallets x 48 cases", quantity cell blank → 336 cases, 48/pallet).
            if not any(prod.get(k) for k in ("quantity_case", "quantity_bottles",
                                             "quantity_pallets", "quantity_ftl")):
                _pm = re.search(r"(\d[\d,]*)\s*pallets?\s*(?:of|[x×*])\s*(\d[\d,]*)\s*cases?",
                                row_text, re.IGNORECASE)
                if _pm:
                    try:
                        _npal = int(_pm.group(1).replace(",", ""))
                        _mc = int(_pm.group(2).replace(",", ""))
                        prod["quantity_case"] = float(_npal * _mc)
                        prod["quantity_unit"] = "cases"
                        prod["cases_per_pallet"] = float(_mc)
                    except ValueError:
                        pass

            # (g) Customs tier stated in the row note (Ashmere "T2. Ready stock").
            #     The structured reader has no customs column, so read a clearly
            #     stated T1/T2 off the row text — EXCEPT where the row forbids
            #     inheriting it or says it does not apply (A16 "The T2 on Kestrel
            #     does not apply"). Gives the Excel rows the same customs status
            #     their Northwick twins carry (Samuel 2026-11-27 parity).
            if not prod.get("custom_status"):
                _negc = re.search(
                    r"do\s*n(?:o|')t\s+(?:assume|copy)"
                    r"|not\s+(?:to\s+be\s+)?(?:assumed?|inherit(?:ed)?|copied)"
                    r"|does\s+not\s+apply|not\s+supplied|unconfirmed|not\s+agreed"
                    r"|pending", row_text, re.IGNORECASE)
                _ct = re.search(r"\bT([12])\b", row_text)
                if _ct and not _negc:
                    prod["custom_status"] = f"T{_ct.group(1)}"

            # Keep the row when it carries ANY commercial signal — a price, a
            # price-on-request marker, a quantity, or an EAN. A no-price,
            # no-quantity line falls through to the LLM/flagging path — UNLESS we
            # have deliberately flagged it for review (e.g. a sold-out row whose
            # price we cleared as historical, Ashmere A14): a flagged row must stay
            # visible so it can be reviewed and excluded, never silently dropped
            # here just because it now has no quotable figure (Samuel 2026-11-28).
            _has_price = (prod.get("price_per_unit") is not None
                          or prod.get("price_per_case") is not None)
            _has_qty = any(prod.get(k) for k in
                           ("quantity_case", "quantity_bottles",
                            "quantity_pallets", "quantity_ftl"))
            if not (_has_price or price_on_request or _has_qty
                    or prod.get("ean_code") or prod.get("needs_manual_review")):
                continue

            # Per-row category on the structured path (parity with the text path):
            # classify from the FULL row text incl. the notes column so a candle →
            # cosmetics and a perfume gift set (its "sprays" live in the notes) →
            # perfumes are not filed under wines & spirits (Samuel 2026-11-28:
            # A07/B03 candles, A08 Mica discovery set). row_text is guaranteed here,
            # unlike downstream where only mapped fields survive.
            if not prod.get("category_slug"):
                from core.category_classifier import detect_row_category as _drc
                _slug = (_category_slug(prod.get("category"))
                         or _drc(f"{row_text} {prod.get('product_name') or ''}"))
                if _slug:
                    prod["category_slug"] = _slug
            if (not prod.get("product_type")) and re.search(r"\bcandles?\b", row_text, re.IGNORECASE):
                prod["product_type"] = "candle"
            # Perfume / cosmetic discriminators the sheet left in the NAME rather than a
            # Type column (Ashmere C02 "… EDP 100 ml TESTER", C03 "… RETAIL") must be
            # captured into their own fields: product_family_id now strips format / size /
            # retail-state tokens from the FAMILY identity (so a tester and a retail unit
            # of the same perfume roll up to one family), so the SKU / peer key stays
            # distinct only if perfume_format and retail_state carry the split. Without
            # this the tester and the retail line would collapse to one SKU (Samuel
            # 2026-11-28).
            if (prod.get("category_slug") or "").lower() in ("perfumes", "cosmetics"):
                _nm = f"{prod.get('product_name') or ''} {row_text}".lower()
                if not prod.get("perfume_format"):
                    _pf = re.search(r"\b(edp|edt|edc|edf|parfum|extrait|cologne)\b", _nm)
                    if _pf:
                        prod["perfume_format"] = _pf.group(1).upper()
                if not prod.get("retail_state"):
                    if re.search(r"\btester\b", _nm):
                        prod["retail_state"] = "tester"
                    elif re.search(r"\bsample\b", _nm):
                        prod["retail_state"] = "sample"
                    elif re.search(r"\bminiature\b", _nm):
                        prod["retail_state"] = "miniature"
                    elif re.search(r"\bretail\b", _nm):
                        prod["retail_state"] = "retail"
            prod["_structured_source"] = f"{ws.title}:row{r}"
            # Keep the full row text on the row so downstream deterministic passes
            # (co-equal delivery split, etc.) can read an annotation that lives in
            # a column we did not map to a field (Samuel 2026-11-27 parity).
            prod["_row_text"] = row_text
            products.append(prod)

    if recognised_sheets == 0 or not products:
        return None
    # Tag every structured row: its price basis comes from the COLUMN HEADER
    # ("Net EUR per case"), not the row text, so the text-based absent-basis pass
    # must skip it (else it would flag a clean per-case price as unconfirmed).
    for _p in products:
        _p["_structured_row"] = True
        # A column literally headed "Reference" maps to supplier_reference, but on
        # a sheet whose row code IS that column (Ashmere "Ready stock": B01…B10)
        # the row has no other product_reference. Promote a row-code-shaped
        # supplier_reference to product_reference so the row carries its own ticket
        # (Samuel 2026-11-27: B-refs were invisible). supplier_reference is kept.
        if not _p.get("product_reference"):
            _sref = str(_p.get("supplier_reference") or "").strip()
            if re.fullmatch(r"[A-Za-z]{1,3}\d{1,3}", _sref):
                _p["product_reference"] = _sref
    _split_structured_alt_delivery(products)
    logger.info(f"[xlsx_structured] extracted {len(products)} product(s) from "
                f"{recognised_sheets} structured sheet(s)")
    return products


# ── Co-equal dual-incoterm leg from a row annotation (Samuel 2026-11-27) ─────
# A structured row can carry its alternative delivery term in a free note column
# the text-path split never sees: Ashmere A10 Rowan "EUR 132/case EXW Ghent …
# Alternatively DAP Lille at EUR 138/case and EUR 23/loose bottle. SAME stock,
# not additional." The base leg is kept AND a co-equal DAP leg is emitted — both
# carry the SAME stock (one pool, counted once), each comparable on its own
# incoterm. Mirrors the text-path co-equal pool so both readers behave alike.
_ALT_DELIVERY_RE = re.compile(
    r"alternativ\w*\s+(?:(EXW|DAP|DDP|FCA|CIF|FOB|CFR|CPT)\s+)?"
    r"([A-Z][A-Za-z]+)?\s*(?:at|:)?\s*"
    r"(?:EUR|USD|GBP|CHF|€|\$|£)?\s*([\d.,]+)\s*(?:/|\bper\b)\s*case",
    re.IGNORECASE)
_ALT_INCOTERM_RE = re.compile(
    r"\b(EXW|DAP|DDP|FCA|CIF|FOB|CFR|CPT)\b", re.IGNORECASE)
_ALT_LOOSE_RE = re.compile(
    r"(?:EUR|USD|GBP|CHF|€|\$|£)?\s*([\d.,]+)\s*(?:/|\bper\b)\s*loose",
    re.IGNORECASE)


def _split_structured_alt_delivery(products: List[Dict[str, Any]]) -> None:
    import copy as _copy
    for o in list(products):
        rt = str(o.get("_row_text") or "")
        if not rt:
            continue
        m = _ALT_DELIVERY_RE.search(rt)
        if not m:
            continue
        # Only a SHARED-stock alternative (not an additional parcel) is co-equal.
        if not re.search(r"same\s+stock|not\s+additional|share[sd]?\s+stock",
                         rt, re.IGNORECASE):
            continue
        incoterm = (m.group(1) or "").upper()
        if not incoterm:
            mi = _ALT_INCOTERM_RE.search(rt[m.start():])
            incoterm = (mi.group(1).upper() if mi else "")
        place = (m.group(2) or "").strip()
        try:
            alt_case = float(str(m.group(3)).replace(",", "."))
        except (TypeError, ValueError):
            continue
        if not incoterm or incoterm == (o.get("incoterm") or "").upper():
            continue
        alt = _copy.deepcopy(o)
        alt["incoterm"] = incoterm
        if place:
            alt["location"] = place
        alt["price_per_case"] = alt_case
        alt["price_per_case_eur"] = alt_case if (o.get("currency") == "EUR") else alt.get("price_per_case_eur")
        # Loose-bottle prices: the first belongs to the base leg, the second to
        # the alternative (A10 "EUR 22 per loose bottle … Alternatively … EUR 23
        # per loose bottle"). Fill each leg's per-unit price when it is missing.
        _loose = _ALT_LOOSE_RE.findall(rt)
        try:
            if _loose and o.get("price_per_unit") is None:
                o["price_per_unit"] = float(str(_loose[0]).replace(",", "."))
            if len(_loose) >= 2:
                alt["price_per_unit"] = float(str(_loose[1]).replace(",", "."))
        except (TypeError, ValueError):
            pass
        pool = o.get("product_reference") or o.get("supplier_reference") or (o.get("brand") or "stock")
        note_base = (f"One stock pool ({pool} counted once): this "
                     f"{(o.get('incoterm') or '').upper()} and the {incoterm} offer are the "
                     f"SAME stock — take from either, not both.")
        note_alt = (f"One stock pool ({pool} counted once): this {incoterm} and the "
                    f"{(o.get('incoterm') or '').upper()} offer are the SAME stock — "
                    f"take from either, not both.")
        for tgt, nt in ((o, note_base), (alt, note_alt)):
            cc = tgt.get("commercial_conditions") or ""
            if "one stock pool" not in cc.lower():
                tgt["commercial_conditions"] = (cc + " | " if cc else "") + nt
            tgt["_stock_pool"] = str(pool)
            tgt["_structured_source"] = (tgt.get("_structured_source") or "") + f"|pool:{incoterm if tgt is alt else (o.get('incoterm') or '')}"
        products.append(alt)


def extract_freetext_sheets(file_path: str,
                            sheet_names: List[str]) -> List[Dict[str, Any]]:
    """Deterministic read of named free-text 'stock circular' sheets, using the
    extra dialect (Ticket / What is on the floor / Commercial figure / …). This is
    the COMPLETENESS BACKSTOP (Samuel 2026-11-26): the LLM parses the embedded
    free-text prices better, but it non-deterministically drops a whole line
    (Dune/Plover vanished between runs). The caller keeps the LLM rows and only
    ADDS a backstop row for a ticket the LLM missed — so a line is never silently
    lost, and a good LLM row is never replaced. Returns raw dicts (pre-clean)."""
    try:
        rows = extract_structured_products(
            file_path, extra_synonyms=_FREETEXT_SYNONYMS,
            only_sheets=set(sheet_names)) or []
    except Exception as e:
        logger.warning(f"[xlsx_structured] free-text backstop read failed: {e}")
        return []
    return rows


def unreadable_product_sheet_names(file_path: str) -> List[str]:
    """Names of sheets that look like product tables (>=3 priced rows) but whose
    header the structured reader cannot map — the caller routes these to the LLM
    fallback and merges, so a recognised sheet keeps its clean extraction while an
    unfamiliar sheet is still captured (Samuel 2026-11-25)."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(file_path, data_only=True, read_only=False)
    except Exception:
        return []
    out = []
    for ws in wb.worksheets:
        if ws.max_row is None or ws.max_row < 2:
            continue
        if _detect_header_row(ws):
            continue
        if _sheet_product_row_count(ws) >= 3:
            out.append(ws.title)
    return out


def write_sheets_subset(file_path: str, sheet_names: List[str]) -> Optional[str]:
    """Write the named sheets to a new temp .xlsx (values only) and return its path,
    so the fallback reader can process just the unreadable sheets."""
    try:
        import openpyxl, tempfile
        src = openpyxl.load_workbook(file_path, data_only=True, read_only=False)
        out = openpyxl.Workbook()
        out.remove(out.active)
        for name in sheet_names:
            if name not in src.sheetnames:
                continue
            ws_src = src[name]
            ws_new = out.create_sheet(title=name[:31])
            for row in ws_src.iter_rows(values_only=True):
                ws_new.append(list(row))
        if not out.sheetnames:
            return None
        fd = tempfile.NamedTemporaryFile(prefix="unread_", suffix=".xlsx", delete=False)
        fd.close()
        out.save(fd.name)
        return fd.name
    except Exception as e:
        logger.warning(f"[xlsx_structured] write_sheets_subset failed: {e}")
        return None
