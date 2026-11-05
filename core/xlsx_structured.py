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
    "units per case": "units_per_case", "unit per case": "units_per_case",
    "bottles per case": "units_per_case", "pieces per case": "units_per_case",
    "pack size": "units_per_case", "pack": "units_per_case",
    "abv": "alcohol_percent", "alcohol": "alcohol_percent",
    "alcohol percent": "alcohol_percent", "alc vol": "alcohol_percent",
    "alcohol by volume": "alcohol_percent", "abv vol": "alcohol_percent",
    "available quantity": "_quantity", "quantity available": "_quantity",
    "quantity": "_quantity", "qty": "_quantity", "stock": "_quantity",
    "available": "_quantity", "stock quantity": "_quantity",
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
    "size": "_size", "volume": "_size", "format": "_size", "capacity": "_size",
    "category": "_category", "type": "_category",
    "notes": "notes", "note": "notes", "comment": "notes", "comments": "notes",
    "remarks": "notes",
    "supplier reference": "supplier_reference", "supplier ref": "supplier_reference",
    "reference": "supplier_reference", "product reference": "product_reference",
    "sku": "product_reference",
}

# Headers we recognise but intentionally ignore (keeps the match score honest).
_IGNORE_HEADERS = {"test id", "id", "no", "line", "row", "index"}

# Fields that make a row a real product line (need at least one).
_PRICE_FIELDS = {"price_per_unit", "price_per_case"}


def _match_header(norm: str) -> Optional[str]:
    if not norm:
        return None
    if norm in _IGNORE_HEADERS:
        return "_ignore"
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
            size_raw: Any = None
            any_value = False

            for col, field in colmap.items():
                cell = ws.cell(row=r, column=col)
                val = cell.value
                if val is None or (isinstance(val, str) and not val.strip()):
                    continue
                any_value = True

                if field == "alcohol_percent":
                    prod["alcohol_percent"] = _scale_percent(cell)
                elif field in ("price_per_unit", "price_per_case", "units_per_case",
                               "moq_cases"):
                    prod[field] = _to_float(val)
                elif field == "_quantity":
                    qty_val = _to_float(val)
                elif field == "quantity_unit":
                    qty_unit = str(val).strip()
                elif field in ("offer_date", "valid_until", "best_before_date"):
                    prod[field] = _iso_date(val)
                elif field == "_size":
                    size_raw = val
                    prod["packaging_raw"] = str(val).strip()
                    ml = _parse_size_ml(val)
                    if ml is not None:
                        prod["unit_volume_ml"] = ml
                elif field == "_category":
                    slug = _category_slug(val)
                    if slug:
                        prod["category_slug"] = slug
                    prod["category"] = str(val).strip()
                elif field == "vintage":
                    prod["vintage"] = str(int(_to_float(val))) if _to_float(val) else str(val).strip()
                elif field == "ean_code":
                    # preserve verbatim (leading zeros matter)
                    prod["ean_code"] = (str(int(val)) if isinstance(val, float) and val.is_integer()
                                        else str(val).strip())
                elif field in ("_ignore",):
                    continue
                else:
                    prod[field] = str(val).strip()

            if not any_value:
                continue
            # A product line needs a name/brand and a price.
            if not (prod.get("product_name") or prod.get("brand")):
                continue
            if prod.get("price_per_unit") is None and prod.get("price_per_case") is None:
                continue

            _assign_quantity(prod, qty_val, qty_unit)
            prod.setdefault("category_slug", _category_slug(prod.get("category")))
            prod["_structured_source"] = f"{ws.title}:row{r}"
            products.append(prod)

    if recognised_sheets == 0 or not products:
        return None
    logger.info(f"[xlsx_structured] extracted {len(products)} product(s) from "
                f"{recognised_sheets} structured sheet(s)")
    return products
