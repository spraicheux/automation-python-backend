"""core/email_table.py

Deterministic recovery for a flattened HTML price table pasted into an email
body.

Why this exists
---------------
Some suppliers (EPIC Singapore / Brevo newsletters) send the offer as an HTML
<table>. By the time the plain-text body reaches us every cell sits on its own
line, so the row structure is gone:

    PRODUCTS / BTL TYPE / REF / NRF / BTLS PER CASE / SIZE (CL) / VOL % /
    T1 / T2 / QUANTITY / PRICE / DELIVERY / PER LOAD / STATUS      <- header
    ABSOLUT BLUE / GLASS / REF / 120 / 5 / 40 / T2 / 1,560 cases /
    € 56.00 / Ready Stock                                          <- one SKU
    …

Handed to the LLM, the columns drift: prices land on the wrong brand, a phantom
"price × quantity" row appears, and SKUs go missing (Samuel EPIC 2026-11-17 —
11/12 with Absolut 37.5cl and Chivas 37.5cl lost and a bogus €1752 row).

The table is perfectly regular, so we reconstruct it DETERMINISTICALLY: find the
column-label header, then group the body into fixed-width records and validate
every field by type. If the shape does NOT match exactly we return None and the
caller falls back to the normal LLM path — so no other email is affected.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# The column labels this table uses, each on its own line. We require a strong
# subset to be present as standalone lines before we trust the format.
_HEADER_LABELS = [
    "PRODUCTS", "BTL TYPE", "REF / NRF", "BTLS PER CASE", "SIZE (CL)",
    "VOL %", "T1 / T2", "QUANTITY", "PRICE", "DELIVERY", "PER LOAD", "STATUS",
]
# Labels that, present together, make this unmistakably the flattened table.
_SIGNATURE = ["BTLS PER CASE", "SIZE (CL)", "VOL %", "QUANTITY", "PRICE", "T1 / T2"]

# Body layout: 10 values per SKU, in this fixed order (PER LOAD / STATUS collapse
# into the single trailing "Ready Stock" token).
_FIELDS_PER_RECORD = 10

_CUR_SYMBOL = {"€": "EUR", "$": "USD", "£": "GBP", "¥": "JPY"}
_CUR_WORD = {"eur": "EUR", "usd": "USD", "gbp": "GBP", "hkd": "HKD",
             "sgd": "SGD", "chf": "CHF", "aed": "AED"}
_T_RE = re.compile(r"^T\s*/?\s*[12]$|^T[12]$", re.IGNORECASE)
_QTY_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(cases?|cartons?|bottles?|btls?|pcs?|pieces?|units?)",
                     re.IGNORECASE)
_PRICE_RE = re.compile(
    r"(€|£|\$|¥|\b(?:eur|usd|gbp|hkd|sgd|chf|aed)\b)\s*([\d][\d,]*(?:\.\d+)?)",
    re.IGNORECASE)
_INT_RE = re.compile(r"^\d[\d,]*$")
_NUM_RE = re.compile(r"^\d[\d,]*(?:[.,]\d+)?$")


def _clean_lines(text: str) -> List[str]:
    """Strip email quote markers ('> ') and whitespace; drop blank lines."""
    out = []
    for ln in (text or "").splitlines():
        s = re.sub(r"^[>\s]+", "", ln).strip()
        if s:
            out.append(s)
    return out


def _num(s: str) -> Optional[float]:
    """A count whose comma is a thousands separator ('1,560' -> 1560)."""
    try:
        return float(str(s).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _size_cl_to_ml(raw: str) -> Optional[float]:
    v = None
    try:
        v = float(str(raw).replace(",", "."))  # "37,5" -> 37.5 (cl is decimal)
    except (TypeError, ValueError):
        return None
    return round(v * 10, 2)  # cl -> ml


def _parse_price(tok: str):
    m = _PRICE_RE.search(tok or "")
    if not m:
        return (None, None)
    sym = m.group(1).lower()
    cur = _CUR_SYMBOL.get(m.group(1)) or _CUR_WORD.get(sym)
    return (cur, _num(m.group(2)))


def _looks_like_record(chunk: List[str]) -> bool:
    """Every field must be the right TYPE for the chunk to be a real SKU row
    (this is what tells a product row apart from the trailing footer prose)."""
    if len(chunk) != _FIELDS_PER_RECORD:
        return False
    brand, btltype, ref, per_case, size, vol, tier, qty, price, _status = chunk
    if not brand or not re.search(r"[A-Za-z]", brand) or len(brand) > 60:
        return False
    if _INT_RE.match(brand):            # a number is never the brand
        return False
    if not _INT_RE.match(per_case.strip()):
        return False
    if not _NUM_RE.match(size.strip()):
        return False
    if not _NUM_RE.match(vol.strip()):
        return False
    if not _T_RE.match(tier.strip()):
        return False
    if not _QTY_RE.search(qty):
        return False
    if not _PRICE_RE.search(price):
        return False
    return True


def parse_flattened_price_table(text: str) -> Optional[List[Dict[str, Any]]]:
    """Return a list of raw product dicts when `text` is the recognised
    flattened price table, else None (caller then uses the LLM path)."""
    lines = _clean_lines(text)
    if len(lines) < len(_SIGNATURE) + _FIELDS_PER_RECORD:
        return None

    # 1) Signature: the key column labels must appear as standalone lines.
    label_pos: Dict[str, int] = {}
    for i, s in enumerate(lines):
        if s in _HEADER_LABELS and s not in label_pos:
            label_pos[s] = i
    if not all(sig in label_pos for sig in _SIGNATURE):
        return None

    # 2) Body starts right after the last contiguous header label.
    header_end = max(label_pos.values())
    body = lines[header_end + 1:]

    # 3) Group into fixed-width records; stop at the first chunk that is not a
    #    valid SKU row (that is where the table ends and the footer begins).
    products: List[Dict[str, Any]] = []
    i = 0
    while i + _FIELDS_PER_RECORD <= len(body):
        chunk = body[i:i + _FIELDS_PER_RECORD]
        if not _looks_like_record(chunk):
            break
        brand, btltype, ref, per_case, size, vol, tier, qty, price, status = chunk
        cur, price_val = _parse_price(price)
        qm = _QTY_RE.search(qty)
        qty_num = _num(qm.group(1)) if qm else None
        qty_word = (qm.group(2).lower() if qm else "")
        prod: Dict[str, Any] = {
            "product_name": brand.strip(),
            "units_per_case": _num(per_case),
            "unit_volume_ml": _size_cl_to_ml(size),
            "alcohol_percent": _num(vol),
            "custom_status": "T" + re.sub(r"\D", "", tier),
            "currency": cur,
            "price_per_case": price_val,       # wholesale quote is per case
            "category_slug": "wines_spirits",
            "bottle_or_can_type": btltype.strip().lower() or None,
            "refillable_status": ("refillable" if ref.strip().upper() == "REF"
                                  else "non-refillable" if "NRF" in ref.upper()
                                  else None),
        }
        if qty_num is not None:
            if qty_word.startswith(("bottle", "btl", "piece", "pc", "unit")):
                prod["quantity_bottles"] = qty_num
                prod["quantity_unit"] = "bottles"
            else:
                prod["quantity_case"] = qty_num
                prod["quantity_unit"] = "cases"
        if status and status.strip():
            prod["commercial_conditions"] = status.strip()
        products.append(prod)
        i += _FIELDS_PER_RECORD

    # Require several clean records so we never hijack a stray near-match.
    if len(products) < 3:
        return None
    logger.info(f"[email_table] flattened price table recognised — "
                f"{len(products)} product(s) parsed deterministically")
    return products
