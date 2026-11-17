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


# ───────────────────────────────────────────────────────────────────────────
# One-product-per-line price lists (FBC-style)
# ───────────────────────────────────────────────────────────────────────────
# A broker emails a plain list, one product per line, every line anchored by an
# "@ <currency> <price> per case/bottle" clause:
#     Jack Daniels 12x70cl NRF FTL @ USD 112 per case, on floor, DAP Newcorp
#     Smirnoff 70cl T2 FTL @ Euro 3.5 per bottle, 3 weeks Ex Loen
# Fed to the LLM this over-splits: it emits a clean row AND a degenerate
# duplicate (the whole line as the name, a borrowed location, no price), so a
# 13-line list became 17 rows (Samuel FBC 2026-11-17). Every line is regular, so
# we parse it deterministically and bypass the LLM. If the lines do NOT look like
# this list we return None and the caller uses the LLM path.

_LL_PRICE_RE = re.compile(
    r'@\s*(€|£|\$|eur|euro|usd|gbp|hkd|chf|sgd)\s*([0-9]+(?:[.,][0-9]+)?)'
    r'\s*(?:per\s+(case|bottle|btl|pc|piece)|/\s*(btl|bottle|case|pc|piece))?',
    re.IGNORECASE)
_LL_CUR = {"€": "EUR", "eur": "EUR", "euro": "EUR", "$": "USD", "usd": "USD",
           "£": "GBP", "gbp": "GBP", "hkd": "HKD", "chf": "CHF", "sgd": "SGD"}
# A size unit may be glued to "x<n>" (70clx12) or "x<n>%" (70clx40% = size×ABV),
# so the unit is "ended" by x / punctuation / whitespace / end, not only \b.
_LL_SIZE_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*(cl|ml|l)(?=x|[\s,.;:/)%]|$)',
                         re.IGNORECASE)
_LL_PACK_A = re.compile(r'(\d+)\s*x\s*\d+(?:[.,]\d+)?\s*(?:cl|ml|l)(?=x|[\s,.;:/)%]|$)',
                        re.IGNORECASE)                                            # 12x70cl
_LL_PACK_B = re.compile(r'\d+(?:[.,]\d+)?\s*(?:cl|ml|l)\s*x\s*(\d+)(?!\s*%)',
                        re.IGNORECASE)                                            # 70clx12 (not 70clx40%)
_LL_QTY_RE = re.compile(r'(\d[\d,]*)\s*(cases?|cartons?|btls?|bottles?|pcs?|pieces?)\b',
                        re.IGNORECASE)
_LL_FTL_RE = re.compile(r'\b(?:ftl|full\s*truck(?:load)?|full\s*load)\b', re.IGNORECASE)
_LL_T_RE = re.compile(r'\bT([12])\b')
_LL_INCO_RE = re.compile(
    r'\b(DAP|DDP|EXW|FCA|CIF|CFR|CNF|FOB|CPT|CIP|DPU)\b\s*([A-Z][A-Za-z]+)',
    re.IGNORECASE)
_LL_EX_RE = re.compile(r'\bEx\s+([A-Z][A-Za-z]+)')   # "Ex Loen" / "Ex Miami" -> EXW


def _ll_size_ml(raw: str, unit: str) -> Optional[float]:
    v = _num(raw)
    if v is None:
        return None
    u = unit.lower()
    return v * 1000 if u == "l" else v * 10 if u == "cl" else v


def _ll_parse_line(line: str) -> Optional[Dict[str, Any]]:
    pm = _LL_PRICE_RE.search(line)
    if not pm:
        return None
    cur = _LL_CUR.get(pm.group(1).lower())
    price = _num(pm.group(2))
    basis = (pm.group(3) or pm.group(4) or "").lower()
    prod: Dict[str, Any] = {"currency": cur, "category_slug": "wines_spirits"}
    if price is not None:
        if basis in ("case",):
            prod["price_per_case"] = price
        elif basis in ("bottle", "btl", "pc", "piece"):
            prod["price_per_unit"] = price
        else:  # unstated basis next to the amount → leave as unit price
            prod["price_per_unit"] = price

    sm = _LL_SIZE_RE.search(line)
    if sm:
        prod["unit_volume_ml"] = _ll_size_ml(sm.group(1), sm.group(2))
    pa = _LL_PACK_A.search(line) or _LL_PACK_B.search(line)
    if pa:
        prod["units_per_case"] = _num(pa.group(1))

    qm = _LL_QTY_RE.search(line)
    if qm:
        n = _num(qm.group(1)); w = qm.group(2).lower()
        if w.startswith(("btl", "bottle")):
            prod["quantity_bottles"] = n; prod["quantity_unit"] = "bottles"
        elif w.startswith(("pc", "piece")):
            prod["quantity_bottles"] = n; prod["quantity_unit"] = "pieces"
        else:
            prod["quantity_case"] = n; prod["quantity_unit"] = "cases"
    elif _LL_FTL_RE.search(line):
        prod["quantity_ftl"] = 1.0; prod["quantity_unit"] = "ftl"

    tm = _LL_T_RE.search(line)
    if tm:
        prod["custom_status"] = "T" + tm.group(1)

    im = _LL_INCO_RE.search(line)
    if im:
        prod["incoterm"] = im.group(1).upper()
        prod["location"] = im.group(2)
    else:
        em = _LL_EX_RE.search(line)
        if em:
            prod["incoterm"] = "EXW"
            prod["location"] = em.group(1)

    # Name = the text before the first of: pack (NxNcl), a bare size, " FTL ",
    # a quantity, or the "@". A real product name never carries price syntax.
    cut = len(line)
    for m in (_LL_PACK_A.search(line), _LL_PACK_B.search(line), sm, qm, pm,
              _LL_FTL_RE.search(line)):
        if m:
            cut = min(cut, m.start())
    name = re.sub(r'[\s,;:-]+$', '', line[:cut]).strip()
    if not name or len(name) > 60 or not re.search(r'[A-Za-z]', name):
        return None
    prod["product_name"] = name
    return prod


def parse_line_list_offer(text: str) -> Optional[List[Dict[str, Any]]]:
    """Return raw product dicts when `text` is a one-product-per-line price list
    (every product line anchored by an '@ <currency> <price>' clause), else None.
    Strictly guarded: the priced lines must be the clear majority of content so a
    prose email that merely mentions one price is never hijacked."""
    lines = _clean_lines(text)
    if len(lines) < 4:
        return None
    priced = [ln for ln in lines if _LL_PRICE_RE.search(ln)]
    if len(priced) < 3:
        return None
    # Priced lines must dominate the body (exclude obvious header/footer lines).
    content = [ln for ln in lines
               if not re.match(r'^(de|à|objet|date|r[ée]pondre|hello|dear|best|'
                               r'thanks|regards|www\.|http|if you|click|unsubscribe|'
                               r'>?\s*$)', ln, re.IGNORECASE)
               and "@" not in ln[:3]]
    if not content or len(priced) / max(len(content), 1) < 0.6:
        return None
    products = [p for p in (_ll_parse_line(ln) for ln in priced) if p]
    if len(products) < 3 or len(products) != len(priced):
        return None  # every priced line must parse, else fall back to the LLM
    logger.info(f"[email_table] line-list price offer recognised — "
                f"{len(products)} product(s) parsed deterministically")
    return products


# ───────────────────────────────────────────────────────────────────────────
# Pipe-delimited numbered lists (AQT-WhatsApp-style)
# ───────────────────────────────────────────────────────────────────────────
# A numbered offer, ONE product per line, fields separated by "|":
#   04) Veuve Clicquot Yellow Label | 750ml | 6 btls/case | 24 cases total … |
#       DAP Riga: EUR 144/case; EXW Breda: price TBC | MOQ 6 cases
# The LLM splits the two-delivery-option line into two rows (double-counting the
# one 24-case lot — Samuel AQT-WA 2026-11-17). One source line = one product, so
# we parse deterministically: each numbered line stays exactly one row.

_PIPE_NUM_RE = re.compile(r'^\s*\d{1,2}[).]\s*(.+)$')
_PIPE_PACK_HINT = re.compile(
    r'/\s*case|per\s*case|btls?\s*/|cans?\s*/|inner\s*box|individual|single|each|'
    r'no\s*case\s*pack', re.IGNORECASE)
_PIPE_QTY_HINT = re.compile(
    r'\d.*\b(cases?|bottles?|btls?|cans?|pieces?|pcs?|pallets?|truck|ftl)\b|'
    r'full\s*truck', re.IGNORECASE)
_PIPE_SIZE_RE = re.compile(r'^\s*\d+(?:[.,]\d+)?\s*(ml|cl|l|g)\b', re.IGNORECASE)
_PIPE_T_RE = re.compile(r'^\s*T\s*/?\s*([12])\s*$', re.IGNORECASE)
_PIPE_FTL_RE = re.compile(r'\bfull\s*truck|\bftl\b', re.IGNORECASE)


def parse_pipe_list_offer(text: str) -> Optional[List[Dict[str, Any]]]:
    """Return raw product dicts when `text` is a pipe-delimited numbered offer
    (one product per 'NN) a | b | c' line), else None. Each line → exactly one
    row, so a multi-delivery-option line is never split/double-counted. Reuses
    the Excel cell parsers for price / pack / quantity so behaviour matches."""
    from core.xlsx_structured import (_parse_price_cell, _parse_pack,
                                      _parse_quantity_cell, _norm)
    lines = _clean_lines(text)
    numbered = []
    for ln in lines:
        m = _PIPE_NUM_RE.match(ln)
        if m and "|" in ln:
            numbered.append(m.group(1))
    if len(numbered) < 3:
        return None

    products: List[Dict[str, Any]] = []
    for body in numbered:
        fields = [f.strip() for f in body.split("|")]
        if len(fields) < 2:
            return None  # not really a pipe row → bail to the LLM
        name = re.sub(r'\[[^\]]*\]', '', fields[0])       # drop "[spelling as sent]"
        name = re.sub(r'["“”]', '', name).strip()          # drop quote marks anywhere
        if not name or len(name) > 70:
            return None
        prod: Dict[str, Any] = {"product_name": name, "category_slug": "wines_spirits"}
        for f in fields[1:]:
            low = f.lower()
            if _PIPE_T_RE.match(f):
                prod["custom_status"] = "T" + re.sub(r"\D", "", f)
                continue
            if low.startswith("moq"):
                prod["moq_text"] = f
                continue
            # price: ONLY a field with a real currency token or an explicit
            # on-request/TBC clause (a bare number is a size/pack/quantity, not a
            # price — never let it masquerade as one).
            pc = _parse_price_cell(f)
            if pc["currency"] or pc["on_request"]:
                if pc["currency"] and not prod.get("currency"):
                    prod["currency"] = pc["currency"]
                if pc["on_request"]:
                    prod.setdefault("error_flags", []).append("Price on request retained")
                    prod["needs_manual_review"] = True
                elif pc["price_per_case"] is not None or pc["price_per_unit"] is not None:
                    if pc["price_per_case"] is not None:
                        prod["price_per_case"] = pc["price_per_case"]
                    if pc["price_per_unit"] is not None:
                        prod["price_per_unit"] = pc["price_per_unit"]
                elif pc["basis_ambiguous"] and pc["price_value"] is not None:
                    cur = (prod.get("currency") or "").strip()
                    prod["commercial_conditions"] = (
                        f"As quoted: {cur} {pc['price_value']:g}".strip()
                        + " — price basis not stated; excluded from price comparison.")
                    prod.setdefault("error_flags", []).append(
                        "price basis not stated — amount retained as-quoted, excluded from comparison")
                    prod["needs_manual_review"] = True
                elif pc["price_value"] is not None:
                    prod["price_per_unit"] = pc["price_value"]
                # a delivery term may also live inside the price field (Veuve:
                # "DAP Riga: EUR 144/case; EXW Breda: price TBC").
                if not prod.get("incoterm"):
                    im = _LL_INCO_RE.search(f)
                    if im:
                        prod["incoterm"] = im.group(1).upper()
                        prod["location"] = im.group(2)
                    else:
                        em = _LL_EX_RE.search(f)
                        if em:
                            prod["incoterm"] = "EXW"
                            prod["location"] = em.group(1)
                continue
            if _PIPE_SIZE_RE.match(f) and len(f) <= 16:
                sm = _LL_SIZE_RE.search(f)
                if sm:
                    prod["unit_volume_ml"] = _ll_size_ml(sm.group(1), sm.group(2))
                continue
            if _PIPE_PACK_HINT.search(f):
                prod["units_per_case"] = _parse_pack(f)
                continue
            if _PIPE_FTL_RE.search(f) and not _PIPE_QTY_HINT.search(re.sub(_PIPE_FTL_RE.pattern, "", f, flags=re.IGNORECASE)):
                prod["quantity_ftl"] = 1.0; prod["quantity_unit"] = "ftl"
                continue
            if _PIPE_QTY_HINT.search(f):
                n, u = _parse_quantity_cell(f)
                if n is not None:
                    un = _norm(u)
                    if un in ("bottle", "bottles", "btl", "btls"):
                        prod["quantity_bottles"] = n; prod["quantity_unit"] = "bottles"
                    elif un in ("piece", "pieces", "pcs", "pc", "can", "cans"):
                        prod["quantity_bottles"] = n; prod["quantity_unit"] = "pieces"
                    elif un in ("pallet", "pallets"):
                        prod["quantity_pallets"] = n; prod["quantity_unit"] = "pallets"
                    else:
                        prod["quantity_case"] = n; prod["quantity_unit"] = "cases"
                continue
            im = _LL_INCO_RE.search(f)
            if im:
                prod["incoterm"] = im.group(1).upper(); prod["location"] = im.group(2)
                continue
            em = _LL_EX_RE.search(f)
            if em:
                prod["incoterm"] = "EXW"; prod["location"] = em.group(1)
                continue
        products.append(prod)

    if len(products) < 3 or len(products) != len(numbered):
        return None
    logger.info(f"[email_table] pipe-delimited numbered offer recognised — "
                f"{len(products)} product(s) parsed deterministically")
    return products
