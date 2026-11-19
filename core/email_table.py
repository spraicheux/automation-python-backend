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
            # The table's column is just "PRICE" with no stated basis; a
            # wholesale offer priced per case (BTLS PER CASE + quantity in cases)
            # is the standard reading, but it is NOT explicit in the source — so
            # flag the basis as assumed (Samuel 2026-11-19, EPIC).
            "commercial_conditions": "Price basis assumed PER CASE — the source "
                                     "column is labelled only 'PRICE' (per-case is "
                                     "the standard wholesale reading given BTLS/CASE "
                                     "and case quantities, but not explicitly stated).",
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
            _existing_cc = prod.get("commercial_conditions")
            prod["commercial_conditions"] = (
                f"{status.strip()} | {_existing_cc}" if _existing_cc else status.strip())
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
# Availability / lead time a broker states inline ("on floor" = in stock now,
# "3 weeks", "4-5 weeks") — a delivery field that was being dropped (Samuel
# 2026-11-19: FBC "on floor" and week ranges missing from the export).
_LL_AVAIL_RE = re.compile(r'\bon\s+floor\b', re.IGNORECASE)
_LL_LEAD_RE = re.compile(r'\b(\d+(?:\s*[-–]\s*\d+)?)\s*weeks?\b', re.IGNORECASE)


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

    # Availability / lead time ("on floor", "3 weeks", "4-5 weeks") — kept in
    # lead_time so the export shows it next to the delivery term (Samuel
    # 2026-11-19). These sit after the "@" price clause, so they never bleed
    # into the product name (the name is cut at/before the price clause below).
    _avail = []
    if _LL_AVAIL_RE.search(line):
        _avail.append("on floor")
    _lm = _LL_LEAD_RE.search(line)
    if _lm:
        _avail.append(re.sub(r"\s+", "", _lm.group(1)) + " weeks")
    if _avail:
        prod["lead_time"] = ", ".join(_avail)

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
# An explicit NON-quantity ("case quantity not stated", "qty not given"). The
# word "case"/"quantity" inside it must not be read as a real count — stripped
# before the qty-hint test (Samuel 2026-11-19: Rémy "1 full truck; case quantity
# not stated" was stored as "1 cases").
_QTY_NEG_RE = re.compile(
    r'\b(?:\w+\s+){0,3}not\s+(?:stated|given|specified|provided|known|available)\b',
    re.IGNORECASE)

# "MOQ 6 cases", "MOQ 2 inner boxes", "MOQ 1 pallet", "MOQ 1 inner box (12 pieces)"
_MOQ_NUM_UNIT_RE = re.compile(
    r'([\d][\d.,]*)\s*(pallets?|full\s*trucks?|trucks?|ftl|inner\s*box(?:es)?|'
    r'cases?|bottles?|btls?|pieces?|pcs?|units?)', re.IGNORECASE)
# The whole "MOQ …" clause, so it can be lifted out of a multi-purpose field
# ("30 cases available; MOQ 2 pallets (48 cases/pallet)").
_MOQ_CLAUSE_RE = re.compile(
    r'MOQ[:\s]*[\d][\d.,]*\s*(?:pallets?|full\s*trucks?|trucks?|ftl|'
    r'inner\s*box(?:es)?|cases?|bottles?|btls?|pieces?|pcs?|units?)'
    r'(?:\s*\([^)]*\))?', re.IGNORECASE)


def _apply_moq(prod: Dict[str, Any], moq_str: str) -> None:
    """Parse a 'MOQ N <unit>' string into the verbatim moq_text AND the numeric
    moq_cases/moq_pallets/moq_bottles/moq_ftl field, mirroring the xlsx parser so
    a WhatsApp MOQ persists and shows exactly like its spreadsheet twin (Samuel
    2026-11-19: WhatsApp MOQs were captured as moq_text then dropped because only
    numeric moq_* fields survived clean_product_data)."""
    s = re.sub(r'(?i)^\s*moq[:\s]*', '', (moq_str or '').strip()).strip()
    if not s:
        return
    prod["moq_text"] = s  # verbatim source wording for the export
    m = _MOQ_NUM_UNIT_RE.search(s)
    if not m:
        return
    try:
        val = float(m.group(1).replace(",", ""))
    except (TypeError, ValueError):
        return
    u = m.group(2).lower()
    if u.startswith("pallet"):
        prod["moq_pallets"] = val; prod["moq_unit"] = "pallets"
    elif u in ("ftl",) or "truck" in u:
        prod["moq_ftl"] = val; prod["moq_unit"] = "ftl"
    elif u.startswith("piece") or u in ("pcs", "pc"):
        prod["moq_bottles"] = val; prod["moq_unit"] = "pieces"
    elif u.startswith("bottle") or u.startswith("btl"):
        prod["moq_bottles"] = val; prod["moq_unit"] = "bottles"
    elif "inner" in u:  # inner box(es) — a pack unit the source states explicitly
        prod["moq_cases"] = val; prod["moq_unit"] = "inner boxes"
    else:  # cases, units → the trading case count
        prod["moq_cases"] = val; prod["moq_unit"] = "cases"


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
                _apply_moq(prod, f)
                continue
            # MOQ embedded in a multi-purpose field ("30 cases available; MOQ 2
            # pallets (48 cases/pallet)"): lift the MOQ clause out, then let the
            # remainder parse as the quantity it also carries (Samuel 2026-11-19,
            # Perrier-Jouët Belle Epoque).
            if "moq" in low:
                _mc = _MOQ_CLAUSE_RE.search(f)
                if _mc:
                    _apply_moq(prod, _mc.group(0))
                    f = (f[:_mc.start()] + f[_mc.end():]).strip(" ;,|").strip()
                    low = f.lower()
                    if not f:
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
                # A SECOND delivery option quoted TBC / on-request in the same
                # field ("DAP Riga: EUR 144/case; EXW Breda: price TBC") — keep it
                # visible as a TBC alternative instead of dropping it (Samuel
                # 2026-11-19, Veuve Clicquot).
                _alt = re.search(
                    r'[;/]\s*((?:EXW|DAP|DDP|FCA|CIF|CFR|FOB|CPT|Ex)\b[^;/]*?'
                    r'(?:price\s*tbc|\btbc\b|on\s*request|to be confirmed)[^;/]*)',
                    f, re.IGNORECASE)
                if _alt:
                    _an = "Also available " + _alt.group(1).strip()
                    prod["commercial_conditions"] = (
                        (prod.get("commercial_conditions") + " | " + _an)
                        if prod.get("commercial_conditions") else _an)
                    prod["needs_manual_review"] = True
                continue
            # A standalone delivery-term field ("DAP Riga? only if full truck",
            # "location Loendersloot; Incoterm not stated") — extract incoterm +
            # location and keep any conditional / not-stated qualifier, so a
            # per-line term is not lost and then back-filled with a document
            # default (Samuel 2026-11-19: Rémy Martin conditional DAP Riga and
            # Beefeater's Loendersloot-with-no-incoterm were shown FCA Antwerp).
            _im0 = _LL_INCO_RE.search(f)
            _em0 = _LL_EX_RE.search(f) if not _im0 else None
            _locm0 = re.search(r'location[:\s]+([A-Z][A-Za-z]+)', f, re.IGNORECASE)
            if (_im0 or _em0 or _locm0) and not _LL_PRICE_RE.search(f):
                if _im0 and not prod.get("incoterm"):
                    prod["incoterm"] = _im0.group(1).upper(); prod["location"] = _im0.group(2)
                elif _em0 and not prod.get("incoterm"):
                    prod["incoterm"] = "EXW"; prod["location"] = _em0.group(1)
                if _locm0 and not prod.get("location"):
                    prod["location"] = _locm0.group(1)
                if re.search(r'only if|conditional|\?|\btbc\b|to be confirmed|not stated', f, re.IGNORECASE):
                    _dn = "Delivery note: " + f.strip().strip('"').strip("'").strip('“”')
                    prod["commercial_conditions"] = (
                        (prod.get("commercial_conditions") + " | " + _dn)
                        if prod.get("commercial_conditions") else _dn)
                    prod["needs_manual_review"] = True
                continue
            if _PIPE_SIZE_RE.match(f) and len(f) <= 16:
                sm = _LL_SIZE_RE.search(f)
                if sm:
                    prod["unit_volume_ml"] = _ll_size_ml(sm.group(1), sm.group(2))
                else:
                    # A weight ("210g") is a solid's size — store grams, never ml
                    # (Samuel 2026-11-19: Nishane candle size was blank on WhatsApp).
                    _gm = re.search(r'(\d+(?:[.,]\d+)?)\s*g\b', f, re.IGNORECASE)
                    if _gm:
                        prod["size_weight_g"] = float(_gm.group(1).replace(",", "."))
                continue
            if _PIPE_PACK_HINT.search(f):
                _pk = _parse_pack(f)
                if _pk:
                    prod["units_per_case"] = _pk
                    # An inner-box pack ("4 inner boxes x 12 pieces = 48 pieces")
                    # carries the per-inner-box count (12, via _parse_pack) AND
                    # the piece total (48) in ONE field, and is priced per piece.
                    # Capture that total as the quantity here (the fall-through
                    # that used to do it no longer runs now that _parse_pack
                    # returns a value), and tag the MOQ/export in inner boxes
                    # (Samuel 2026-11-19: Montblanc shown "per piece").
                    if re.search(r'inner\s*box', f, re.IGNORECASE):
                        _tot = re.search(r'=\s*(\d[\d,]*)\s*(?:pieces?|pcs?)',
                                         f, re.IGNORECASE)
                        if _tot and not any(prod.get(k) for k in (
                                "quantity_bottles", "quantity_case",
                                "quantity_pallets", "quantity_ftl")):
                            prod["quantity_bottles"] = _num(_tot.group(1).replace(",", ""))
                            prod["quantity_unit"] = "pieces"
                    continue
                # Not a real case pack (an individual-piece line like "single
                # candle") — fall through so the quantity branch can capture a
                # "= N pieces" TOTAL that would otherwise be swallowed here.
            if _PIPE_FTL_RE.search(f):
                # Strip the FTL token AND any explicit "quantity not stated"
                # negation, then test what remains: only a REAL count (e.g.
                # "800 cases FTL") should fall through to the quantity branch.
                # "1 full truck; case quantity not stated" leaves no count, so it
                # is a full-truck quantity with the case count unknown.
                _resid = _QTY_NEG_RE.sub(" ",
                    re.sub(_PIPE_FTL_RE.pattern, " ", f, flags=re.IGNORECASE))
                if not _PIPE_QTY_HINT.search(_resid):
                    prod["quantity_ftl"] = 1.0; prod["quantity_unit"] = "ftl"
                    if _QTY_NEG_RE.search(f):
                        _qn = "Case quantity not stated by supplier"
                        prod["commercial_conditions"] = (
                            (prod.get("commercial_conditions") + " | " + _qn)
                            if prod.get("commercial_conditions") else _qn)
                        prod["needs_manual_review"] = True
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


# ───────────────────────────────────────────────────────────────────────────
# Generic delimited table (schema engine) — any delimiter, columns classified
# by CONTENT rather than a hard-coded layout. This is what lets a tabular
# supplier we have never seen extract deterministically instead of via the LLM
# (Samuel 2026-11-18 schema-cache). Guarded + gated by the count self-check, so
# a mis-detected table is caught, never silently trusted.
# ───────────────────────────────────────────────────────────────────────────
_DELIMS = ["|", "\t", ";"]


def _classify_cell(cell: str, price_fn, pack_hint, size_re, qty_hint) -> str:
    s = (cell or "").strip()
    if not s:
        return "empty"
    if _PIPE_T_RE.match(s):
        return "customs"
    pc = price_fn(s)
    if pc["currency"] or pc["on_request"]:
        return "price"
    if size_re.match(s) and len(s) <= 16:
        return "size"
    if pack_hint.search(s):
        return "pack"
    if qty_hint.search(s) or _PIPE_FTL_RE.search(s):
        return "qty"
    if _LL_INCO_RE.search(s) or _LL_EX_RE.search(s):
        return "incoterm"
    letters = sum(c.isalpha() for c in s)
    digits = sum(c.isdigit() for c in s)
    if letters >= 2 and letters >= digits:
        return "name"
    return "other"


def _detect_delimiter(lines: List[str]):
    """Return (delimiter, [data rows as cell-lists]) for the most consistent
    delimited table among the lines, or (None, [])."""
    best = (None, [])
    for d in _DELIMS:
        rows = []
        for ln in lines:
            if ln.count(d) >= 2:
                cells = [c.strip() for c in ln.split(d)]
                rows.append(cells)
        if len(rows) < 3:
            continue
        # keep the dominant column count (most rows agree on it)
        from collections import Counter
        counts = Counter(len(r) for r in rows)
        col_n, n_rows = counts.most_common(1)[0]
        if col_n < 3 or n_rows < 3:
            continue
        kept = [r for r in rows if len(r) == col_n]
        if len(kept) > len(best[1]):
            best = (d, kept)
    return best


def parse_delimited_table_offer(text: str) -> Optional[List[Dict[str, Any]]]:
    """Return raw product dicts when `text` contains a consistent delimited
    table whose columns we can classify to a name + a price, else None. Columns
    are classified by the content of their cells (majority vote), so no fixed
    layout is required."""
    from core.xlsx_structured import (_parse_price_cell, _parse_pack,
                                      _parse_quantity_cell, _norm)
    lines = _clean_lines(text)
    delim, rows = _detect_delimiter(lines)
    if not delim or len(rows) < 3:
        return None
    col_n = len(rows[0])

    # Drop a header row (cells are mostly labels, not data) from the data set.
    def _is_header(r):
        return sum(1 for c in r
                   if _classify_cell(c, _parse_price_cell, _PIPE_PACK_HINT,
                                     _PIPE_SIZE_RE, _PIPE_QTY_HINT)
                   in ("name", "empty")) >= max(1, col_n - 1)
    data = rows[1:] if rows and _is_header(rows[0]) else rows
    if len(data) < 3:
        data = rows

    # Classify each column by majority vote across the data rows.
    roles: List[str] = []
    for c in range(col_n):
        from collections import Counter
        votes = Counter(
            _classify_cell(r[c] if c < len(r) else "", _parse_price_cell,
                           _PIPE_PACK_HINT, _PIPE_SIZE_RE, _PIPE_QTY_HINT)
            for r in data)
        votes.pop("empty", None)
        roles.append(votes.most_common(1)[0][0] if votes else "other")

    # Need a name column and a price column, or this is not an offer table.
    if "name" not in roles or "price" not in roles:
        return None
    name_col = roles.index("name")

    products: List[Dict[str, Any]] = []
    for r in data:
        if len(r) != col_n:
            continue
        prod: Dict[str, Any] = {"category_slug": "wines_spirits"}
        name = re.sub(r'["“”]', '', r[name_col]).strip()
        if not name or len(name) > 70 or not re.search(r"[A-Za-z]", name):
            continue
        prod["product_name"] = name
        for c, role in enumerate(roles):
            if c == name_col or c >= len(r):
                continue
            cell = r[c]
            if role == "price":
                pc = _parse_price_cell(cell)
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
                elif pc["price_value"] is not None and not pc["basis_ambiguous"]:
                    prod["price_per_unit"] = pc["price_value"]
            elif role == "size":
                sm = _LL_SIZE_RE.search(cell)
                if sm:
                    prod["unit_volume_ml"] = _ll_size_ml(sm.group(1), sm.group(2))
            elif role == "pack":
                prod["units_per_case"] = _parse_pack(cell)
            elif role == "qty":
                if _PIPE_FTL_RE.search(cell):
                    prod["quantity_ftl"] = 1.0; prod["quantity_unit"] = "ftl"
                else:
                    n, u = _parse_quantity_cell(cell)
                    if n is not None:
                        un = _norm(u)
                        if un in ("bottle", "bottles", "btl", "btls", "piece",
                                  "pieces", "pcs", "pc", "can", "cans"):
                            prod["quantity_bottles"] = n
                            prod["quantity_unit"] = "pieces" if un.startswith(("pc", "pie", "can")) else "bottles"
                        elif un in ("pallet", "pallets"):
                            prod["quantity_pallets"] = n; prod["quantity_unit"] = "pallets"
                        else:
                            prod["quantity_case"] = n; prod["quantity_unit"] = "cases"
            elif role == "customs":
                prod["custom_status"] = "T" + re.sub(r"\D", "", cell)
            elif role == "incoterm":
                im = _LL_INCO_RE.search(cell)
                if im:
                    prod["incoterm"] = im.group(1).upper(); prod["location"] = im.group(2)
                else:
                    em = _LL_EX_RE.search(cell)
                    if em:
                        prod["incoterm"] = "EXW"; prod["location"] = em.group(1)
        products.append(prod)

    if len(products) < 3:
        return None
    logger.info(f"[email_table] generic delimited table ('{delim}') recognised — "
                f"{len(products)} product(s), roles={roles}")
    return products
