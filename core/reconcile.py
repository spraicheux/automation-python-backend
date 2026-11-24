"""core/reconcile.py

Row-level self-check (Tier 2). The count self-check (core/completeness) says a
job's row count doesn't match the source; this says WHICH rows.

Price is the one field that survives extraction mostly intact, so we join the
raw source to the stored rows on price:

  * a source price line with no matching stored row  -> a MISSED product (we
    report the raw line, e.g. "Glenfiddich 18 … EUR 534/case");
  * a stored row whose price matches no source line  -> a PHANTOM / over-split.

A source product line with no price at all ("call for quotation", "price on
request") can't join on price, so it falls back to a brand + quantity signature.

Deterministic, no network. Intended to run after persistence and record the
result on the source so a human (and the export) sees exactly what was dropped.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from core.completeness import _is_header_footer_line

_CUR = {
    "€": "EUR", "£": "GBP", "$": "USD", "¥": "JPY",
    "eur": "EUR", "euro": "EUR", "euros": "EUR", "usd": "USD", "gbp": "GBP",
    "hkd": "HKD", "chf": "CHF", "sgd": "SGD", "jpy": "JPY", "aed": "AED",
    "cny": "CNY", "cad": "CAD", "aud": "AUD",
}
# A word-currency may sit directly against a digit ("11,25Euro") but not inside
# another word ("neuro"), so bound it by non-letters rather than \b.
_CUR_ALT = (r"€|£|\$|¥|(?<![a-z])(?:eur|euro|usd|gbp|hkd|chf|sgd|jpy|aed|cny|cad|aud)(?![a-z])")
# A price token, currency either before or after the number. The gap is SPACES
# only (not tab/newline) so a tab-separated quantity never glues to the next
# cell's currency symbol — "1215<TAB>$89.25" is 89.25, not 1215 (Samuel
# 2026-11-20: URD tab-table read every Qnty/cs column as the price).
_PRICE_RE = re.compile(
    rf"(?P<c1>{_CUR_ALT})[  ]*(?P<a1>\d[\d.,]*)"
    rf"|(?P<a2>\d[\d.,]*)[  ]*(?P<c2>{_CUR_ALT})",
    re.IGNORECASE)
# A quantity token ("300 cases", "1,560 cases", "9,600 bottles", "48 pieces").
_QTY_RE = re.compile(
    r"\b(\d[\d,]*)\s*(cases?|cartons?|bottles?|btls?|pcs?|pieces?|cans?|pallets?|units?)\b",
    re.IGNORECASE)
# An on-request / no-price marker.
_NOPRICE_RE = re.compile(
    r"on\s*request|call\s*for\s*quot|quotation|price\s*tbc|\btbc\b|p\.?o\.?a\.?",
    re.IGNORECASE)
# A priced line that is a COMMERCIAL-TERMS header, not a product — it carries a
# price (a minimum order value, an order threshold) but names no goods, so it must
# not be read as a missed product line (Samuel 2026-11-21: Fenwick "Minimum order
# value EUR 1,850" reported as a missed source line).
_TERMS_HEADER_RE = re.compile(
    r"minimum\s+order|min\.?\s*order\b|\bmov\b|order\s+value|"
    r"minimum\s+order\s+value|subject\s+to\s+(?:prior\s+sale|unsold|confirmation)|"
    r"^\s*contact\b|payment\s+terms|terms\s*:", re.IGNORECASE)
# A pack-format product line: "<name> 24x33 cl cans" / "<name> 4×6×33 cl bottles".
_FORMAT_RE = re.compile(
    r"^(.*?)(\d+)\s*[x×*]\s*(\d+)(?:\s*[x×*]\s*(\d+))?\s*cl\s*(?:cans?|bottles?)\b",
    re.IGNORECASE)


def _amt_variants(raw: str) -> List[float]:
    """Plausible numeric values for a raw amount token under BOTH conventions —
    English '1,234.50' (comma = thousands) and European '7,05' / '341,25' (comma =
    decimal). The self-check joins a source price to a stored row if EITHER reading
    agrees, so a European 'Euro 7,05' still matches its 7.05 row instead of being
    read as 705 and falsely reported missed (Samuel 2026-11-20: ERKAN beer offer,
    every line priced '7,05 Euro per case')."""
    s = str(raw).strip()
    out: set = set()
    try:                                   # English: strip thousands commas
        v = round(float(s.replace(",", "")), 2)
        if v > 0:
            out.add(v)
    except (TypeError, ValueError):
        pass
    # European decimal ONLY when a comma is followed by exactly two digits and no
    # dot-decimal is present ("7,05", "10,20", "341,25") — never for "1,800".
    if re.search(r",\d{2}\b", s) and not re.search(r"\.\d", s):
        try:
            v = round(float(s.replace(".", "").replace(",", ".")), 2)
            if v > 0:
                out.add(v)
        except (TypeError, ValueError):
            pass
    return list(out)


def _cur(raw: Any) -> Optional[str]:
    if not raw:
        return None
    return _CUR.get(str(raw).strip().lower())


def _line_prices(line: str) -> List[Tuple[float, Optional[str]]]:
    out = []
    for m in _PRICE_RE.finditer(line):
        c = _cur(m.group("c1") or m.group("c2"))
        for a in _amt_variants(m.group("a1") or m.group("a2")):
            out.append((a, c))
    return out


def _prices_overlap(a: List[Tuple[float, Optional[str]]],
                    b) -> bool:
    for (av, ac) in a:
        for (bv, bc) in b:
            if abs(av - bv) < 0.01 and (ac is None or bc is None or ac == bc):
                return True
    return False


def _brand_of(line: str) -> Optional[str]:
    try:
        from core.brand_gazetteer import extract_brand_prefix
        # try the line as-is and with a leading number/bullet stripped
        for cand in (line, re.sub(r"^\s*\d+[).]\s*", "", line)):
            hit = extract_brand_prefix(cand.strip().strip('"“”').lstrip("> ").strip())
            if hit:
                return hit[0]
    except Exception:
        pass
    return None


def _qty_of(line: str) -> Optional[int]:
    m = _QTY_RE.search(line)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def _row_label(r: Dict[str, Any]) -> str:
    name = ((r.get("brand") or "") + " " + (r.get("product_name") or "")).strip()
    price = r.get("price_per_case") or r.get("price_per_unit")
    cur = r.get("currency") or ""
    return f"{name or 'Not Found'} ({cur} {price})".strip()


def reconcile_source_vs_rows(source_text: str,
                             rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Match raw-source product lines to stored rows by price (with a brand+qty
    fallback for no-price lines). Returns the exact missed source lines and
    phantom stored rows."""
    if not source_text:
        return {"missed": [], "phantom": [], "source_lines": 0, "matched": 0}

    # ── Source side ──────────────────────────────────────────────────────────
    try:
        from core.inquiry import is_strong_buy_line
    except Exception:
        is_strong_buy_line = lambda _t: False
    priced, noprice = [], []
    for raw in source_text.split("\n"):
        line = re.sub(r"^[>\s]+", "", raw).strip()
        if not line:
            continue
        # A buy-side ENQUIRY line ("a customer wants … can you source?") is a
        # request, not a sellable offer: it is deliberately not stored, so it must
        # NOT be reported as a "missed" sale (Samuel 2026-11-24: North Pier 2024;
        # the same point he raised for Alder A07's buying enquiry).
        if is_strong_buy_line(line):
            continue
        pr = _line_prices(line)
        if pr and _TERMS_HEADER_RE.search(line):
            # a priced commercial-terms header (MOV / "subject to prior sale" /
            # contact line) names no goods — not a product, not a missed line.
            continue
        if pr:
            # a real price token (currency next to a number) is a strong
            # product-line signal — never filter it as header/footer.
            priced.append({"line": line, "prices": pr, "matched": False})
        elif (_NOPRICE_RE.search(line) and _brand_of(line)
                and not _is_header_footer_line(line)):
            # a genuine product line that just withholds the price
            noprice.append({"line": line, "brand": _brand_of(line),
                            "qty": _qty_of(line), "matched": False})

    # ── Stored side ──────────────────────────────────────────────────────────
    sigs = []
    for r in rows:
        ps = []
        c = _cur(r.get("currency"))
        for f in ("price_per_unit", "price_per_case",
                  "price_per_unit_eur", "price_per_case_eur"):
            v = r.get(f)
            try:
                if v and float(v) > 0:
                    ps.append((round(float(v), 2), c))
            except (TypeError, ValueError):
                pass
        if not ps:
            # A basis-unconfirmed row carries its quoted amount in the conditions,
            # NOT in a price field (kept out of Best Prices). Join it on that
            # amount so its priced source line is not falsely reported missed
            # (Samuel 2026-11-20: Woodford "USD 96, basis unconfirmed" showed as
            # "not imported although it is visible").
            cc = str(r.get("commercial_conditions") or "")
            mq = re.search(r"as quoted:\s*([^\d]*?)([\d][\d.,]*)", cc, re.IGNORECASE)
            if mq:
                qc = c or _cur((mq.group(1) or "").strip())
                for a in _amt_variants(mq.group(2)):
                    ps.append((a, qc))
        sigs.append({"row": r, "prices": ps, "matched": False,
                     "brand": (r.get("brand") or "").strip().lower(),
                     "qty": r.get("quantity_case") or r.get("quantity_bottles")})

    # ── Pass 1: priced lines match on price (consume each stored row once) ───
    for s in priced:
        for rs in sigs:
            if rs["matched"] or not rs["prices"]:
                continue
            if _prices_overlap(s["prices"], rs["prices"]):
                s["matched"] = rs["matched"] = True
                break

    # ── Pass 2: any still-unmatched source line matches on BRAND against a
    # remaining stored row. This covers a product that is correctly stored
    # WITHOUT a price — an ambiguous basis ("USD 75, per bottle or case?") or an
    # on-request line — whose price can't join, so it must fall back to brand
    # (Samuel 2026-11-18: Beefeater was wrongly reported missed). A genuinely
    # dropped product has no stored row of that brand, so it still shows missed.
    for s in (priced + noprice):
        if s["matched"]:
            continue
        b = (s.get("brand") or _brand_of(s["line"]) or "").strip().lower()
        if not b:
            continue
        for rs in sigs:
            if rs["matched"]:
                continue
            if rs["brand"] and b in rs["brand"]:
                s["matched"] = rs["matched"] = True
                break

    # ── Twin pack formats at ONE price ───────────────────────────────────────
    # Two formats of a product quoted at the same price consume one priced line
    # each in Pass 1, leaving the second row looking like a phantom. It is real
    # when its price matches a source line AND that same line names its quantity
    # (Samuel 2026-11-21: Westhaven Brooklyn "63 trays of singles and 47 trays of
    # the wrapped packs", both €11.52 — the 47 row is not a phantom).
    for rs in sigs:
        if rs["matched"] or not rs["prices"]:
            continue
        q = rs.get("qty")
        if not q:
            continue
        try:
            qn = str(int(float(q)))
        except (TypeError, ValueError):
            continue
        for s in priced:
            if _prices_overlap(rs["prices"], s["prices"]) and re.search(rf"\b{qn}\b", s["line"]):
                rs["matched"] = True
                break

    missed = [s["line"] for s in (priced + noprice) if not s["matched"]]
    phantom = [_row_label(rs["row"]) for rs in sigs
               if not rs["matched"] and rs["prices"]]
    total_src = len(priced) + len(noprice)

    # ── Format completeness (price-INDEPENDENT) ──────────────────────────────
    # In an 'ON THE FLOOR'-style list, every "<name> A×B cl cans/bottles" product
    # line is its own SKU, even when two formats of one product share a price.
    # Match each pack-format line to a row by name-tokens + unit count; an
    # unmatched format is MISSED although its price sits on a twin — so an 8-row
    # import of a 9-format source no longer reports "0 missed" (Samuel 2026-11-20:
    # Carlsberg Lager 24×33 vs 4×6×33, both €7.05).
    _fmt = []
    for raw in source_text.split("\n"):
        fm = _FORMAT_RE.search(raw.strip())
        if fm:
            a = int(fm.group(2)); c = int(fm.group(3)); d = fm.group(4)
            _fmt.append((raw.strip(), fm.group(1), a * c if d else a))
    if len(_fmt) >= 3:
        _rs = []
        for r in rows:
            _tk = set(t for t in re.findall(
                r"[a-z0-9]+", ((r.get("brand") or "") + " " + (r.get("product_name") or "")).lower())
                if len(t) > 2)
            try:
                _u = float(r.get("units_per_case")) if r.get("units_per_case") else None
            except (TypeError, ValueError):
                _u = None
            _rs.append([_tk, _u, False])
        for line, name, units in _fmt:
            ntoks = set(t for t in re.findall(r"[a-z0-9]+", name.lower()) if len(t) > 2)
            hit = False
            for rsig in _rs:
                if rsig[2] or not ntoks or not (ntoks <= rsig[0]):
                    continue
                if rsig[1] is None or rsig[1] == float(units):
                    rsig[2] = True; hit = True; break
            if not hit and line not in missed:
                missed.append(line)

    return {
        "missed": missed,
        "phantom": phantom,
        "source_lines": total_src,
        "matched": total_src - len(missed),
    }
