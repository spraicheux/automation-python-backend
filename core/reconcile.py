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
_CUR_ALT = r"€|£|\$|¥|\b(?:eur|euro|usd|gbp|hkd|chf|sgd|jpy|aed|cny|cad|aud)\b"
# A price token, currency either before or after the number.
_PRICE_RE = re.compile(
    rf"(?P<c1>{_CUR_ALT})\s*(?P<a1>\d[\d.,]*)"
    rf"|(?P<a2>\d[\d.,]*)\s*(?P<c2>{_CUR_ALT})",
    re.IGNORECASE)
# A quantity token ("300 cases", "1,560 cases", "9,600 bottles", "48 pieces").
_QTY_RE = re.compile(
    r"\b(\d[\d,]*)\s*(cases?|cartons?|bottles?|btls?|pcs?|pieces?|cans?|pallets?|units?)\b",
    re.IGNORECASE)
# An on-request / no-price marker.
_NOPRICE_RE = re.compile(
    r"on\s*request|call\s*for\s*quot|quotation|price\s*tbc|\btbc\b|p\.?o\.?a\.?",
    re.IGNORECASE)


def _amt(raw: str) -> Optional[float]:
    """English number format: comma = thousands, dot = decimal."""
    try:
        return round(float(str(raw).replace(",", "").strip()), 2)
    except (TypeError, ValueError):
        return None


def _cur(raw: Any) -> Optional[str]:
    if not raw:
        return None
    return _CUR.get(str(raw).strip().lower())


def _line_prices(line: str) -> List[Tuple[float, Optional[str]]]:
    out = []
    for m in _PRICE_RE.finditer(line):
        a = _amt(m.group("a1") or m.group("a2"))
        c = _cur(m.group("c1") or m.group("c2"))
        if a is not None and a > 0:
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
    priced, noprice = [], []
    for raw in source_text.split("\n"):
        line = re.sub(r"^[>\s]+", "", raw).strip()
        if not line:
            continue
        pr = _line_prices(line)
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

    missed = [s["line"] for s in (priced + noprice) if not s["matched"]]
    phantom = [_row_label(rs["row"]) for rs in sigs
               if not rs["matched"] and rs["prices"]]
    total_src = len(priced) + len(noprice)
    return {
        "missed": missed,
        "phantom": phantom,
        "source_lines": total_src,
        "matched": total_src - len(missed),
    }
