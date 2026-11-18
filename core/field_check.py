"""core/field_check.py

Field-accuracy self-check (Samuel 2026-11-18). The count self-check
(core.completeness → import_incomplete) says HOW MANY rows; core.reconcile says
WHICH source lines were missed. This says a stored row's FIELDS are internally
inconsistent in a way that signals a wrong extraction — the class of error a
row-count check can never see (a price with no currency, a currency invented on
a price-less row, a case price that isn't unit×pack, grams stored as ml).

Deterministic, no network, no source text needed — it inspects the stored rows
only, so it runs on EVERY ingest path (xlsx, PDF, email, WhatsApp). Runs after
persistence and records the issues on source_files.reconciliation so the
dashboard banner and the export show exactly which field looks wrong.
"""
from __future__ import annotations

from typing import Any, Dict, List

_KNOWN_CUR = {
    "EUR", "USD", "GBP", "HKD", "CHF", "SGD", "JPY", "AED", "CNY", "CAD",
    "AUD", "NZD", "SEK", "DKK", "NOK", "PLN", "ZAR",
}


def _pos(v: Any) -> bool:
    try:
        return isinstance(v, (int, float)) and float(v) > 0
    except (TypeError, ValueError):
        return False


def _name(r: Dict[str, Any]) -> str:
    n = ((r.get("brand") or "") + " " + (r.get("product_name") or "")).strip()
    return n or "a row"


def check_field_accuracy(rows: List[Dict[str, Any]]) -> List[str]:
    """Return a list of human-readable field-consistency issues (empty = clean)."""
    issues: List[str] = []
    for r in rows:
        name = _name(r)
        ppu, ppc = r.get("price_per_unit"), r.get("price_per_case")
        has_price = _pos(ppu) or _pos(ppc)
        cur = (str(r.get("currency") or "")).strip()
        cc = str(r.get("commercial_conditions") or "").lower()
        upc = r.get("units_per_case")

        # A price must name its currency.
        if has_price and not cur:
            issues.append(f"{name}: has a price but no currency")

        # A currency with no price (and no amount retained as-quoted) is
        # invented — a genuinely price-less row carries no currency.
        if cur and not has_price and "as quoted" not in cc:
            issues.append(f"{name}: currency {cur} but no price in the row")

        # A per-case price must equal unit price × case pack (small tolerance),
        # or the case price is fabricated / the pack is wrong.
        if _pos(ppu) and _pos(ppc) and _pos(upc) and float(upc) > 1:
            expected = float(ppu) * float(upc)
            if abs(float(ppc) - expected) > max(0.02, expected * 0.02):
                issues.append(
                    f"{name}: case price {float(ppc):g} does not match "
                    f"unit {float(ppu):g} x pack {float(upc):g} ({expected:.2f})")

        # A size is a volume OR a weight, never both.
        if _pos(r.get("size_weight_g")) and _pos(r.get("unit_volume_ml")):
            issues.append(f"{name}: has both a gram weight and an ml volume")

        # Currency must be a known ISO code.
        if cur and cur.upper() not in _KNOWN_CUR:
            issues.append(f"{name}: unrecognized currency {cur!r}")

    return issues
