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

        # A quantity expressed as BOTH N cases and N×pack units is the same
        # goods counted twice (Samuel 2026-11-19 Dom Pérignon: "4 cases + 24
        # units" where 24 = 4 × 6 is one order, not 48 bottles). Only fires when
        # the unit count is exactly the case×pack equivalent — a genuine "4
        # cases plus 2 loose" (bottles ≠ case×pack) is left alone.
        qc, qb = r.get("quantity_case"), r.get("quantity_bottles")
        if _pos(qc) and _pos(qb) and _pos(upc) and float(upc) > 1:
            if abs(float(qb) - float(qc) * float(upc)) < 0.5:
                issues.append(
                    f"{name}: quantity shown as both {float(qc):g} cases and "
                    f"{float(qb):g} units — the units equal the cases × pack, "
                    f"so this may double-count the same goods")

    return issues


# Standard incoterms, as whole-word tokens, for source-fidelity matching.
_INCOTERMS = ("EXW", "FCA", "FOB", "CIF", "CFR", "CPT", "CIP", "DAP", "DDP",
              "DPU", "DAT", "FAS")


def _windows(source: str, name: str, radius: int = 200) -> List[str]:
    """The slice(s) of `source` that describe this product. In a line/pipe list
    each offer is its own line, so a product's incoterm, size and conditions sit
    on that one line — we return the matching line(s) and NOT a character window,
    which in a dense list bleeds into the neighbour above/below and would hide an
    inherited incoterm behind a neighbour's real one. For a product whose token
    only appears mid-prose (no line match), fall back to a ±radius char window.
    Empty ⇒ the product could not be located, so the caller skips rather than
    risk a false "not in source" flag."""
    import re
    src = source or ""
    # Anchor on the longest alphabetic token in the name (brand/product word).
    toks = [t for t in re.split(r"[^A-Za-z0-9]+", name) if len(t) >= 4]
    toks.sort(key=len, reverse=True)
    for tok in toks[:2]:
        t = tok.lower()
        lines = [ln for ln in src.splitlines() if t in ln.lower()]
        if lines:
            return lines
    # Prose fallback: character window around the first token hit.
    low = src.lower()
    for tok in toks[:2]:
        i = low.find(tok.lower())
        if i >= 0:
            return [src[max(0, i - radius): i + len(tok) + radius]]
    return []


def check_source_fidelity(rows: List[Dict[str, Any]],
                          source_text: str) -> List[str]:
    """Verify stored FIELDS against the SOURCE, not just against each other.

    This is the check the internal pass (check_field_accuracy) structurally
    cannot do and the reason those rows were green: an inherited incoterm, a
    grams-as-ml size, an assumed per-case basis and a dropped price option are
    each internally consistent — they only disagree with the source. Each test
    below is conservative (it skips when it cannot locate the product in the
    source) so a corrected row never trips it; verified on prod to flag the old
    wrong rows and pass the fixed ones with zero false positives.
    """
    import re
    issues: List[str] = []
    src = source_text or ""
    # The source-vs-stored tests (A, B) need the source text; the marker test (C)
    # reads the row's OWN recorded conditions, so it must still run for an Excel
    # offer that has no free-text source — otherwise a basis-unconfirmed Excel row
    # stays silently green (Samuel 2026-11-20: Larch Pier Plymouth).
    have_src = bool(src.strip())

    try:
        from core.deterministic_rules import _has_mixed_incoterms
        mixed = _has_mixed_incoterms(src)
    except Exception:
        mixed = False

    # (C) markers are collapsed so one assumption shared by every row of a file
    # (EPIC's per-case basis) reads as a single line, not N identical ones.
    _marker_rows: Dict[str, List[str]] = {}

    for r in rows:
        name = _name(r)
        inco = (str(r.get("incoterm") or "")).strip().upper()
        cc = str(r.get("commercial_conditions") or "")
        cc_low = cc.lower()
        wins = _windows(src, name) if have_src else []

        # (A) Inherited incoterm. In a MIXED-incoterm document each line must
        # state its own term; if the row carries one the source never states
        # near this product, it bled down from a header (Beefeater/Rémy shown
        # FCA Antwerp when the source gave neither). EXW is matched in its
        # written forms too — "Ex works", "Ex warehouse", "Ex <Place>" (FBC
        # writes it "Ex Loen"/"Ex Miami", not "EXW"), else every such line is a
        # false "inherited" flag.
        if mixed and inco in _INCOTERMS and wins:
            if inco == "EXW":
                pat = r"\bEXW\b|\bex[\s\-]+(?:works?|warehouse|[A-Z])"
            else:
                pat = rf"\b{inco}\b"
            if not any(re.search(pat, w, re.IGNORECASE) for w in wins):
                issues.append(
                    f"{name}: incoterm {inco} is not stated in the source for "
                    f"this line (likely inherited from the document header)")

        # (B) Grams stored as ml. If the row carries a volume but the source
        # gives that same number as a weight for this product, the unit is
        # wrong (Nishane 210 g read as 210 ml).
        ml = r.get("unit_volume_ml")
        if _pos(ml) and wins:
            n = f"{float(ml):g}"
            g_hit = any(re.search(rf"\b{re.escape(n)}\s*g\b", w, re.IGNORECASE)
                        for w in wins)
            ml_hit = any(re.search(rf"\b{re.escape(n)}\s*ml\b", w, re.IGNORECASE)
                         for w in wins)
            if g_hit and not ml_hit:
                issues.append(
                    f"{name}: size stored as {n} ml but the source states "
                    f"{n} g (a weight, not a volume)")

        # (C) Assumed / qualified / dropped-option markers the parser recorded
        # must be SURFACED, not left silently green. These values are genuine
        # assumptions or carry an alternative the source leaves open (EPIC's
        # assumed per-case basis; a "price TBC" option kept as a note), so a
        # reviewer has to confirm them against the source.
        for marker, why in (
            ("assumed", "a basis/value was assumed, not stated in the source"),
            ("price tbc", "the source leaves a price to be confirmed"),
            ("price to be confirmed", "the source leaves a price to be confirmed"),
            ("also available", "the source offers an alternative not priced here"),
            ("conditional", "the source makes this offer conditional"),
            ("subject to confirmation", "the source makes this subject to confirmation"),
            ("price basis not stated",
             "price basis unconfirmed — quoted amount kept, bottle/case price left blank"),
            ("basis not stated",
             "price basis unconfirmed — quoted amount kept, bottle/case price left blank"),
        ):
            if marker in cc_low:
                _marker_rows.setdefault(why, []).append(name)
                break

    # Emit one collapsed line per distinct marker reason: a single assumption
    # shared by every row (EPIC) becomes "12 rows: …", a one-off stays named.
    for why, names in _marker_rows.items():
        uniq = list(dict.fromkeys(names))
        if len(uniq) == 1:
            issues.append(f"{uniq[0]}: {why} — confirm against the source")
        else:
            shown = ", ".join(uniq[:3])
            more = f" (+{len(uniq) - 3} more)" if len(uniq) > 3 else ""
            issues.append(f"{len(uniq)} rows — {why}; confirm against the "
                          f"source: {shown}{more}")

    return issues
