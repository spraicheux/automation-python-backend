"""core/offer_reconcile.py

Deterministic post-extraction reconciliation passes for TEXT offers (email /
WhatsApp). Each operates on the list of extracted offer dicts (and, where needed,
the original source text) and returns what it changed, so the worker can sync the
DB. No LLM — these correct systematic extraction errors from the source itself,
the way the project's other completeness passes do (Samuel 2026-11-19 round).

Every function:
  * mutates offer dicts in place and marks changed ones with `_dirty=True`,
  * returns the set of uids to DROP (for merges / phantom removal), or 0/None,
  * is conservative: it only acts on an unambiguous signal, so a correct file is
    left untouched (verified against the other acceptance files).

Passes:
  merge_alt_delivery_lots   — one lot quoted under two delivery terms → one row
                              (Jägermeister 1,200 cs, Moët 85 cs double-count)
  drop_summary_rows         — header/aggregate rows that repeat a series total
                              (Octomore "180 cases" summaries)
  fix_inner_box_from_text   — inner-box pack/MOQ + no fabricated case price
                              (Montblanc tester on the WhatsApp path)
  recover_loose_pieces_qty  — "N pieces available" the extractor left blank
                              (L'Oréal 36 pieces)
  flag_unstated_price_basis — price basis the source does NOT state → review
                              (Beefeater USD 75, bottle-or-case)
  repair_case_prices_from_source — re-pair each row's per-case price to its
                              source block by the (correct) per-bottle price
                              (EPIC/Grant's one-product price shift)
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

_NOT = {"", "not found", "not stated", "none", "n/a", "-", "—"}


def _clean(v) -> str:
    s = (str(v).strip() if v is not None else "")
    return "" if s.lower() in _NOT else s


def _norm(s) -> str:
    return re.sub(r"\s+", " ",
                  re.sub(r"[^a-z0-9. ]", "", (s or "").lower().replace("’", "").replace("'", ""))).strip()


def _ident(o: Dict[str, Any]) -> str:
    br = _clean(o.get("brand")); nm = _clean(o.get("product_name"))
    return (br + " " + nm).strip() if br else nm


def _primary_qty(o: Dict[str, Any]) -> Optional[float]:
    for f in ("quantity_case", "quantity_bottles", "quantity_ftl", "quantity_pallets"):
        v = o.get(f)
        try:
            if v and float(v) > 0:
                return float(v)
        except (TypeError, ValueError):
            pass
    return None


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mark(o): o["_dirty"] = True


# ─────────────────────────────────────────────────────────────────────────────
def merge_alt_delivery_lots(offers: List[Dict[str, Any]]) -> Set[str]:
    """One physical lot quoted under two delivery terms (same product, same
    quantity, same — or one missing — case price) is ONE offer, not two. Keep the
    richest row, fold the alternative incoterm/location into its conditions, drop
    the rest. Two genuinely different SKUs (Tanqueray 1L vs 70cl: different
    quantity AND price) are left as two rows."""
    dropped: Set[str] = set()
    groups: Dict[tuple, list] = defaultdict(list)
    for o in offers:
        groups[(_norm(o.get("brand")), _norm(o.get("product_name")))].append(o)
    for g in groups.values():
        if len(g) < 2:
            continue
        by_lot: Dict[tuple, list] = defaultdict(list)
        for o in g:
            by_lot[(_primary_qty(o), _f(o.get("units_per_case")))].append(o)
        for lot in by_lot.values():
            if len(lot) < 2:
                continue
            # require a real shared quantity so two no-quantity rows never merge
            if _primary_qty(lot[0]) is None:
                continue
            pcs = {round(_f(o.get("price_per_case")), 2) for o in lot if _f(o.get("price_per_case"))}
            locs = {(_norm(o.get("incoterm")), _norm(o.get("location"))) for o in lot}
            # Same product, same quantity, same pack. When the case prices differ
            # it is STILL one lot if the delivery terms differ — a single stock
            # quoted under two incoterms at two prices (Samuel 2026-11-20:
            # Drambuie, 364 cases, "both options are for the same stock": EUR
            # 79.20/case EXW Breda OR EUR 83.40/case DAP Lille). Only two rows with
            # the SAME delivery and different prices are genuinely separate.
            if len(pcs) > 1 and len(locs) < 2:
                continue
            keep = sorted(lot, key=lambda o: (
                bool(_f(o.get("price_per_case"))),
                bool(_clean(o.get("lead_time"))),
                not bool(o.get("needs_manual_review"))), reverse=True)[0]
            alts = []
            for o in lot:
                if o is keep:
                    continue
                dropped.add(o.get("uid"))
                term = " ".join(x for x in (_clean(o.get("incoterm")), _clean(o.get("location"))) if x)
                if term:
                    apc = _f(o.get("price_per_case")); acur = _clean(o.get("currency"))
                    if apc and len(pcs) > 1:
                        # delivery-specific alternate price → state it explicitly
                        term += f" ({(acur + ' ') if acur else ''}{apc:g}/case)"
                    elif not apc and not _f(o.get("price_per_unit")):
                        term += " (price TBC)"
                    alts.append(term)
            if alts:
                note = "Also available: " + " ; ".join(dict.fromkeys(alts))
                # When the lot shares ONE case price, state it on the alternative
                # term too — same lot, same price, not a second quote (Jägermeister
                # EXW confirmed same EUR 78/case). When the alternates carry their
                # own prices (above) the note already shows them, so skip this.
                kpc = _f(keep.get("price_per_case")); kcur = _clean(keep.get("currency"))
                if kpc and kcur and len(pcs) <= 1 and not any("price TBC" in a for a in alts):
                    note += f" (same {kcur} {kpc:g}/case)"
                cc = _clean(keep.get("commercial_conditions"))
                keep["commercial_conditions"] = (cc + " | " if cc else "") + note
            _mark(keep)
    if dropped:
        offers[:] = [o for o in offers if o.get("uid") not in dropped]
    return dropped


# ─────────────────────────────────────────────────────────────────────────────
def _tok_prefix_header(parent_id: str, child_id: str) -> bool:
    """True when child_id is a MORE-SPECIFIC edition of parent_id: same leading
    tokens, and the child extends the parent's last token with a '.N' version
    (Octomore 17 → Octomore 17.1, same length) or appends further tokens
    (Octomore 17 → Octomore 17.3 Islay Barley)."""
    pt = parent_id.split(); ct = child_id.split()
    if len(pt) == 0 or len(ct) < len(pt):
        return False
    for i in range(len(pt) - 1):
        if ct[i] != pt[i]:
            return False
    last = pt[-1]; clast = ct[len(pt) - 1]
    extends = clast.startswith(last + ".")          # 17 → 17.1
    if len(ct) > len(pt):
        return clast == last or extends
    return extends                                   # same length: must extend


def drop_summary_rows(offers: List[Dict[str, Any]]) -> Set[str]:
    """Drop header/summary rows that repeat a series' whole allocation instead of
    being a real line — e.g. an Octomore offer where "Octomore 17", "Octomore 16"
    and "Bruichladdich Octomore" each carry the full 180-case headline while the
    six real editions (17.1…16.3) carry their own 20/40 cases."""
    dropped: Set[str] = set()
    ids = [(o, _norm(_ident(o))) for o in offers]
    # Phase A — "series header" rows: a more-specific edition exists in ≥2 other
    # rows, and this row's quantity exceeds each of them (it is an aggregate).
    for o, idnt in ids:
        if not idnt:
            continue
        pq = _primary_qty(o)
        children = [c for c, cid in ids if c is not o and _tok_prefix_header(idnt, cid)]
        if len(children) >= 2:
            cq = [q for q in (_primary_qty(c) for c in children) if q]
            if pq and cq and pq > max(cq):
                dropped.add(o.get("uid"))
    # Phase B — a brand/range catch-all with NO version number whose quantity ≈
    # the sum of ≥2 numbered siblings that share its range token. Count only rows
    # not already dropped in Phase A, so the real editions are the siblings.
    live = [(o, cid) for o, cid in ids if o.get("uid") not in dropped]
    for o, idnt in live:
        pname = _clean(o.get("product_name"))
        if re.search(r"\d", pname):
            continue
        toks = [t for t in idnt.split() if len(t) > 3]
        if not toks:
            continue
        rng = toks[-1]
        sibs = [c for c, cid in live if c is not o
                and rng in cid.split() and re.search(r"\d", _clean(c.get("product_name")))]
        if len(sibs) >= 2:
            s = sum((_primary_qty(x) or 0) for x in sibs)
            pq = _primary_qty(o)
            if pq and s > 0 and pq >= 0.9 * s and pq > max((_primary_qty(x) or 0) for x in sibs):
                dropped.add(o.get("uid"))
    if dropped:
        offers[:] = [o for o in offers if o.get("uid") not in dropped]
    return dropped


# ─────────────────────────────────────────────────────────────────────────────
_INNERBOX_RE = re.compile(
    r"(\d[\d,]*)\s*inner\s*box(?:es)?\s*[x×*]\s*(\d[\d,]*)\s*(?:pieces?|pcs?|pc)\b", re.IGNORECASE)
_INNERBOX_EQ_RE = re.compile(
    r"=\s*(\d[\d,]*)\s*(?:pieces?|pcs?|pc)\b", re.IGNORECASE)
_INNERBOX_MOQ_RE = re.compile(
    r"moq\s*(\d[\d,]*)\s*inner\s*box(?:es)?\b", re.IGNORECASE)
_PER_PIECE_RE = re.compile(r"(?:per|/)\s*(?:piece|pc|pcs)\b", re.IGNORECASE)
# Reversed inner-box phrasing (Nishane 2026-11-20: "6 pieces/inner box … 5 inner
# boxes available … EUR 38/piece") — the pieces-per-box and the available box
# count are stated separately, not as "N inner boxes x M pieces".
_PIECES_PER_BOX_RE = re.compile(
    r"(\d[\d,]*)\s*(?:pieces?|pcs?|pc)\s*(?:/|per)\s*inner\s*box", re.IGNORECASE)
_BOXES_AVAIL_RE = re.compile(
    r"(\d[\d,]*)\s*inner\s*box(?:es)?\s*(?:available|avail|in\s*stock|offered)", re.IGNORECASE)
_ANY_BOXES_RE = re.compile(r"(\d[\d,]*)\s*inner\s*box(?:es)?\b", re.IGNORECASE)


def _boxes_available(ln: str) -> Optional[int]:
    """The available inner-box count on a line, preferring an explicit
    'N inner boxes available' and never mistaking the 'MOQ 1 inner box' count."""
    m = _BOXES_AVAIL_RE.search(ln)
    if m:
        return _int(m.group(1))
    for mm in _ANY_BOXES_RE.finditer(ln):
        if "moq" not in ln[max(0, mm.start() - 6):mm.start()].lower():
            return _int(mm.group(1))
    return None


def _innerbox_counts(ln: str):
    """(pieces_per_box, total_pieces) from either inner-box phrasing:
    'N inner boxes x M pieces [= T pieces]' (Montblanc) or 'M pieces/inner box …
    N inner boxes available' (Nishane). (None, None) when it isn't one."""
    m = _INNERBOX_RE.search(ln)
    if m:
        boxes = _int(m.group(1)); per = _int(m.group(2))
    else:
        pm = _PIECES_PER_BOX_RE.search(ln)
        if not pm:
            return None, None
        per = _int(pm.group(1)); boxes = _boxes_available(ln)
    if not per:
        return None, None
    eq = _INNERBOX_EQ_RE.search(ln)
    total = _int(eq.group(1)) if eq else (boxes * per if boxes else None)
    return per, total


def _int(s) -> Optional[int]:
    try:
        return int(str(s).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _source_line_for(o: Dict[str, Any], lines: List[str]) -> Optional[str]:
    """The source line whose text contains the offer's brand/product tokens."""
    toks = [t for t in re.findall(r"[a-z0-9]+", _ident(o).lower()) if len(t) > 2]
    if not toks:
        return None
    best, score = None, 0
    for ln in lines:
        low = ln.lower()
        s = sum(1 for t in toks if t in low)
        if s > score:
            best, score = ln, s
    return best if score >= 1 else None


def fix_inner_box_from_text(offers: List[Dict[str, Any]], source_text: str) -> int:
    """When the source sells a product in INNER BOXES, carry that through: pack =
    pieces per inner box, quantity in pieces, MOQ in inner boxes, and NO
    fabricated per-case price. Handles both the straight phrasing ("4 inner boxes
    x 12 pieces = 48 pieces", Montblanc) and the reversed one ("6 pieces/inner
    box … 5 inner boxes available … EUR 38/piece", Nishane — Samuel 2026-11-20),
    the way the xlsx/pipe parsers already do for the structured path."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        ln = _source_line_for(o, lines)
        if not ln:
            continue
        per, total = _innerbox_counts(ln)
        if not per:
            continue
        o["units_per_case"] = float(per)          # pieces per inner box
        o["quantity_unit"] = "pieces"
        if total:
            o["quantity_bottles"] = float(total)
        o["quantity_case"] = None                 # not cases
        mm = _INNERBOX_MOQ_RE.search(ln)
        if mm:
            o["moq_cases"] = float(_int(mm.group(1)))
            o["moq_unit"] = "inner boxes"
        # per-piece pricing: drop any fabricated per-case price, and the now-stale
        # "case price = unit × pack" info flag that would otherwise mislead.
        if _PER_PIECE_RE.search(ln):
            o["price_per_case"] = None
            o["price_per_case_eur"] = None
            o["error_flags"] = [f for f in (o.get("error_flags") or [])
                                if "price_per_case calculated" not in str(f).lower()]
        _mark(o)
        changed += 1
    return changed


# ─────────────────────────────────────────────────────────────────────────────
_LOOSE_PIECES_RE = re.compile(r"(\d[\d,]*)\s*(?:pieces?|pcs?|pc)\s+available", re.IGNORECASE)


def recover_loose_pieces_qty(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Fill a blank quantity for a loose-piece line the extractor left empty —
    the source states "N pieces available" (L'Oréal 36 pieces) but no quantity
    reached the record. Only acts when the row truly has NO quantity."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        if _primary_qty(o) is not None:
            continue
        ln = _source_line_for(o, lines)
        if not ln or "inner box" in ln.lower():
            continue
        m = _LOOSE_PIECES_RE.search(ln)
        if not m:
            continue
        n = _int(m.group(1))
        if n:
            o["quantity_bottles"] = float(n)
            o["quantity_unit"] = "pieces"
            _mark(o)
            changed += 1
    return changed


# ─────────────────────────────────────────────────────────────────────────────
_BASIS_UNSTATED_RE = re.compile(
    r"basis\s+not\s+(?:stated|given|specified)|"
    r"(?:per\s+bottle\s+or\s+case)|(?:bottle\s+or\s+case)|"
    r"per\s+\w+\s+or\s+per\s+\w+|"   # "per bottle or per case" (Woodford, Samuel 2026-11-20)
    r"price\s+basis\s+not\s+stated", re.IGNORECASE)


def flag_unstated_price_basis(offers: List[Dict[str, Any]], source_text: str) -> int:
    """When the source explicitly says the price basis is NOT stated (Beefeater
    "USD 75.00, basis not stated — per bottle or case"), the record must not pass
    a green self-check as a confident case price. Flag it for review and note it,
    so the ambiguity is visible instead of silently assumed."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        ln = _source_line_for(o, lines)
        if not ln or not _BASIS_UNSTATED_RE.search(ln):
            continue
        already = ("as quoted" in (str(o.get("commercial_conditions") or "")).lower()
                   and not _f(o.get("price_per_case"))
                   and not _f(o.get("price_per_unit")))
        if already:
            continue
        # Keep the quoted amount + currency, but CLEAR both price fields so the
        # amount is excluded from Best Prices / alerts until the supplier
        # confirms the basis (Samuel 2026-11-20: Beefeater USD 75, Rémy XO
        # EUR 138 — source leaves bottle-or-case unspecified).
        cur = _clean(o.get("currency"))
        amt = _f(o.get("price_per_case")) or _f(o.get("price_per_unit"))
        if amt is None:
            # The LLM sometimes leaves the price blank PRECISELY because the basis
            # is unstated (Beefeater "Price USD 75.00, basis not stated" extracted
            # as no price). Recover the quoted amount + currency from the source
            # line so the "as quoted" note keeps the number, not a bare "amount".
            mm = re.search(r"(USD|EUR|GBP|CHF|US\$|\$|€|£)\s*([\d][\d.,]*)", ln, re.I)
            if mm:
                amt = _f(mm.group(2).replace(",", ""))
                if not cur:
                    cur = {"$": "USD", "US$": "USD", "€": "EUR", "£": "GBP"}.get(
                        mm.group(1), mm.group(1).upper())
        for f in ("price_per_unit", "price_per_case",
                  "price_per_unit_eur", "price_per_case_eur"):
            o[f] = None
        o["needs_manual_review"] = True
        quoted = f"{cur} {amt:g}".strip() if amt else "amount"
        flags = list(o.get("error_flags") or [])
        fnote = (f"{quoted} quoted — price basis not stated (per bottle or case); "
                 f"excluded from price comparison until confirmed")
        if fnote not in flags:
            flags.append(fnote)
        o["error_flags"] = flags
        cc = _clean(o.get("commercial_conditions"))
        if "as quoted" not in cc.lower():
            o["commercial_conditions"] = ((cc + " | " if cc else "")
                                          + f"as quoted: {quoted} (price basis not stated)")
        _mark(o)
        changed += 1
    return changed


# ─────────────────────────────────────────────────────────────────────────────
_LOOSE_ADDON_RE = re.compile(
    r"(\d[\d,]*)\s*cases?\s*(?:\+|plus|and)\s*(\d[\d,]*)\s*(?:loose\s*)?(?:bottles?|btls?|pieces?|pcs?)\b",
    re.IGNORECASE)


def recover_loose_bottles_addon(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Capture a 'N cases + M loose bottles' addon the extractor dropped
    (Perrier-Jouët: 10 cases + 2 loose bottles). Keeps the case count and records
    the loose bottles so the quantity reads '10 cases + 2 units', not just 10."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        qc = _f(o.get("quantity_case"))
        if not qc or _f(o.get("quantity_bottles")):
            continue
        ln = _source_line_for(o, lines)
        if not ln:
            continue
        m = _LOOSE_ADDON_RE.search(ln)
        if not m:
            continue
        cases = _int(m.group(1)); loose = _int(m.group(2))
        if cases and loose and float(cases) == qc:
            o["quantity_bottles"] = float(loose)
            cc = _clean(o.get("commercial_conditions"))
            if "loose" not in cc.lower():
                o["commercial_conditions"] = ((cc + " | " if cc else "")
                    + f"+ {loose} loose bottle" + ("s" if loose != 1 else ""))
            _mark(o)
            changed += 1
    return changed


def _twin_key_fields(supplier, brand, vol) -> tuple:
    v = _f(vol)
    return (_norm(supplier), _norm(brand), round(v) if v else 0)


def apply_basis_from_twin(offers: List[Dict[str, Any]], twin_map: Dict[tuple, set]) -> int:
    """Cross-source basis sync (Samuel 2026-11-20): when the SAME supplier quotes
    a product with an unstated price basis in a text/WhatsApp twin (prices
    cleared), a spreadsheet row for the same item that merely DERIVED a case
    price (price_per_case == unit × pack) is the same ambiguous quote — clear it
    too and keep the quoted amount. Keyed on (supplier, brand, volume) AND the
    quoted amount, so a different SKU of the same brand (Rémy XO vs VSOP) with its
    own price is never touched, and an INDEPENDENT case price is left alone."""
    changed = 0
    for o in offers:
        pc = _f(o.get("price_per_case")); pu = _f(o.get("price_per_unit"))
        upc = _f(o.get("units_per_case"))
        if not (pc or pu):
            continue
        amounts = twin_map.get(_twin_key_fields(o.get("supplier_name"),
                                                o.get("brand"), o.get("unit_volume_ml")))
        if not amounts:
            continue
        if not any(abs(a - x) < 0.5 for a in amounts for x in (pu or 0.0, pc or 0.0)):
            continue  # not the same quoted amount → different line
        if pc and pu and upc and abs(pc - pu * upc) > 0.5:
            continue  # an independent case price (not derived) → leave it
        cur = _clean(o.get("currency")); amt = pu or pc
        for f in ("price_per_unit", "price_per_case",
                  "price_per_unit_eur", "price_per_case_eur"):
            o[f] = None
        o["needs_manual_review"] = True
        note = (f"as quoted: {cur + ' ' if cur else ''}{amt:g} (price basis not "
                f"stated — matches the supplier's other quote for this item)")
        cc = _clean(o.get("commercial_conditions"))
        if "as quoted" not in cc.lower():
            o["commercial_conditions"] = (cc + " | " if cc else "") + note
        _mark(o)
        changed += 1
    return changed


_NO_CURRENCY_RE = re.compile(
    r"call\s+for\s+quotation|price\s+on\s+request|on\s+request|"
    r"no\s+(?:amount|price)?\s*/?\s*(?:and\s*)?currency|no\s+amount",
    re.IGNORECASE)


def clear_currency_on_request(offers: List[Dict[str, Any]], source_text: str) -> int:
    """A 'call for quotation' / 'price on request' line with no amount should not
    carry an (inherited/default) currency (Samuel 2026-11-20: Cody, JD & Cola —
    no currency was quoted). Only clears when the row has NO price at all."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        if _f(o.get("price_per_case")) or _f(o.get("price_per_unit")):
            continue
        if not _clean(o.get("currency")):
            continue
        ln = _source_line_for(o, lines)
        if ln and _NO_CURRENCY_RE.search(ln):
            o["currency"] = None
            o["needs_manual_review"] = True
            _mark(o)
            changed += 1
    return changed


# ─────────────────────────────────────────────────────────────────────────────
_COND_DELIVERY_RE = re.compile(
    r"((?:EXW|FCA|FOB|CIF|CFR|CPT|DAP|DDP|DAT|DPU)\b[^.;\n]*?\bonly\s+if\b[^.;\n]*|"
    r"\bonly\s+if\s+(?:a\s+)?full\s*(?:truck|load|ftl)[^.;\n]*)", re.IGNORECASE)


def capture_conditional_incoterm(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Preserve a CONDITIONAL delivery term the extractor flattened to a plain
    incoterm — "DAP Riga only if full truck" stored as just incoterm=DAP,
    location=Riga, losing the condition (Samuel 2026-11-20: Rémy Martin XO). Keep
    the structured incoterm/location untouched and add the exact conditional as a
    delivery note on the commercial conditions, so it shows on the live drawer and
    the export. Only acts when the source line carries the condition and the row
    does not already record it."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        cc = _clean(o.get("commercial_conditions"))
        if "only if" in cc.lower():
            continue
        ln = _source_line_for(o, lines)
        if not ln:
            continue
        m = _COND_DELIVERY_RE.search(ln)
        if not m:
            continue
        phrase = re.sub(r"\s+", " ", m.group(1).strip().rstrip(".,; "))
        o["commercial_conditions"] = (cc + " | " if cc else "") + f"Delivery: {phrase}"
        _mark(o)
        changed += 1
    return changed


# ─────────────────────────────────────────────────────────────────────────────
_BLOCK_PU_RE = re.compile(r"(?:USD|EUR|GBP|CHF|US\$|\$|€|£)\s*([\d][\d.,]*)\s*per\s*bottle", re.IGNORECASE)
_BLOCK_PC_RE = re.compile(r"(?:USD|EUR|GBP|CHF|US\$|\$|€|£)\s*([\d][\d.,]*)\s*per\s*case", re.IGNORECASE)


def _num(s) -> Optional[float]:
    try:
        return round(float(str(s).replace(",", "")), 2)
    except (TypeError, ValueError):
        return None


def _parse_price_blocks(source_text: str) -> List[Dict[str, Any]]:
    """Parse a 'USD x per bottle / USD y per case' price list into per-product
    blocks: {name, pack, per_bottle, per_case}. Pack is the standalone integer a
    few lines above 'per bottle' (the 'REF / 6 / 70 CL' shape)."""
    lines = [re.sub(r"^[>\s]+", "", l).strip() for l in str(source_text).splitlines()]
    blocks = []
    for i, ln in enumerate(lines):
        mb = _BLOCK_PU_RE.search(ln)
        if not mb:
            continue
        pu = _num(mb.group(1))
        pc = None
        for j in range(i + 1, min(len(lines), i + 4)):
            mc = _BLOCK_PC_RE.search(lines[j])
            if mc:
                pc = _num(mc.group(1)); break
        # name: nearest preceding line with ≥3 letters that isn't a spec/number
        name = None; pack = None
        for j in range(i - 1, max(-1, i - 22), -1):
            c = lines[j].strip()
            if not c:
                continue
            if pack is None and re.fullmatch(r"\d{1,2}", c):
                pack = int(c); continue
            if re.fullmatch(r"(?:REF|NRF|GBX|NGBX|T[12]|\d+\s*(?:cl|ml|l)|\d+\s*%|\d{1,3}\s*%?)", c, re.IGNORECASE):
                continue
            if re.search(r"[A-Za-z]{3}", c) and not _BLOCK_PU_RE.search(c) and not _BLOCK_PC_RE.search(c) \
               and "cases" not in c.lower() and "per " not in c.lower():
                name = c; break
        if pu or pc:
            blocks.append({"name": name, "pack": pack, "per_bottle": pu, "per_case": pc})
    return blocks


def repair_case_prices_from_source(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Re-pair each row's per-case price (and pack) to its source block. In the
    EPIC/Grant's list every per-BOTTLE price was correct but every per-CASE price
    was shifted onto the next product. Match a row to its source block by the
    (trustworthy) per-bottle price, confirmed by the product name, then set the
    case price and pack from that block. Only changes a field that disagrees."""
    if not source_text:
        return 0
    blocks = _parse_price_blocks(source_text)
    if len(blocks) < 2:
        return 0
    changed = 0
    for o in offers:
        pu = _num(o.get("price_per_unit"))
        nm = _norm(_ident(o))
        cand = None
        # 1) exact per-bottle match (the correct field), confirmed by a name token
        for b in blocks:
            if b["per_bottle"] is not None and pu is not None and abs(b["per_bottle"] - pu) < 0.01:
                bn = _norm(b["name"])
                if bn and nm and (bn.split()[0] in nm or nm.split()[0] in bn):
                    cand = b; break
        # 2) fall back to a strong name match
        if cand is None:
            for b in blocks:
                bn = _norm(b["name"])
                if not bn:
                    continue
                shared = len(set(bn.split()) & set(nm.split()))
                if shared >= 2:
                    cand = b; break
        if not cand:
            continue
        dirty = False
        # EUR conversion factor from the (correct, unchanged) per-bottle pair, so
        # a corrected per-case price keeps a consistent EUR value for Best Prices.
        pu0 = _f(o.get("price_per_unit")); pue0 = _f(o.get("price_per_unit_eur"))
        factor = (pue0 / pu0) if (pu0 and pue0 and pu0 > 0) else None
        if cand["per_case"] is not None and _num(o.get("price_per_case")) != cand["per_case"]:
            o["price_per_case"] = cand["per_case"]
            o["price_per_case_eur"] = round(cand["per_case"] * factor, 4) if factor else None
            dirty = True
        if cand["pack"] and _f(o.get("units_per_case")) != float(cand["pack"]):
            o["units_per_case"] = float(cand["pack"])
            dirty = True
        if cand["per_bottle"] is not None and _num(o.get("price_per_unit")) != cand["per_bottle"]:
            o["price_per_unit"] = cand["per_bottle"]
            o["price_per_unit_eur"] = None
            dirty = True
        if dirty:
            _mark(o); changed += 1
    return changed


# ─────────────────────────────────────────────────────────────────────────────
# A case written "A x B x <size>" holds A*B saleable units (4 x 6 x 33cl = 24
# cans), not B. The extractor read units_per_case = B and set the per-"unit"
# price to case/A — i.e. the inner (six-pack) price, not the per-can price
# (Samuel 2026-11-20: ERKAN beer offer).
_MULTIPACK_RE = re.compile(
    r"\b(\d{1,2})\s*[x×*]\s*(\d{1,2})\s*[x×*]\s*\d[\d.,]*\s*(?:cl|ml|l)\b", re.IGNORECASE)


def fix_multipack_units(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Correct the pack of a multi-pack case "A x B x size": units_per_case = A*B
    and per-unit price = case price / (A*B), so the unit price is per can, not per
    inner six-pack. Matched by the inner count B the extractor mis-stored, so a
    normal "24 x 33cl" case (single multiplier) is never touched."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for ln in lines:
        m = _MULTIPACK_RE.search(ln)
        if not m:
            continue
        a = _int(m.group(1)); b = _int(m.group(2))
        if not a or not b or a < 2:
            continue
        units = a * b
        ltoks = set(re.findall(r"[a-z0-9]+", ln.lower()))
        best = None; score = 0
        for o in offers:
            if _f(o.get("units_per_case")) != float(b):   # the mis-stored inner count
                continue
            ot = set(re.findall(r"[a-z0-9]+", _ident(o).lower()))
            s = sum(1 for t in ot if len(t) > 2 and t in ltoks)
            if s > score:
                best, score = o, s
        if not best or score < 1:
            continue
        best["units_per_case"] = float(units)
        pc = _f(best.get("price_per_case"))
        if pc:
            best["price_per_unit"] = round(pc / units, 4)
            pce = _f(best.get("price_per_case_eur"))
            best["price_per_unit_eur"] = round(pce / units, 4) if pce else None
        _mark(best); changed += 1
    return changed


# ─────────────────────────────────────────────────────────────────────────────
_PER_LOAD_RE = re.compile(r"(\d[\d.,]*)\s*cases?\s*per\s*load", re.IGNORECASE)
_AVAIL_QTY_RE = re.compile(r"(\d[\d.,]*)\s*cases?\s*(?:available|in\s*stock|offered)", re.IGNORECASE)


def separate_load_capacity(offers: List[Dict[str, Any]], source_text: str) -> int:
    """"N cases per load" is a truckload CAPACITY, not available stock — keep them
    apart (Samuel 2026-11-20: ERKAN Carlsberg lines stored "2882 cases per load" as
    the quantity). When a product block states a per-load figure and NO "cases
    available", clear the quantity the extractor took from it and note the load
    capacity, flagged for review. A block that states real "cases available" is
    left untouched."""
    if not source_text:
        return 0
    blocks = [b for b in re.split(r"\n\s*\n", str(source_text)) if b.strip()]
    changed = 0
    for b in blocks:
        ml = _PER_LOAD_RE.search(b)
        if not ml or _AVAIL_QTY_RE.search(b):
            continue
        load_n = _int(ml.group(1))
        if not load_n:
            continue
        btoks = set(re.findall(r"[a-z0-9]+", b.lower()))
        for o in offers:
            if _f(o.get("quantity_case")) != float(load_n):
                continue
            ot = set(t for t in re.findall(r"[a-z0-9]+", _ident(o).lower()) if len(t) > 2)
            if not (ot & btoks):
                continue
            o["quantity_case"] = None
            o["needs_manual_review"] = True
            cc = _clean(o.get("commercial_conditions"))
            if "per load" not in cc.lower():
                o["commercial_conditions"] = (cc + " | " if cc else "") + (
                    f"{load_n} cases per load (truckload capacity; "
                    f"available stock not stated)")
            _mark(o); changed += 1
    return changed


# ─────────────────────────────────────────────────────────────────────────────
# ERKAN-style "ON THE FLOOR" beer list: every priced product line must become its
# OWN row, including two pack formats of the same product at the same price
# (Samuel 2026-11-20: Carlsberg Lager 24×33 vs 4×6×33, both €7.05 — the model
# merged them into one). Deterministically recover a missing format and carry each
# block's BBD breakdown + bottle/can type + load capacity.
_FLOOR_PRODUCT_RE = re.compile(
    r"^(.*?)(\d+)\s*[x×*]\s*(\d+)(?:\s*[x×*]\s*(\d+))?\s*cl\s*(cans?|bottles?)\b",
    re.IGNORECASE)
_FLOOR_PRICE_RE = re.compile(
    r"(\d[\d.,]*)\s*euro\s*per\s*case\s*"
    r"(exworks?|exw|ex|fca|fob|cif|cfr|cpt|dap|ddp|dat|dpu)\b\s*([A-Za-z][A-Za-z .]*)",
    re.IGNORECASE)
_FLOOR_DATE_RE = re.compile(
    r"\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}")


def _eu_num(s) -> Optional[float]:
    """A number under either convention: '7,05'→7.05, '1.890'→1890, '4.911'→4911."""
    t = str(s).strip()
    if re.search(r",\d{1,2}\b", t) and "." not in t:
        t = t.replace(".", "").replace(",", ".")
    else:
        t = t.replace(",", "")
    try:
        return round(float(t), 4)
    except (TypeError, ValueError):
        return None


def _floor_bbd(block: str) -> str:
    """The best-before breakdown in a block — the dated lines (and any 'N cases
    <date>' split), joined. Skips the product, availability, status and price
    lines so only the BBD detail is captured (Amsterdam Navigator's four dates)."""
    out = []
    for ln in block.splitlines():
        s = ln.strip().strip(" ,.")
        if not s:
            continue
        low = s.lower()
        if ("euro" in low or low.endswith("status") or "status" in low
                or "per load" in low or "available" in low or "on the floor" in low):
            continue
        if "bbd" in low or _FLOOR_DATE_RE.search(s):
            s2 = re.sub(r"(?i)\bbbd\b\s*:?", "", s).strip(" ,.-:")
            if s2:
                out.append(re.sub(r"\s{2,}", " ", s2))
    return "; ".join(out)


def _floor_parse(block: str) -> Optional[Dict[str, Any]]:
    pm = None
    for ln in block.splitlines():
        pm = _FLOOR_PRODUCT_RE.search(ln.strip())
        if pm:
            break
    if not pm:
        return None
    name = pm.group(1).strip(" ,-")
    a = _int(pm.group(2)); c = _int(pm.group(3)); d = _int(pm.group(4))
    if d:                                  # A × C × D  → A*C units, D cl
        units = a * c; vol_cl = d; inner = c
        pack_disp = f"{a} × {c} × {d}cl"
    else:                                  # A × C      → A units, C cl
        units = a; vol_cl = c; inner = None
        pack_disp = f"{a} × {c}cl"
    container = pm.group(5).lower().rstrip("s")       # can / bottle
    pm2 = _FLOOR_PRICE_RE.search(block)
    pc = inco = loc = None
    if pm2:
        pc = _eu_num(pm2.group(1))
        inco = pm2.group(2).upper()
        inco = "EXW" if inco in ("EX", "EXWORK", "EXWORKS") else inco
        loc = pm2.group(3).strip(" .,")
    pl = _PER_LOAD_RE.search(block); av = _AVAIL_QTY_RE.search(block)
    return {
        "name": name, "units": units, "vol_ml": vol_cl * 10, "inner": inner,
        "container": container, "pack_disp": f"{pack_disp} {container}",
        "pc": pc, "inco": inco, "loc": loc,
        "per_load": _int(pl.group(1)) if pl else None,
        "avail": _int(av.group(1)) if av else None,
        "bbd": _floor_bbd(block),
    }


def _floor_toks(name: str):
    return set(t for t in re.findall(r"[a-z0-9]+", (name or "").lower()) if len(t) > 2)


def _floor_enrich(o: Dict[str, Any], sp: Dict[str, Any]) -> None:
    if sp["container"] and not _clean(o.get("bottle_or_can_type")):
        o["bottle_or_can_type"] = sp["container"]
    if sp["vol_ml"] and not _f(o.get("unit_volume_ml")):
        o["unit_volume_ml"] = float(sp["vol_ml"])
    o["packaging_raw"] = sp["pack_disp"]
    if sp["inco"] and not _clean(o.get("incoterm")):
        o["incoterm"] = sp["inco"]
    if sp["loc"] and not _clean(o.get("location")):
        o["location"] = sp["loc"]
    cc = _clean(o.get("commercial_conditions"))
    notes = []
    if sp["per_load"] and not sp["avail"]:
        o["quantity_case"] = None
        o["needs_manual_review"] = True
        if "per load" not in cc.lower():
            notes.append(f"{sp['per_load']} cases per load (truckload capacity; "
                         f"available stock not stated)")
    if sp["bbd"] and "bbd" not in cc.lower():
        notes.append(f"BBD: {sp['bbd']}")
    if notes:
        o["commercial_conditions"] = (cc + " | " if cc else "") + " | ".join(notes)
    _mark(o)


def reconcile_floor_list(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Make every priced product line in an 'ON THE FLOOR' list its own row.
    Recovers a pack format the extractor merged away (same product + price,
    different pack) and enriches every row with its block's bottle/can type, BBD
    breakdown and load capacity. Returns the number of rows enriched or added."""
    changed = 0
    if not source_text:
        return changed
    blocks = [b for b in re.split(r"\n\s*\n", str(source_text)) if b.strip()]
    priced = [b for b in blocks if re.search(r"euro\s*per\s*case", b, re.IGNORECASE)]
    if len(priced) < 3:
        return changed              # not a floor list → leave everything alone
    parsed = [sp for sp in (_floor_parse(b) for b in priced) if sp]
    if len(parsed) < 3:
        return changed
    used = set()
    for sp in parsed:
        stoks = _floor_toks(sp["name"])
        match = None
        for i, o in enumerate(offers):
            if i in used:
                continue
            _ovol = _f(o.get("unit_volume_ml"))
            if (stoks and stoks <= _floor_toks(_ident(o))
                    and _f(o.get("units_per_case")) == float(sp["units"])
                    and _ovol in (None, float(sp["vol_ml"]))):
                match = i; break
        if match is not None:
            used.add(match)
            _floor_enrich(offers[match], sp)
            changed += 1
        else:
            # Recover: clone a sibling row of the same product (for currency /
            # price), or build a fresh one, then set this format's fields.
            tmpl = next((o for o in offers if stoks and stoks <= _floor_toks(_ident(o))), None)
            new = dict(tmpl) if tmpl else {}
            new.pop("_dirty", None)
            if not tmpl:
                parts = sp["name"].split()
                new["brand"] = parts[0] if parts else sp["name"]
                new["product_name"] = " ".join(parts[1:]) or sp["name"]
                new["currency"] = "EUR"
            new["uid"] = f"floor-{abs(hash((sp['name'], sp['pack_disp']))) % (10 ** 9)}"
            new["units_per_case"] = float(sp["units"])
            if sp["pc"]:
                new["price_per_case"] = sp["pc"]
                new["price_per_unit"] = round(sp["pc"] / sp["units"], 4) if sp["units"] else None
                new["price_per_case_eur"] = None
                new["price_per_unit_eur"] = None
            for k in ("quantity_bottles", "quantity_pallets", "quantity_ftl"):
                new.pop(k, None)
            new["quantity_case"] = float(sp["avail"]) if sp["avail"] else None
            new["commercial_conditions"] = None
            _floor_enrich(new, sp)
            offers.append(new)
            changed += 1
    return changed


# ─────────────────────────────────────────────────────────────────────────────
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_INNERBOX_OF_RE = re.compile(r"(\w+)\s+inner\s*box(?:es)?\s+of\s+(\w+)", re.IGNORECASE)
_LOOSE_PLUS_RE = re.compile(r"plus\s+(\w+)\s+loose", re.IGNORECASE)
_MIN_INNER_RE = re.compile(r"minimum\s+(\w+)\s+inner", re.IGNORECASE)


def _wordnum(tok) -> Optional[int]:
    t = str(tok).strip().lower()
    if t.isdigit():
        return int(t)
    return _WORDNUM.get(t)


def fix_inner_box_of_n(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Handle the "N inner boxes of M candles, plus K loose candles" phrasing with
    spelled-out numbers (Samuel 2026-11-20: Lalique — "Six inner boxes of four
    candles, plus two loose candles" = 26 pieces). Sets quantity in pieces, pack =
    pieces per inner box, MOQ in inner boxes, and drops a fabricated case price for
    per-piece pricing — like fix_inner_box_from_text but for the 'of' wording."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        ln = _source_line_for(o, lines)
        if not ln:
            continue
        m = _INNERBOX_OF_RE.search(ln)
        if not m:
            continue
        boxes = _wordnum(m.group(1)); per = _wordnum(m.group(2))
        if not boxes or not per:
            continue
        total = boxes * per
        lm = _LOOSE_PLUS_RE.search(ln)
        loose = _wordnum(lm.group(1)) if lm else 0
        if loose:
            total += loose
        o["units_per_case"] = float(per)
        o["quantity_unit"] = "pieces"
        o["quantity_bottles"] = float(total)
        o["quantity_case"] = None
        if re.search(r"per\s+(?:candle|piece|pc)\b", ln, re.IGNORECASE):
            o["price_per_case"] = None
            o["price_per_case_eur"] = None
        mm = _MIN_INNER_RE.search(ln)
        if mm and _wordnum(mm.group(1)):
            o["moq_cases"] = float(_wordnum(mm.group(1)))
            o["moq_unit"] = "inner boxes"
        cc = _clean(o.get("commercial_conditions"))
        note = f"{boxes} inner boxes of {per}" + (f" + {loose} loose" if loose else "")
        if "inner box" not in cc.lower():
            o["commercial_conditions"] = (cc + " | " if cc else "") + note
        _mark(o)
        changed += 1
    return changed


# ─────────────────────────────────────────────────────────────────────────────
# Recover the structured fields of a TAB-separated price table the free-text
# extractor flattened (Samuel 2026-11-20: URD Exports — every row lost Qnty/cs,
# location and category; 6 brands came through "Not Found").
_CATEGORY_WORDS = {
    "rum": ("Spirits", "Rum"), "tequila": ("Spirits", "Tequila"),
    "whisky": ("Spirits", "Whisky"), "whiskey": ("Spirits", "Whisky"),
    "liqueur": ("Spirits", "Liqueur"), "liquer": ("Spirits", "Liqueur"),
    "gin": ("Spirits", "Gin"), "vodka": ("Spirits", "Vodka"),
    "cognac": ("Spirits", "Cognac"), "brandy": ("Spirits", "Brandy"),
    "champagne": ("Wine", "Champagne"), "wine": ("Wine", None),
    "beer": ("Beer", None), "minis": (None, None), "miniatures": (None, None),
}
_SUBCAT_IN_NAME = (
    ("vermouth", ("Wine", "Vermouth")), ("tequila", ("Spirits", "Tequila")),
    ("cognac", ("Spirits", "Cognac")), ("whisky", ("Spirits", "Whisky")),
    ("rum", ("Spirits", "Rum")), ("gin", ("Spirits", "Gin")),
    ("vodka", ("Spirits", "Vodka")),
)
_EX_LOC_RE = re.compile(r"^(EXW?|FCA|FOB|CIF|CFR|CPT|DAP|DDP|DAT|DPU)\s+([A-Za-z][A-Za-z .'\-]+?)$",
                        re.IGNORECASE)
_MULTIWORD_BRAND_LEAD = {"don", "grey", "jack", "johnnie", "jim", "maker", "wild",
                         "four", "glen", "royal", "famous", "old"}


def _split_lead_brand(name: str):
    """(brand, remaining_name) by taking the leading alphabetic token(s) — two for
    a known compound lead ("Don Julio"), else one — provided it carries no digit.
    Cinzano Vermouth Rosso 12/100/15 → ("Cinzano", "Vermouth Rosso 12/100/15")."""
    parts = str(name).split()
    if not parts:
        return None, None
    n = 1
    if (len(parts) >= 2 and parts[0].lower() in _MULTIWORD_BRAND_LEAD
            and not re.search(r"\d", parts[1])):
        n = 2
    brand = " ".join(parts[:n])
    if re.search(r"\d", brand) or len(brand) < 2:
        return None, None
    return brand, " ".join(parts[n:]).strip(" ,-")


def _tab_price(s: str):
    m = re.search(r"(USD|EUR|GBP|CHF|US\$|\$|€|£)\s*([\d][\d.,]*)"
                  r"|([\d][\d.,]*)\s*(USD|EUR|GBP|CHF|€|£|\$)", str(s), re.IGNORECASE)
    if not m:
        return None, None
    curtok = (m.group(1) or m.group(4) or "").strip()
    amt = _num(m.group(2) or m.group(3))
    cur = {"$": "USD", "US$": "USD", "€": "EUR", "£": "GBP"}.get(curtok, curtok.upper()) if curtok else None
    return cur, amt


def recover_tab_table_fields(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Parse a [name, Qnty/cs, Price] tab-table and backfill the matching rows:
    quantity (cases), location/incoterm, brand (gazetteer-missed names) and
    category. The column header is a bare 'Price' with no per-bottle/per-case
    label, so the amount is basis-UNCONFIRMED — keep it as a quoted note and leave
    both price fields blank (the universal rule Samuel set)."""
    if not source_text:
        return 0
    raw = [re.sub(r"^[>\s]+", "", l) for l in str(source_text).splitlines()]
    has_header = any("\t" in l and re.search(r"qnty|qty", l, re.I) and re.search(r"price", l, re.I)
                     for l in raw)
    if not has_header:
        return 0
    location = None; incoterm = None
    rows_tab = []                  # (name, qty, price_raw, (cat, subcat))
    cat = (None, None)
    for l in raw:
        if not l.strip():
            continue
        first = l.split("\t")[0].strip()
        em = _EX_LOC_RE.match(first)
        if em and location is None and len(first.split()) <= 4:
            incoterm = "EXW" if em.group(1).upper() == "EX" else em.group(1).upper()
            location = em.group(2).strip()
            continue
        if re.match(r"^(product|qnty|qty)\b", first, re.IGNORECASE):
            continue
        parts = [p.strip() for p in l.split("\t")]
        nonempty = [p for p in parts if p]
        if len(nonempty) == 1 and nonempty[0].lower() in _CATEGORY_WORDS:
            cat = _CATEGORY_WORDS[nonempty[0].lower()]
            continue
        if len(parts) >= 2 and len(parts[0]) > 3 and re.search(r"[A-Za-z]{3}", parts[0]):
            qty = None; price_raw = None
            for p in parts[1:]:
                if re.search(r"[$€£]", p):
                    price_raw = p
                elif qty is None and re.fullmatch(r"[\d.,]+", p):
                    qty = _int(p)
            rows_tab.append((parts[0], qty, price_raw, cat))
    if not rows_tab:
        return 0
    changed = 0
    for o in offers:
        toks = [t for t in re.findall(r"[a-z0-9]+", _ident(o).lower()) if len(t) > 2]
        if not toks:
            continue
        best = None; score = 0
        for row in rows_tab:
            low = row[0].lower()
            sc = sum(1 for t in toks if t in low)
            if sc > score:
                best, score = row, sc
        if not best or score < 1:
            continue
        name, qty, price_raw, c = best
        dirty = False
        if qty and _primary_qty(o) is None:
            o["quantity_case"] = float(qty); o["quantity_unit"] = "cases"; dirty = True
        if location and not _clean(o.get("location")):
            o["location"] = location; dirty = True
        if incoterm and not _clean(o.get("incoterm")):
            o["incoterm"] = incoterm; dirty = True
        subcat = None
        for kw, cc2 in _SUBCAT_IN_NAME:
            if kw in name.lower():
                subcat = cc2; break
        c = subcat or c
        if c[0] and not _clean(o.get("category")):
            o["category"] = c[0]; dirty = True
        if c[1] and not _clean(o.get("subcategory")):
            o["subcategory"] = c[1]
            o["sub_category"] = c[1]   # schema key used by clean_product_data
            dirty = True
        if not _clean(o.get("brand")):
            br, rest = _split_lead_brand(name)
            if br:
                o["brand"] = br
                if rest:
                    o["product_name"] = rest
                o["needs_manual_review"] = True
                dirty = True
        if price_raw:
            cur, amt = _tab_price(price_raw)
            if amt is not None:
                for f in ("price_per_unit", "price_per_case",
                          "price_per_unit_eur", "price_per_case_eur"):
                    o[f] = None
                if cur and not _clean(o.get("currency")):
                    o["currency"] = cur
                o["needs_manual_review"] = True
                cc = _clean(o.get("commercial_conditions"))
                quoted = f"{(cur + ' ') if cur else ''}{amt:g}"
                if "as quoted" not in cc.lower():
                    o["commercial_conditions"] = (cc + " | " if cc else "") + (
                        f"as quoted: {quoted} (price basis not stated — "
                        f"column 'Price', per bottle or case?)")
                dirty = True
        if dirty:
            _mark(o); changed += 1
    return changed
