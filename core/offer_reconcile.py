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
            if len(pcs) > 1:
                continue  # different case prices → different lots, leave them
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
                    if not _f(o.get("price_per_case")) and not _f(o.get("price_per_unit")):
                        term += " (price TBC)"
                    alts.append(term)
            if alts:
                note = "Also available: " + " ; ".join(dict.fromkeys(alts))
                # The merge requires a single shared case price, so when the kept
                # row carries one, state it on the alternative term too — it is
                # the SAME lot at the SAME price, not a second quote (Samuel
                # 2026-11-20: Jägermeister EXW confirmed same EUR 78/case).
                kpc = _f(keep.get("price_per_case")); kcur = _clean(keep.get("currency"))
                if kpc and kcur and not any("price TBC" in a for a in alts):
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
