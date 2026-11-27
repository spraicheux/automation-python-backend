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
def merge_alt_delivery_lots(offers: List[Dict[str, Any]], source_text: str = "") -> Set[str]:
    """One physical lot quoted under two delivery terms (same product, same
    quantity, same — or one missing — case price) is ONE offer, not two. Keep the
    richest row, fold the alternative incoterm/location into its conditions, drop
    the rest. Two genuinely different SKUs (Tanqueray 1L vs 70cl: different
    quantity AND price) are left as two rows."""
    dropped: Set[str] = set()
    _alt_lines = ([l.strip() for l in str(source_text).splitlines() if l.strip()]
                  if source_text else None)
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
            # Two distinct delivery terms at two prices → two comparable siblings
            # sharing one pool, not a fold (Samuel 2026-11-26 Moss). Same-term or
            # same-price alternates still fold into a note below.
            others = [o for o in lot if o is not keep and not o.get("_pool_linked")]
            lot_tofold = _split_or_fold_siblings(keep, others, _alt_lines)
            alts = []
            for o in lot_tofold:
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
_CUR_TOKEN_RE = re.compile(
    r"\b(USD|EUR|GBP|HKD|CHF|SGD|AED|JPY|CNY|RMB|CAD|AUD|NZD|SEK|DKK|NOK|PLN|ZAR)\b",
    re.IGNORECASE)


def fix_currency_from_source_line(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Correct a priced row whose source line names ONE currency that differs from
    the stored one, then recompute its EUR comparable via FX. The LLM sometimes
    labels a clearly-foreign price EUR (Samuel 2026-11-21: Fenwick Courvoisier
    "USD 138/case DAP Lille" stored as €138/case). Conservative: acts only when
    the matched line carries exactly one currency token and the row has a price."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        if not (_f(o.get("price_per_unit")) or _f(o.get("price_per_case"))):
            continue
        ln = _source_line_for(o, lines)
        if not ln:
            continue
        found = {m.group(1).upper().replace("RMB", "CNY")
                 for m in _CUR_TOKEN_RE.finditer(ln)}
        if len(found) != 1:
            continue
        src_cur = found.pop()
        cur_now = _clean(o.get("currency")).upper()
        if not src_cur or src_cur == cur_now:
            continue
        o["currency"] = src_cur
        fx = None
        if src_cur == "EUR":
            fx = 1.0
        else:
            try:
                from core.fx_rates import get_eur_rate
                fx, _eff, _fsrc, _fb = get_eur_rate(
                    src_cur, o.get("offer_date") or o.get("date_received"))
                if fx:
                    o["fx_source"] = _fsrc
            except Exception:
                fx = None
        if fx:
            o["fx_rate"] = fx
            for nat, eur in (("price_per_unit", "price_per_unit_eur"),
                             ("price_per_case", "price_per_case_eur")):
                v = _f(o.get(nat))
                o[eur] = round(v * fx, 4) if v is not None else None
        else:
            # No rate available — never leave a stale same-number EUR comparable
            # under a now-foreign currency; blank it until FX is known.
            o["price_per_unit_eur"] = None
            o["price_per_case_eur"] = None
        _mark(o)
        changed += 1
    return changed


def _combined_name_tokens(o: Dict[str, Any]) -> tuple:
    """Sorted word tokens of brand + product_name — so a brand-split duplicate
    ('Punt' + 'e Mes vermouth') and its clean twin ('Punt e Mes' + 'Vermouth')
    share one key."""
    toks = [t for t in re.findall(r"[a-z0-9]+", _norm(_ident(o))) if t]
    return tuple(sorted(toks))


def collapse_identical_commercial_lines(offers: List[Dict[str, Any]]) -> Set[str]:
    """Collapse rows that are the SAME commercial line split in two by the
    extractor — identical product (by combined brand+name tokens), quantity,
    pack, volume, incoterm, location and currency, with compatible prices (equal,
    or both blank). Keep the row with the better-formed brand (Samuel 2026-11-21:
    Fenwick Punt e Mes appeared twice for the same 43 cases). Two real lots that
    differ in quantity, price or delivery are NOT touched (that is merge_alt's
    job, and Asahi's two pack formats differ in quantity)."""
    dropped: Set[str] = set()
    groups: Dict[tuple, list] = defaultdict(list)
    for o in offers:
        key = (_combined_name_tokens(o), _primary_qty(o),
               _f(o.get("units_per_case")), _f(o.get("unit_volume_ml")),
               _norm(o.get("incoterm")), _norm(o.get("location")),
               _clean(o.get("currency")).upper())
        groups[key].append(o)
    for key, g in groups.items():
        if len(g) < 2 or key[1] is None:   # need a real shared quantity
            continue
        pus = {round(_f(o.get("price_per_unit")), 2) for o in g if _f(o.get("price_per_unit"))}
        pcs = {round(_f(o.get("price_per_case")), 2) for o in g if _f(o.get("price_per_case"))}
        if len(pus) > 1 or len(pcs) > 1:    # genuinely different prices → keep both
            continue
        # Keep the row whose brand holds the most of the combined tokens (the
        # clean "Punt e Mes", not the split "Punt"); carry review across.
        keep = sorted(g, key=lambda o: (
            len(re.findall(r"[a-z0-9]+", _norm(o.get("brand")))),
            bool(_f(o.get("price_per_case")) or _f(o.get("price_per_unit")))),
            reverse=True)[0]
        for o in g:
            if o is keep:
                continue
            if o.get("needs_manual_review"):
                keep["needs_manual_review"] = True
            dropped.add(o.get("uid"))
        _mark(keep)
    if dropped:
        offers[:] = [o for o in offers if o.get("uid") not in dropped]
    return dropped


_INCOTERM_WORD_RE = re.compile(
    r"\b(EXW|FCA|FOB|CIF|CFR|CPT|CIP|DAP|DDP|DPU|DAT|FAS)\b|"
    r"\bex[\s-]*(?:works?|warehouse|cellars?|[A-Z][a-z]+)\b|"
    r"\b(?:collected|collection|delivered|delivery|free\s+carrier|franco|"
    r"in\s+bond|ex\s+bond)\b", re.IGNORECASE)


def clear_unsourced_incoterm(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Clear an incoterm the source never states for a line. The LLM stamps EXW
    on a line that only names a stock LOCATION (Samuel 2026-11-21: Westhaven
    Galliano "52 cases in Antwerp" shown as EXW Antwerp — the source gives no
    delivery term). Conservative: acts only when the matched source line names the
    row's own location (so we know we found the right line) and carries NO
    incoterm token or delivery verb at all."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        inco = _clean(o.get("incoterm"))
        loc = _clean(o.get("location"))
        if not inco or not loc:
            continue
        ln = _source_line_for(o, lines)
        if not ln or loc.lower() not in ln.lower():
            continue
        if _INCOTERM_WORD_RE.search(ln):
            continue   # the source DOES carry a delivery term/verb — keep it
        o["incoterm"] = None
        flags = list(o.get("error_flags") or [])
        fnote = (f"incoterm not stated in the source for this line "
                 f"(was {inco}); shown as location only")
        if fnote not in flags:
            flags.append(fnote)
        o["error_flags"] = flags
        _mark(o)
        changed += 1
    return changed


_TRAY_FMT_RE = re.compile(
    r"(\d[\d,]*)\s*trays?\s+of\s+(?:the\s+)?"
    r"(singles?|wrapped(?:\s+(?:six[\s-]?packs?|packs?))?)", re.IGNORECASE)


def recover_dual_tray_format(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Recover a SECOND pack format the extractor collapsed into one row. A
    supplier offers one product in two tray formats at the same price (Samuel
    2026-11-21: Westhaven Brooklyn — "63 trays of singles AND 47 trays of the
    wrapped packs", both €11.52/tray; only the 63 came through). When the matched
    source line lists ≥2 "N trays of <format>" counts and only one row exists,
    clone it for each missing count and fold the format into both names."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    # Group rows by product so we can see how many formats already exist.
    by_prod: Dict[tuple, list] = defaultdict(list)
    for o in offers:
        by_prod[(_norm(o.get("brand")), _norm(o.get("product_name")))].append(o)
    new_rows: list = []
    for (bn, pn), g in list(by_prod.items()):
        ln = _source_line_for(g[0], lines)
        if not ln:
            continue
        fmts = [(m.group(1).replace(",", ""), re.sub(r"\s+", " ", m.group(2).strip().lower()))
                for m in _TRAY_FMT_RE.finditer(ln)]
        # unique by format label
        seen = {}
        for qty, lab in fmts:
            seen.setdefault(lab, qty)
        if len(seen) < 2:
            continue
        existing_qtys = {int(_primary_qty(o)) for o in g if _primary_qty(o)}
        base = dict(g[0])   # pristine snapshot BEFORE any _label_fmt relabels g[0]
        for lab, qty in seen.items():
            q = int(qty)
            if q in existing_qtys:
                # label the existing row's format if not already present
                for o in g:
                    if _primary_qty(o) and int(_primary_qty(o)) == q:
                        _label_fmt(o, lab)
                continue
            clone = dict(base)
            clone.pop("_dirty", None)
            uid = base.get("uid") or ""
            clone["uid"] = f"{uid}__fmt_{q}"
            clone["quantity_case"] = float(q)
            clone["quantity_bottles"] = None
            clone["quantity_pallets"] = None
            _label_fmt(clone, lab)
            clone["needs_manual_review"] = True
            _mark(clone)
            new_rows.append(clone)
            changed += 1
    if new_rows:
        offers.extend(new_rows)
    return changed


def _label_fmt(o: Dict[str, Any], lab: str) -> None:
    nm = _clean(o.get("product_name"))
    if lab and lab not in nm.lower():
        o["product_name"] = f"{nm} ({lab})" if nm else lab
        _mark(o)


_WHOLE_LOT_RE = re.compile(
    r"whole\s+lot|full\s+(?:lot|load|truck|pallet\s+load)|entire\s+lot|"
    r"take\s+the\s+whole|complete\s+lot", re.IGNORECASE)


def annotate_whole_lot(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Carry a 'whole lot only' condition onto an alternative delivery option the
    merge folded into a note (Samuel 2026-11-21: Westhaven Cazadores — "EUR 93 per
    case delivered Lille if you take the whole lot"). The option already survives
    in the 'Also available' note; this adds the condition the source attached."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        cc = _clean(o.get("commercial_conditions"))
        if "also available" not in cc.lower() or "whole lot" in cc.lower():
            continue
        ln = _source_line_for(o, lines)
        if not ln or not _WHOLE_LOT_RE.search(ln):
            continue
        o["commercial_conditions"] = cc + " — whole lot only"
        _mark(o)
        changed += 1
    return changed


_PACKING_RE = re.compile(
    r"(\d[\d,]*)\s*(loose|wrapped)\s+"
    r"(cans?|bottles?|six[\s-]?packs?|packs?|trays?)", re.IGNORECASE)


def annotate_pack_format(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Label two lots of the SAME product that differ only by pack FORMAT so the
    buyer can tell them apart (Samuel 2026-11-21: Fenwick Asahi — "24 loose cans"
    vs "4 wrapped six-packs", both €9.84/case, stored identically). Matches each
    row to its own source line by product tokens AND its quantity, then folds the
    packing phrase into product_name + packaging_raw. Only fires when ≥2 rows
    share brand+product_name and the source gives each a distinct packing."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    by_prod: Dict[tuple, list] = defaultdict(list)
    for o in offers:
        by_prod[(_norm(o.get("brand")), _norm(o.get("product_name")))].append(o)
    changed = 0
    for g in by_prod.values():
        if len(g) < 2:
            continue
        seen_fmt = set()
        plan = []
        for o in g:
            q = _primary_qty(o)
            toks = [t for t in re.findall(r"[a-z0-9]+", _ident(o).lower()) if len(t) > 2]
            cand = [ln for ln in lines
                    if sum(1 for t in toks if t in ln.lower()) >= 1
                    and (q is None or re.search(rf"\b{int(q)}\b", ln))]
            if len(cand) != 1:
                plan = []
                break
            pm = _PACKING_RE.search(cand[0])
            if not pm:
                plan = []
                break
            fmt = re.sub(r"\s+", " ", pm.group(0).strip().lower())
            if fmt in seen_fmt:
                plan = []
                break
            seen_fmt.add(fmt)
            plan.append((o, pm.group(0).strip()))
        if len(plan) < 2:
            continue
        for o, fmt in plan:
            nm = _clean(o.get("product_name"))
            if fmt.lower() not in nm.lower():
                o["product_name"] = f"{nm} ({fmt})" if nm else fmt
            pr = _clean(o.get("packaging_raw"))
            if fmt.lower() not in pr.lower():
                o["packaging_raw"] = (pr + " | " if pr else "") + fmt
            _mark(o)
            changed += 1
    return changed


_DIM_FMT_RE = re.compile(
    r'(\d+\s*[x×]\s*\d+(?:\s*[x×]\s*\d+)?\s*(?:ml|cl|l|litre|liter|lt)\b)',
    re.IGNORECASE)


def _dim_format(line: str) -> Optional[str]:
    m = _DIM_FMT_RE.search(line or "")
    return re.sub(r"\s+", " ", m.group(1).strip().lower()) if m else None


def fold_distinct_pack_formats(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Two lots of the SAME product that differ only by a DIMENSIONAL pack format
    ("24 x 33cl" vs "4 x 6 x 33cl", both 24/case at the same price) extract to
    identical rows, so the display duplicate-guard then hides one. Fold each row's
    distinct source format into product_name + packaging_raw so both stay visible
    (Samuel 2026-11-22: MERIDIAN Carlsberg Lager — "Do not merge with line 1").
    Complements annotate_pack_format, which only handles loose/wrapped wording."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    by_prod: Dict[tuple, list] = defaultdict(list)
    for o in offers:
        by_prod[(_norm(o.get("brand")), _norm(o.get("product_name")))].append(o)
    changed = 0
    for g in by_prod.values():
        if len(g) < 2:
            continue
        toks = [t for t in re.findall(r"[a-z0-9]+", _ident(g[0]).lower()) if len(t) > 2]
        if not toks:
            continue
        # Lines that name THIS product (all identity tokens present), each with a
        # dimensional format — gathered as the distinct formats the source gives.
        fmts: list = []
        for ln in lines:
            low = ln.lower()
            if all(t in low for t in toks):
                f = _dim_format(ln)
                if f and f not in fmts:
                    fmts.append(f)
        # Fold only when the source offers at least one distinct format per row.
        if len(fmts) < len(g) or len(fmts) < 2:
            continue
        # Already distinct by packaging_raw? then nothing to do.
        existing = {_norm(o.get("packaging_raw")) for o in g}
        if "" not in existing and len(existing) >= len(g):
            continue
        for o, fmt in zip(g, fmts):
            nm = _clean(o.get("product_name"))
            if fmt not in nm.lower():
                o["product_name"] = f"{nm} ({fmt})" if nm else fmt
            pr = _clean(o.get("packaging_raw"))
            if fmt not in pr.lower():
                o["packaging_raw"] = (pr + " | " if pr else "") + fmt
            _mark(o)
            changed += 1
    return changed


_FTL_QTY_RE = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*(?:x\s*)?ftl\b|\b(\d+)\s*full\s*(?:truck|load)',
    re.IGNORECASE)


def capture_ftl_quantity(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Preserve a quantity stated in full truckloads ("1 FTL; case quantity not
    stated") the extractor left blank (Samuel 2026-11-22: Rémy Martin XO). Only
    sets it when the row carries no other quantity, so a real case / bottle /
    pallet count is never overwritten."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        if _f(o.get("quantity_ftl")):
            continue
        if any(_f(o.get(k)) for k in
               ("quantity_case", "quantity_bottles", "quantity_pallets")):
            continue
        ln = _source_line_for(o, lines)
        if not ln:
            continue
        m = _FTL_QTY_RE.search(ln)
        if not m:
            continue
        try:
            val = float(m.group(1) or m.group(2))
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        o["quantity_ftl"] = val
        o["quantity_unit"] = "ftl"
        _mark(o)
        changed += 1
    return changed


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
_PER_PIECE_RE = re.compile(r"(?:per|/)\s*(?:piece|pc|pcs|candle)s?\b", re.IGNORECASE)
# Reversed inner-box phrasing (Nishane 2026-11-20: "6 pieces/inner box … 5 inner
# boxes available … EUR 38/piece") — the pieces-per-box and the available box
# count are stated separately, not as "N inner boxes x M pieces".
_PIECES_PER_BOX_RE = re.compile(
    r"(\d[\d,]*)\s*(?:pieces?|pcs?|pc|candles?)\s*(?:/|per)\s*inner(?:\s*box)?", re.IGNORECASE)
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


_INNERBOX_OF_RE = re.compile(
    r"(?<![\d.])(\d[\d,]*)\s+inner\s*box(?:es)?\s+of\s+(\d[\d,]*)", re.IGNORECASE)  # "3 inner boxes of 8"
_BOX_SIZE_OF_RE = re.compile(
    r"inner\s*box(?:es)?\s*of\s*(\d[\d,]*)", re.IGNORECASE)              # "inner boxes of 8 candles"
_INNERS_COUNT_RE = re.compile(r"(\d[\d,]*)\s*inners?\b", re.IGNORECASE)  # "3 inners"


def _innerbox_counts(ln: str):
    """(pieces_per_box, total_pieces) from any inner-box phrasing:
    'N inner boxes x M pieces [= T pieces]' (Montblanc); 'M pieces/inner box …
    N inner boxes available' (Nishane); 'N inner boxes of M' or 'inner boxes of M
    … N inners' (Samuel 2026-11-25: Estuary/NF Linen). (None, None) otherwise."""
    boxes = per = None
    m = _INNERBOX_RE.search(ln)
    pm = _PIECES_PER_BOX_RE.search(ln)
    bs = _BOX_SIZE_OF_RE.search(ln)
    if m:                                 # "N inner boxes x M pieces"
        boxes = _int(m.group(1)); per = _int(m.group(2))
    elif pm:                              # "M pieces/candles per inner [box]"
        per = _int(pm.group(1))
    elif bs:                              # "inner boxes of M [candles]"
        per = _int(bs.group(1))
    else:
        mo = _INNERBOX_OF_RE.search(ln)   # "N inner boxes of M" (last resort)
        if mo:
            boxes = _int(mo.group(1)); per = _int(mo.group(2))
    if not per:
        return None, None
    if boxes is None:                     # count from "N inners" / "N inner boxes available"
        ic = _INNERS_COUNT_RE.search(ln)
        boxes = _int(ic.group(1)) if ic else _boxes_available(ln)
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
        # A "+ N loose <piece>" addon on the same line (Voluspa 2026-11-21:
        # "3 inner boxes plus one loose candle" = 3×8 + 1 = 25, not 24).
        lm = _LOOSE_PLUS_RE.search(ln)
        if lm and total is not None:
            _lp = _wordnum(lm.group(1))
            if _lp:
                total += _lp
        # The pieces-per-inner-box is NOT a case count: a per-piece item must
        # show EMPTY case fields (Samuel 2026-11-23 — Nishane still read
        # "6/case" because units_per_case kept the inner-box size).
        o["units_per_case"] = None
        o["quantity_unit"] = "pieces"
        if total:
            o["quantity_bottles"] = float(total)
        o["quantity_case"] = None                 # not cases
        # Label the pack an inner box (not a case). Overwrite a generic
        # "bottle"/"piece" the extractor left — that blocked the inner-box label
        # and left the drawer reading "6/case" (Samuel 2026-11-23: Nishane
        # packaging_raw was "bottle").
        _ib_label = f"{int(per)} pieces per inner box"
        _praw_cur = _clean(o.get("packaging_raw")).lower()
        if (not _praw_cur) or _praw_cur in (
                "bottle", "bottles", "piece", "pieces", "pc", "pcs",
                "candle", "candles"):
            o["packaging_raw"] = _ib_label
        mm = _INNERBOX_MOQ_RE.search(ln)
        if mm:
            o["moq_cases"] = float(_int(mm.group(1)))
            o["moq_unit"] = "inner boxes"
        # per-piece pricing: drop any fabricated per-case price, and the now-stale
        # "case price = unit × pack" info flag that would otherwise mislead. "each"
        # and "apiece" are per-piece too (Samuel 2026-11-25: W06 "GBP 7.10 each").
        if _PER_PIECE_RE.search(ln) or re.search(r"\b(?:each|apiece)\b", ln, re.IGNORECASE):
            o["price_per_case"] = None
            o["price_per_case_eur"] = None
            o["error_flags"] = [f for f in (o.get("error_flags") or [])
                                if "price_per_case calculated" not in str(f).lower()]
            # Recover the EXPLICIT per-piece amount the source states ("GBP 7.10
            # each", "GBP 7.40 per candle") — the extractor sometimes stored a
            # wrong/bled per-unit price (Samuel 2026-11-25: W06 Linen kept GBP 0.50
            # instead of 7.10; D07 Linen 7.40). Take the amount adjacent to the
            # per-piece word, overriding the stored value.
            mp = re.search(
                r"(USD|EUR|GBP|CHF|HKD|SGD|\$|€|£)\s*([\d][\d.,]*)\s*"
                r"(?:each|per\s+(?:candle|piece|pc|bottle|unit))"
                r"|([\d][\d.,]*)\s*(?:each|per\s+(?:candle|piece|pc))", ln, re.IGNORECASE)
            if mp:
                _amt = mp.group(2) or mp.group(3)
                try:
                    _val = round(float(str(_amt).replace(",", ".").replace(" ", "")), 4)
                    if _val > 0:
                        o["price_per_unit"] = _val
                        o["price_per_unit_eur"] = None   # recomputed downstream
                        _cw = mp.group(1)
                        if _cw and not _clean(o.get("currency")):
                            o["currency"] = _CUR_WORD.get(_cw.lower(), _cw.upper())
                except ValueError:
                    pass
            o["_unit_priced"] = True       # per-piece basis — absent-basis pass must not clear it
        _mark(o)
        changed += 1
    return changed


# ─────────────────────────────────────────────────────────────────────────────
_LOOSE_PIECES_RE = re.compile(r"(\d[\d,]*)\s*(?:pieces?|pcs?|pc)\s+available", re.IGNORECASE)
# A retail line sells individual units and states the count directly: "15
# individual bottles at USD 26 per bottle", "8 bottles USD 28/bottle", "21
# individual candles" (Samuel 2026-11-26: Northwick W04 Linden, W20 Cedar, W17
# Silk). Only when a RETAIL signal is present, so "6 bottles per case" (a pack
# spec) is never read as the quantity.
_RETAIL_QTY_RE = re.compile(
    r"(\d[\d,]*)\s+(?:individual\s+)?(?:bottles?|pieces?|candles?|sprays?|sets?)\b"
    r"(?=.*(?:per\s+(?:bottle|piece|candle|set|spray|unit)"
    r"|/\s*(?:bottle|piece|candle|set|spray)|each\b|retail|individual|tester|unboxed|intact))",
    re.IGNORECASE)


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
        m = _LOOSE_PIECES_RE.search(ln) or _RETAIL_QTY_RE.search(ln)
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
    # "per bottle or case", "bottle or a case", "whether that's a bottle or a
    # case" (Samuel 2026-11-21: Westhaven Galliano "hasn't said whether that's a
    # bottle or a case") — the optional 'a' before each noun was the gap.
    r"(?:per\s+)?(?:a\s+)?bottles?\s+or\s+(?:a\s+)?cases?|"
    r"per\s+\w+\s+or\s+per\s+\w+|"   # "per bottle or per case" (Woodford, Samuel 2026-11-20)
    r"price\s+basis\s+not\s+stated", re.IGNORECASE)

# A price basis IS stated when the line names the unit the amount is for — "per
# case", "/btl", "the case", "each". Shared by flag_absent_price_basis (below)
# and the self-check (field_check.check_source_fidelity) so the two never drift.
_BASIS_UNIT = (r"(?:cases?|cs|bottles?|btls?|bot|units?|pieces?|pcs?|pc|cans?|"
               r"kegs?|sets?|candles?|litres?|liters?|ltr|l)")
_BASIS_STATED_RE = re.compile(
    # an optional modifier word allows "per gift set" / "per loose bottle"
    r"\bper\s+(?:\w+\s+)?" + _BASIS_UNIT + r"\b"   # per case / per bottle / per gift set
    r"|/\s*" + _BASIS_UNIT + r"\b"        # /case  /cs  /btl  /can  /keg
    r"|\bthe\s+" + _BASIS_UNIT + r"\b"    # "the case" (Northbank "EUR 9,36 the case")
    r"|\ba\s+" + _BASIS_UNIT + r"\b"      # "a case" (Southglass "EUR 9.12 a case")
    r"|\beach\b|\bapiece\b|\bp\s*/?\s*c\b",
    re.IGNORECASE)
# A currency amount sits on the line (symbol/code before the number, or a number
# followed by a written currency word). Comma-decimals (8,99) are fine here.
_AMOUNT_RE = re.compile(
    r"(?:USD|EUR|GBP|CHF|HKD|SGD|AED|US\$|\$|€|£)\s*\d"
    r"|\d[\d.,]*\s*(?:euros?|dollars?|pounds?|usd|eur|gbp|chf|hkd)\b",
    re.IGNORECASE)
# A line that OPENS with an explicit stock count — "23800 bottles …", "1650
# cases …" — the quantity the extractor must preserve verbatim. Tolerates the
# "> " / bullet quote markers a forwarded email puts in front of each line.
_LEAD_QTY_RE = re.compile(
    r"^[\s>*•·\-]*(\d[\d, ]*\d|\d)\s*(cases?|bottles?|btls?|btl|units?|pcs?|pieces?|cs)\b",
    re.IGNORECASE)
_CAPACITY_RE = re.compile(
    r"per\s+(?:full\s+)?(?:load|truck|pallet|container)|truckload|\bcapacity\b",
    re.IGNORECASE)
# The price amount + its currency as WRITTEN on the line (symbol/code before the
# number, or a number followed by a currency word). Used to quote the amount the
# supplier actually wrote — never a figure the LLM derived (Samuel 2026-11-24:
# Clase Azul "121 Euro" had been quoted back as EUR 726 = 121×6).
_SRC_PRICE_RE = re.compile(
    r"(USD|EUR|GBP|CHF|HKD|SGD|AED|US\$|\$|€|£)\s*([\d][\d.,]*)"
    r"|([\d][\d.,]*)\s*(euros?|dollars?|pounds?|usd|eur|gbp|chf|hkd)\b",
    re.IGNORECASE)
_CUR_WORD = {"euro": "EUR", "euros": "EUR", "dollar": "USD", "dollars": "USD",
             "pound": "GBP", "pounds": "GBP", "$": "USD", "us$": "USD",
             "€": "EUR", "£": "GBP"}


def _norm_amt(s: str) -> Optional[float]:
    """Parse a price string with a European-or-US decimal mark ("8,99", "74.50",
    "1,630" thousands) into a float."""
    s = s.strip()
    if "," in s and "." in s:
        s = (s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".")
             else s.replace(",", ""))
    elif "," in s:
        head, tail = s.rsplit(",", 1)
        s = head.replace(",", "") + "." + tail if len(tail) in (1, 2) else s.replace(",", "")
    return _f(s)


def _quoted_from_source(ln: str, o: Dict[str, Any]) -> tuple:
    """(amount, currency) the supplier WROTE on this line. Falls back to the row's
    own figure only when the line has no parseable price."""
    if ln:
        m = _SRC_PRICE_RE.search(ln)
        if m:
            raw = m.group(2) or m.group(3)
            amt = _norm_amt(raw) if raw else None
            sym = (m.group(1) or m.group(4) or "").lower()
            cur = _CUR_WORD.get(sym, sym.upper()) if sym else _clean(o.get("currency"))
            if amt is not None:
                return amt, cur
    return (_f(o.get("price_per_case")) or _f(o.get("price_per_unit"))), _clean(o.get("currency"))


# An EXPLICITLY-APPLICABLE price basis — anchored to a currency, a figure/price
# word, or a column heading — is valid even when not physically next to the amount
# (Samuel 2026-11-26: "the basis has to be explicitly applicable to it, not
# adjacent"; e.g. Reed "the figure is EUR per case", a "Net EUR per case" column).
# A PACK spec ("6 bottles per case") is NOT anchored this way, so it is not a basis.
_BASIS_CLAUSE_RE = re.compile(
    r"(?:USD|EUR|GBP|CHF|HKD|SGD|CAD|AUD|\$|€|£|net|figure|price|amount|quote|rate|"
    r"quotation|desk\s+quote)\s*(?:is\s+)?(?:[\w.,$€£]+\s+){0,3}?(?:per|/)\s*"
    r"(?:case|bottle|btl|unit|piece|pc|candle|set|keg|six-?pack|carton)",
    re.IGNORECASE)


def _basis_stated_near_price(ln: str) -> bool:
    """True when the price basis is stated FOR the amount: either a basis word next
    to the price, OR an explicitly-applicable clause / column heading elsewhere on
    the line ('the figure is EUR per case', 'Net EUR per case', 'EUR 0.52 per
    bottle'). A pack spec ('6 bottles per case') is NOT such a clause and does not
    count (Samuel 2026-11-25/26)."""
    for m in _SRC_PRICE_RE.finditer(ln):
        win = ln[max(0, m.start() - 6): m.end() + 22]
        if _BASIS_STATED_RE.search(win):
            return True
    return bool(_BASIS_CLAUSE_RE.search(ln))


def flag_absent_price_basis(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Price basis ASSUMED from convention. When a text offer quotes a bare amount
    with NO per-unit/per-case word on the line ("at 121 Euro", "116 USD"), the
    extractor guesses per-case and back-fills the other side — a guess the source
    does not support (Samuel 2026-11-24: Sabina "at 121 Euro" became EUR 121/case
    AND EUR 20.17/bottle; Clase Azul, a per-bottle super-premium, read as €20/btl).
    Keep the quoted amount + currency, clear BOTH comparables, flag for review.

    This is the gap that let the URD/HAZ basis rule regress: flag_unstated_price_basis
    only fires when the supplier WRITES "basis not stated" / "per bottle or case".
    Real suppliers omit the basis silently, so this keys on the ABSENCE of a basis
    word near the amount, not the presence of a disclaimer."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        if o.get("_structured_row") or o.get("_unit_priced"):
            continue                      # structured (columnar basis) or already priced per unit
        pc = _f(o.get("price_per_case")); pu = _f(o.get("price_per_unit"))
        if not (pc or pu):
            continue                      # no price (or already cleared) to judge
        if "as quoted" in (str(o.get("commercial_conditions") or "")).lower():
            continue                      # explicit-phrase rule already handled it
        # A UNIQUE source line (one line holds every identity token) can be read
        # for the amount; the fuzzy best-overlap matcher returns the SAME line for
        # every duplicate-named lot, so it must NEVER drive the amount — it put
        # "EUR 145" on all six Chivas lots (caught on prod, 2026-11-24). For a
        # non-unique row the LLM's own per-row field is the quoted amount.
        toks = [t for t in re.findall(r"[a-z0-9]+", _ident(o).lower()) if len(t) > 2]
        hits = [l for l in lines if toks and all(t in l.lower() for t in toks)]
        ln = hits[0] if len(hits) == 1 else _source_line_for(o, lines)
        if not ln or not _AMOUNT_RE.search(ln):
            continue                      # can't see the amount on the line → don't touch
        if _basis_stated_near_price(ln):
            continue                      # basis IS stated next to the price → green
        # Unique line → quote the amount the supplier WROTE (fixes a derived field,
        # e.g. Clase Azul 726 = 121×6). Non-unique → the row's own field is right.
        if len(hits) == 1:
            amt, cur = _quoted_from_source(ln, o)
        else:
            amt, cur = (pc or pu), _clean(o.get("currency"))
        for f in ("price_per_unit", "price_per_case",
                  "price_per_unit_eur", "price_per_case_eur"):
            o[f] = None
        o["needs_manual_review"] = True
        quoted = f"{cur} {amt:g}".strip() if amt else "amount"
        flags = [f for f in (o.get("error_flags") or [])
                 if "calculated from price_per_case" not in str(f).lower()
                 and "price_per_case calculated" not in str(f).lower()]
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


def fix_quantity_from_source(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Preserve an explicit stock count the extractor mangled. The LLM silently
    turned "23800 bottles … lead time 7 days" into 3,400 (= 23800 ÷ 7), yet the
    sibling line "20100 bottles" came through intact — an arithmetic hallucination,
    not a deterministic bug, so it can't be prompt-tuned away (Samuel 2026-11-24).
    When a line OPENS with "<N> cases/bottles" and the stored quantity differs,
    restore the stated number and flag it. Capacity lines ("per load") are left to
    capture_ftl_quantity."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        # Overwriting a quantity is destructive, so demand an UNAMBIGUOUS match:
        # exactly ONE source line that contains every identity token of this row.
        # The fuzzy best-overlap matcher (_source_line_for) collapses duplicate-
        # named lots onto one line — five Chivas lots all matched the "240 cases"
        # line — so it must never drive a correction (caught in offline test,
        # 2026-11-24). A unique match is the only safe anchor.
        toks = [t for t in re.findall(r"[a-z0-9]+", _ident(o).lower()) if len(t) > 2]
        if not toks:
            continue
        hits = [ln for ln in lines if all(t in ln.lower() for t in toks)]
        if len(hits) != 1:
            continue
        ln = hits[0]
        if _CAPACITY_RE.search(ln):
            continue
        m = _LEAD_QTY_RE.match(ln)
        if not m:
            continue
        src_n = _int(m.group(1))
        if not src_n:
            continue
        unit = m.group(2).lower()
        fld = "quantity_case" if (unit.startswith("case") or unit == "cs") else "quantity_bottles"
        cur = _f(o.get(fld))
        if cur is None or abs(cur - src_n) < 0.5:
            continue                      # no stored value to correct, or already right
        o[fld] = float(src_n)
        flags = list(o.get("error_flags") or [])
        fnote = (f"quantity corrected to {src_n} {unit} stated in the source "
                 f"(extractor stored {cur:g})")
        if fnote not in flags:
            flags.append(fnote)
        o["error_flags"] = flags
        o["needs_manual_review"] = True
        _mark(o)
        changed += 1
    return changed


# A line that NAMES a customs tier only to forbid inheriting it, or says the
# status is pending / not supplied / not agreed — so the named tier is NOT this
# line's own status (Samuel 2026-11-25: Fallow "do not assume T2 from W01").
_CUSTOMS_NEGATED_RE = re.compile(
    # "do not assume" AND "do not copy" (Samuel 2026-11-27: Northwick W14 Willow
    # "Customs unconfirmed: do not copy T2 from the Kestrel offers").
    r"do\s*n(?:o|')t\s+(?:assume|copy)"
    r"|not\s+(?:to\s+be\s+)?(?:assumed?|inherit(?:ed)?|copied)"
    r"|customs\s+status\s+(?:still\s+)?pending|status\s+(?:still\s+)?pending"
    r"|customs\s+status\s+not\s+(?:supplied|stated|agreed|confirmed|provided|given)"
    r"|customs\s+status\s+not\s+supplied\s+for\s+this\s+item"
    r"|customs\s+(?:status\s+)?unconfirmed"
    r"|t\s*[12]\b[^.]{0,40}does\s+not\s+apply"
    r"|no\s+(?:general\s+)?customs\s+status",
    re.IGNORECASE)


def _unique_source_line(o: Dict[str, Any], lines: List[str]) -> Optional[str]:
    """The ONE source line that contains every identity token of this row, or None.
    An amount-changing pass must use this, never the fuzzy best-overlap matcher,
    which returns the same line for every duplicate-named lot and would corrupt it
    (Samuel: _source_line_for put one price on all six Chivas lots)."""
    toks = [t for t in re.findall(r"[a-z0-9]+", _ident(o).lower()) if len(t) > 2]
    if not toks:
        return None
    hits = [l for l in lines if all(t in l.lower() for t in toks)]
    return hits[0] if len(hits) == 1 else None


def _row_line_by_amount(o: Dict[str, Any], lines: List[str]) -> Optional[str]:
    """Like _unique_source_line but, when two lots share the name (Grey Inlet
    Eider Rock C-83 open vs C-84 closed), disambiguate by the row's own price /
    quantity number so each lot resolves to its OWN line. Returns None if still
    ambiguous — the caller then does nothing, never acts on the wrong lot."""
    toks = [t for t in re.findall(r"[a-z0-9]+", _ident(o).lower()) if len(t) > 2]
    if not toks:
        return None
    cand = [l for l in lines if all(t in l.lower() for t in toks)]
    if len(cand) == 1:
        return cand[0]
    nums = []
    for fld in ("price_per_case", "price_per_unit", "quantity_case", "quantity_bottles"):
        v = _f(o.get(fld))
        if v:
            nums.append(str(int(v)) if float(v) == int(v) else f"{v:g}")
    # Word/decimal boundaries so a quantity/price number is not matched inside a
    # customs token ("T2") or another number ("27", "2.70", "43%").
    matched = [l for l in cand
               if any(re.search(rf"(?<![\w.]){re.escape(n)}(?![\w.])", l) for n in nums)]
    return matched[0] if len(matched) == 1 else None


# ── Superseded / withdrawn price (Samuel 2026-11-25) ─────────────────────────
# Estuary Birch "Superseded: current price EUR 60 per case; 54 was the previous
# quote" was stored as 54. When the source states the CURRENT price and marks the
# other withdrawn, the current figure wins and the row stays usable.
_CURRENT_PRICE_RE = re.compile(
    r"(?:current|new|correct|revised)\s+price\s*(?:is\s*)?"
    r"(?P<cur>USD|EUR|GBP|CHF|HKD|SGD|\$|€|£)?\s*(?P<amt>\d[\d.,]*)",
    re.IGNORECASE)
_WITHDRAWN_RE = re.compile(
    r"\bwithdrawn\b|\bsuperseded\b|must\s+not\s+be\s+used|no\s+longer\s+valid"
    r"|previous\s+quote|earlier\b[^.]*\bquote",
    re.IGNORECASE)


def apply_superseded_price(offers: List[Dict[str, Any]], source_text: str) -> int:
    """When the source states a CURRENT price and marks an earlier one withdrawn /
    superseded, the current figure wins. If the stored row kept the withdrawn
    amount, correct every price field proportionally and note it; the row stays
    usable because the source is explicit about which price applies."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        ln = _unique_source_line(o, lines)
        if not ln:
            continue
        low = ln.lower()
        if not (_WITHDRAWN_RE.search(low) and _CURRENT_PRICE_RE.search(ln)):
            continue
        m = _CURRENT_PRICE_RE.search(ln)
        try:
            cur_amt = round(float(m.group("amt").replace(",", "")), 2)
        except (TypeError, ValueError):
            continue
        old = _f(o.get("price_per_case"))
        if old is None or old <= 0 or abs(old - cur_amt) < 0.01:
            continue                      # already current, or no case price to fix
        ratio = cur_amt / old
        o["price_per_case"] = cur_amt
        for f in ("price_per_unit", "price_per_unit_eur", "price_per_case_eur"):
            v = _f(o.get(f))
            if v is not None:
                o[f] = round(v * ratio, 4)
        flags = list(o.get("error_flags") or [])
        note = (f"used the stated current price "
                f"{(m.group('cur') + ' ') if m.group('cur') else ''}{cur_amt:g} per case "
                f"(the earlier {old:g} was withdrawn)")
        if note not in flags:
            flags.append(note)
        o["error_flags"] = flags
        _mark(o)
        changed += 1
    return changed


# ── No available stock: reserved / sold out / nothing to offer (Samuel 2026-11-25:
# Estuary Raven "Reserved in full; no stock available for new orders"). ──────────
_NO_STOCK_RE = re.compile(
    r"reserved\s+in\s+full|fully\s+reserved|reserved\b[^.]*\bno\s+stock"
    r"|no\s+stock\s+(?:available|left|remaining)(?:\s+for\s+new\s+orders)?"
    r"|nothing\s+(?:left|available)\s+to\s+(?:offer|sell)"
    r"|not\s+available\s+for\s+new\s+orders|sold\s+out|all\s+sold\b"
    # "All 70 cases SOLD", "Nothing available", "0 available", "no stock offered"
    # (Samuel 2026-11-26: Northwick W12 Quartz — all 70 sold, previous EUR 36 —
    # stayed green with 70 "available"; mirror Ashmere A14 Quartz).
    r"|all\s+[\d,]+\s+(?:cases?|cs|bottles?|btls?|units?|pcs?|pieces?)\s+sold"
    r"|\bnothing\s+available\b|\b0\s+available\b|\bzero\s+available\b"
    r"|no\s+stock\s+offered\b|all\s+\d[\d,]*\s+sold\b"
    # "Prior parcel closed, no balance; retained for accounts" — a closed parcel
    # with no sellable balance (Samuel 2026-11-25: Grey Inlet C-84 Eider Rock,
    # stored green at a USD 2/case accounts placeholder).
    r"|parcel\s+closed|closed\b[^.]*\bno\s+balance|no\s+balance\b"
    r"|retained\s+for\s+accounts|for\s+accounts\s+only",
    re.IGNORECASE)


def flag_no_available_stock(offers: List[Dict[str, Any]], source_text: str) -> int:
    """A line that says the stock is fully reserved / sold out / nothing left to
    offer has NO quantity available for a new order — flag it for review so it is
    kept out of Best Prices and active comparisons, with its price retained for
    reference (exclusion = needs_manual_review)."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        if o.get("_no_stock_flagged"):
            continue
        ln = _row_line_by_amount(o, lines)
        if not ln or not _NO_STOCK_RE.search(ln):
            continue
        flags = list(o.get("error_flags") or [])
        note = ("no stock available for new orders (reserved / sold) — kept for "
                "reference, excluded from active comparisons")
        if note not in flags:
            flags.append(note)
        o["error_flags"] = flags
        o["needs_manual_review"] = True
        o["_no_stock_flagged"] = True
        _mark(o)
        changed += 1
    return changed


# ── Row-level expiry overrides the circular's general validity (Samuel
# 2026-11-27): Northwick W22 Moor "This particular offer expired 25 November 2026
# and has not been renewed"; Ashmere B09 Hearth "This quote expired 25 November
# 2026. No renewal." An offer whose own line says it has expired is stale and must
# not read as live, even though the header validity is 4 Dec. ────────────────────
_EXPIRED_RE = re.compile(
    r"\bexpired\b[^.]*?(\d{1,2}\s+[A-Za-z]+\s+20\d\d|\d{4}-\d\d-\d\d)"
    r"|has\s+not\s+been\s+renewed|\bno\s+renewal\b", re.IGNORECASE)


def flag_expired_offer(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Flag a row whose own source line states it has expired and not been renewed
    — kept visible for reference, excluded from active comparisons (its price is
    stale)."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        if o.get("_expired_flagged"):
            continue
        ln = _row_line_by_amount(o, lines) or _unique_source_line(o, lines)
        if not ln:
            continue
        m = _EXPIRED_RE.search(ln)
        if not m:
            continue
        _edate = ""
        try:
            _edate = (m.group(1) or "").strip()
        except (IndexError, AttributeError):
            _edate = ""
        flags = list(o.get("error_flags") or [])
        note = (f"offer expired{(' ' + _edate) if _edate else ''} and not renewed "
                f"— stale; excluded from active comparisons, retained for reference")
        if note not in flags:
            flags.append(note)
        o["error_flags"] = flags
        o["needs_manual_review"] = True
        o["_expired_flagged"] = True
        _mark(o)
        changed += 1
    return changed


# ── Lot total vs breakdown conflict (Samuel 2026-11-25): Estuary D12 Fern
# "Declared total 102 cases … BBD lots: 44 + 60 = 104"; D08 Pale Ferry balance 96
# vs manifest 60 + 40 = 100. A stated total that disagrees with its own breakdown
# must not pass as a confirmed quantity. ──────────────────────────────────────────
_BREAKDOWN_SCOPE_RE = re.compile(
    r"(?:bbd\s+lots?|manifest|breakdown|lots?\s*:|split\s+as|comprising|made\s+up\s+of)(.*)",
    re.IGNORECASE)
_BREAKDOWN_QTY_RE = re.compile(r"(\d[\d,]*)\s*(?:cases?|cs|dated)\b", re.IGNORECASE)


def flag_lot_total_conflict(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Flag a row whose stated total quantity disagrees with its own lot/manifest
    breakdown — the quantity is not confirmed and must go to review."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        q = _f(o.get("quantity_case"))
        if not q:
            continue
        ln = _row_line_by_amount(o, lines) or _unique_source_line(o, lines)
        if not ln:
            continue
        lots = []
        ms = _BREAKDOWN_SCOPE_RE.search(ln)
        if ms:
            for x in _BREAKDOWN_QTY_RE.findall(ms.group(1)):
                try:
                    lots.append(int(x.replace(",", "")))
                except ValueError:
                    pass
        # Enumerated per-lot form the scope regex misses: "Lot J1: 49 cases. Lot
        # J2: 71 cases" (Ashmere A12) / "Lot N1=38 cases, N2=61 cases" (Northwick
        # W10). Each lot is a label (letter+digits) then : or = then N cases.
        if len(lots) < 2:
            lots = [int(x.replace(",", "")) for x in re.findall(
                r"(?:lot\s+)?[A-Za-z]{1,3}\d+\s*[:=]\s*(\d[\d,]*)\s*cases?\b",
                ln, re.IGNORECASE)]
        if len(lots) >= 2 and sum(lots) != int(q):
            flags = list(o.get("error_flags") or [])
            note = (f"declared total {int(q)} cases ≠ lot breakdown "
                    f"{' + '.join(str(x) for x in lots)} = {sum(lots)} — "
                    f"quantity conflict, confirm with supplier")
            if note not in flags:
                flags.append(note)
            o["error_flags"] = flags
            o["needs_manual_review"] = True
            _mark(o)
            changed += 1
    return changed


# ── Retail unit price, not a case (Samuel 2026-11-25): Dune "EUR 21 per gift set
# … This is not a case price" → 3/case, EUR 63/case; Mere "14 pieces at USD 23 per
# bottle" → invented 14/case; Amber "EUR per candle; stock is pieces". The item is
# sold as individual sets/candles/pieces; a case price here is invented. ─────────
_RETAIL_UNIT_SIGNAL_RE = re.compile(
    r"not\s+a\s+case\s+price|this\s+is\s+not\s+a\s+case"
    r"|per\s+gift\s*set|per\s+set\b|per\s+candle\b|price\s+is\s+eur\s+per\s+candle"
    # slash / plural forms the "per <unit>" branch missed: "EUR 23/set", "GBP
    # 6.50 each", "13 sets at EUR 23/set", "8 individual bottles" (Samuel
    # 2026-11-26: Northwick W07 Mica, W04 Linden, W20 Cedar).
    r"|/\s*(?:gift\s*)?set\b|/\s*candle\b|/\s*spray\b"
    r"|\d+\s+(?:intact\s+)?sets?\b|\bsets?\s+at\b|intact\s+set"
    r"|\d+\s+individual\s+(?:bottles?|candles?|pieces?)"
    r"|retail\s+box|retail\s+bottle|retail\s+boxed|\d+\s+pieces?\b|stock\s+is\s+pieces",
    re.IGNORECASE)
_CASE_BASIS_SIGNAL_RE = re.compile(
    r"per\s+case|/\s*case|\bthe\s+case\b|\bcases?\s+(?:of|@)\b"
    r"|per\s+six-?pack|per\s+pack\b",
    re.IGNORECASE)


def honor_retail_unit_price(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Honour an item sold per individual set / candle / piece, dropping an invented
    case price + pack and recovering the per-unit amount from the source when the
    extractor blanked it. Only fires when the line signals a retail unit AND states
    no case basis — so normal 'per case' lines are untouched."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        ln = _unique_source_line(o, lines)
        if not ln:
            continue
        low = ln.lower()
        if not _RETAIL_UNIT_SIGNAL_RE.search(low) or _CASE_BASIS_SIGNAL_RE.search(low):
            continue
        unit_word = ("sets" if re.search(r"gift\s*set|\bset\b", low)
                     else "pieces")
        pu = _f(o.get("price_per_unit"))
        pc = _f(o.get("price_per_case"))
        # An EXPLICIT per-unit amount on the line ("$24 each", "EUR 21 per gift set")
        # is authoritative — override a wrong/bled stored value, not just fill a
        # blank (Samuel 2026-11-25: Mere "$24 each" had been stored as 0.52).
        _mex = re.search(
            r"(USD|EUR|GBP|CHF|HKD|SGD|CAD|AUD|\$|€|£)\s*([\d][\d.,]*)\s*"
            r"(?:each|apiece"
            r"|per\s+(?:gift\s*set|set|candle|piece|pc|bottle|unit)"
            r"|/\s*(?:gift\s*set|set|candle|piece|pc|bottle|unit))",   # "EUR 23/set"
            ln, re.IGNORECASE)
        if _mex:
            try:
                _ev = round(float(_mex.group(2).replace(",", "")), 4)
                if _ev > 0 and _ev != pu:
                    o["price_per_unit"] = _ev
                    o["price_per_unit_eur"] = None   # recomputed downstream
                    pu = _ev
                    _cw = _mex.group(1)
                    if _cw and not _clean(o.get("currency")):
                        o["currency"] = _CUR_WORD.get(_cw.lower(), _cw.upper())
            except ValueError:
                pass
        # Recover the per-unit price from the source line when it was dropped
        # (Amber: price blanked). Prefer a price with a per-set/per-candle basis.
        if pu is None:
            amt, cur = _quoted_from_source(ln, o)
            if not amt:
                # A retail price is often a bare decimal sitting among integer
                # quantities / weights ("… | 25 | 9.8 | …", Amber per-candle). Pick
                # a decimal that is NOT a weight / volume / ABV, so 9.8 is read as
                # the price while 200 g / 25 cases / 4.5% are not.
                for _mm in re.finditer(
                        r"(?<![\d.])(\d+\.\d+)(?!\s*(?:g|kg|mg|ml|cl|l|%|cm|yo))",
                        ln, re.IGNORECASE):
                    try:
                        amt = round(float(_mm.group(1)), 4)
                        break
                    except ValueError:
                        continue
            if amt:
                pu = amt
                o["price_per_unit"] = amt
                if cur and not _clean(o.get("currency")):
                    o["currency"] = cur
        # A per-set / retail item whose ONLY amount the extractor stored as a
        # "case" price (Samuel 2026-11-26: Mica Garden Set — "EUR 23/set" landed
        # in price_per_case, shown as EUR 23/case) must move to the unit price.
        if pu is None and pc is not None:
            o["price_per_unit"] = pc
            o["price_per_unit_eur"] = None
            pu = pc
        # An invented case price (per-unit × pack) is wrong for a retail unit.
        if pc is not None:
            o["price_per_case"] = None
            o["price_per_case_eur"] = None
        o["units_per_case"] = None
        o["quantity_unit"] = unit_word
        # A per-piece item has no case quantity. When the real count is already in
        # quantity_bottles, a leftover quantity_case (the LLM read "8 bottles" as
        # "1 case") would show a phantom "1 case" beside the pieces — drop it
        # (Samuel 2026-11-27: Northwick W20 Cedar, quantity_case=1 + 8 bottles).
        if _f(o.get("quantity_bottles")) is not None:
            o["quantity_case"] = None
        flags = [f for f in (o.get("error_flags") or [])
                 if "assumed per case" not in str(f).lower()
                 and "price ambiguous" not in str(f).lower()
                 and "price_per_case calculated" not in str(f).lower()]
        _u = unit_word[:-1] if unit_word.endswith("s") else unit_word
        note = f"priced per {_u}; sold as individual {unit_word}, not a case"
        if note not in flags:
            flags.append(note)
        o["error_flags"] = flags
        o["_unit_priced"] = True          # basis is per unit — absent-basis pass must not clear it
        _mark(o)
        changed += 1
    return changed


# ── Supplier from an explicit 'Supplier:' line (Samuel 2026-11-25): Estuary rows
# took the sheet TITLE as supplier, then 'Not Found'; the circular states
# "Supplier: Estuary Ledger Merchants BV". ──────────────────────────────────────
_SUPPLIER_LINE_RE = re.compile(r"\bsupplier\s*[:\-]\s*(.+)", re.IGNORECASE)
_SUPPLIER_JUNK_RE = re.compile(
    r"—|–|\d{4}|additional\s+stock|desk\s+parcels|circular|stock\s+(?:note|list)"
    r"|price\s+list|offer\s+circular", re.IGNORECASE)


def fill_supplier_from_source(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Read an explicit 'Supplier: X' line from the document and apply it to rows
    that have no real supplier (missing, 'Not Found', or a title-derived junk value
    carrying a dash / date / sheet name)."""
    if not source_text:
        return 0
    sup = None
    for raw in str(source_text).splitlines():
        m = _SUPPLIER_LINE_RE.search(raw.strip())
        if not m:
            continue
        cand = m.group(1).split("|")[0].strip()
        cand = re.split(r"\s+(?:circular|ref\b|offer\b|stock\b|price\b|subject\b)",
                        cand, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if len(cand) >= 3 and re.search(r"[A-Za-z]", cand):
            sup = cand
            break
    if not sup:
        return 0
    changed = 0
    for o in offers:
        cur = _clean(o.get("supplier_name"))
        if cur and cur.lower() not in ("not found", "unknown", ""):
            if not _SUPPLIER_JUNK_RE.search(cur):
                continue                  # already a real supplier
        o["supplier_name"] = sup
        _mark(o)
        changed += 1
    return changed


# ── Price basis / stock unit the source explicitly leaves UNCONFIRMED (Samuel
# 2026-11-25: Cairn W05 "Desk figure GBP 3.60 … the desk hasn't confirmed what the
# price covers … the warehouse hasn't identified what the balance counts"). The
# extractor assumed per-bottle + per-case + 315 cases + EUR; all unsupported. ────
_PRICE_UNCONFIRMED_RE = re.compile(
    r"(?:has\s*n(?:o|')t|have\s*n(?:o|')t|not)\s+confirmed\s+what\s+the\s+"
    r"(?:price|figure|amount)\s+(?:covers|is\s+for|applies\s+to)"
    r"|what\s+the\s+(?:price|figure)\s+covers\b"
    r"|price\s+basis\s+(?:unconfirmed|not\s+confirmed)",
    re.IGNORECASE)
_UNIT_UNCONFIRMED_RE = re.compile(
    r"(?:has\s*n(?:o|')t|have\s*n(?:o|')t|not)\s+(?:identified|confirmed)\s+what\s+"
    r"the\s+(?:balance|quantity|figure|stock)\s+(?:counts|is|means)"
    r"|what\s+the\s+balance\s+counts\b"
    r"|(?:quantity|stock)\s+unit\s+(?:unconfirmed|not\s+confirmed)",
    re.IGNORECASE)


def flag_unconfirmed_basis_unit(offers: List[Dict[str, Any]], source_text: str) -> int:
    """When the source EXPLICITLY says the price basis and/or the stock unit is
    unconfirmed, the extractor's per-bottle/per-case/quantity assumptions are
    unsupported: keep the amount as quoted, clear the comparables, drop the guessed
    quantity unit, flag for review (kept out of Best Prices)."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        ln = _unique_source_line(o, lines)
        if not ln:
            continue
        price_unconf = bool(_PRICE_UNCONFIRMED_RE.search(ln))
        unit_unconf = bool(_UNIT_UNCONFIRMED_RE.search(ln))
        if not (price_unconf or unit_unconf):
            continue
        flags = [f for f in (o.get("error_flags") or [])
                 if "calculated from" not in str(f).lower()]
        if price_unconf:
            amt = _f(o.get("price_per_unit")) or _f(o.get("price_per_case"))
            cur = _clean(o.get("currency"))
            for f in ("price_per_unit", "price_per_case",
                      "price_per_unit_eur", "price_per_case_eur"):
                o[f] = None
            quoted = f"{cur} {amt:g}".strip() if amt else "amount"
            note = (f"{quoted} quoted — the source does not confirm what the price "
                    f"covers (per bottle or per case); excluded from comparison")
            if note not in flags:
                flags.append(note)
            cc = _clean(o.get("commercial_conditions"))
            if "as quoted" not in cc.lower():
                o["commercial_conditions"] = ((cc + " | " if cc else "")
                                              + f"as quoted: {quoted} (price basis unconfirmed)")
        if unit_unconf:
            o["quantity_unit"] = None
            note = ("stock unit unconfirmed — the source does not state what the "
                    "balance counts (cases or bottles)")
            if note not in flags:
                flags.append(note)
        o["error_flags"] = flags
        o["needs_manual_review"] = True
        _mark(o)
        changed += 1
    return changed


def clear_unsourced_custom_status(offers: List[Dict[str, Any]], source_text: str) -> int:
    """Drop a T1/T2 customs status the source never states for a line. The LLM
    stamped T2 on nine Sabina rows (Chivas, Hennessy, Veuve, Havana, Baileys, JD
    Honey) when only Proper12 carried it (Samuel 2026-11-24). Mirror of
    clear_unsourced_incoterm: act only when the matched source line for THIS
    product carries no T1/T2 token of its own."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        cs = _clean(o.get("custom_status"))
        m = re.search(r"t\s*([12])\b", cs, re.IGNORECASE) if cs else None
        if not m:
            continue
        tier = m.group(1)
        ln = _source_line_for(o, lines)
        if not ln:
            continue                      # line not found → leave as-is
        # The line can NAME T{tier} only to forbid inheriting it ("do not assume T2
        # from W01"), or declare the status pending/not-supplied. That is NOT the
        # line stating its own status — clear it (Samuel 2026-11-25: Fallow W16
        # "Customs status still pending; do not assume T2 from W01" kept T2).
        _negated = bool(_CUSTOMS_NEGATED_RE.search(ln))
        if re.search(rf"\bT\s*{tier}\b", ln, re.IGNORECASE) and not _negated:
            continue                      # line genuinely states T{tier}
        o["custom_status"] = None
        flags = list(o.get("error_flags") or [])
        fnote = ((f"customs status pending / not to be inherited — cleared per the "
                  f"source for this line")
                 if _negated else
                 (f"customs status T{tier} not stated in the source for this line "
                  f"(inherited from another line)"))
        if fnote not in flags:
            flags.append(fnote)
        o["error_flags"] = flags
        o["needs_manual_review"] = True
        _mark(o)
        changed += 1
    return changed


_PARCEL_RE = re.compile(r"\bparcel\s+([A-Z]{1,4}[-\s]?\d{1,6}[A-Z]?)\b", re.IGNORECASE)


def _parcel_label(o: Dict[str, Any], lines: Optional[List[str]]) -> Optional[str]:
    """The parcel/lot id the source names for a row ("Parcel MB71"), used to tell
    the reviewer the two delivery options are literally one stock."""
    if not lines:
        return None
    ln = _unique_source_line(o, lines) or _row_line_by_amount(o, lines)
    if not ln:
        return None
    m = _PARCEL_RE.search(ln)
    return m.group(1).upper().replace(" ", "") if m else None


def _blank_quantity(o: Dict[str, Any]) -> None:
    for k in ("quantity_case", "quantity_bottles", "quantity_pallets",
              "quantity_ftl", "quantity_unit"):
        o[k] = None


_POOL_QTY_FIELDS = ("quantity_case", "quantity_bottles", "quantity_pallets",
                    "quantity_ftl", "quantity_unit", "units_per_case")


def _link_shared_pool(keep: Dict[str, Any], alt: Dict[str, Any],
                      lines: Optional[List[str]]) -> None:
    """One physical stock offered under TWO DISTINCT delivery terms, each with its
    own price, is TWO INDEPENDENTLY COMPARABLE offers sharing ONE stock pool
    (Samuel 2026-11-26: Moss Bridge Whisky EXW Ghent EUR 120 vs DAP Lille EUR 126,
    parcel MB71; Rowan Ferry Whisky). Both legs are full, co-equal offers — each
    carries the SAME shared quantity and competes in its own incoterm peer group —
    linked by the parcel id and marked "counted once". This is safe because nothing
    sums quantity across rows (Best Prices / benchmarks / alerts rank price only),
    so a buyer sees availability on either leg without the pool being double-counted.
    Clears the dual-delivery review markers — both terms are explicitly offered, so
    neither row is an ambiguity (the gate re-flags either independently if needed)."""
    # Share the ONE pool's stock onto BOTH legs (the source says it is the same
    # stock), taking the non-empty value from either — so each leg is a full
    # comparable offer showing the same availability (incl. "9 cases + 5 loose"),
    # regardless of which leg was chosen as the richer "keep".
    for k in _POOL_QTY_FIELDS:
        v = keep.get(k) if keep.get(k) not in (None, "", 0) else alt.get(k)
        if v not in (None, "", 0):
            keep[k] = v
            alt[k] = v
    parcel = _parcel_label(keep, lines) or _parcel_label(alt, lines)
    pc = f"parcel {parcel} " if parcel else ""
    kterm = " ".join(x for x in (_clean(keep.get("incoterm")), _clean(keep.get("location"))) if x) or "the primary"
    aterm = " ".join(x for x in (_clean(alt.get("incoterm")), _clean(alt.get("location"))) if x) or "the alternative"
    knote = (f"One stock pool ({pc}counted once): this {kterm} offer and the "
             f"{aterm} offer are the SAME stock — take from either, not both.")
    anote = (f"One stock pool ({pc}counted once): this {aterm} offer and the "
             f"{kterm} offer are the SAME stock — take from either, not both.")
    for o, note in ((keep, knote), (alt, anote)):
        cc = _clean(o.get("commercial_conditions"))
        if "one stock pool" not in cc.lower():
            o["commercial_conditions"] = (cc + " | " if cc else "") + note
        ef = o.get("error_flags")
        if isinstance(ef, list):
            o["error_flags"] = [f for f in ef if not (isinstance(f, str) and (
                "multiple incoterms detected" in f
                or "alternative delivery term" in f
                or "confirm which applies" in f
                or "recover alternatives" in f))]
        # a split-by-incoterm pair is intentional; let the gate decide review
        o["needs_manual_review"] = False
        o["_pool_linked"] = True
        if parcel:
            o["_stock_pool"] = parcel   # machine link between the co-equal legs
        _mark(o)


def _split_or_fold_siblings(keep: Dict[str, Any], others: List[Dict[str, Any]],
                            lines: Optional[List[str]]) -> List[Dict[str, Any]]:
    """Partition the non-kept rows of a same-stock group into those that SPLIT off
    as their own comparable sibling (distinct incoterm AND distinct case price)
    and those that FOLD into `keep` as a note. Returns the still-to-fold rows."""
    kinc = _norm(keep.get("incoterm")); kpc = _f(keep.get("price_per_case"))
    to_fold: List[Dict[str, Any]] = []
    for o in others:
        oinc = _norm(o.get("incoterm")); opc = _f(o.get("price_per_case"))
        if (oinc and kinc and oinc != kinc
                and opc is not None and kpc is not None
                and round(opc, 2) != round(kpc, 2)):
            _link_shared_pool(keep, o, lines)
        else:
            to_fold.append(o)
    return to_fold


_SAME_STOCK_RE = re.compile(
    r"same\s+stock\s+(?:either\s+way|both\s+ways|regardless)"
    r"|same\s+stock\b|same\s+(?:physical\s+)?lot\s+either|one\s+(?:and\s+the\s+same|single)\s+lot"
    # "Same MK884 allocation … Alternative for whole parcel only" — one parcel
    # quoted a second way (Samuel 2026-11-25: Grey Inlet C-81/C-81B Morrow Key Rye).
    r"|same\s+[A-Z0-9-]{2,}\s+(?:allocation|parcel|lot)"
    r"|alternative\s+for\s+(?:the\s+)?whole\s+parcel"
    r"|alternative\s+for\s+(?:the\s+)?same\s+(?:parcel|stock|lot)",
    re.IGNORECASE)


def merge_same_stock_restatement(offers: List[Dict[str, Any]], source_text: str) -> Set[str]:
    """One lot quoted under two delivery options that the source explicitly calls
    the SAME stock ("…deliver Lille at EUR 53.40/case instead … Same stock either
    way", Samuel 2026-11-24 Whitley Neill) must count once. The extractor sometimes
    splits it into two rows and stamps BOTH with the base delivery term, so
    merge_alt_delivery_lots can't tell them apart (same incoterm, two prices). This
    uses the source phrase as the authority: keep the richest row, fold the other
    price/term into a note, and count the stock once. Returns dropped uids."""
    if not source_text:
        return set()
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    dropped: Set[str] = set()
    groups: Dict[tuple, list] = defaultdict(list)
    for o in offers:
        groups[(_norm(o.get("brand")), _norm(o.get("product_name")))].append(o)
    for g in groups.values():
        if len(g) < 2:
            continue
        # The "same stock / alternative for the whole parcel" phrase can sit on
        # EITHER row of the pair (Grey Inlet C-81B carries it, not C-81), so scan
        # every source line that names this product, not just the first row's line.
        toks = [t for t in re.findall(r"[a-z0-9]+", _ident(g[0]).lower()) if len(t) > 2]
        glines = [l for l in lines if toks and all(t in l.lower() for t in toks)]
        if not any(_SAME_STOCK_RE.search(l) for l in glines):
            continue
        # Same physical stock → at most one distinct quantity across the pair (an
        # alternative-terms row often carries no quantity of its own).
        qtys = {_primary_qty(o) for o in g if _primary_qty(o) is not None}
        if len(qtys) > 1:
            continue
        keep = sorted(g, key=lambda o: (
            bool(_f(o.get("price_per_case"))),
            bool(_clean(o.get("lead_time"))),
            not bool(o.get("needs_manual_review"))), reverse=True)[0]
        # Distinct delivery terms at distinct prices become their own comparable
        # siblings (Samuel 2026-11-26 Moss); same-term restatements still fold.
        others = [o for o in g if o is not keep and not o.get("_pool_linked")]
        to_fold = _split_or_fold_siblings(keep, others, lines)
        alts = []
        for o in to_fold:
            dropped.add(o.get("uid"))
            term = " ".join(x for x in (_clean(o.get("incoterm")), _clean(o.get("location"))) if x)
            apc = _f(o.get("price_per_case")); acur = _clean(o.get("currency"))
            label = (term or "alternative delivery")
            if apc:
                label += f" ({(acur + ' ') if acur else ''}{apc:g}/case)"
            alts.append(label)
        if alts:
            note = ("Also available (same stock, counted once): "
                    + " ; ".join(dict.fromkeys(alts)))
            cc = _clean(keep.get("commercial_conditions"))
            if "same stock" not in cc.lower():
                keep["commercial_conditions"] = (cc + " | " if cc else "") + note
            _mark(keep)
        if dropped:
            offers[:] = [o for o in offers if o.get("uid") not in dropped]
    return dropped


_SAME_PRODUCT_RE = re.compile(
    r"\bsame\s+(?:beer|product|item|wine|spirit|gin|vodka|whisky|whiskey|rum|lager|"
    r"brand|one|liquor|bottle)\b", re.IGNORECASE)


def resolve_supplier_as_brand(offers: List[Dict[str, Any]], source_text: str) -> int:
    """A product's brand is never the supplier's own company name. When the
    extractor stamps the supplier as the brand — because the line refers back to a
    previous product ("Same beer, but the wrapped 4 six-packs…" → the Carlsberg
    line above; Samuel 2026-11-24) — inherit the brand from the nearest PRECEDING
    offer with a real brand, and replace an anaphora placeholder product name ("Same
    Beer") with that product. The pack format stays distinct, so the wrapped line
    remains its own row."""
    if not source_text or not offers:
        return 0
    sup = _norm(offers[0].get("supplier_name"))
    if not sup:
        return 0
    # The brand may be the supplier name WITHOUT its legal suffix ("Southglass
    # Drinks" vs "Southglass Drinks BV" — Samuel 2026-11-24), so match on the
    # suffix-stripped form too.
    sup_core = re.sub(r"\s*\b(bv|bvba|ltd|limited|llc|gmbh|inc|incorporated|sa|nv|"
                      r"srl|ag|plc|corp|co)\b\s*$", "", sup).strip()

    def _is_sup(b) -> bool:
        bn = _norm(b)
        return bool(bn) and (bn == sup or bn == sup_core)

    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]

    def _idx(o):
        # A supplier-as-brand row has a polluted identity (the brand is the supplier
        # name, which also appears in the header line), so locate it by its PRODUCT
        # tokens only; donors (real brands) use the normal matcher.
        if _is_sup(o.get("brand")):
            toks = [t for t in re.findall(r"[a-z0-9]+", str(o.get("product_name") or "").lower())
                    if len(t) > 2]
            ln, best = None, 0
            for L in lines:
                s = sum(1 for t in toks if t in L.lower())
                if s > best:
                    ln, best = L, s
            ln = ln if best >= 1 else None
        else:
            ln = _source_line_for(o, lines)
        try:
            return lines.index(ln) if ln else -1
        except ValueError:
            return -1
    idxmap = {id(o): _idx(o) for o in offers}
    changed = 0
    for o in offers:
        if not _is_sup(o.get("brand")):
            continue                      # brand isn't the supplier → nothing to fix
        my = idxmap[id(o)]
        if my < 0:
            continue
        donor, best = None, -1
        for other in offers:
            if other is o:
                continue
            ob = _norm(other.get("brand"))
            if not ob or _is_sup(ob):
                continue
            oi = idxmap[id(other)]
            if 0 <= oi <= my and oi > best:
                donor, best = other, oi
        if not donor:
            continue
        new_brand = donor.get("brand")
        o["brand"] = new_brand
        pn = str(o.get("product_name") or "")
        if _SAME_PRODUCT_RE.search(pn) or _norm(pn) in ("same beer", "same product", "same"):
            o["product_name"] = donor.get("product_name") or new_brand
        flags = list(o.get("error_flags") or [])
        fnote = (f"brand resolved to {new_brand} from a 'same product' reference "
                 f"(was the supplier name); kept as a separate pack format")
        if fnote not in flags:
            flags.append(fnote)
        o["error_flags"] = flags
        _mark(o)
        changed += 1
    return changed


def drop_buy_inquiry_rows(offers: List[Dict[str, Any]], source_text: str) -> Set[str]:
    """Remove a buy-side ENQUIRY line the extractor stored as a sellable offer. In a
    multi-line WhatsApp/email that is mostly a sell offer, one line can be a request
    ("a customer wants 90 cases of North Pier rosé 2024 at around EUR 39/case. Can
    you source that vintage for us?" — Samuel 2026-11-24). The message-level RFQ
    classifier can't see it (the whole message carries prices), so match the row's
    own source line against the strong buy-phrase detector and drop it. Returns the
    set of dropped uids; the offers list is filtered in place."""
    if not source_text:
        return set()
    try:
        from core.inquiry import is_strong_buy_line
    except Exception:
        return set()
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    dropped: Set[str] = set()
    for o in offers:
        ln = _source_line_for(o, lines)
        if ln and is_strong_buy_line(ln):
            uid = o.get("uid")
            if uid is not None:
                dropped.add(uid)
            o["_buy_inquiry_drop"] = True
    if dropped:
        offers[:] = [o for o in offers if o.get("uid") not in dropped]
    else:
        # No uids yet (pre-save) — filter by the marker instead.
        before = len(offers)
        offers[:] = [o for o in offers if not o.get("_buy_inquiry_drop")]
        if len(offers) != before:
            dropped.add("_marker")
    return dropped


# A name that is really the circular's banner, not a product: "Supplier Offer",
# "Supplier Estuary Ledger Merchants BV", "Circular EL-925", "Subject to unsold".
# "supplier"/"circular" may be followed by the company/ref text the LLM swept in.
_HEADER_NAME_RE = re.compile(
    r"^(?:supplier\b.*|offer|circular\b.*|price\s*list|pricelist|"
    r"subject\s+to\s+unsold.*|not\s+found|n/?a|unknown)$", re.IGNORECASE)
_CIRCULAR_REF_RE = re.compile(r"^[A-Z]{1,4}[-\s]?\d{2,6}[A-Z]?$", re.IGNORECASE)


def drop_nonproduct_header_rows(offers: List[Dict[str, Any]],
                               source_text: str = "") -> Set[str]:
    """Drop a row the extractor fabricated from a circular's HEADER / supplier
    line, not a product (Samuel 2026-11-26: "Supplier Offer / EL-925", currency
    EUR, no price, no quantity — the "Supplier: Estuary Ledger Merchants BV |
    Circular EL-925 | Subject to unsold" banner read as an item). It inflated the
    row count by one with no sellable content. Conservative: only a row with NO
    price AND NO quantity whose own identity is a header token (or an empty name
    carrying only a circular reference). A real price-on-request product (Sedge,
    15 cases) keeps its quantity and name, so it is never touched."""
    dropped: Set[str] = set()
    before = len(offers)
    keep: List[Dict[str, Any]] = []
    for o in offers:
        has_price = _f(o.get("price_per_unit")) or _f(o.get("price_per_case"))
        has_qty = any(_f(o.get(k)) for k in (
            "quantity_case", "quantity_bottles", "quantity_pallets", "quantity_ftl"))
        nm = _clean(o.get("product_name"))
        ref = _clean(o.get("product_reference")) or _clean(o.get("supplier_reference"))
        is_header_name = bool(nm) and bool(_HEADER_NAME_RE.match(nm))
        is_bare_ref = (not nm or nm.lower() in ("not found", "unknown")) and \
            bool(ref) and bool(_CIRCULAR_REF_RE.match(ref))
        if (not has_price) and (not has_qty) and (is_header_name or is_bare_ref):
            uid = o.get("uid")
            if uid is not None:
                dropped.add(uid)
            continue
        keep.append(o)
    if len(keep) != before:
        offers[:] = keep
        if not dropped:
            dropped.add("_marker")
    return dropped


_NO_UNIT_LABEL_RE = re.compile(
    r"no\s+unit\s+label"
    r"|has\s+no\s+unit\b"
    r"|without\s+a\s+unit\s+label"
    r"|hasn'?t\s+identified\s+what\s+the\s+balance"
    r"|has\s+not\s+identified\s+what\s+the\s+balance"
    r"|balance[^.]*?\bno\s+unit\b"
    r"|balance[^.]*?(?:unit\s+(?:is\s+)?(?:unknown|not\s+(?:stated|given|labelled|labeled)|unconfirmed))"
    r"|what\s+the\s+balance\s+counts"
    # "Neither the balance unit nor the price unit has been confirmed" (Samuel
    # 2026-11-26: Northwick W05 Slate) — the unit of the 275 balance is unknown.
    r"|neither\s+the\s+balance\s+unit"
    r"|balance\s+unit[^.]*?(?:nor|not\s+(?:been\s+)?confirmed|unconfirmed)",
    re.IGNORECASE)
_BAL_NUM_RE = re.compile(r"\bbalance\s+(?:is\s+)?([\d][\d,]*)\b", re.IGNORECASE)


def flag_unlabeled_quantity_balance(offers: List[Dict[str, Any]],
                                    source_text: str) -> int:
    """A stock balance the SOURCE itself says carries no unit label must not be
    stored as a confident quantity in a fabricated unit (Samuel 2026-11-26: Plover
    Lane Soap "360 … Warehouse balance has no unit label" was stored as 30 cases /
    360 pieces; North Fen Cairn "Balance 315 … hasn't identified what the balance
    counts"). The PRICE basis may still be explicit (USD 1.25 per piece) and is
    kept as quoted; but the quantity unit is genuinely unconfirmed, so clear the
    invented quantity fields, keep the figure in a note, and flag for review —
    consistent with the rule that an unconfirmed stock unit triggers review."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        ln = _unique_source_line(o, lines) or _row_line_by_amount(o, lines) \
            or _source_line_for(o, lines)
        if not ln or not _NO_UNIT_LABEL_RE.search(ln):
            continue
        # already handled (idempotent re-run / twin)
        cc_low = _clean(o.get("commercial_conditions")).lower()
        if "unit label" in cc_low or "unit unconfirmed" in cc_low:
            continue
        bal = None
        m = _BAL_NUM_RE.search(ln)
        if m:
            bal = m.group(1).replace(",", "")
        else:
            for k in ("quantity_bottles", "quantity_case"):
                v = _f(o.get(k))
                if v:
                    bal = f"{v:g}"; break
        # Keep the balance VISIBLE but with its unit marked UNCONFIRMED — the
        # number stays (275), it is NOT presented as cases, and needs_manual_review
        # keeps it out of usable/confirmed stock (Samuel 2026-11-26: "the number
        # must remain visible as an unconfirmed balance, not blanked, not cases").
        o["quantity_bottles"] = None
        o["quantity_pallets"] = None
        o["quantity_ftl"] = None
        if bal:
            try:
                o["quantity_case"] = float(bal)
            except (TypeError, ValueError):
                o["quantity_case"] = None
            o["quantity_unit"] = "unconfirmed"
        else:
            o["quantity_case"] = None
            o["quantity_unit"] = None
        where = f"Warehouse balance {bal}" if bal else "The warehouse balance"
        note = (f"{where} — stock unit unconfirmed (no unit label in the source); "
                f"shown as an unconfirmed balance, not confirmed cases.")
        cc = _clean(o.get("commercial_conditions"))
        o["commercial_conditions"] = (cc + " | " if cc else "") + note
        # The same line may also leave the price BASIS unconfirmed and the
        # extractor may have dropped the amount entirely, so the row reads "no
        # price" although one is stated (Samuel 2026-11-26: Northwick W05 Slate —
        # "GBP 4.20. Neither the balance unit nor the price unit confirmed").
        # Keep the quoted amount so it is neither lost nor reported as missing.
        if not (_f(o.get("price_per_unit")) or _f(o.get("price_per_case"))):
            _pm = re.search(r"\b(GBP|EUR|USD|CHF|HKD|SGD|CAD|AUD)\s*([\d][\d.,]*)", ln)
            if _pm and "as quoted:" not in o["commercial_conditions"].lower():
                _cw = _pm.group(1).upper(); _amt = _pm.group(2).replace(",", "").rstrip(".")
                o["commercial_conditions"] += (
                    f" | as quoted: {_cw} {_amt} (price basis unconfirmed — "
                    f"per bottle or per case not stated)")
                if not _clean(o.get("currency")):
                    o["currency"] = _cw
                o["error_flags"] = [f for f in (o.get("error_flags") or [])
                    if not (isinstance(f, str) and (
                        "no price stated" in f.lower()
                        or "price on request" in f.lower()
                        or "but no price in the row" in f.lower()))]
        ef = o.get("error_flags")
        if not isinstance(ef, list):
            ef = []
        _msg = (f"stock quantity unit unconfirmed — the source states a balance "
                f"with no unit label; quantity left blank, flagged for review")
        if _msg not in ef:
            ef.append(_msg)
        o["error_flags"] = ef
        o["needs_manual_review"] = True
        _mark(o)
        changed += 1
    return changed


def clear_contradictory_currency_flag(offers: List[Dict[str, Any]],
                                      source_text: str = "") -> int:
    """Remove a stale "currency normalized/normalised to EUR" note from a row whose
    stored currency is NOT EUR (Samuel 2026-11-26: Plover shown USD 1.25 still
    carried a "normalized to EUR from the source line" flag — a contradiction that
    reads as if the row were converted). The gate strips this only when it
    neutralises a currency; a confirmed non-EUR row needs the same cleanup."""
    changed = 0
    for o in offers:
        cur = _clean(o.get("currency")).upper()
        if not cur or cur == "EUR":
            continue
        ef = o.get("error_flags")
        if not isinstance(ef, list):
            continue
        new = [f for f in ef if not (isinstance(f, str)
               and ("normalized to eur" in f.lower()
                    or "normalised to eur" in f.lower()))]
        if len(new) != len(ef):
            o["error_flags"] = new
            _mark(o)
            changed += 1
    return changed


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
        _qb = _f(o.get("quantity_bottles")); _upc = _f(o.get("units_per_case"))
        if not qc:
            continue
        ln = _source_line_for(o, lines)
        if not ln:
            continue
        m = _LOOSE_ADDON_RE.search(ln)
        if not m:
            continue
        cases = _int(m.group(1)); loose = _int(m.group(2))
        if not (cases and loose and float(cases) == qc):
            continue
        _did = False
        # The primary count is CASES ("9 cases plus 5 loose bottles"); the
        # extractor sometimes mislabels it "bottles" off the "… loose bottles" /
        # "SAME 59 bottles" tail (Samuel 2026-11-27: Rowan W08 showed "9 bottles").
        # Correct the unit so it reads "9 cases + 5 loose bottles". Always runs,
        # even when the loose count is already right.
        if str(o.get("quantity_unit") or "").lower() in ("bottles", "bottle", "", "none"):
            o["quantity_unit"] = "cases"
            _did = True
        # A real bottle count stays; but a FABRICATED one that is exactly
        # cases × pack is the double-count the extractor made (54 = 9×6, the +5
        # loose lost) — override it with the stated loose count.
        if not _qb or (_upc and abs(_qb - qc * _upc) < 0.5):
            if o.get("quantity_bottles") != float(loose):
                o["quantity_bottles"] = float(loose)
                _did = True
        cc = _clean(o.get("commercial_conditions"))
        if "loose" not in cc.lower():
            o["commercial_conditions"] = ((cc + " | " if cc else "")
                + f"+ {loose} loose bottle" + ("s" if loose != 1 else ""))
            _did = True
        if _did:
            _mark(o)
            changed += 1
    return changed


_PACK_BASIS_RE = re.compile(
    r"per\s+(?:a\s+)?(?:six[\s-]?pack|\d+[\s-]?pack|tray)"
    r"|wrapped\s+in\s+sixes|in\s+sixes\b|six[\s-]?packs?\s+in\s+a\s+case"
    r"|\d+\s+six[\s-]?packs?\s+(?:in|per)\s+a?\s*case",
    re.IGNORECASE)


def fix_wrapped_pack_unit_price(offers: List[Dict[str, Any]], source_text: str) -> int:
    """A wrapped multipack priced PER PACK (Samuel 2026-11-26: Northwick W02 /
    Estuary D02 — "EUR 2.70 per six-pack, FOUR six-packs in a case") stores the
    case price correctly (4 × 2.70 = 10.80) but leaves price_per_unit as the
    per-PACK figure (2.70), so the self-check raises a FALSE conflict (2.70 × 24
    cans ≠ 10.80). When the line states a pack basis and the row has a case price
    over a larger can pack, set the per-unit price to case ÷ pack (per can) — the
    real per-can price — which clears the false conflict."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    changed = 0
    for o in offers:
        ppc = _f(o.get("price_per_case")); upc = _f(o.get("units_per_case"))
        ppu = _f(o.get("price_per_unit"))
        if not (ppc and upc and upc > 1):
            continue
        per_can = round(ppc / upc, 4)
        if ppu is not None and abs(ppu - per_can) <= 0.01:
            continue  # already per-can
        ln = (_unique_source_line(o, lines) or _row_line_by_amount(o, lines)
              or _source_line_for(o, lines))
        if not ln or not _PACK_BASIS_RE.search(ln):
            continue
        o["price_per_unit"] = per_can
        o["price_per_unit_eur"] = None
        cc = _clean(o.get("commercial_conditions"))
        if "per pack" not in cc.lower() and "per six-pack" not in cc.lower():
            o["commercial_conditions"] = (cc + " | " if cc else "") + (
                f"price quoted per pack; per-can price = case ÷ pack ({per_can:g})")
        o["error_flags"] = [f for f in (o.get("error_flags") or [])
            if not (isinstance(f, str) and "does not match unit" in f.lower())]
        o["needs_manual_review"] = False   # no real conflict; gate re-decides
        _mark(o)
        changed += 1
    return changed


_LEAD_TICKET_RE = re.compile(r"^([A-Za-z]{1,4}\d{1,4}(?:-[A-Za-z0-9]{1,3})?)\b")


def fix_mislabeled_product_reference(offers: List[Dict[str, Any]],
                                     source_text: str) -> int:
    """The extractor sometimes stores the CIRCULAR's own number (NF1126, EL-925,
    AQ1126) as a row's product_reference instead of the line's own ticket (W02),
    so the reconciler cannot join the row and reports it "not imported" although it
    is present (Samuel 2026-11-26: Northwick W02). Re-derive the ticket from the
    row's source line when the stored reference is really the document id."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    header = " ".join(lines[:4])
    circ = {m.upper() for m in re.findall(r"\b([A-Z]{1,4}-?\d{3,6})\b", header)}
    if not circ:
        return 0
    changed = 0
    for o in offers:
        ref = str(o.get("product_reference") or "").strip().upper()
        if not ref or ref.replace("-", "") not in {c.replace("-", "") for c in circ}:
            continue
        ln = _unique_source_line(o, lines) or _row_line_by_amount(o, lines)
        if not ln:
            continue
        m = _LEAD_TICKET_RE.match(ln)
        if m and m.group(1).upper().replace("-", "") != ref.replace("-", ""):
            o["product_reference"] = m.group(1)
            _mark(o)
            changed += 1
    return changed


def assign_reference_from_source(offers: List[Dict[str, Any]],
                                 source_text: str) -> int:
    """Give a row its own ticket when the extractor dropped it (Samuel 2026-11-27:
    Northwick W01/W07/W08/W12/W15/W20/W22 came back with no reference, so the
    reconciliation could not key them). Match the row to its unique source line and
    read the leading 'Wxx —' ticket. Only fills an EMPTY reference — never
    overwrites one the extractor already captured. The leading token must look like
    a line reference (letter+digits) and NOT be the circular id."""
    if not source_text:
        return 0
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    header = " ".join(lines[:4])
    circ = {m.upper().replace("-", "")
            for m in re.findall(r"\b([A-Z]{1,4}-?\d{3,6})\b", header)}
    _LEAD = r"^([A-Za-z]{1,3}\d{1,4})\b[\s)\].:—-]"
    # The row tickets this circular actually uses (W01…W22 / A01…A16) — the set a
    # stored reference must belong to. Anything else (a parcel id NF83, the
    # circular id NF1126) is not a row ticket and should be replaced by the row's
    # own leading ticket (Samuel 2026-11-27: Rowan W08 stored the parcel "NF83").
    lead_set = set()
    for ln in lines:
        mm = re.match(_LEAD, ln)
        if mm:
            lead_set.add(mm.group(1).upper())
    changed = 0
    for o in offers:
        cur = str(o.get("product_reference") or "").strip().upper()
        if cur and cur in lead_set:
            continue  # already a valid row ticket — leave it
        ln = _unique_source_line(o, lines) or _row_line_by_amount(o, lines)
        if not ln:
            continue
        m = re.match(_LEAD, ln)
        if not m:
            continue
        tok = m.group(1)
        if tok.upper().replace("-", "") in circ:
            continue
        if tok.upper() == cur:
            continue
        o["product_reference"] = tok
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
        # A row whose OWN source confirms the basis — a per-set/per-candle/each
        # price (honor_retail / inner-box), or a structured column header — is a
        # CONFIRMED quote and must never be overridden by a twin. This is what
        # Samuel 2026-11-26 required: "a later import must not silently change
        # another import's confirmed or unconfirmed price basis." It is also what
        # made a duplicate resend / a stale earlier copy wrongly blank Dune & Linen.
        if o.get("_unit_priced") or o.get("_structured_row"):
            continue
        # Cross-source sync is for the SAME NAMED supplier only. Two rows that both
        # lack a supplier name must never be treated as twins — otherwise North
        # Fen could reach into Estuary (Samuel 2026-11-26: "North Fen's explicit USD
        # must not resolve Estuary's ambiguous dollar"). An empty supplier key is
        # unmatchable here and unregistered on the lookup side.
        if not _norm(o.get("supplier_name")):
            continue
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
_PER_LOAD_RE = re.compile(
    r"(\d[\d.,]*)\s*cases?\s*per\s*(?:full\s+)?(?:load|truck)"
    # "A full truck holds 2700 cases. This is capacity only" (Samuel 2026-11-27:
    # Northwick W09 Osprey stayed GREEN with no stock, while its Ashmere twin A11
    # — "2800 cases per full truck" — was correctly flagged REVIEW).
    r"|(?:a\s+)?(?:full\s+)?truck\s+holds?\s+(\d[\d.,]*)\s*cases?",
    re.IGNORECASE)
_AVAIL_QTY_RE = re.compile(r"(\d[\d.,]*)\s*cases?\s*(?:available|in\s*stock|offered)", re.IGNORECASE)
_PACK_SIG_RE = re.compile(r"(\d+)\s*[x×*]\s*(\d+)(?:\s*[x×*]\s*(\d+))?\s*cl", re.IGNORECASE)


def _pack_sig(text) -> Optional[str]:
    """A pack-format signature ('4x6x33', '24x33') from any text, so a capacity
    line can be matched to the RIGHT pack-format row (two formats of one product
    share the brand/product tokens but differ by pack)."""
    m = _PACK_SIG_RE.search(str(text or ""))
    if not m:
        return None
    return "x".join(g for g in m.groups() if g)


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
        load_n = _int(ml.group(1) or ml.group(2))
        if not load_n:
            continue
        btoks = set(re.findall(r"[a-z0-9]+", b.lower()))
        bsig = _pack_sig(b)
        for o in offers:
            # The capacity line must NAME this product's BRAND — a shared pack
            # format alone (e.g. 24×33cl) would otherwise tag a DIFFERENT brand's
            # real available stock (Samuel 2026-11-23: the Carlsberg per-load line
            # wrongly flagged Tuborg Green 24×33, which had 792 cases available).
            brand_toks = set(t for t in re.findall(r"[a-z0-9]+", _norm(o.get("brand")).lower())
                             if len(t) > 2)
            if not brand_toks or not (brand_toks & btoks):
                continue
            osig = _pack_sig(o.get("packaging_raw")) or _pack_sig(o.get("product_name"))
            qty_is_cap = _f(o.get("quantity_case")) == float(load_n)
            # A capacity-only block whose brand matches a row that has NO confirmed
            # quantity at all: that row IS the capacity-only offer (its stock count
            # was never supplied), so flag it even when the pack signatures differ
            # (Samuel 2026-11-27: Northwick W09 Osprey).
            _no_qty = not any(_f(o.get(_qk)) for _qk in
                              ("quantity_case", "quantity_bottles",
                               "quantity_pallets", "quantity_ftl"))
            # Within the matched brand, tie the block to its row by the per-load
            # figure OR the exact pack format, so BOTH twins of a "per load"
            # product get the capacity note + review — even the twin the extractor
            # left quantity-less (Samuel 2026-11-23: the 24×33 Carlsberg twin).
            if not (qty_is_cap or (bsig and osig and bsig == osig) or _no_qty):
                continue
            if qty_is_cap:
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
    # Use the LETTER "x" (not "×") as the dimension separator so the folded name
    # canonicalises the SAME as the plain text/fold path, which writes "24 x 33cl"
    # — canonical_product keeps "x" but strips "×", so a floor-list supplier's
    # multi-format SKU ("tuborg green 24 × 33cl") must use "x" to match the same
    # SKU from a text-path supplier ("tuborg green (24 x 33cl)") — Samuel 2026-11-24.
    if d:                                  # A x C x D  → A*C units, D cl
        units = a * c; vol_cl = d; inner = c
        pack_core = f"{a} x {c} x {d}cl"
    else:                                  # A x C      → A units, C cl
        units = a; vol_cl = c; inner = None
        pack_core = f"{a} x {c}cl"
    container = pm.group(5).lower().rstrip("s")       # can / bottle
    pack_disp = f"{pack_core} {container}"
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
        "container": container, "pack_disp": pack_disp, "pack_core": pack_core,
        "pc": pc, "inco": inco, "loc": loc,
        "per_load": _int(pl.group(1)) if pl else None,
        "avail": _int(av.group(1)) if av else None,
        "bbd": _floor_bbd(block),
    }


def _floor_toks(name: str):
    return set(t for t in re.findall(r"[a-z0-9]+", (name or "").lower()) if len(t) > 2)


_PACK_IN_NAME_RE = re.compile(
    r"\s*\d+\s*[x×*]\s*\d+(?:\s*[x×*]\s*\d+)?\s*cl(?:\s*(?:cans?|bottles?))?\s*$",
    re.IGNORECASE)


def _floor_setname(o: Dict[str, Any], sp: Dict[str, Any], multi: bool = False) -> None:
    """Fold the pack format into the product name ONLY when the same product has
    more than one format in this list (Carlsberg Lager 24×33 vs 4×6×33) so the two
    stay DISTINCT SKUs. A single-format product keeps its clean base name — baking
    the format into every name (Samuel 2026-11-24) diverged this supplier's keys
    from the same product quoted by another supplier on the plain text path
    (Erkan "carlsberg 24 50cl" never matched Meridian "carlsberg"); the pack format
    is still kept in packaging_raw either way."""
    base = _PACK_IN_NAME_RE.sub("", _clean(o.get("product_name"))).strip() or _clean(o.get("product_name"))
    o["product_name"] = (f"{base} {sp['pack_core']}").strip() if multi else base


def _floor_enrich(o: Dict[str, Any], sp: Dict[str, Any], multi: bool = False) -> None:
    _floor_setname(o, sp, multi)
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
    # A product is MULTI-format only when it appears with >1 distinct pack in this
    # list; only then is the pack baked into the name (see _floor_setname).
    _fmt_by_name: Dict[frozenset, set] = defaultdict(set)
    for sp in parsed:
        _fmt_by_name[frozenset(_floor_toks(sp["name"]))].add(sp["pack_core"])
    def _is_multi(sp):
        return len(_fmt_by_name[frozenset(_floor_toks(sp["name"]))]) > 1
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
            _floor_enrich(offers[match], sp, _is_multi(sp))
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
            _floor_enrich(new, sp, _is_multi(sp))
            offers.append(new)
            changed += 1
    return changed


# ─────────────────────────────────────────────────────────────────────────────
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
# "N inner boxes/cartons of M [candles]" — box OR carton, digit or spelled number
# (Samuel 2026-11-24: Fog Bell "7 inner cartons of four").
_INNERBOX_OF_RE = re.compile(
    r"(\w+)\s+inner\s*(?:box(?:es)?|cartons?)\s+of\s+(\w+)", re.IGNORECASE)
# "+ / plus / and one loose <piece>" (Fog Bell "and one loose piece").
_LOOSE_PLUS_RE = re.compile(
    r"(?:plus|and|\+)\s+(\w+)\s+(?:loose|individual|single)", re.IGNORECASE)
# "minimum one [sealed] inner" — tolerate an adjective before "inner".
_MIN_INNER_RE = re.compile(r"minimum\s+(\w+)\s+(?:\w+\s+)?inner", re.IGNORECASE)


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
        # Pieces-per-inner is NOT a case count — a per-piece item shows EMPTY case
        # fields, with the inner-box size kept in packaging_raw (same as
        # fix_inner_box_from_text; Nishane/Lumière 2026-11-23).
        o["units_per_case"] = None
        o["quantity_unit"] = "pieces"
        o["quantity_bottles"] = float(total)
        o["quantity_case"] = None
        _praw_cur = _clean(o.get("packaging_raw")).lower()
        if (not _praw_cur) or _praw_cur in (
                "bottle", "bottles", "piece", "pieces", "pc", "pcs", "candle", "candles"):
            o["packaging_raw"] = f"{int(per)} pieces per inner box"
        # Per-piece pricing: "per piece/candle" OR a bare "EUR 14.20 each" (Fog Bell).
        if re.search(r"per\s+(?:candle|piece|pc)\b|\beach\b", ln, re.IGNORECASE):
            o["price_per_case"] = None
            o["price_per_case_eur"] = None
            o["error_flags"] = [f for f in (o.get("error_flags") or [])
                                if "price_per_case calculated" not in str(f).lower()]
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
