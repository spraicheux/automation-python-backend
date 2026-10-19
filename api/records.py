import re
import unicodedata
from datetime import datetime, timedelta
from fastapi import APIRouter, Query, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, and_
from sqlalchemy.sql.expression import nulls_last
from typing import Optional, List
from core.database import get_db
from core.normalization import peer_group_id, peer_group_id_from_ean, is_peer_group_qualified
from models.offer_item import OfferItemDB

router = APIRouter()

# ── Search term matching (R-01) ─────────────────────────────────────────────
# Both apostrophe AND accent variants are folded, on the query and the stored
# value, so "jack daniels" matches "Jack Daniel's" and "bacardi" matches
# "Bacardí". supplier_name is intentionally NOT searched — otherwise a supplier
# like "Bacardi Martini" pulls every unrelated brand it carries into a
# brand/product query. Supplier filtering has its own dedicated facet.
_APOS_CHARS = ["'", "’", "‘", "ʼ", "`", "´"]


def _ascii_fold(s: str) -> str:
    s = str(s or "")
    for a in _APOS_CHARS:
        s = s.replace(a, "")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# Accented chars → ASCII, for folding the stored column in SQL (translate is
# case-sensitive, so include both cases). Built once; both strings same length.
_ACC_FROM = "áàâäãåçéèêëíìîïñóòôöõúùûüýÁÀÂÄÃÅÇÉÈÊËÍÌÎÏÑÓÒÔÖÕÚÙÛÜÝ"
_ACC_TO = _ascii_fold(_ACC_FROM)


def _fold_col(col):
    expr = col
    for a in _APOS_CHARS:
        expr = func.replace(expr, a, "")
    return func.translate(expr, _ACC_FROM, _ACC_TO)


def _term_match(term: str):
    like = f"%{_ascii_fold(term)}%"
    return or_(
        _fold_col(OfferItemDB.brand).ilike(like),
        _fold_col(OfferItemDB.product_name).ilike(like),
        OfferItemDB.ean_code.ilike(f"%{term}%"),
        OfferItemDB.supplier_reference.ilike(f"%{term}%"),
    )


# ── Volume-alias normalisation ──────────────────────────────────────────────
_VOL_ALIAS = {
    # Litres → ml
    "1l": 1000, "1.5l": 1500, "2l": 2000, "0.7l": 700, "0.75l": 750,
    # Centilitres → ml  (most common in EU trade)
    "1cl": 10, "3cl": 30, "4cl": 40, "5cl": 50, "7cl": 70, "10cl": 100,
    "20cl": 200, "25cl": 250, "35cl": 350, "37.5cl": 375, "50cl": 500,
    "70cl": 700, "75cl": 750, "100cl": 1000, "150cl": 1500, "200cl": 2000,
}
# e.g. "100cl" → 1000  /  "1L" → 1000  /  "1000ml" → 1000
def _token_to_ml(tok: str) -> Optional[float]:
    t = tok.strip().lower().replace(",", ".")
    if m := re.fullmatch(r"(\d+(?:\.\d+)?)\s*ml", t):
        return float(m.group(1))
    if m := re.fullmatch(r"(\d+(?:\.\d+)?)\s*cl", t):
        return float(m.group(1)) * 10
    if m := re.fullmatch(r"(\d+(?:\.\d+)?)\s*l", t):
        return float(m.group(1)) * 1000
    return _VOL_ALIAS.get(t)

def _parse_search_tokens(raw: str):
    """
    Break a free-text search string into structured constraints.
    Returns (text_terms: list[str], volume_ml: float|None, customs: str|None).
    """
    if not raw:
        return [], None, None
    text_terms, volume_ml, customs = [], None, None
    for tok in raw.split():
        if tok.upper() in ("T1", "T2"):
            customs = tok.upper()
            continue
        ml = _token_to_ml(tok)
        if ml is not None:
            volume_ml = ml
            continue
        text_terms.append(tok)
    return text_terms, volume_ml, customs


_WINDOW_DAYS = {"1M": 30, "3M": 90, "6M": 182, "12M": 365, "24M": 730, "ALL": None}


@router.get("/benchmarks")
async def get_benchmarks(
    window: str = Query("12M", description="Time window: 1M | 3M | 6M | 12M | 24M | ALL"),
    db: Session = Depends(get_db),
):
    """
    Historical benchmark per genuinely comparable peer group.
    Peer key = LOWER(brand) + LOWER(product_name) + unit_volume_ml + units_per_case.

    Returns:
    {
      "window": "12M",
      "window_days": 365,
      "peers": {
        "grey goose|original|700|6": {
          "low_eur":  15.15,
          "low_uid":  "abc-…",
          "avg_eur":  16.42,
          "samples":  8,
          "since":    "2025-09-23"
        },
        ...
      }
    }
    """
    window = window.upper() if window else "12M"
    days = _WINDOW_DAYS.get(window, 365)
    cutoff = datetime.utcnow() - timedelta(days=days) if days else None

    today = datetime.utcnow().strftime('%Y-%m-%d')
    q = (db.query(OfferItemDB)
         .filter(OfferItemDB.price_per_unit_eur.isnot(None))
         # Exclude offers whose valid_until has already passed.
         .filter(or_(OfferItemDB.valid_until.is_(None),
                     OfferItemDB.valid_until == '',
                     OfferItemDB.valid_until >= today)))
    if cutoff:
        q = q.filter(func.coalesce(OfferItemDB.offer_date, OfferItemDB.created_at) >= cutoff)

    rows = q.all()
    peers = {}
    for r in rows:
        # Peer identity comes from the ONE canonicalisation defined in
        # core.normalization. The full commercial key includes location,
        # ABV, vintage, age_statement, edition — so an EXW Rotterdam 40%
        # offer never gets peered with an EXW Dubai 43% offer.
        # Key MUST match the Best Prices roll-up (sku_identity/peer_group_id)
        # so the "12M LOW" badge in All Offers and the Best Prices benchmark
        # describe the SAME peer — including perfume/cosmetics discriminators.
        # Must reproduce EXACTLY what OfferItemDB.to_dict() emits as
        # `peer_group_id` (the dashboard looks up benchmarks[item.peer_group_id]),
        # which includes ean_code. Omitting it produced a different key and left
        # the All Offers "12M LOW" blank while Best Prices showed the history.
        # EAN-primary key (matches models.offer_item.to_dict and Best Prices):
        # same EAN + incoterm + location is ONE peer even if shade/range/name
        # differ, so the All Offers "12M LOW" badge and Best Prices describe the
        # SAME peer. Falls back to the attribute key when no valid EAN.
        key = peer_group_id_from_ean(r.ean_code, r.incoterm, r.location) or peer_group_id(
            r.brand, r.product_name,
            unit_volume_ml=r.unit_volume_ml,
            units_per_case=r.units_per_case,
            incoterm=r.incoterm,
            location=r.location,
            alcohol_percent=r.alcohol_percent,
            vintage=r.vintage,
            ean_code=r.ean_code,
            category_slug=getattr(r, 'category_slug', None),
            perfume_format=getattr(r, 'perfume_format', None),
            retail_state=getattr(r, 'retail_state', None),
            gender=getattr(r, 'gender', None),
            shade=getattr(r, 'shade', None),
            product_type=getattr(r, 'product_type', None),
            size_weight_g=getattr(r, 'size_weight_g', None),
            range_name=getattr(r, 'range_name', None),
        )
        entry = peers.setdefault(key, {"samples": [], "all_qualified": True})
        entry["samples"].append((r.price_per_unit_eur, r.uid))
        # A peer group is "qualified" only when EVERY row it contains has
        # both a known incoterm and a known location. One unqualified row
        # poisons the whole group — otherwise a trusted NEW XM LOW could
        # fire when the current row is Rotterdam-EXW but the "prior best"
        # is a row with no known origin, and that's exactly the falsely-
        # precise trading signal the client asked us to avoid.
        if not is_peer_group_qualified(r.incoterm, r.location):
            entry["all_qualified"] = False

    out = {}
    for k, v in peers.items():
        samples = v["samples"]
        if not samples:
            continue
        # Sort ascending so [0] is the group min and [1] is the second-lowest
        # (used as prior-low when the current row IS the group min).
        samples.sort(key=lambda t: t[0])
        low_price, low_uid = samples[0]
        second_low_price, second_low_uid = (samples[1] if len(samples) > 1
                                            else (None, None))
        avg_price = sum(p for p, _ in samples) / len(samples)
        out[k] = {
            "low_eur": round(low_price, 4),
            "low_uid": low_uid,
            # The 2nd lowest lets the frontend show a genuine PRIOR low when
            # the current row is itself the group minimum — otherwise a row
            # would show up as "12M Low = itself".
            "second_low_eur": round(second_low_price, 4) if second_low_price is not None else None,
            "second_low_uid": second_low_uid,
            "avg_eur": round(avg_price, 4),
            "samples": len(samples),
            # True only when every row in this group has known incoterm AND
            # known location. Callers must downgrade "NEW XM LOW" and other
            # confident signals when this is False.
            "is_qualified": v["all_qualified"],
        }

    return {
        "window": window,
        "window_days": days,
        "since": cutoff.isoformat() if cutoff else None,
        "peers": out,
        "peer_count": len(out),
    }


@router.get("/records")
async def get_records(
    skip: int = Query(0),
    limit: int = Query(24),
    source_file_id: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    sub_category: Optional[str] = Query(None),
    category_slug: Optional[str] = Query(None, description="wines_spirits | perfumes | cosmetics"),
    db: Session = Depends(get_db),
):
    # Same-file duplicate guard — only collapses TRUE extractor duplicates
    # (LLM sometimes emits the same line twice within a batch). A supplier
    # can legitimately list the same SKU multiple times in the same file
    # with a different price, quantity, incoterm, location, EAN, lot
    # reference or expiry — those are separate commercial lines and MUST
    # remain distinct offers. The fingerprint below includes every field
    # that could vary between two legitimate lines of the same SKU; if
    # every one of them matches, the row is a real duplicate.
    subquery = db.query(
        OfferItemDB.uid,
        func.row_number().over(
            partition_by=(
                # Product identity
                func.lower(func.coalesce(OfferItemDB.brand, '')),
                func.lower(func.coalesce(OfferItemDB.product_name, '')),
                func.coalesce(OfferItemDB.unit_volume_ml, 0),
                func.coalesce(OfferItemDB.units_per_case, 0),
                # Supplier + source file
                func.coalesce(OfferItemDB.supplier_name, ''),
                OfferItemDB.source_file_id,
                # Commercial fields that can legitimately differ across
                # lots of the same SKU within one file:
                func.upper(func.coalesce(OfferItemDB.incoterm, '')),
                func.lower(func.coalesce(OfferItemDB.location, '')),
                func.coalesce(OfferItemDB.price_per_unit, 0),
                func.coalesce(OfferItemDB.price_per_case, 0),
                func.coalesce(OfferItemDB.currency, ''),
                func.coalesce(OfferItemDB.quantity_case, 0),
                # quantity_unit MUST be part of the fingerprint — "11 btls",
                # "11 cs" and "11 pallets" all have quantity_case=11 but
                # are completely different commercial lines.
                func.lower(func.coalesce(OfferItemDB.quantity_unit, '')),
                func.coalesce(OfferItemDB.ean_code, ''),
                func.coalesce(OfferItemDB.product_reference, ''),
                func.coalesce(OfferItemDB.valid_until, ''),
                func.coalesce(OfferItemDB.best_before_date, ''),
                func.coalesce(OfferItemDB.vintage, ''),
                func.coalesce(OfferItemDB.custom_status, ''),
            ),
            order_by=[
                nulls_last(OfferItemDB.price_per_unit_eur.asc()),
                nulls_last(OfferItemDB.price_per_case_eur.asc()),
                OfferItemDB.created_at.desc()
            ]
        ).label("rn")
    )

    if source_file_id:
        subquery = subquery.filter(OfferItemDB.source_file_id == source_file_id)

    if search:
        text_terms, volume_ml, customs = _parse_search_tokens(search)
        # Volume token → exact ml match
        if volume_ml is not None:
            subquery = subquery.filter(OfferItemDB.unit_volume_ml == volume_ml)
        # Customs token (T1/T2) — match structured field OR product_name fallback
        # for rows where T1/T2 hasn't been backfilled into custom_status yet.
        if customs:
            subquery = subquery.filter(or_(
                OfferItemDB.custom_status == customs,
                OfferItemDB.product_name.ilike(f"%{customs}%"),
            ))
        # Remaining text terms — each must match a product-identity field
        # (apostrophe-folded; supplier_name excluded — see _term_match).
        for term in text_terms:
            subquery = subquery.filter(_term_match(term))

    if category:
        subquery = subquery.filter(OfferItemDB.category.ilike(category))

    if sub_category:
        subquery = subquery.filter(OfferItemDB.sub_category.ilike(sub_category))

    if category_slug:
        subquery = subquery.filter(OfferItemDB.category_slug == category_slug.lower().strip())

    subquery = subquery.subquery()

    query = db.query(OfferItemDB).join(
        subquery, OfferItemDB.uid == subquery.c.uid
    ).filter(
        subquery.c.rn == 1
    ).order_by(
        OfferItemDB.created_at.desc()
    )

    total = query.count()
    rows = query.offset(skip).limit(limit).all()

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "records": [row.to_dict() for row in rows],
    }


@router.get("/best-prices")
async def get_best_prices(
    skip: int = Query(0),
    limit: int = Query(24),
    search: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    sub_category: Optional[str] = Query(None),
    category_slug: Optional[str] = Query(None, description="wines_spirits | perfumes | cosmetics"),
    db: Session = Depends(get_db),
):
    """
    Returns products that appear under multiple suppliers,
    showing all supplier offers side by side so buyers can compare.
    Grouped by: product_name + unit_volume_ml + category + sub_category + brand
    Only groups with 2+ distinct suppliers are returned.
    """
    from sqlalchemy import case, cast, String

    # Build a subquery that counts distinct suppliers per product group.
    # Incoterm is part of the group key so an EXW €20 and DAP €21 offer for
    # the same product don't get lined up as apples-to-apples comparisons.
    # NOTE: sub_category is deliberately NOT a grouping key. It is a free-form
    # label that suppliers populate inconsistently (e.g. one puts "EDT"/"EDP"
    # in it, another leaves it null), which would split the same product from
    # two suppliers into separate 1-supplier groups and drop it from Best
    # Prices entirely (Samuel: YSL Midnight Halo). The real physical-SKU split
    # happens later via sku_identity, which carries the actual discriminators.
    group_cols = [
        func.lower(func.coalesce(OfferItemDB.product_name, '')),
        func.coalesce(OfferItemDB.unit_volume_ml, 0),
        func.lower(func.coalesce(OfferItemDB.category, '')),
        func.lower(func.coalesce(OfferItemDB.brand, '')),
        func.upper(func.coalesce(OfferItemDB.incoterm, '')),
    ]

    multi_supplier_subq = (
        db.query(
            func.lower(func.coalesce(OfferItemDB.product_name, '')).label("pn"),
            func.coalesce(OfferItemDB.unit_volume_ml, 0).label("vol"),
            func.lower(func.coalesce(OfferItemDB.category, '')).label("cat"),
            func.lower(func.coalesce(OfferItemDB.brand, '')).label("brand"),
            func.upper(func.coalesce(OfferItemDB.incoterm, '')).label("inco"),
            func.count(func.distinct(
                func.coalesce(OfferItemDB.supplier_name, OfferItemDB.sender_email, '')
            )).label("supplier_count")
        )
        .group_by(*group_cols)
        .having(
            func.count(func.distinct(
                func.coalesce(OfferItemDB.supplier_name, OfferItemDB.sender_email, '')
            )) >= 2
        )
        .subquery()
    )

    query = (
        db.query(OfferItemDB)
        .join(
            multi_supplier_subq,
            (func.lower(func.coalesce(OfferItemDB.product_name, '')) == multi_supplier_subq.c.pn) &
            (func.coalesce(OfferItemDB.unit_volume_ml, 0) == multi_supplier_subq.c.vol) &
            (func.lower(func.coalesce(OfferItemDB.category, '')) == multi_supplier_subq.c.cat) &
            (func.lower(func.coalesce(OfferItemDB.brand, '')) == multi_supplier_subq.c.brand) &
            (func.upper(func.coalesce(OfferItemDB.incoterm, '')) == multi_supplier_subq.c.inco)
        )
        # Zero / null EUR prices are not actionable and must not surface as
        # "€0.00" offers in Best Prices or its benchmark history (Samuel: the
        # Baileys €0.00 rows). They are flagged "Unusable: no price" elsewhere.
        .filter(OfferItemDB.price_per_unit_eur.isnot(None))
        .filter(OfferItemDB.price_per_unit_eur > 0)
    )

    if search:
        text_terms, volume_ml, customs = _parse_search_tokens(search)
        if volume_ml is not None:
            query = query.filter(OfferItemDB.unit_volume_ml == volume_ml)
        if customs:
            query = query.filter(or_(
                OfferItemDB.custom_status == customs,
                OfferItemDB.product_name.ilike(f"%{customs}%"),
            ))
        for term in text_terms:
            query = query.filter(_term_match(term))

    if category:
        query = query.filter(OfferItemDB.category.ilike(category))

    if sub_category:
        query = query.filter(OfferItemDB.sub_category.ilike(sub_category))

    if category_slug:
        query = query.filter(OfferItemDB.category_slug == category_slug.lower().strip())

    query = query.order_by(
        func.lower(func.coalesce(OfferItemDB.product_name, '')),
        func.coalesce(OfferItemDB.unit_volume_ml, 0),
        nulls_last(OfferItemDB.price_per_unit_eur.asc())
    )

    # Fetch ALL rows in the multi-supplier universe (bounded — only products
    # with 2+ suppliers qualify), then paginate at the GROUP level after the
    # sku/EAN roll-up below. Paginating rows here cut groups at page
    # boundaries, so "Load more" could land entirely inside already-shown
    # groups and appear to do nothing.
    rows = query.all()

    # ── Best Prices: two-level roll-up ───────────────────────────────
    # Level 1 (outer): sku_identity — gather all competing offers for the
    # same physical SKU so a buyer sees every option side by side.
    # Level 2 (inner): peer_group_id — determine the trusted Best Price
    # per commercial peer. Same SKU at EXW Rotterdam and DAP Paris shows
    # BOTH prices with their own trusted-best label; we never mash them
    # into a single "best" that mixes commercial terms.
    #
    # EAN reconciliation. sku_identity is attribute-only, so a supplier
    # who left EAN blank still merges with an EAN-bearing supplier of the
    # same product. Within an SKU group we then look at the distinct EANs
    # present:
    #   - 0 or 1 distinct EAN → clean merge (client's fallback rule).
    #   - ≥ 2 distinct EANs   → ean_conflict=True; buyer sees Needs Review.
    from collections import defaultdict
    from core.normalization import (
        sku_identity, ean_key, peer_group_id, is_peer_group_qualified,
    )

    sku_groups = defaultdict(list)
    for row in rows:
        sku = sku_identity(
            row.brand, row.product_name,
            unit_volume_ml=row.unit_volume_ml,
            units_per_case=row.units_per_case,
            alcohol_percent=row.alcohol_percent,
            vintage=row.vintage,
            # Phase 3 M1: perfume/cosmetics discriminators are part of the
            # attribute-only SKU key so EDT vs EDP, retail vs tester, 91 vs
            # 116 shades don't collapse into one SKU when EAN is missing on
            # either side. Same EAN across suppliers is reconciled in the
            # union-find pass below regardless of these values.
            category_slug=getattr(row, 'category_slug', None),
            perfume_format=getattr(row, 'perfume_format', None),
            retail_state=getattr(row, 'retail_state', None),
            gender=getattr(row, 'gender', None),
            shade=getattr(row, 'shade', None),
            product_type=getattr(row, 'product_type', None),
            size_weight_g=getattr(row, 'size_weight_g', None),
            range_name=getattr(row, 'range_name', None),
        )
        sku_groups[sku].append(row)

    # ── EAN-first reconciliation across attribute-differing sku_identity
    # buckets (Phase 3 M1, client requirement §5 + §4-attribute-conflict).
    # Same EAN + one side blank on gender/retail_state → merge (§5 fallback).
    # Same EAN + BOTH sides populated with different CRITICAL attribute
    # values (unit_volume_ml, perfume_format, retail_state, product_type)
    # → do NOT merge; those are material contradictions that shouldn't be
    # collapsed into one SKU without human review. Genuinely-different-EANs
    # rows stay split and surface the ean_conflict flag downstream.
    _CRITICAL_ATTRS = ("unit_volume_ml", "perfume_format", "retail_state",
                       "product_type", "shade", "size_weight_g")
    def _attr_conflict(a_row, b_row):
        for attr in _CRITICAL_ATTRS:
            av = getattr(a_row, attr, None)
            bv = getattr(b_row, attr, None)
            # Only a mismatch when BOTH sides are populated and differ.
            # Blank vs populated is not a conflict (client §5 fallback).
            if av not in (None, "", 0, 0.0) and bv not in (None, "", 0, 0.0):
                # numeric compare with a tiny epsilon for ml/g
                if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                    if abs(av - bv) > 0.5:
                        return attr, av, bv
                elif str(av).strip().lower() != str(bv).strip().lower():
                    return attr, av, bv
        return None

    parent = {k: k for k in sku_groups}
    def _find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb
    ean_to_sku_keys = defaultdict(set)
    for sku_k, rows_in in sku_groups.items():
        for r in rows_in:
            ek = ean_key(r.ean_code)
            if ek:
                ean_to_sku_keys[ek].add(sku_k)
    ean_attribute_conflicts = {}  # sku_key → list of (ean, attr, va, vb) rejected unions
    for ek, ks in ean_to_sku_keys.items():
        ks = list(ks)
        base = ks[0]
        base_rep = sku_groups[base][0]  # exemplar row
        for i in range(1, len(ks)):
            other_rep = sku_groups[ks[i]][0]
            conflict = _attr_conflict(base_rep, other_rep)
            if conflict:
                attr, va, vb = conflict
                # Do not union — record the conflict for downstream flagging.
                ean_attribute_conflicts.setdefault(base, []).append(
                    {"ean": ek, "attribute": attr, "values": [va, vb]}
                )
                ean_attribute_conflicts.setdefault(ks[i], []).append(
                    {"ean": ek, "attribute": attr, "values": [va, vb]}
                )
            else:
                _union(base, ks[i])
    merged_groups = defaultdict(list)
    merged_attribute_conflicts = defaultdict(list)
    for sku_k, rows_in in sku_groups.items():
        root = _find(sku_k)
        merged_groups[root].extend(rows_in)
        if sku_k in ean_attribute_conflicts:
            merged_attribute_conflicts[root].extend(ean_attribute_conflicts[sku_k])
    sku_groups = merged_groups

    grouped_list = []
    for sku_key, rows_in in sku_groups.items():
        # EAN reconciliation across the SKU cluster
        distinct_eans = {ean_key(r.ean_code) for r in rows_in}
        known_eans = sorted(e for e in distinct_eans if e)
        ean_conflict = len(known_eans) > 1

        # Sub-group by commercial peer for trusted-best-per-peer.
        peer_buckets = defaultdict(list)
        for r in rows_in:
            pg = peer_group_id(
                r.brand, r.product_name,
                unit_volume_ml=r.unit_volume_ml,
                units_per_case=r.units_per_case,
                incoterm=r.incoterm, location=r.location,
                alcohol_percent=r.alcohol_percent, vintage=r.vintage,
                ean_code=r.ean_code,
            )
            peer_buckets[pg].append(r)

        peers = []
        # SKU-level headline: "lowest QUALIFIED nominal price". Precision
        # matters here — this is the min across peer groups whose commercial
        # terms are known (is_peer_group_qualified). An unqualified offer
        # at a lower raw price is intentionally EXCLUDED, so the label
        # reads "lowest qualified nominal" rather than "lowest nominal"
        # (there could be a lower unqualified offer sitting in the data).
        # It is NOT "Trusted Best Price": €20 EXW Rotterdam and €21 DAP
        # Paris still can't be traded against each other without freight /
        # landed-cost normalisation. Each peer group has its own trusted
        # best; the SKU-level number is nominal-only and clearly labelled.
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
        lowest_qualified_nominal_price = None
        lowest_qualified_nominal_peer = None
        for pg_key, pg_rows in peer_buckets.items():
            pg_sorted = sorted(pg_rows, key=lambda r: r.price_per_unit_eur or float('inf'))
            head = pg_sorted[0]
            qualified = all(
                is_peer_group_qualified(r.incoterm, r.location) for r in pg_rows
            )
            def _is_expired(r):
                vu = r.valid_until
                if not vu:
                    return False
                vu_str = str(vu)[:10]
                return vu_str < today_str
            def _offer_dict(r):
                d = r.to_dict()
                d['is_expired'] = _is_expired(r)
                # Implausible pack guard: a full-size perfume/cosmetic does not
                # come in a case of hundreds. A mis-extracted units_per_case
                # (e.g. 380) produced absurd case prices (€12,730). Treat such a
                # row as loose for display — no case price, no case pack.
                try:
                    _upc = float(d.get('units_per_case')) if d.get('units_per_case') is not None else None
                except (TypeError, ValueError):
                    _upc = None
                try:
                    _pce = float(d.get('price_per_case_eur')) if d.get('price_per_case_eur') is not None else None
                except (TypeError, ValueError):
                    _pce = None
                if (d.get('category_slug') in ('perfumes', 'cosmetics')
                        and ((_upc is not None and _upc > 60)
                             or (_pce is not None and _pce > 2000))):
                    d['price_per_case'] = None
                    d['price_per_case_eur'] = None
                    d['units_per_case'] = None
                return d
            offers_with_flag = [_offer_dict(r) for r in pg_sorted]
            # Best price: prefer non-expired offers for the headline
            active_rows = [r for r in pg_sorted if not _is_expired(r)]
            best_row = active_rows[0] if active_rows else pg_sorted[0]
            entry = {
                "peer_group_id": pg_key,
                "incoterm": head.incoterm,
                "location": head.location,
                "is_qualified": qualified,
                "has_active_offers": len(active_rows) > 0,
                "supplier_count": len({(r.supplier_name or r.sender_email or '') for r in pg_rows}),
                "best_price_eur": best_row.price_per_unit_eur,
                "best_uid": best_row.uid,
                "best_supplier": best_row.supplier_name,
                "offers": offers_with_flag,
            }
            peers.append(entry)
            # Headline uses active (non-expired) offers only
            if qualified and best_row.price_per_unit_eur is not None and active_rows:
                if lowest_qualified_nominal_price is None or best_row.price_per_unit_eur < lowest_qualified_nominal_price:
                    lowest_qualified_nominal_price = best_row.price_per_unit_eur
                    lowest_qualified_nominal_peer = pg_key

        # Sort peers: qualified first, then by best price ascending.
        peers.sort(key=lambda p: (not p["is_qualified"],
                                  p["best_price_eur"] or float('inf')))

        head_row = rows_in[0]
        supplier_count = len({(r.supplier_name or r.sender_email or '') for r in rows_in})

        # EAN conflict rule: when two suppliers disagree on the EAN for
        # the same-attribute SKU, we DO NOT surface a SKU-level headline
        # at all. Per-peer trusted bests continue to render inside the
        # card, but the cross-peer comparison is disabled until the
        # conflict is resolved (client's rule).
        if ean_conflict:
            lowest_qualified_nominal_price = None
            lowest_qualified_nominal_peer = None

        # Same-EAN attribute-conflict rule (client §4): when the same EAN
        # appears with a materially different critical attribute (volume,
        # perfume_format, retail_state, product_type, shade, size_weight_g),
        # we did NOT union the sku_identity clusters upstream, but we still
        # want the UI to render both under the same product name with a
        # visible "Needs Review — same EAN, conflicting attributes" flag
        # and no trusted Best Price until the conflict is resolved.
        ean_attr_conflicts = merged_attribute_conflicts.get(sku_key, [])
        if ean_attr_conflicts:
            lowest_qualified_nominal_price = None
            lowest_qualified_nominal_peer = None

        grouped_list.append({
            "sku_identity": sku_key,
            "brand": head_row.brand,
            "product_name": head_row.product_name,
            "category": head_row.category,
            "sub_category": head_row.sub_category,
            "unit_volume_ml": head_row.unit_volume_ml,
            "units_per_case": head_row.units_per_case,
            "known_eans": known_eans,
            "ean_conflict": ean_conflict,
            "ean_attribute_conflicts": ean_attr_conflicts,
            "supplier_count": supplier_count,
            "peer_count": len(peers),
            # Nominal, cross-peer headline. Not a trusted trading signal —
            # it does not correct for freight or terms, and it stays null
            # when peer groups have unknown terms, the SKU has an EAN
            # conflict, OR a same-EAN attribute conflict is unresolved.
            "lowest_qualified_nominal_price_eur": lowest_qualified_nominal_price,
            "lowest_qualified_nominal_peer_id": lowest_qualified_nominal_peer,
            "peers": peers,
        })

    return {
        "total": len(grouped_list),          # real comparable-GROUP count
        "skip": skip,
        "limit": limit,
        "groups": grouped_list[skip:skip + limit],   # paginate groups, not rows
    }