from datetime import datetime, timedelta
from fastapi import APIRouter, Query, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, and_
from sqlalchemy.sql.expression import nulls_last
from typing import Optional
from core.database import get_db
from core.normalization import peer_group_id, is_peer_group_qualified
from models.offer_item import OfferItemDB

router = APIRouter()


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

    q = db.query(OfferItemDB).filter(OfferItemDB.price_per_unit_eur.isnot(None))
    if cutoff:
        q = q.filter(func.coalesce(OfferItemDB.offer_date, OfferItemDB.created_at) >= cutoff)

    rows = q.all()
    peers = {}
    for r in rows:
        # Peer identity comes from the ONE canonicalisation defined in
        # core.normalization. The full commercial key includes location,
        # ABV, vintage, age_statement, edition — so an EXW Rotterdam 40%
        # offer never gets peered with an EXW Dubai 43% offer.
        key = peer_group_id(
            r.brand, r.product_name,
            unit_volume_ml=r.unit_volume_ml,
            units_per_case=r.units_per_case,
            incoterm=r.incoterm,
            location=r.location,
            alcohol_percent=r.alcohol_percent,
            vintage=r.vintage,
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
        search_term = f"%{search}%"
        subquery = subquery.filter(
            or_(
                OfferItemDB.product_name.ilike(search_term),
                OfferItemDB.brand.ilike(search_term)
            )
        )

    if category:
        subquery = subquery.filter(OfferItemDB.category.ilike(category))

    if sub_category:
        subquery = subquery.filter(OfferItemDB.sub_category.ilike(sub_category))

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
    group_cols = [
        func.lower(func.coalesce(OfferItemDB.product_name, '')),
        func.coalesce(OfferItemDB.unit_volume_ml, 0),
        func.lower(func.coalesce(OfferItemDB.category, '')),
        func.lower(func.coalesce(OfferItemDB.sub_category, '')),
        func.lower(func.coalesce(OfferItemDB.brand, '')),
        func.upper(func.coalesce(OfferItemDB.incoterm, '')),
    ]

    multi_supplier_subq = (
        db.query(
            func.lower(func.coalesce(OfferItemDB.product_name, '')).label("pn"),
            func.coalesce(OfferItemDB.unit_volume_ml, 0).label("vol"),
            func.lower(func.coalesce(OfferItemDB.category, '')).label("cat"),
            func.lower(func.coalesce(OfferItemDB.sub_category, '')).label("subcat"),
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
            (func.lower(func.coalesce(OfferItemDB.sub_category, '')) == multi_supplier_subq.c.subcat) &
            (func.lower(func.coalesce(OfferItemDB.brand, '')) == multi_supplier_subq.c.brand) &
            (func.upper(func.coalesce(OfferItemDB.incoterm, '')) == multi_supplier_subq.c.inco)
        )
    )

    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                OfferItemDB.product_name.ilike(search_term),
                OfferItemDB.brand.ilike(search_term)
            )
        )

    if category:
        query = query.filter(OfferItemDB.category.ilike(category))

    if sub_category:
        query = query.filter(OfferItemDB.sub_category.ilike(sub_category))

    query = query.order_by(
        func.lower(func.coalesce(OfferItemDB.product_name, '')),
        func.coalesce(OfferItemDB.unit_volume_ml, 0),
        nulls_last(OfferItemDB.price_per_unit_eur.asc())
    )

    total = query.count()
    rows = query.offset(skip).limit(limit).all()

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
        )
        sku_groups[sku].append(row)

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
        lowest_qualified_nominal_price = None
        lowest_qualified_nominal_peer = None
        for pg_key, pg_rows in peer_buckets.items():
            pg_sorted = sorted(pg_rows, key=lambda r: r.price_per_unit_eur or float('inf'))
            head = pg_sorted[0]
            qualified = all(
                is_peer_group_qualified(r.incoterm, r.location) for r in pg_rows
            )
            entry = {
                "peer_group_id": pg_key,
                "incoterm": head.incoterm,
                "location": head.location,
                "is_qualified": qualified,
                "supplier_count": len({(r.supplier_name or r.sender_email or '') for r in pg_rows}),
                "best_price_eur": head.price_per_unit_eur,
                "best_uid": head.uid,
                "best_supplier": head.supplier_name,
                "offers": [r.to_dict() for r in pg_sorted],
            }
            peers.append(entry)
            if qualified and head.price_per_unit_eur is not None:
                if lowest_qualified_nominal_price is None or head.price_per_unit_eur < lowest_qualified_nominal_price:
                    lowest_qualified_nominal_price = head.price_per_unit_eur
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
            "supplier_count": supplier_count,
            "peer_count": len(peers),
            # Nominal, cross-peer headline. Not a trusted trading signal —
            # it does not correct for freight or terms, and it stays null
            # when peer groups have unknown terms OR the SKU has an EAN
            # conflict. Real trusted bests live inside each peer block.
            "lowest_qualified_nominal_price_eur": lowest_qualified_nominal_price,
            "lowest_qualified_nominal_peer_id": lowest_qualified_nominal_peer,
            "peers": peers,
        })

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "groups": grouped_list,
    }