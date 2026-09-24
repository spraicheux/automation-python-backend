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
    # Deduplicate by GENUINELY comparable identity so 70cl and 20cl of the same
    # product name never collapse into one row (see client feedback: bottle size
    # and packaging define distinct products). Partition on brand + name + volume
    # + units_per_case + supplier so different suppliers of the same reference
    # also stay visible.
    subquery = db.query(
        OfferItemDB.uid,
        func.row_number().over(
            partition_by=(
                func.lower(func.coalesce(OfferItemDB.brand, '')),
                func.lower(func.coalesce(OfferItemDB.product_name, '')),
                func.coalesce(OfferItemDB.unit_volume_ml, 0),
                func.coalesce(OfferItemDB.units_per_case, 0),
                func.coalesce(OfferItemDB.supplier_name, ''),
                OfferItemDB.source_file_id,
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

    # Group results by SKU identity (level B) + incoterm+location — so two
    # rows are shown side-by-side ONLY when they're the same physical SKU
    # AND the commercial terms match. Raw string comparison of brand /
    # product misses spelling variants (Baileys vs Bailey's), so the
    # Python-side key uses the canonical sku_identity from
    # core.normalization.
    from collections import defaultdict
    from core.normalization import sku_identity, is_peer_group_qualified

    groups = defaultdict(list)
    for row in rows:
        sku = sku_identity(
            row.brand, row.product_name,
            unit_volume_ml=row.unit_volume_ml,
            units_per_case=row.units_per_case,
            alcohol_percent=row.alcohol_percent,
            vintage=row.vintage,
            ean_code=row.ean_code,
        )
        # A Best Price comparison is anchored on the commercial terms, so
        # the group key IS the peer_group_id (SKU + incoterm + location).
        # Two same-SKU offers at different incoterms show up as separate
        # rows, which is the honest read.
        key = (sku, (row.incoterm or '').upper(), (row.location or '').lower())
        groups[key].append(row)

    grouped_list = []
    for key, rows_in in groups.items():
        rows_sorted = sorted(rows_in, key=lambda r: r.price_per_unit_eur or float('inf'))
        items_sorted = [r.to_dict() for r in rows_sorted]
        # A group is qualified only when every row it contains has known
        # incoterm AND known location. The Best Price signal degrades to
        # informational otherwise.
        is_qualified = all(
            is_peer_group_qualified(r.incoterm, r.location) for r in rows_in
        )
        head = rows_sorted[0]
        grouped_list.append({
            "sku_identity": key[0],
            "product_name": head.product_name,
            "brand": head.brand,
            "category": head.category,
            "sub_category": head.sub_category,
            "unit_volume_ml": head.unit_volume_ml,
            "incoterm": head.incoterm,     # comparison anchor
            "location": head.location,     # part of the anchor now
            "supplier_count": len(rows_in),
            "best_price_eur": head.price_per_unit_eur,
            "is_qualified": is_qualified,
            "offers": items_sorted,
        })

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "groups": grouped_list,
    }