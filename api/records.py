from datetime import datetime, timedelta
from fastapi import APIRouter, Query, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, and_
from sqlalchemy.sql.expression import nulls_last
from typing import Optional
from core.database import get_db
from core.normalization import canonical_key
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
        # Use the canonical key so "Baileys" / "Bailey's" / "Baileys Original"
        # collapse into one peer group. Incoterm is part of the key so EXW €20
        # and DAP €21 aren't treated as apples-to-apples until landed-cost
        # normalisation exists.
        # NOTE: location is NOT in the key yet — EXW Rotterdam vs EXW Dubai are
        # treated as peers today. This is a known conservative gap; see the
        # deterministic_rules note on document-level location inheritance.
        key = canonical_key(r.brand, r.product_name, r.unit_volume_ml,
                            r.units_per_case, r.incoterm)
        p = peers.get(key)
        if p is None:
            peers[key] = {
                "low_eur": r.price_per_unit_eur,
                "low_uid": r.uid,
                "sum_eur": r.price_per_unit_eur,
                "samples": 1,
            }
        else:
            if r.price_per_unit_eur < p["low_eur"]:
                p["low_eur"] = r.price_per_unit_eur
                p["low_uid"] = r.uid
            p["sum_eur"] += r.price_per_unit_eur
            p["samples"] += 1

    out = {
        k: {
            "low_eur": round(v["low_eur"], 4),
            "low_uid": v["low_uid"],
            "avg_eur": round(v["sum_eur"] / v["samples"], 4),
            "samples": v["samples"],
        }
        for k, v in peers.items() if v["samples"] >= 1
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

    # Group results by product key for the response
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        key = (
            (row.product_name or '').lower(),
            row.unit_volume_ml or 0,
            (row.category or '').lower(),
            (row.sub_category or '').lower(),
            (row.brand or '').lower(),
            (row.incoterm or '').upper(),
        )
        groups[key].append(row.to_dict())

    grouped_list = []
    for key, items in groups.items():
        items_sorted = sorted(items, key=lambda x: x.get('price_per_unit_eur') or float('inf'))
        grouped_list.append({
            "product_name": items[0].get("product_name"),
            "brand": items[0].get("brand"),
            "category": items[0].get("category"),
            "sub_category": items[0].get("sub_category"),
            "unit_volume_ml": items[0].get("unit_volume_ml"),
            "incoterm": items[0].get("incoterm"),  # comparison anchor
            "supplier_count": len(items),
            "best_price_eur": items_sorted[0].get("price_per_unit_eur"),
            "offers": items_sorted,
        })

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "groups": grouped_list,
    }