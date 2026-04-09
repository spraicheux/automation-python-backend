from fastapi import APIRouter, Query, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from sqlalchemy.sql.expression import nulls_last
from typing import Optional
from core.database import get_db
from models.offer_item import OfferItemDB

router = APIRouter()


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
    subquery = db.query(
        OfferItemDB.uid,
        func.row_number().over(
            partition_by=func.coalesce(func.lower(OfferItemDB.product_name), OfferItemDB.uid),
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

    # Build a subquery that counts distinct suppliers per product group
    group_cols = [
        func.lower(func.coalesce(OfferItemDB.product_name, '')),
        func.coalesce(OfferItemDB.unit_volume_ml, 0),
        func.lower(func.coalesce(OfferItemDB.category, '')),
        func.lower(func.coalesce(OfferItemDB.sub_category, '')),
        func.lower(func.coalesce(OfferItemDB.brand, '')),
    ]

    multi_supplier_subq = (
        db.query(
            func.lower(func.coalesce(OfferItemDB.product_name, '')).label("pn"),
            func.coalesce(OfferItemDB.unit_volume_ml, 0).label("vol"),
            func.lower(func.coalesce(OfferItemDB.category, '')).label("cat"),
            func.lower(func.coalesce(OfferItemDB.sub_category, '')).label("subcat"),
            func.lower(func.coalesce(OfferItemDB.brand, '')).label("brand"),
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
            (func.lower(func.coalesce(OfferItemDB.brand, '')) == multi_supplier_subq.c.brand)
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