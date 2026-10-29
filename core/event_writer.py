"""
Persist trading events to the price_events table.

Called after each ingest batch commits so the lineage of best-price events
survives further ingest (R-12). The function reuses the same eligibility
rules the live `/api/alerts` endpoint applies, but writes rows idempotently
(dedup key: peer_group_id + current_uid + previous_uid).
"""
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.normalization import (
    peer_group_id, peer_group_id_from_ean,
    is_peer_group_qualified, ean_key,
)
from models.offer_item import OfferItemDB
from models.price_event import PriceEventDB

logger = logging.getLogger(__name__)


def _parse_valid_until(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    s = str(v).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s[:19].replace("Z", ""))
    except ValueError:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None


def record_price_events(db: Session, lookback_days: int = 365) -> list:
    """Scan the full offer_items table and insert any new trading events.

    Dedup is enforced by the unique index on (peer_group_id, current_uid,
    previous_uid), so re-running after another batch is safe and additive:
    events already written stay as-is, new ones get fresh ids.

    Returns the list of newly-written PriceEventDB rows (ids + metadata).
    """
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    all_rows = (
        db.query(OfferItemDB)
          .filter(OfferItemDB.price_per_unit_eur.isnot(None))
          .all()
    )

    # Cross-category EAN guard (mirrors api/alerts).
    _ean_cats = defaultdict(set)
    for r in all_rows:
        ek = ean_key(r.ean_code)
        cs = getattr(r, "category_slug", None)
        if ek and cs:
            _ean_cats[ek].add(cs)
    _cross_cat_eans = {e for e, cs in _ean_cats.items() if len(cs) > 1}

    by_key = defaultdict(list)
    for r in all_rows:
        k = (peer_group_id_from_ean(
                r.ean_code, r.incoterm, r.location, r.custom_status,
                r.perfume_format, r.retail_state,
             )
             or peer_group_id(
                 r.brand, r.product_name,
                 unit_volume_ml=r.unit_volume_ml,
                 units_per_case=r.units_per_case,
                 incoterm=r.incoterm,
                 location=r.location,
                 alcohol_percent=r.alcohol_percent,
                 vintage=r.vintage,
                 category_slug=getattr(r, "category_slug", None),
                 perfume_format=getattr(r, "perfume_format", None),
                 retail_state=getattr(r, "retail_state", None),
                 gender=getattr(r, "gender", None),
                 shade=getattr(r, "shade", None),
                 product_type=getattr(r, "product_type", None),
                 size_weight_g=getattr(r, "size_weight_g", None),
                 range_name=getattr(r, "range_name", None),
                 custom_status=getattr(r, "custom_status", None),
             ))
        by_key[k].append(r)

    written = []

    for k, rows in by_key.items():
        if len(rows) < 2:
            continue
        if any(ean_key(r.ean_code) in _cross_cat_eans for r in rows):
            continue
        is_qualified = all(is_peer_group_qualified(r.incoterm, r.location) for r in rows)
        if not is_qualified:
            continue

        # Walk rows in INGEST order (created_at) and emit an event each
        # time a new row's price beats (or competitively matches) the
        # best eligible row we had seen before this batch arrived.
        # Samuel's "After02 A01 25 → B01 24 improvement" is defined by
        # ingest batches, not by the source's offer_date — Beta's
        # 2026-10-28 offer is older than Alpha's 2026-10-29 offer, but
        # it ARRIVED after Alpha, which is what "after02" means.
        rows_sorted = sorted(rows, key=lambda x: (x.created_at or datetime.min, x.uid))

        today = datetime.utcnow()

        def _eligible(r):
            """Can this row take part in a trading comparison today?"""
            if r.offer_date is None:
                return False  # U01 — no commercial offer date, never eligible
            if r.price_per_unit_eur is None or r.price_per_unit_eur <= 0:
                return False
            if getattr(r, "needs_manual_review", False):
                return False
            if getattr(r, "superseded_by_uid", None):
                return False
            vu = _parse_valid_until(getattr(r, "valid_until", None))
            # H01/H02 valid_until 2026-09-30 vs reference clock 2026-10-29
            # → expired, never eligible for trading events.
            if vu is not None and vu < today:
                return False
            return True

        best_prior = None   # the cheapest eligible row seen so far
        for cur in rows_sorted:
            if not _eligible(cur):
                continue

            if best_prior is None:
                best_prior = cur
                continue

            prev = best_prior
            cur_price = cur.price_per_unit_eur
            prev_price = prev.price_per_unit_eur
            same_supplier = ((cur.supplier_name or "").strip().lower()
                             == (prev.supplier_name or "").strip().lower())

            etype = None
            if cur_price < prev_price - 0.001:
                etype = "new_best_price"
            else:
                if not same_supplier and cur_price <= prev_price * 1.05:
                    etype = "competitive_supplier"

            if etype:
                delta_abs = cur_price - prev_price
                delta_pct = (delta_abs / prev_price) * 100 if prev_price else None

                snap = {
                    "members": [
                        {
                            "uid": r.uid,
                            "supplier": r.supplier_name,
                            "reference": r.supplier_reference,
                            "price_eur": r.price_per_unit_eur,
                            "offer_date": r.offer_date.isoformat() if r.offer_date else None,
                            "source_filename": r.source_filename,
                            "superseded_by_uid": r.superseded_by_uid,
                        }
                        for r in rows_sorted
                        if r.offer_date is None or r.offer_date <= cur.offer_date
                    ],
                }
                event = PriceEventDB(
                    peer_group_id=k,
                    event_type=etype,
                    current_uid=cur.uid,
                    current_supplier=cur.supplier_name,
                    current_price_eur=cur_price,
                    current_offer_date=cur.offer_date,
                    current_source_file=cur.source_filename,
                    previous_uid=prev.uid,
                    previous_supplier=prev.supplier_name,
                    previous_price_eur=prev_price,
                    previous_offer_date=prev.offer_date,
                    previous_source_file=prev.source_filename,
                    delta_pct=delta_pct,
                    delta_abs=delta_abs,
                    message=(
                        f"{cur.supplier_name or 'Unknown supplier'} @ €{cur_price:.2f} — "
                        f"{etype} for {cur.brand or ''} {cur.product_name or ''}. "
                        f"Previous best: {prev.supplier_name or 'unknown'} €{prev_price:.2f} "
                        f"({delta_pct:+.1f}%)" if delta_pct is not None else ""
                    ),
                    peer_snapshot_json=json.dumps(snap),
                )
                try:
                    db.add(event)
                    db.commit()
                    db.refresh(event)
                    written.append(event)
                    logger.info(
                        f"R-12 event {event.id}: {prev.supplier_reference or prev.uid}"
                        f" €{prev_price:.2f} → {cur.supplier_reference or cur.uid}"
                        f" €{cur_price:.2f} ({delta_pct:+.2f}%)"
                    )
                except IntegrityError:
                    db.rollback()  # already recorded — dedup kick-in, that's fine.
                except Exception as exc:
                    db.rollback()
                    logger.warning(f"R-12 event insert failed for peer {k}: {exc}")

            # Advance best_prior whenever cur price undercuts it so a later
            # row competes against the new best, not the stale one.
            if cur_price < best_prior.price_per_unit_eur - 0.001:
                best_prior = cur

    return written
