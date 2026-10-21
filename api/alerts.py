"""
Basic trading alerts (Milestone 5).

Two alert types:
- new_best_price: the latest offer for a canonical product is the lowest we've
  ever seen at its incoterm.
- competitive_supplier: a supplier's newest offer for a product they hadn't
  previously priced is now the peer-group best.

MVP: computed on demand from the database. No background job, no email
delivery yet — surfaced through /api/alerts for the dashboard to display.
When we're ready to push these out, add an email/webhook fan-out on top.
"""
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from core.database import get_db
from core.normalization import peer_group_id, peer_group_id_from_ean, is_peer_group_qualified
from models.offer_item import OfferItemDB

router = APIRouter()


@router.get("/alerts")
def list_alerts(
    days: int = Query(7, description="Look at offers arrived in the last N days"),
    limit: int = Query(50),
    db: Session = Depends(get_db),
):
    """
    Return recent trading alerts. Each alert is one product event.

    Payload:
      [{
        type: "new_best_price" | "competitive_supplier",
        peer_group_id: "...",
        product: {brand, name, size_ml, upc, incoterm},
        current: {uid, supplier, price_eur, offer_date},
        previous_best: {uid, supplier, price_eur, offer_date, delta_pct},
        message: "…"
      }, ...]
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    all_rows = (db.query(OfferItemDB)
                  .filter(OfferItemDB.price_per_unit_eur.isnot(None))
                  .all())

    # Group all rows by canonical key.
    # EAN-primary: when a valid EAN is present, records that share the same
    # EAN + incoterm + location are the same peer regardless of how each
    # supplier formatted the product name. This is the fix for cross-supplier
    # alerts not firing when name normalisation differs between files.
    # Falls back to the full attribute-based peer_group_id when EAN is absent.
    by_key = defaultdict(list)
    for r in all_rows:
        # Peer key MUST match the Best Prices roll-up (api/records.get_best_prices
        # → sku_identity/peer_group_id) or alerts fire on groups Best Prices does
        # not treat as comparable. In particular perfume/cosmetics discriminators
        # (format, retail state, gender, shade, product type, size, range) and the
        # category slug keep EDP vs EDT (e.g. YSL Midnight Halo) in separate peers.
        k = (peer_group_id_from_ean(r.ean_code, r.incoterm, r.location, r.custom_status)
             or peer_group_id(
                 r.brand, r.product_name,
                 unit_volume_ml=r.unit_volume_ml,
                 units_per_case=r.units_per_case,
                 incoterm=r.incoterm,
                 location=r.location,
                 alcohol_percent=r.alcohol_percent,
                 vintage=r.vintage,
                 category_slug=getattr(r, 'category_slug', None),
                 perfume_format=getattr(r, 'perfume_format', None),
                 retail_state=getattr(r, 'retail_state', None),
                 gender=getattr(r, 'gender', None),
                 shade=getattr(r, 'shade', None),
                 product_type=getattr(r, 'product_type', None),
                 size_weight_g=getattr(r, 'size_weight_g', None),
                 range_name=getattr(r, 'range_name', None),
                 custom_status=getattr(r, 'custom_status', None),
             ))
        by_key[k].append(r)

    alerts = []
    # Track (peer_group_id, supplier, alert_type) so we don't emit the same
    # supplier/product situation twice — most recent event per identity wins.
    seen = set()
    for k, rows in by_key.items():
        if len(rows) < 2:
            continue  # No prior peer to beat — not an alert-worthy event

        # Chronological order
        rows = sorted(rows, key=lambda x: x.offer_date or x.created_at or datetime.min)
        latest = rows[-1]

        # Freshness gate: use offer_date ONLY — never fall back to
        # created_at.  A historical file ingested today for validation
        # purposes has created_at=today but offer_date=April/June/Sept.
        # Using created_at as a fallback would fire stale "alerts" for
        # every historical low in those files; that's not actionable.
        # If offer_date is absent we have no way to know whether the
        # offer is genuinely fresh — skip it (conservative, safe).
        if not latest.offer_date or latest.offer_date < cutoff:
            continue
        # Received-date fallback guard: ingest once stored offer_date =
        # datetime.utcnow() when the extractor returned no document date.
        # Those rows carry offer_date == date_received to the microsecond and
        # a non-midnight time component. They are NOT a verified commercial
        # offer date, so they must not fire 'new best price' / freshness
        # alerts. Detection is deterministic from stored data (no new field):
        od = latest.offer_date
        dr = latest.date_received
        if dr is not None and od == dr and (
                od.hour or od.minute or od.second or od.microsecond):
            continue

        prior = rows[:-1]
        best_prior = min(prior, key=lambda x: x.price_per_unit_eur or float("inf"))
        best_prior_price = best_prior.price_per_unit_eur or float("inf")
        cur_price = latest.price_per_unit_eur

        if cur_price is None or best_prior_price is None or best_prior_price == float("inf"):
            continue

        alert_type = None
        message = None
        if cur_price < best_prior_price - 0.001:
            # Strict new low
            alert_type = "new_best_price"
            delta_pct = ((cur_price - best_prior_price) / best_prior_price) * 100
            message = (f"{latest.supplier_name or 'Unknown supplier'} @ €{cur_price:.2f} — "
                       f"new best for {latest.brand or ''} {latest.product_name or ''}. "
                       f"Previous best: {best_prior.supplier_name or 'unknown'} €{best_prior_price:.2f} "
                       f"({delta_pct:.1f}%)")
        else:
            # Did a NEW supplier just come in with a competitive offer?
            latest_supplier = (latest.supplier_name or "").lower().strip()
            prior_suppliers = {(p.supplier_name or "").lower().strip() for p in prior}
            if latest_supplier and latest_supplier not in prior_suppliers:
                # Competitive if within 5% of the prior best
                if cur_price <= best_prior_price * 1.05:
                    alert_type = "competitive_supplier"
                    delta_pct = ((cur_price - best_prior_price) / best_prior_price) * 100
                    message = (f"New competitive supplier for {latest.brand or ''} {latest.product_name or ''}: "
                               f"{latest.supplier_name} @ €{cur_price:.2f} "
                               f"({'+' if delta_pct >= 0 else ''}{delta_pct:.1f}% vs "
                               f"{best_prior.supplier_name} at €{best_prior_price:.2f})")

        if not alert_type:
            continue

        # A trading alert must not fire on a peer group that contains any
        # unqualified row (incoterm or location unknown on ANY row here).
        # Otherwise a "NEW LOW" could claim Rotterdam-EXW beats a prior
        # that has no known origin — a falsely precise signal.
        is_qualified = all(
            is_peer_group_qualified(r.incoterm, r.location) for r in rows
        )
        if not is_qualified:
            continue

        # Deduplicate: same peer group + supplier + alert type → collapse to
        # the newest event (the sort at the end promotes the latest).
        dedup_key = (k, (latest.supplier_name or "").lower().strip(), alert_type)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        delta_pct = ((cur_price - best_prior_price) / best_prior_price) * 100

        alerts.append({
            "type": alert_type,
            "peer_group_id": k,
            "created_at": (latest.offer_date or latest.created_at).isoformat(),
            "product": {
                "brand": latest.brand,
                "name": latest.product_name,
                "size_ml": latest.unit_volume_ml,
                "units_per_case": latest.units_per_case,
                "incoterm": latest.incoterm,
                # SKU discriminators so two look-alike cards (same brand/name,
                # different EAN or retail state) are distinguishable at a glance
                # (Samuel #12: Angel Fantasm €43 vs €51.50, different EANs).
                "ean": latest.ean_code,
                "retail_state": getattr(latest, "retail_state", None),
            },
            "current": {
                "uid": latest.uid,
                "supplier": latest.supplier_name,
                "price_eur": cur_price,
                "offer_date": (latest.offer_date or latest.created_at).isoformat(),
                "source_filename": latest.source_filename,
                "custom_status": latest.custom_status,
            },
            "previous_best": {
                "uid": best_prior.uid,
                "supplier": best_prior.supplier_name,
                "price_eur": best_prior_price,
                "offer_date": (best_prior.offer_date or best_prior.created_at).isoformat() if best_prior.offer_date or best_prior.created_at else None,
                "source_filename": best_prior.source_filename,
                "delta_pct": round(delta_pct, 2),
            },
            "message": message,
        })

    # Newest alert first
    alerts.sort(key=lambda a: a["created_at"], reverse=True)
    return {"count": len(alerts), "alerts": alerts[:limit], "window_days": days}
