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
from core.normalization import peer_group_id, is_peer_group_qualified
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

    # Group all rows by canonical key
    by_key = defaultdict(list)
    for r in all_rows:
        k = peer_group_id(
            r.brand, r.product_name,
            unit_volume_ml=r.unit_volume_ml,
            units_per_case=r.units_per_case,
            incoterm=r.incoterm,
            location=r.location,
            alcohol_percent=r.alcohol_percent,
            vintage=r.vintage,
        )
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

        if not (latest.offer_date or latest.created_at) or \
           (latest.offer_date or latest.created_at) < cutoff:
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
            message = (f"New best price on {latest.brand or ''} {latest.product_name or ''}: "
                       f"€{cur_price:.2f} — {delta_pct:.1f}% below prior best (€{best_prior_price:.2f}) "
                       f"from {best_prior.supplier_name or 'unknown'}")
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
            },
            "current": {
                "uid": latest.uid,
                "supplier": latest.supplier_name,
                "price_eur": cur_price,
                "offer_date": (latest.offer_date or latest.created_at).isoformat(),
            },
            "previous_best": {
                "uid": best_prior.uid,
                "supplier": best_prior.supplier_name,
                "price_eur": best_prior_price,
                "offer_date": (best_prior.offer_date or best_prior.created_at).isoformat() if best_prior.offer_date or best_prior.created_at else None,
                "delta_pct": round(delta_pct, 2),
            },
            "message": message,
        })

    # Newest alert first
    alerts.sort(key=lambda a: a["created_at"], reverse=True)
    return {"count": len(alerts), "alerts": alerts[:limit], "window_days": days}
