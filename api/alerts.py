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
from core.normalization import peer_group_id, peer_group_id_from_ean, is_peer_group_qualified, ean_key
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

    # Cross-category EAN conflict guard (Samuel 2026-10-28). A valid GS1 EAN
    # identifies one physical product; if the same EAN carries two different
    # `category_slug` values in the DB that is a classification inconsistency,
    # not a reason to compare prices. The records endpoint already surfaces
    # this as a read-only `cross_category_ean_conflict` flag. Alerts must not
    # compare offers across the boundary — otherwise a Perfumes offer would
    # fire a "new best" against a Cosmetics benchmark even though the UI shows
    # the EAN as cross-category (Samuel's current example). We suppress every
    # alert whose peer group contains any EAN that appears under more than one
    # category_slug across the full table.
    _ean_cats = defaultdict(set)
    for r in all_rows:
        ek = ean_key(r.ean_code)
        cs = getattr(r, "category_slug", None)
        if ek and cs:
            _ean_cats[ek].add(cs)
    _cross_cat_eans = {e for e, cs in _ean_cats.items() if len(cs) > 1}

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
        k = (peer_group_id_from_ean(r.ean_code, r.incoterm, r.location, r.custom_status, r.perfume_format, r.retail_state)
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

        # Cross-category EAN exclusion: if any row in this peer group carries
        # an EAN that is also tagged with a different category_slug elsewhere
        # in the table, the two sides are not comparable until the mismatch is
        # resolved. Skip the whole group — the records endpoint still shows
        # the conflict flag so the operator can clean it up.
        if any(ean_key(r.ean_code) in _cross_cat_eans for r in rows):
            continue

        # Chronological order BY offer_date only — Samuel R-06/R-12: a row
        # whose source gave no commercial offer_date (U01) must not be
        # treated as "the latest commercial event". Undated rows sort to
        # the start (datetime.min) so they only ever appear as prior
        # history, never as the alert's "current". The prior-set filter
        # below also excludes them from best-prior computation.
        rows = sorted(rows, key=lambda x: x.offer_date or datetime.min)
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

        # Same-batch guard: a supplier can list the same SKU twice at
        # different prices in a single file (Samuel: Dior J'Adore €89 vs €96,
        # Balkan Luxury, both W39, same offer date). Those are the same
        # commercial batch, not a trading event — the "previous best peer"
        # must come from a DIFFERENT supplier OR a DIFFERENT offer date
        # (older-than-current). Rows with the same supplier and same date
        # are same-batch duplicates and excluded from the prior-set.
        _latest_sup = (latest.supplier_name or "").strip().lower()
        _latest_date = latest.offer_date
        prior = [
            r for r in rows[:-1]
            if ((r.supplier_name or "").strip().lower() != _latest_sup)
            or (r.offer_date is not None and _latest_date is not None
                and r.offer_date < _latest_date)
        ]
        if not prior:
            continue

        # Eligibility filter for the best-prior comparison (Samuel R-07/R-12):
        # a trading alert's "previous best" must be a trusted, actionable price.
        # Exclude from the prior set:
        #   • zero / unknown prices (price_per_unit_eur <= 0)
        #   • needs_manual_review rows (POR retained, zero price retained,
        #     explicit-vs-derived ppu discrepancy, implausible ABV, …)
        #   • undated rows (offer_date is None — "U01 ineligible under
        #     declared policy", Samuel R-12)
        #   • expired rows (valid_until < latest.offer_date — a 2026-09-30
        #     expired low must not win the current best)
        def _parse_valid_until(v):
            # valid_until is stored as a free-form string (model col String(64)).
            # Accept "YYYY-MM-DD" and ISO-with-time; return None on anything else
            # so an unparseable value never pretends to be a current offer.
            if v is None or (isinstance(v, str) and not v.strip()):
                return None
            if isinstance(v, datetime):
                return v
            s = str(v).strip()
            try:
                return datetime.fromisoformat(s[:19].replace("Z", ""))
            except ValueError:
                try:
                    return datetime.strptime(s[:10], "%Y-%m-%d")
                except ValueError:
                    return None

        def _is_commercially_eligible(r) -> bool:
            if r.price_per_unit_eur is None or r.price_per_unit_eur <= 0:
                return False
            if getattr(r, "needs_manual_review", False):
                return False
            if r.offer_date is None:
                return False
            # R-07: a row the supplier explicitly superseded is not an
            # eligible current price for new-best/competitive comparisons.
            if getattr(r, "superseded_by_uid", None):
                return False
            vu = _parse_valid_until(getattr(r, "valid_until", None))
            if vu is not None and _latest_date is not None:
                if vu < _latest_date:
                    return False
            return True
        prior_eligible = [r for r in prior if _is_commercially_eligible(r)]
        if not prior_eligible:
            continue
        best_prior = min(prior_eligible, key=lambda x: x.price_per_unit_eur or float("inf"))
        best_prior_price = best_prior.price_per_unit_eur or float("inf")
        cur_price = latest.price_per_unit_eur

        if cur_price is None or best_prior_price is None or best_prior_price == float("inf"):
            continue
        # The latest offer itself must also be commercially eligible — a
        # POR / zero / review row cannot ANNOUNCE a new best price.
        if not _is_commercially_eligible(latest):
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
