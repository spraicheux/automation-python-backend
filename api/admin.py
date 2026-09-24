"""
One-off admin endpoints. Currently:

  POST /api/admin/backfill-case-supplier
     Applies the same €/case sanity guard + supplier metadata re-derivation
     to already-ingested rows. Idempotent: safe to run more than once.
     Gated by the app's admin auth token to avoid public triggering.

Not routed publicly on the dashboard — hit with curl once after a deploy.
"""
import os
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session

from core.database import get_db
from core.deterministic_rules import detect_supplier_from_metadata
from models.offer_item import OfferItemDB


router = APIRouter()


def _require_admin(x_admin_token: str = Header(None)):
    expected = os.getenv("ADMIN_TOKEN", "valid-token")
    if not x_admin_token or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Missing / invalid admin token")


@router.post("/admin/backfill-case-supplier", dependencies=[Depends(_require_admin)])
def backfill_case_supplier(apply: bool = False, db: Session = Depends(get_db)):
    """
    Fix two client-reported bugs on already-ingested rows:

      1. €/case: rows where the LLM stored a "Total Price" column as
         price_per_case get their price_per_case cleared.
      2. Supplier: rows with supplier_name null / "Not Found" get supplier
         re-derived from filename suffix + sender_email.

    Dry-run by default. Pass ?apply=true to write.
    """
    case_changes = []
    supplier_changes = []

    # ── €/case sanity ──────────────────────────────────────────────
    rows = (db.query(OfferItemDB)
              .filter(OfferItemDB.price_per_case.isnot(None),
                      OfferItemDB.price_per_unit.isnot(None),
                      OfferItemDB.quantity_case.isnot(None))
              .all())
    for r in rows:
        upc = r.units_per_case
        qc = r.quantity_case
        ppc = r.price_per_case
        ppu = r.price_per_unit
        if not (ppc and ppu and qc):
            continue
        if upc not in (None, 1, 1.0):
            continue
        if qc <= 1:
            continue
        expected_total = qc * ppu
        if expected_total <= 0:
            continue
        rel_error = abs(ppc - expected_total) / expected_total
        # tight tolerance (2%) — genuine per-case prices don't accidentally
        # equal qty × unit_price to 2%.
        if rel_error < 0.02:
            case_changes.append({
                "uid": r.uid,
                "brand": r.brand,
                "product": r.product_name,
                "qty": qc,
                "unit_price": ppu,
                "old_case_price": ppc,
                "new_case_price": None,
                "reason": "price_per_case == qty × unit_price (mis-mapped Total column)",
            })
            if apply:
                r.price_per_case = None
                r.price_per_case_eur = None
        elif upc in (1, 1.0) and abs(ppc - ppu) < 0.01:
            case_changes.append({
                "uid": r.uid,
                "brand": r.brand,
                "product": r.product_name,
                "old_case_price": ppc,
                "new_case_price": None,
                "reason": "units_per_case=1 and case_price==unit_price (no real case)",
            })
            if apply:
                r.price_per_case = None
                r.price_per_case_eur = None

    # ── supplier re-derivation ────────────────────────────────────
    missing = (db.query(OfferItemDB)
                 .filter((OfferItemDB.supplier_name.is_(None)) |
                         (OfferItemDB.supplier_name == "") |
                         (OfferItemDB.supplier_name.ilike("not found")))
                 .all())
    for r in missing:
        s = detect_supplier_from_metadata(
            text="",
            source_filename=r.source_filename,
            sender_email=r.sender_email,
        )
        if s:
            supplier_changes.append({
                "uid": r.uid,
                "file": r.source_filename,
                "sender_email": r.sender_email,
                "old_supplier": r.supplier_name,
                "new_supplier": s,
            })
            if apply:
                r.supplier_name = s

    if apply:
        db.commit()

    return {
        "applied": apply,
        "case_price_fixes": len(case_changes),
        "case_price_samples": case_changes[:20],
        "supplier_fixes": len(supplier_changes),
        "supplier_samples": supplier_changes[:20],
    }
