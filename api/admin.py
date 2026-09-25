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
                "old_quantity_unit": r.quantity_unit,
                "new_quantity_unit": "bottles",
                "reason": "price_per_case == qty × unit_price (mis-mapped Total column); quantity was bottles",
            })
            if apply:
                r.price_per_case = None
                r.price_per_case_eur = None
                if not (r.quantity_unit or "").strip():
                    r.quantity_unit = "bottles"
        elif upc in (1, 1.0) and abs(ppc - ppu) < 0.01:
            case_changes.append({
                "uid": r.uid,
                "brand": r.brand,
                "product": r.product_name,
                "old_case_price": ppc,
                "new_case_price": None,
                "old_quantity_unit": r.quantity_unit,
                "new_quantity_unit": "bottles",
                "reason": "units_per_case=1 and case_price==unit_price (no real case); quantity was bottles",
            })
            if apply:
                r.price_per_case = None
                r.price_per_case_eur = None
                if not (r.quantity_unit or "").strip():
                    r.quantity_unit = "bottles"

    # ── quantity_unit tidy-up ─────────────────────────────────────
    # For rows where price_per_case was already cleared in a prior run but
    # quantity_unit is still null (e.g. the MIX SPIRITS HNS backfill run
    # before the column existed), infer the unit from the same heuristic:
    # units_per_case = 1 AND quantity_case > 1 AND no per-case price → the
    # quantity was expressed in bottles.
    unit_backfill = 0
    unit_rows = (db.query(OfferItemDB)
                   .filter((OfferItemDB.quantity_unit.is_(None)) |
                           (OfferItemDB.quantity_unit == ""))
                   .filter(OfferItemDB.quantity_case.isnot(None),
                           OfferItemDB.quantity_case > 1,
                           OfferItemDB.units_per_case.in_([1, 1.0]),
                           OfferItemDB.price_per_case.is_(None))
                   .all())
    for r in unit_rows:
        unit_backfill += 1
        if apply:
            r.quantity_unit = "bottles"

    # ── category slug backfill (Phase 3 M1) ───────────────────────
    # Existing W&S rows have `category` set (e.g. "Spirits", "Wine")
    # but no machine slug. Fold the free-form label into the canonical
    # slug so /api/records?category_slug=wines_spirits catches them.
    from core.category_classifier import _normalize_category as _cat_norm
    slug_backfill = 0
    slug_rows = (db.query(OfferItemDB)
                   .filter((OfferItemDB.category_slug.is_(None)) |
                           (OfferItemDB.category_slug == ""))
                   .filter(OfferItemDB.category.isnot(None))
                   .all())
    for r in slug_rows:
        slug = _cat_norm(r.category)
        if slug:
            slug_backfill += 1
            if apply:
                r.category_slug = slug

    # ── units_per_case tidy-up ────────────────────────────────────
    # Rows that already went through the "no case pack" cleanup still
    # carry units_per_case=1 as a legacy fallback. Clear those to null
    # so the dashboard reads "loose bottles" and the SKU key no longer
    # invents a phantom 1-bottle case.
    upc_backfill = 0
    upc_rows = (db.query(OfferItemDB)
                  .filter(OfferItemDB.units_per_case.in_([1, 1.0]),
                          OfferItemDB.quantity_unit == "bottles",
                          OfferItemDB.price_per_case.is_(None))
                  .all())
    for r in upc_rows:
        upc_backfill += 1
        if apply:
            r.units_per_case = None

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
        "quantity_unit_backfill": unit_backfill,
        "units_per_case_backfill": upc_backfill,
        "category_slug_backfill": slug_backfill,
    }
