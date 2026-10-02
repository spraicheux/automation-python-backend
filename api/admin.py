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


@router.post("/admin/backfill-doc-defaults", dependencies=[Depends(_require_admin)])
def backfill_doc_defaults(
    category_slug: str = "perfumes",
    apply: bool = False,
    db: Session = Depends(get_db),
):
    """
    Re-apply deterministic document-header defaults (incoterm / location /
    supplier / currency) to already-ingested rows where the LLM output
    "Not Found" and the earlier pass's blank-check didn't treat it as
    empty. Deterministic — no OpenAI calls.
    """
    from core.deterministic_rules import (
        detect_document_incoterm, detect_document_location,
        detect_supplier_from_metadata,
    )
    rows = (db.query(OfferItemDB)
              .filter(OfferItemDB.category_slug == category_slug)
              .all())
    changes = {"incoterm": 0, "location": 0, "supplier": 0}
    sample = []
    # Group by source filename so we only derive the header defaults once per file.
    by_file = {}
    for r in rows:
        by_file.setdefault(r.source_filename, []).append(r)
    for fname, group in by_file.items():
        if not fname:
            continue
        # Treat the first row's extracted values + filename as a cheap
        # proxy for the document header. For FBC Trades specifically the
        # filename alone resolves to EXW / Rotterdam / FBC Trades via the
        # existing deterministic detectors.
        hay = f"{fname} " + " ".join(
            (f"{r.incoterm} {r.location} {r.supplier_name} {r.product_name}")
            for r in group[:5]
        )
        doc_inco = detect_document_incoterm(hay) or "EXW" if "exw" in hay.lower() else detect_document_incoterm(hay)
        doc_loc = detect_document_location(hay)
        doc_sup = detect_supplier_from_metadata(hay, source_filename=fname, sender_email=(group[0].sender_email or ""))
        for r in group:
            if doc_inco and str(r.incoterm or "").strip().lower() in ("", "not found"):
                if apply: r.incoterm = doc_inco
                changes["incoterm"] += 1
                if len(sample) < 10:
                    sample.append({"uid": r.uid, "field": "incoterm", "from": r.incoterm, "to": doc_inco, "file": fname})
            if doc_loc and str(r.location or "").strip().lower() in ("", "not found"):
                if apply: r.location = doc_loc
                changes["location"] += 1
            if doc_sup and str(r.supplier_name or "").strip().lower() in ("", "not found"):
                if apply: r.supplier_name = doc_sup
                changes["supplier"] += 1
    if apply:
        db.commit()
    return {"applied": apply, "category_slug": category_slug,
            "rows_examined": len(rows), "changes": changes, "sample": sample}


@router.post("/admin/backfill-loose-case-zero", dependencies=[Depends(_require_admin)])
def backfill_loose_case_zero(
    category_slug: str = "perfumes",
    apply: bool = False,
    db: Session = Depends(get_db),
):
    """
    Null out zero-value price_per_case / price_per_case_eur / units_per_case
    on loose-unit rows (units_per_case null / 0 / 1). Client requirement §3:
    €/CASE column should render "—" not "€0.00" for rows that have no
    real case pack.
    """
    rows = (db.query(OfferItemDB)
              .filter(OfferItemDB.category_slug == category_slug)
              .all())
    changed = 0
    for r in rows:
        upc = r.units_per_case
        touched = False
        if upc in (None, 0, 0.0, 1, 1.0):
            if r.price_per_case in (0, 0.0):
                if apply:
                    r.price_per_case = None
                touched = True
            if r.price_per_case_eur in (0, 0.0):
                if apply:
                    r.price_per_case_eur = None
                touched = True
            if upc in (0, 0.0):
                if apply:
                    r.units_per_case = None
                touched = True
        if touched:
            changed += 1
    if apply:
        db.commit()
    return {"applied": apply, "category_slug": category_slug,
            "rows_examined": len(rows), "rows_changed": changed}


@router.post("/admin/backfill-ean-and-retail", dependencies=[Depends(_require_admin)])
def backfill_ean_and_retail(
    category_slug: str = "perfumes",
    apply: bool = False,
    db: Session = Depends(get_db),
):
    """
    Re-run EAN check-digit repair and null-out inferred retail_state on
    already-ingested rows. Both are deterministic (no LLM calls), so this
    is idempotent and doesn't spend OpenAI credits.

    EAN repair uses core.openai_client._repair_ean — same algorithm the
    ingest pipeline now runs post-extraction.

    retail_state null-out: the ingest rule was "default to retail unless
    explicitly tester/sample/miniature" which silently manufactured an
    SKU discriminator. Client wants retail_state to be null when the
    source didn't say. This backfill nulls any 'retail' value on rows in
    the given category — 'tester' / 'sample' / 'miniature' are always
    kept (those came from explicit source words per Rule 0.23).

    Dry-run by default. Pass ?apply=true to write.
    """
    from core.openai_client import _repair_ean
    import re as _re
    import json as _json

    def _load_flags(raw):
        # error_flags is stored as a JSON-string Text column (see
        # models/offer_item.py:97). to_dict() deserialises with
        # json.loads. Mirror that here — never assign a Python list to
        # a Text column, always serialise back to string.
        if not raw: return []
        try:
            v = _json.loads(raw) if isinstance(raw, str) else raw
            return list(v) if isinstance(v, list) else []
        except Exception:
            return []

    def _dump_flags(lst):
        return _json.dumps(lst) if lst else None

    rows = (db.query(OfferItemDB)
              .filter(OfferItemDB.category_slug == category_slug)
              .all())

    ean_stats = {"scanned": 0, "already_valid": 0, "repaired": 0, "flagged": 0}
    ean_samples = []
    retail_stats = {"scanned": 0, "nulled": 0, "kept_tester_sample": 0}

    for r in rows:
        if r.ean_code:
            ean_stats["scanned"] += 1
            digits = _re.sub(r"\D+", "", str(r.ean_code))
            repaired, reason = _repair_ean(digits)
            if repaired is None:
                ean_stats["flagged"] += 1
                if len(ean_samples) < 20:
                    ean_samples.append({
                        "uid": r.uid, "brand": r.brand,
                        "product": r.product_name, "was": r.ean_code,
                        "action": "flagged", "reason": reason,
                    })
                if apply:
                    flags = _load_flags(r.error_flags)
                    tag = f"ean_code failed length + check-digit validation ({reason})"
                    if tag not in flags:
                        flags.append(tag)
                        r.error_flags = _dump_flags(flags)
                    r.needs_manual_review = True
            elif repaired == digits:
                ean_stats["already_valid"] += 1
            else:
                ean_stats["repaired"] += 1
                if len(ean_samples) < 20:
                    ean_samples.append({
                        "uid": r.uid, "brand": r.brand,
                        "product": r.product_name, "was": r.ean_code,
                        "now": repaired, "action": "repaired", "reason": reason,
                    })
                if apply:
                    r.ean_code = repaired
                    flags = _load_flags(r.error_flags)
                    tag = f"ean_code repaired by backfill ({reason})"
                    if tag not in flags:
                        flags.append(tag)
                        r.error_flags = _dump_flags(flags)

        if r.retail_state:
            retail_stats["scanned"] += 1
            if str(r.retail_state).strip().lower() == "retail":
                retail_stats["nulled"] += 1
                if apply:
                    r.retail_state = None
            else:
                retail_stats["kept_tester_sample"] += 1

    if apply:
        db.commit()

    return {
        "applied": apply,
        "category_slug": category_slug,
        "rows_examined": len(rows),
        "ean": ean_stats,
        "retail_state": retail_stats,
        "ean_samples": ean_samples,
    }


@router.post("/admin/purge-by-filename", dependencies=[Depends(_require_admin)])
def purge_by_filename(pattern: str, apply: bool = False, db: Session = Depends(get_db)):
    """
    Delete every offer_items row (and the associated source_file entries)
    where `source_filename ILIKE pattern`. Intended for cleaning up
    duplicate test ingests before a fresh run. Dry-run by default;
    ?apply=true actually deletes.

    Example — delete every ingest of the FBC perfumes PDF:
      POST /api/admin/purge-by-filename?pattern=%25perfumes%25&apply=true
    """
    if not pattern or len(pattern) < 3:
        raise HTTPException(status_code=400, detail="pattern too short — refuse to delete broadly")

    from models.source_file import SourceFileDB
    rows = (db.query(OfferItemDB)
              .filter(OfferItemDB.source_filename.ilike(pattern))
              .all())
    files = (db.query(SourceFileDB)
               .filter(SourceFileDB.source_filename.ilike(pattern))
               .all())

    samples = [{"uid": r.uid, "brand": r.brand, "product": r.product_name,
                "source": r.source_filename} for r in rows[:10]]
    filenames = sorted({r.source_filename for r in rows if r.source_filename})

    if apply:
        for r in rows:
            db.delete(r)
        for f in files:
            db.delete(f)
        db.commit()

    return {
        "applied": apply,
        "pattern": pattern,
        "offers_deleted": len(rows) if apply else 0,
        "offers_matching": len(rows),
        "source_files_deleted": len(files) if apply else 0,
        "distinct_source_filenames": filenames,
        "sample": samples,
    }


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
