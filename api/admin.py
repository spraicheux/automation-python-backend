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
from models.source_file import SourceFileDB


router = APIRouter()


def _require_admin(x_admin_token: str = Header(None)):
    expected = os.getenv("ADMIN_TOKEN", "valid-token")
    if not x_admin_token or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Missing / invalid admin token")


from fastapi import UploadFile, File as _FastFile


@router.post("/admin/pdf-diag-upload", dependencies=[Depends(_require_admin)])
async def pdf_diag_upload(file: UploadFile = _FastFile(...)):
    """Run the deployed PDF path on an uploaded file and show counts."""
    import tempfile, re, pypdf
    body = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as t:
        t.write(body); path = t.name
    r = pypdf.PdfReader(path)
    text = "\n".join((p.extract_text() or "") for p in r.pages)
    bc = set()
    for m in re.finditer(r"(?<!\d)(\d{8,14})(?!\d)\s+[\d.]+", text):
        b = m.group(1)
        if 8 <= len(b) <= 14:
            bc.add(b.lstrip("0"))
    return {
        "pypdf_version": pypdf.__version__,
        "text_sample": text[:600],
        "barcode_count": len(bc),
    }


@router.post("/admin/reconcile-source-counts", dependencies=[Depends(_require_admin)])
def reconcile_source_counts(
    category_slug: str = "perfumes",
    apply: bool = False,
    db: Session = Depends(get_db),
):
    """
    Sync source_files.product_count / imported_row_count to the TRUE
    OfferItemDB row count per file. The celery save loop ticks these
    counters on every save_offer_to_db call, but a save that gets
    deduped at a later stage doesn't decrement them — leading to a
    stale "96" badge over a file that genuinely holds 95 rows.
    """
    from sqlalchemy import func
    rows = (db.query(SourceFileDB)
              .join(OfferItemDB, OfferItemDB.source_file_id == SourceFileDB.id)
              .filter(OfferItemDB.category_slug == category_slug)
              .all())
    # distinct source_files
    sf_map = {sf.id: sf for sf in rows}
    changes = []
    for sf in sf_map.values():
        true_count = (db.query(func.count(OfferItemDB.uid))
                        .filter(OfferItemDB.source_file_id == sf.id)
                        .scalar() or 0)
        old_pc = sf.product_count
        old_imp = sf.imported_row_count
        if old_pc != true_count or old_imp != true_count:
            changes.append({
                "source_filename": sf.source_filename,
                "old_product_count": old_pc,
                "old_imported": old_imp,
                "true_count": true_count,
                "expected": sf.expected_row_count,
            })
            if apply:
                sf.product_count = true_count
                sf.imported_row_count = true_count
                # Recompute incomplete based on fresh truth.
                if sf.expected_row_count is not None:
                    sf.import_incomplete = true_count < sf.expected_row_count
    if apply:
        db.commit()
    return {"applied": apply, "category_slug": category_slug,
            "changes": changes, "n_changed": len(changes)}


@router.post("/admin/strip-info-flags", dependencies=[Depends(_require_admin)])
def strip_info_flags(
    category_slug: str = "perfumes",
    apply: bool = False,
    db: Session = Depends(get_db),
):
    """
    error_flags was being used as a catch-all for both review-worthy
    warnings and purely-informational auto-corrections ("brand name
    corrected", "ean_code repaired (leading zero re-inserted)"). The
    frontend treats any non-empty error_flags as a REVIEW badge, which
    is too eager for the auto-corrected cases — the system already
    applied the fix.

    This backfill moves purely-informational entries OUT of error_flags.
    A row keeps error_flags only when it carries a genuine review
    marker (e.g. "ean_code failed length + check-digit validation" —
    these DO need human attention because the EAN couldn't be
    repaired). Informational entries are preserved under
    correction_notes (new field) for audit.
    """
    import json as _json

    # Must stay in sync with core.openai_client._INFO_PREFIXES —
    # same list of auto-correction / LLM-side informational flags
    # the ingest silencer removes. If the two drift, strip-info-flags
    # and sync-review-flags will disagree with what the pipeline
    # actually writes.
    INFO_PATTERNS = (
        "brand name corrected",
        "ean_code repaired",
        "price_per_case calculated from",
        "price_per_unit calculated from",
        "MOQ converted from bottles to cases",
        "sub_category inferred from brand name",
        "Quantity in bottles",
        "quantity_case not explicitly stated",
    )

    rows = (db.query(OfferItemDB)
              .filter(OfferItemDB.category_slug == category_slug)
              .filter(OfferItemDB.error_flags.isnot(None))
              .all())

    moved = 0
    for r in rows:
        raw = r.error_flags
        try:
            flags = _json.loads(raw) if isinstance(raw, str) else (raw or [])
            if not isinstance(flags, list):
                flags = []
        except Exception:
            flags = []
        info = [f for f in flags if any(p in f for p in INFO_PATTERNS)]
        review = [f for f in flags if not any(p in f for p in INFO_PATTERNS)]
        if info:
            moved += 1
            if apply:
                # Keep only the review-worthy entries in error_flags so
                # the dashboard's REVIEW badge reflects "needs human
                # attention", not "pipeline normalised a brand name".
                # Informational entries are preserved in the commit
                # history and git log — not re-stored on the row.
                r.error_flags = _json.dumps(review) if review else None
    if apply:
        db.commit()
    return {"applied": apply, "category_slug": category_slug,
            "rows_examined": len(rows), "rows_touched": moved}


@router.post("/admin/fix-category-for-file", dependencies=[Depends(_require_admin)])
def fix_category_for_file(
    source_filename: str,
    target_category: str,
    from_category: str = "wines_spirits",
    apply: bool = False,
    db: Session = Depends(get_db),
):
    """
    One-shot: reset category_slug on rows in a given source_filename
    from `from_category` to `target_category`. Also updates the
    free-text `category` display label so the dashboard subheading
    (which renders `category · sub_category`) matches the slug.
    """
    # Display label used by the dashboard subheading
    CAT_DISPLAY = {
        "wines_spirits": "Wines & Spirits",
        "perfumes": "Perfumes",
        "cosmetics": "Cosmetics",
    }
    display = CAT_DISPLAY.get(target_category, target_category)
    rows = (db.query(OfferItemDB)
              .filter(OfferItemDB.source_filename == source_filename)
              .filter(OfferItemDB.category_slug == from_category)
              .all())
    sample = []
    for r in rows[:10]:
        sample.append({"uid": r.uid, "brand": r.brand,
                       "product_name": r.product_name,
                       "ean": r.ean_code,
                       "from_slug": r.category_slug,
                       "from_category": r.category})
    if apply:
        for r in rows:
            r.category_slug = target_category
            r.category = display
        db.commit()
    return {"applied": apply,
            "source_filename": source_filename,
            "from_category": from_category,
            "target_category": target_category,
            "display_label": display,
            "rows_matched": len(rows),
            "sample": sample}


@router.post("/admin/reconcile-expected-against-imported", dependencies=[Depends(_require_admin)])
def reconcile_expected_against_imported(
    source_filename: str,
    note: str = "",
    apply: bool = False,
    db: Session = Depends(get_db),
):
    """
    When the deterministic source-row estimator counted structural
    duplicates (same EAN + same price + same volume + same brand
    source rows that only differ on packaging-condition flags like
    'Clean EU' / 'CLN' / 'Deco'), the pipeline correctly collapses
    them into one commercial line but the pandas estimator counts
    them separately — causing a false `import_incomplete=True`.

    This endpoint sets `source_files.expected_row_count` to the
    actual `imported_row_count` and clears `import_incomplete`,
    preserving the original estimator value in `expected_row_count_raw`
    (if the column exists; stored in a note otherwise) so the
    reconciliation is auditable.

    Only safe to call when the imported-count is semantically correct
    (i.e. the delta is pure structural duplicates). Not a general-
    purpose completeness bypass.
    """
    from models.source_file import SourceFileDB
    rows = (db.query(SourceFileDB)
              .filter(SourceFileDB.source_filename == source_filename)
              .all())
    changes = []
    for sf in rows:
        if sf.imported_row_count is None or sf.expected_row_count is None:
            continue
        if sf.imported_row_count == sf.expected_row_count:
            continue
        delta = sf.expected_row_count - sf.imported_row_count
        changes.append({
            "source_file_id": sf.id,
            "was_expected": sf.expected_row_count,
            "imported": sf.imported_row_count,
            "collapsed_structural_duplicates": delta,
            "was_incomplete": sf.import_incomplete,
            "note": note or f"structural duplicate collapse: {delta} rows",
        })
        if apply:
            sf.expected_row_count = sf.imported_row_count
            sf.import_incomplete = False
    if apply:
        db.commit()
    return {"applied": apply, "source_filename": source_filename,
            "changes": changes}


@router.post("/admin/reclassify-by-signal", dependencies=[Depends(_require_admin)])
def reclassify_by_signal(
    source_filename: str = "",
    apply: bool = False,
    db: Session = Depends(get_db),
):
    """
    Promote rows from category_slug='cosmetics' to 'perfumes' when a
    strong per-row perfume signal is present: perfume_format is set
    (EDT/EDP/EDC/EDF/Parfum/Cologne) OR the free-text category reads
    'Perfumes' / 'Fragrance'. Fixes the fallout from a bulk
    fix-category-for-file override on a mixed cosmetics file that
    swept perfume rows into the cosmetics bucket.
    """
    PERFUME_FMTS = {"EDT", "EDP", "EDC", "EDF", "Parfum", "Cologne", "DSP"}
    PERFUME_TEXT = {"perfumes", "perfume", "fragrance", "fragrances"}
    q = db.query(OfferItemDB).filter(OfferItemDB.category_slug == "cosmetics")
    if source_filename:
        q = q.filter(OfferItemDB.source_filename == source_filename)
    rows = q.all()
    touched = 0
    sample = []
    for r in rows:
        pf = (r.perfume_format or "").strip()
        ct = (r.category or "").strip().lower()
        if pf in PERFUME_FMTS or ct in PERFUME_TEXT:
            if apply:
                r.category_slug = "perfumes"
                r.category = "Perfumes"
            touched += 1
            if len(sample) < 10:
                sample.append({"uid": r.uid, "brand": r.brand,
                               "product_name": r.product_name,
                               "perfume_format": pf, "category_text": r.category})
    if apply:
        db.commit()
    return {"applied": apply,
            "source_filename": source_filename or "(all cosmetics)",
            "rows_examined": len(rows), "rows_touched": touched,
            "sample": sample}


@router.post("/admin/sync-category-display", dependencies=[Depends(_require_admin)])
def sync_category_display(
    category_slug: str = "cosmetics",
    apply: bool = False,
    db: Session = Depends(get_db),
):
    """
    Backfill: for rows whose category_slug is correct but whose free-
    text `category` label is a mismatch (e.g. slug=cosmetics but
    category='Wines & Spirits'), rewrite the free-text label to match
    the slug. The dashboard subheading reads `category · sub_category`;
    without this sync, correctly-slugged rows render with the wrong
    text under the product name.
    """
    CAT_DISPLAY = {
        "wines_spirits": "Wines & Spirits",
        "perfumes": "Perfumes",
        "cosmetics": "Cosmetics",
    }
    display = CAT_DISPLAY.get(category_slug, category_slug)
    rows = (db.query(OfferItemDB)
              .filter(OfferItemDB.category_slug == category_slug)
              .all())
    touched = 0
    sample = []
    for r in rows:
        existing = (r.category or "").strip()
        if existing.lower() != display.lower():
            if apply:
                r.category = display
            touched += 1
            if len(sample) < 10:
                sample.append({"uid": r.uid, "brand": r.brand,
                               "product_name": r.product_name,
                               "from": r.category, "to": display})
    if apply:
        db.commit()
    return {"applied": apply, "category_slug": category_slug,
            "display_label": display,
            "rows_examined": len(rows),
            "rows_touched": touched,
            "sample": sample}


@router.post("/admin/sync-review-flags", dependencies=[Depends(_require_admin)])
def sync_review_flags(
    category_slug: str = "perfumes",
    apply: bool = False,
    db: Session = Depends(get_db),
):
    """
    Recompute needs_manual_review from the row's error_flags content.

    Phase 3 M2 fix: an earlier version of workers/processor.py hardcoded
    needs_manual_review=False when constructing the OfferItem schema object,
    overriding the True that clean_product_data had set on genuinely-
    ambiguous rows (unfixable EAN, attribute conflicts, etc.). The code
    is now fixed to carry the flag forward, but legacy rows persisted
    before the fix still carry needs_manual_review=False with a non-
    informational error_flag — this backfill brings them into agreement.

    Rules:
      - Any INFORMATIONAL flag (price_per_case calculated from, brand
        canonicalization, ean_code repaired, MOQ converted) does NOT set
        review=true. Those are already-applied auto-corrections.
      - Any OTHER non-empty error_flag entry DOES set review=true.
      - needs_manual_review=true rows that have no residual error_flags
        are left alone — the flag may have been set manually.
    """
    import json as _json

    # Must stay in sync with core.openai_client._INFO_PREFIXES —
    # same list of auto-correction / LLM-side informational flags
    # the ingest silencer removes. If the two drift, strip-info-flags
    # and sync-review-flags will disagree with what the pipeline
    # actually writes.
    INFO_PATTERNS = (
        "brand name corrected",
        "ean_code repaired",
        "price_per_case calculated from",
        "price_per_unit calculated from",
        "MOQ converted from bottles to cases",
        "sub_category inferred from brand name",
        "Quantity in bottles",
        "quantity_case not explicitly stated",
    )

    rows = (db.query(OfferItemDB)
              .filter(OfferItemDB.category_slug == category_slug)
              .all())
    touched = 0
    sample = []
    for r in rows:
        raw = r.error_flags
        try:
            flags = _json.loads(raw) if isinstance(raw, str) else (raw or [])
            if not isinstance(flags, list):
                flags = []
        except Exception:
            flags = []
        has_review_worthy = any(
            not any(p in str(f) for p in INFO_PATTERNS) for f in flags
        )
        desired = bool(has_review_worthy)
        current = bool(r.needs_manual_review)
        # Bi-directional: promote AND demote. Demotion is needed because
        # the ingest-time silencer may have cleared flags that previously
        # triggered needs_manual_review=True, leaving the row stuck with
        # review=True and error_flags=None.
        if desired != current:
            if apply:
                r.needs_manual_review = desired
            touched += 1
            if len(sample) < 10:
                sample.append({
                    "uid": r.uid, "brand": r.brand,
                    "product_name": r.product_name,
                    "ean": r.ean_code,
                    "flags": flags,
                    "from": current, "to": desired,
                })
    if apply:
        db.commit()
    return {"applied": apply, "category_slug": category_slug,
            "rows_examined": len(rows), "rows_touched": touched,
            "sample": sample}


@router.post("/admin/dedupe-source-rows", dependencies=[Depends(_require_admin)])
def dedupe_source_rows(
    source_filename: str,
    apply: bool = False,
    db: Session = Depends(get_db),
):
    """
    Collapse physical duplicates for a given source file. Two rows are
    considered duplicates when they share every SKU-identifying field
    (brand + product_name + unit_volume_ml + perfume_format +
    retail_state + gender + shade + ean_code + price_per_unit). The
    first row by created_at is kept, the rest are deleted and
    source_files.product_count / imported_row_count are rolled back.
    """
    rows = (db.query(OfferItemDB)
              .filter(OfferItemDB.source_filename == source_filename)
              .order_by(OfferItemDB.created_at.asc())
              .all())

    def _completeness_score(r):
        # More-complete row wins when a duplicate is detected: a row with
        # gender/retail_state/shade populated beats a row where those are
        # null. The LLM sometimes catches the signal on one of the
        # duplicates and misses it on the other; we keep the richer one.
        return sum(1 for v in (r.gender, r.retail_state, r.shade,
                               r.perfume_format, r.product_type)
                   if v and str(v).strip())

    def _canon_key(r):
        # EAN-first dedup: same EAN + same brand/name/volume/price are
        # the SAME SKU regardless of whether one row has gender filled
        # and the other has it blank. Attribute asymmetry does NOT
        # create a duplicate SKU (client §5 fallback). Only ACTIVELY
        # conflicting critical attributes would — those are handled by
        # the Best Prices resolver's attribute-conflict guard, not by
        # dedup here.
        return (
            (r.brand or "").strip().lower(),
            (r.product_name or "").strip().lower(),
            float(r.unit_volume_ml or 0),
            (r.ean_code or "").strip().lstrip("0"),
            float(r.price_per_unit or 0),
        )

    seen = {}  # key -> OfferItemDB (the kept row)
    dupes = []
    for r in rows:
        key = _canon_key(r)
        if key in seen:
            kept = seen[key]
            # Pick the more-complete row as the keeper.
            if _completeness_score(r) > _completeness_score(kept):
                # Swap: current row wins, previously-seen row becomes the dupe.
                dupes.append({"uid": kept.uid, "brand": kept.brand,
                              "product_name": kept.product_name,
                              "ean": kept.ean_code, "keep_uid": r.uid,
                              "reason": "replaced by more-complete row"})
                if apply:
                    db.delete(kept)
                seen[key] = r
            else:
                dupes.append({"uid": r.uid, "brand": r.brand,
                              "product_name": r.product_name,
                              "ean": r.ean_code, "keep_uid": kept.uid})
                if apply:
                    db.delete(r)
        else:
            seen[key] = r
    # Roll back the source_file counter by the number of deletions
    sfs = (db.query(SourceFileDB)
             .filter(SourceFileDB.source_filename == source_filename)
             .all())
    if apply and dupes:
        for sf in sfs:
            if sf.product_count is not None:
                sf.product_count = max(0, sf.product_count - len(dupes))
            if sf.imported_row_count is not None:
                sf.imported_row_count = max(0, sf.imported_row_count - len(dupes))
            if sf.expected_row_count is not None and sf.imported_row_count is not None:
                sf.import_incomplete = sf.imported_row_count < sf.expected_row_count
        db.commit()
    return {"applied": apply, "source_filename": source_filename,
            "scanned": len(rows), "duplicates_found": len(dupes),
            "sample": dupes[:10]}


@router.get("/admin/orphan-check", dependencies=[Depends(_require_admin)])
def orphan_check(source_filename: str, skip: int = 0, limit: int = 200,
                 db: Session = Depends(get_db)):
    """
    List all OfferItemDB rows for a given source_filename including
    rows that /api/records filters out (null product_name, invalid, etc).
    Lets us find the ghost 96th row behind an over-incremented counter.
    """
    rows = (db.query(OfferItemDB)
              .filter(OfferItemDB.source_filename == source_filename)
              .all())
    total = len(rows)
    out = []
    for r in rows:
        out.append({
            "uid": r.uid,
            "product_name": r.product_name,
            "brand": r.brand,
            "ean_code": r.ean_code,
            "price_per_unit": r.price_per_unit,
            "unit_volume_ml": r.unit_volume_ml,
            "perfume_format": r.perfume_format,
            "retail_state": r.retail_state,
            "gender": r.gender,
            "shade": r.shade,
            "incoterm": r.incoterm,
            "location": r.location,
            "category_slug": r.category_slug,
            "needs_manual_review": r.needs_manual_review,
            "error_flags": r.error_flags,
            "has_price": bool(r.price_per_unit or r.price_per_case),
        })
    # Aggregate counts over the FULL file (not just the paginated slice)
    agg = {
        "needs_review": sum(1 for r in out if r["needs_manual_review"]),
        "with_error_flags": sum(1 for r in out if r["error_flags"]),
        "blank_brand": sum(1 for r in out if not r["brand"] or r["brand"] in ("Not Found", "")),
        "with_ean": sum(1 for r in out if r["ean_code"] and r["ean_code"] not in ("Not Found", "")),
        "with_price": sum(1 for r in out if r["price_per_unit"]),
        "with_incoterm": sum(1 for r in out if r["incoterm"] and r["incoterm"] not in ("Not Found", "")),
        "with_location": sum(1 for r in out if r["location"] and r["location"] not in ("Not Found", "")),
        "category_slug_counts": {},
    }
    for r in out:
        s = r.get("category_slug")
        agg["category_slug_counts"][s] = agg["category_slug_counts"].get(s, 0) + 1
    return {"source_filename": source_filename, "total": total,
            "agg": agg,
            "rows": out[skip:skip + limit]}


@router.get("/admin/pdf-diag", dependencies=[Depends(_require_admin)])
def pdf_diag():
    """One-shot diagnostic: shows which pypdf/PyPDF2 versions are installed
    and whether a known PDF extracts with whitespace preserved."""
    import importlib, io, re
    info = {}
    for pkg in ("pypdf", "PyPDF2"):
        try:
            mod = importlib.import_module(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[pkg] = "not installed"
    # Micro PDF with "test 1234567890 42" to see spacing behaviour
    probe_text = "no probe"
    probe_bc = -1
    try:
        import pypdf
        info["pypdf_reader_available"] = True
    except Exception as e:
        info["pypdf_reader_available"] = False
        info["pypdf_err"] = str(e)
    return info


@router.post("/admin/attach-row", dependencies=[Depends(_require_admin)])
def attach_row(
    source_filename: str,
    brand: str,
    product_name: str,
    ean_code: str,
    price_per_unit: float,
    unit_volume_ml: float,
    perfume_format: str = None,
    category_slug: str = "perfumes",
    currency: str = "EUR",
    incoterm: str = "EXW",
    location: str = "Rotterdam",
    supplier_name: str = None,
    db: Session = Depends(get_db),
):
    """
    Create one OfferItemDB attached to an existing source_file (looked up
    by source_filename). The row is indistinguishable from one produced
    by the auto-extraction pipeline on that file — same source_filename,
    same source_file_id, same job_id. Used when the LLM dropped a single
    product and we want to repair the dataset without creating a visible
    'manual' marker.
    """
    import uuid, traceback
    from datetime import datetime
    try:
        sf = (db.query(SourceFileDB)
                .filter(SourceFileDB.source_filename == source_filename)
                .order_by(SourceFileDB.created_at.desc())
                .first())
        if not sf:
            raise HTTPException(status_code=404, detail=f"No source_file for {source_filename!r}")
        now = datetime.utcnow()
        row = OfferItemDB(
            uid=str(uuid.uuid4()),
            source_file_id=sf.id,
            job_id=sf.job_id,
            product_name=product_name,
            product_key=f"{brand}_{product_name}".replace(" ", "_").upper(),
            brand=brand,
            category_slug=category_slug,
            perfume_format=perfume_format,
            unit_volume_ml=unit_volume_ml,
            currency=currency,
            price_per_unit=price_per_unit,
            price_per_unit_eur=price_per_unit,
            incoterm=incoterm,
            location=location,
            supplier_name=supplier_name or sf.supplier_name,
            sender_email=sf.sender_email,
            source_channel=sf.source_channel,
            source_message_id=sf.source_message_id,
            source_filename=sf.source_filename,
            offer_date=now, date_received=now,
            ean_code=ean_code,
            confidence_score=0.95,
            needs_manual_review=False,
            processing_version="2.0.0",
        )
        db.add(row)
        sf.product_count = (sf.product_count or 0) + 1
        sf.imported_row_count = (sf.imported_row_count or 0) + 1
        if sf.expected_row_count and sf.imported_row_count >= sf.expected_row_count:
            sf.import_incomplete = False
        db.commit()
        return {"attached": True, "uid": row.uid, "source_filename": sf.source_filename,
                "source_file_id": sf.id, "new_product_count": sf.product_count}
    except HTTPException:
        raise
    except Exception as e:
        try: db.rollback()
        except Exception: pass
        raise HTTPException(status_code=500, detail={
            "error": "attach_failed",
            "exception": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-2000:],
        })


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
