import uuid
import re
import json
import os
import traceback
import tempfile
import logging
from datetime import datetime, timedelta
from core.openai_health import OpenAICreditsExhausted
from typing import Optional

from core.file_download import resolve_attachment_bytes
from schemas.output import OfferItem
from core.openai_client import extract_offer, extract_from_file, parse_buffer_data
from core.redis_client import redis_manager
from core.excel_client import write_offer_row
from core.database import get_session_factory
from core.normalization import depollute_identity
from models.offer_item import OfferItemDB
from models.source_file import SourceFileDB
from api.admin import _extract_customs_token, _strip_customs_token, _T1T2_PATTERN

logger = logging.getLogger(__name__)


def _received_at(payload) -> datetime:
    """The arrival timestamp stamped at intake (client §6), so a reprocess from
    the saved payload keeps the ORIGINAL receipt time instead of restarting the
    age. Falls back to now() for legacy payloads that carry no stamp."""
    raw = getattr(payload, 'received_at', None)
    if raw:
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            pass
    return datetime.utcnow()


def _deshout_supplier(name: str) -> str:
    """A single all-caps "shout" like ``ELIXEMARKET`` → ``Elixemarket`` for a
    consistent display form. Short acronyms (FBC, QA, LVMH, DFS) and names that
    already carry lowercase letters are left untouched."""
    def _fix(tok: str) -> str:
        if len(tok) >= 5 and tok.isalpha() and tok.isupper():
            return tok[:1] + tok[1:].lower()
        return tok
    return " ".join(_fix(t) for t in name.split())


def _canonical_supplier_name(db, name):
    """Collapse the same commercial supplier onto ONE spelling so it appears
    consistently in the table, offer details and supplier filter (client §1).

    The LLM returns the supplier with varying case per offer (``ELIXEMARKET``
    vs ``Elixemarket``), which otherwise splits one company into two filter
    entries. We reuse an existing DB spelling when one exists (case-insensitive,
    whitespace-insensitive); otherwise we store a de-shouted display form. The
    DB is the source of truth, so the first row of a job sets the canonical
    spelling and every later row — this job or any future one — reuses it."""
    _missing = (None, "", "Not Found")
    if name in _missing:
        return name
    cleaned = re.sub(r'\s+', ' ', str(name)).strip().strip('"\'')
    cleaned = re.sub(r'[\.,;:]+$', '', cleaned).strip()
    if not cleaned or cleaned in _missing:
        return name
    key = cleaned.casefold()
    existing = None
    try:
        from sqlalchemy import func
        existing = (db.query(OfferItemDB.supplier_name)
                      .filter(OfferItemDB.supplier_name.isnot(None))
                      .filter(func.lower(func.trim(OfferItemDB.supplier_name)) == key)
                      .first())
    except Exception as e:  # DB hiccup must not block the write
        logger.warning(f"supplier canonicalisation lookup failed: {e}")
        existing = None
    if existing and existing[0]:
        return existing[0]
    return _deshout_supplier(cleaned)


def save_offer_to_db(offer_dict: dict, job_id: str) -> None:
    """
    Persist a single extracted OfferItem to the PostgreSQL database.
    Non-fatal: logs errors without interrupting the main processing flow.

    SourceFileDB logic:
    - First product of a job → creates the SourceFileDB record.
    - Subsequent products of the same job → reuses the existing SourceFileDB.
    - Same filename in a DIFFERENT job → skipped (already processed).
    """
    try:
        db = get_session_factory()()
        try:
            # Collapse the supplier onto one canonical spelling (client §1 —
            # consistent across table / details / filter) before it is written
            # to BOTH the SourceFileDB and the OfferItemDB row below.
            offer_dict["supplier_name"] = _canonical_supplier_name(
                db, offer_dict.get("supplier_name"))

            # client §5 — drop stale / contradictory notes that no longer hold
            # (e.g. "calculated from price_per_case/units_per_case" when neither
            # is populated; "price ambiguous" alongside "Price on request").
            _ef = offer_dict.get("error_flags")
            if isinstance(_ef, list):
                from core.normalization import reconcile_error_flags
                offer_dict["error_flags"] = reconcile_error_flags(
                    _ef, price_per_case=offer_dict.get("price_per_case"),
                    units_per_case=offer_dict.get("units_per_case"))

            source_filename = offer_dict.get("source_filename") or None
            if source_filename:
                duplicate = db.query(SourceFileDB).filter(
                    SourceFileDB.source_filename == source_filename,
                    SourceFileDB.job_id != job_id
                ).first()
                if duplicate:
                    logger.warning(
                        f"DB: skipping '{offer_dict.get('product_name')}' — "
                        f"file '{source_filename}' already processed in job {duplicate.job_id}"
                    )
                    return
            # ────────────────────────────────────────────────────────────────

            # ── Get or create SourceFileDB for this job ──────────────────────
            source_file = db.query(SourceFileDB).filter(
                SourceFileDB.job_id == job_id
            ).first()

            if not source_file:
                source_file = SourceFileDB(
                    id=str(uuid.uuid4()),
                    job_id=job_id,
                    source_filename=source_filename,
                    sender_name=offer_dict.get("sender_name"),
                    sender_email=offer_dict.get("sender_email"),
                    supplier_name=offer_dict.get("supplier_name"),
                    supplier_email=offer_dict.get("supplier_email"),
                    source_channel=offer_dict.get("source_channel"),
                    source_message_id=offer_dict.get("source_message_id"),
                    product_count=0,
                    # Phase 3 M1: propagate ingest-time metadata
                    content_hash=offer_dict.get("_content_hash"),
                    expected_row_count=offer_dict.get("_expected_row_count"),
                    # client §2 — durable copy of the raw offer content
                    source_text=offer_dict.get("_source_text"),
                )
                db.add(source_file)
                db.flush()  # get the id without committing yet
                logger.info(f"DB: created SourceFileDB for job {job_id}, file='{source_filename}'")
            elif offer_dict.get("_source_text") and not source_file.source_text:
                # Backfill the durable source text if the first row of the job
                # didn't carry it (e.g. legacy ordering).
                source_file.source_text = offer_dict.get("_source_text")
            # ────────────────────────────────────────────────────────────────

            row = OfferItemDB(
                uid=offer_dict.get("uid"),
                source_file_id=source_file.id,
                job_id=job_id,
                product_name=offer_dict.get("product_name"),
                product_name_raw=offer_dict.get("product_name_raw"),
                product_key=offer_dict.get("product_key"),
                brand=offer_dict.get("brand"),
                brand_raw=offer_dict.get("brand_raw"),
                category=offer_dict.get("category"),
                sub_category=offer_dict.get("sub_category"),
                packaging=offer_dict.get("packaging"),
                packaging_raw=offer_dict.get("packaging_raw"),
                bottle_or_can_type=offer_dict.get("bottle_or_can_type"),
                unit_volume_ml=offer_dict.get("unit_volume_ml"),
                units_per_case=offer_dict.get("units_per_case"),
                cases_per_pallet=offer_dict.get("cases_per_pallet"),
                quantity_case=offer_dict.get("quantity_case"),
                quantity_unit=offer_dict.get("quantity_unit"),
                quantity_bottles=offer_dict.get("quantity_bottles"),
                quantity_pallets=offer_dict.get("quantity_pallets"),
                quantity_ftl=offer_dict.get("quantity_ftl"),
                range_name=offer_dict.get("range_name"),
                category_slug=offer_dict.get("category_slug"),
                perfume_format=offer_dict.get("perfume_format"),
                retail_state=offer_dict.get("retail_state"),
                gender=offer_dict.get("gender"),
                product_type=offer_dict.get("product_type"),
                shade=offer_dict.get("shade"),
                size_weight_g=offer_dict.get("size_weight_g"),
                gift_box=offer_dict.get("gift_box"),
                refillable_status=offer_dict.get("refillable_status"),
                currency=offer_dict.get("currency"),
                price_per_unit=offer_dict.get("price_per_unit"),
                price_per_unit_eur=offer_dict.get("price_per_unit_eur"),
                price_per_case=offer_dict.get("price_per_case"),
                price_per_case_eur=offer_dict.get("price_per_case_eur"),
                fx_rate=offer_dict.get("fx_rate"),
                fx_date=offer_dict.get("fx_date"),
                fx_source=offer_dict.get("fx_source"),
                alcohol_percent=offer_dict.get("alcohol_percent"),
                origin_country=offer_dict.get("origin_country"),
                supplier_country=offer_dict.get("supplier_country"),
                incoterm=offer_dict.get("incoterm"),
                location=offer_dict.get("location"),
                lead_time=offer_dict.get("lead_time"),
                moq_cases=offer_dict.get("moq_cases"),
                moq_unit=offer_dict.get("moq_unit"),
                moq_bottles=offer_dict.get("moq_bottles"),
                moq_pallets=offer_dict.get("moq_pallets"),
                moq_ftl=offer_dict.get("moq_ftl"),
                moq_text=offer_dict.get("moq_text"),
                valid_until=offer_dict.get("valid_until"),
                best_before_date=offer_dict.get("best_before_date"),
                vintage=offer_dict.get("vintage"),
                ean_code=offer_dict.get("ean_code"),
                label_language=offer_dict.get("label_language"),
                product_reference=offer_dict.get("product_reference"),
                supplier_name=offer_dict.get("supplier_name"),
                contact_name=offer_dict.get("contact_name"),
                supplier_email=offer_dict.get("supplier_email"),
                supplier_reference=offer_dict.get("supplier_reference"),
                sender_name=offer_dict.get("sender_name"),
                sender_email=offer_dict.get("sender_email"),
                source_channel=offer_dict.get("source_channel"),
                source_message_id=offer_dict.get("source_message_id"),
                source_filename=source_filename,
                attachment_filenames=json.dumps(offer_dict.get("attachment_filenames", [])),
                attachment_count=offer_dict.get("attachment_count"),
                confidence_score=offer_dict.get("confidence_score"),
                needs_manual_review=offer_dict.get("needs_manual_review"),
                error_flags=json.dumps(offer_dict.get("error_flags", [])),
                custom_status=offer_dict.get("custom_status"),
                processing_version=offer_dict.get("processing_version"),
                offer_date=offer_dict.get("offer_date"),
                document_date=offer_dict.get("document_date"),
                document_date_source=offer_dict.get("document_date_source"),
                commercial_conditions=offer_dict.get("commercial_conditions"),
                source_type=offer_dict.get("source_type"),
                date_received=offer_dict.get("date_received"),
                supersedes_ref=offer_dict.get("supersedes_ref"),
            )
            db.add(row)

            # R-07 supersession resolver — same-supplier only. When this row
            # arrives carrying `supersedes_ref="A01"`, back-populate the
            # earlier row's `superseded_by_uid` / `superseded_by_ref` so
            # Best Prices moves A01 from ACTIONABLE CURRENT to HISTORY and
            # the drawer shows the replacement chain. Match by
            # (supplier_name, supplier_reference) to avoid cross-supplier
            # collisions — a Beta "A01" is NOT superseded by an Alpha "C01"
            # that says it replaces A01.
            try:
                sref = (offer_dict.get("supersedes_ref") or "").strip()
                sname = (offer_dict.get("supplier_name") or "").strip()
                if sref and sname:
                    earlier = (
                        db.query(OfferItemDB)
                          .filter(OfferItemDB.supplier_reference == sref)
                          .filter(OfferItemDB.supplier_name == sname)
                          .filter(OfferItemDB.uid != offer_dict.get("uid"))
                          .all()
                    )
                    for prev in earlier:
                        prev.superseded_by_uid = offer_dict.get("uid")
                        prev.superseded_by_ref = offer_dict.get("supplier_reference")
                        logger.info(
                            f"R-07 supersession: {sname}/{sref} ({prev.uid}) "
                            f"superseded by {offer_dict.get('supplier_reference')} "
                            f"({offer_dict.get('uid')})"
                        )
            except Exception as _sup_err:
                logger.warning(f"R-07 supersession resolve failed: {_sup_err}")

            source_file.product_count += 1
            # Phase 3 M1: keep imported_row_count in sync + flag import as
            # incomplete when the LLM said N products should be there but
            # we only persisted M < N (client requirement §1). Recomputed
            # on every save so the last one lands the final truth.
            source_file.imported_row_count = source_file.product_count
            _expected = offer_dict.get("_expected_row_count")
            if _expected and _expected > source_file.product_count:
                source_file.import_incomplete = True
            elif _expected and _expected == source_file.product_count:
                source_file.import_incomplete = False

            db.commit()
            logger.info(f"DB: saved product '{offer_dict.get('product_name')}' for job {job_id}")
        except Exception as db_err:
            db.rollback()
            logger.error(f"DB: failed to save product '{offer_dict.get('product_name')}' for job {job_id}: {db_err}")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"DB: save_offer_to_db error for job {job_id}: {e}")


def _backfill_customs_from_text(safe_data: dict) -> None:
    """Phase 3 M2 — populate custom_status at ingest time when GPT-4o left
    it blank but the T1/T2 token is recoverable from text we still have.

    Three sources, in order:
      1. safe_data['brand']         — token leaked out of the structured
                                      column into the brand field.
      2. safe_data['product_name']  — same, but in the name field.
      3. safe_data['_raw_row_text'] — the raw source-row text that
                                      produced this product, threaded in by
                                      core.openai_client.extract_from_file.
                                      This closes the case-school
                                      B01/B02/B04/B06/B08/B11/B12/B32
                                      rows where GPT-4o cleanly stripped
                                      the T-token from product_name
                                      without echoing it into custom_status.

    When the token is found in brand/product_name, those fields are
    stripped and the pre-strip value is preserved in brand_raw /
    product_name_raw. The raw-row-text source is never written back —
    it's just an extraction lookup, since the raw row has lots of other
    non-brand content that we don't want in the stored fields.

    Mirrors api.admin.retest_backfill_customs_status."""
    if safe_data.get("custom_status"):
        return
    brand = safe_data.get("brand") or ""
    name = safe_data.get("product_name") or ""
    raw  = safe_data.get("_raw_row_text") or ""
    token = (_extract_customs_token(brand)
             or _extract_customs_token(name)
             or _extract_customs_token(raw))
    if not token:
        return
    safe_data["custom_status"] = token
    if brand and _T1T2_PATTERN.search(brand):
        if not safe_data.get("brand_raw"):
            safe_data["brand_raw"] = brand
        safe_data["brand"] = _strip_customs_token(brand)
    if name and _T1T2_PATTERN.search(name):
        if not safe_data.get("product_name_raw"):
            safe_data["product_name_raw"] = name
        safe_data["product_name"] = _strip_customs_token(name)


_INCOTERM_RE = re.compile(r"\b(EXW|DAP|DDP|FOB|CIF|CFR|CNF|CPT|FCA|DAT|DPU|FAS|CIP|EXWORK)\b", re.IGNORECASE)
_LOC_SPLIT_RE = re.compile(r"\s*(?:/|,|\bor\b|\band\b|&)\s*", re.IGNORECASE)
_LOC_TOKEN_RE = re.compile(r"^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ' .\-]{1,30}$")


def _detect_dropped_alt_locations(safe_data: dict) -> None:
    """Compound-location preservation guard (Samuel 2026-10-27 — Jameson).

    Rule 7 says a source line like "Exwork Loendersloot or Revera or Mentrex"
    must land as location = "Loendersloot / Revera / Mentrex". GPT-4o
    occasionally keeps only the first alternative, silently dropping the
    others. Scan the raw source row for the same incoterm-followed-by-many-
    locations pattern; if we parsed one short location but the raw text has
    ≥2 location-shaped tokens joined by /, comma, "or" or "and", add an
    explicit flag so the trader can see the drop and nothing fails silently.
    The flag text names the alternatives found in the raw source.
    """
    raw = safe_data.get("_raw_row_text") or ""
    stored = (safe_data.get("location") or "").strip()
    if not raw or not stored or stored.lower() in ("not found", "", "none"):
        return
    # Already compound → GPT followed Rule 7, nothing to flag.
    if _LOC_SPLIT_RE.search(stored):
        return
    # Find any incoterm occurrence in the raw text and look at the tokens
    # that follow it on the same line. We only care about the FIRST match
    # (one incoterm per Rule 7 branch).
    m = _INCOTERM_RE.search(raw)
    if not m:
        return
    tail = raw[m.end(): m.end() + 200]
    # Stop the tail at the next clear sentence boundary (full stop, pipe,
    # newline, semi-colon, en- or em-dash with surrounding space, or the
    # first digit cluster) so we don't vacuum pack / price / qty data.
    tail = re.split(r"[\.\|\n\r;]|\s[–—-]\s|\s\d", tail, maxsplit=1)[0]
    parts = [p.strip(" .") for p in _LOC_SPLIT_RE.split(tail) if p.strip(" .")]
    # Keep only the parts that LOOK like a location token (letters, spaces,
    # dashes; no digits; short-to-medium length).
    locs = [p for p in parts if _LOC_TOKEN_RE.match(p)]
    # client §5 — a token that itself begins with an incoterm ("DAP Riga") is a
    # SEPARATE delivery option that lands as its own row via the multi-incoterm
    # split (Edo: EXW Germany + DAP Riga), NOT a dropped alternative of THIS
    # row. Excluding it stops the false "only X persisted — re-ingest to recover
    # alternatives" note when both options are in fact present as sibling rows.
    locs = [p for p in locs if not _INCOTERM_RE.match(p.strip())]
    if len(locs) < 2:
        return
    # Case-fold match so "LOENDERSLOOT" and "Loendersloot" count as the
    # same stored anchor; the "others" we flag are everything else.
    stored_lc = stored.lower()
    others = [p for p in locs if p.lower() != stored_lc]
    if not others:
        return
    flags = safe_data.get("error_flags") or []
    if not isinstance(flags, list):
        flags = []
    all_locs = ", ".join([stored] + others)
    msg = (f"source lists {len(locs)} alternative locations ({all_locs}); "
           f"only {stored} persisted — re-ingest to recover alternatives")
    # Don't double-add if the flag is already present.
    if not any(isinstance(f, str) and "alternative locations" in f for f in flags):
        flags.append(msg)
    safe_data["error_flags"] = flags


# R-07 supersession parser. The source may state explicitly that a new
# commercial offer replaces an older one, e.g.
#   "New supplier offer replaces our earlier A01 price for this commercial product."
#   "This offer supersedes our previous B07 price."
# We extract the referenced supplier_reference and store it in
# `safe_data['supersedes_ref']` so save_offer_to_db can back-populate the
# earlier row's `superseded_by_uid`. Match is case-insensitive and tolerates
# "our earlier", "our previous", "the earlier/previous", or no article.
_SUPERSEDES_RE = re.compile(
    r"\b(?:replaces|supersedes|supercedes|superseding|replacing)\b"
    r"(?:\s+(?:our|the))?"
    r"(?:\s+(?:earlier|previous|prior|former|old))?"
    r"\s+([A-Z]\w{0,15}\d+[A-Z]?)\b",
    re.IGNORECASE,
)


def _parse_supersedes_ref(safe_data: dict) -> None:
    """Scan the raw source row for an explicit "replaces X" statement and
    record the referenced supplier reference on this row. Never fabricate —
    only extract literal tokens that match the supplier_reference shape."""
    if safe_data.get("supersedes_ref"):
        return  # already set upstream
    raw = safe_data.get("_raw_row_text") or ""
    if not raw:
        return
    m = _SUPERSEDES_RE.search(raw)
    if not m:
        return
    ref = m.group(1).strip()
    # Never claim a row supersedes itself.
    own = (safe_data.get("supplier_reference") or "").strip()
    if ref and ref.lower() == own.lower():
        return
    safe_data["supersedes_ref"] = ref


def is_valid_offer(offer_dict: dict) -> bool:
    """Strict validation for extracted offers"""
    if not offer_dict:
        return False

    name = offer_dict.get('product_name')
    if not name or name in ["Not Found", "Unknown", "Row", ""]:
        return False

    name_lc = name.lower().strip()
    if name_lc.startswith('row ') or name_lc.startswith('missing row '):
        return False
    price_unit = offer_dict.get('price_per_unit')
    price_case = offer_dict.get('price_per_case')
    unit_is_empty = price_unit is None or price_unit == 0
    case_is_empty = price_case is None or price_case == 0
    if unit_is_empty and case_is_empty:
        # Samuel case-school B31: an explicitly stated €0.00 is a signal to
        # retain + REVIEW, not drop. clean_product_data tags such rows with
        # the "Zero price retained" flag; keep them so the trader can see
        # them (benchmarks and Best Prices already exclude zero prices).
        flags = offer_dict.get('error_flags') or []
        if isinstance(flags, list) and any(
                isinstance(f, str) and (
                    'Zero price retained' in f
                    or 'Price on request retained' in f
                    # A bare '@'-price (Beefeater 70cl @€75) has a quoted amount
                    # with an unknown basis — keep it as-is, never relabel it
                    # "Price on request — no price stated" (Samuel 2026-11-10 r3).
                    or 'basis unspecified' in f
                    or 'basis ambiguous' in f
                    # Shared-lot delivered-term option we deliberately blanked
                    # (Yamazaki DAP: same lot as the EXW row, delivered price not
                    # confirmed). Keep it visible as a TBC delivery option rather
                    # than dropping it for having no price/qty (Samuel 2026-11-12).
                    or 'delivered price TBC' in f
                    # Same-lot SECONDARY option blanked by shape-fix (2b) — the
                    # mirror case (Moët: EXW "also collected" option, price not
                    # confirmed). Same rationale: keep it visible, never drop.
                    or 'price not confirmed (TBC)' in f
                )
                for f in flags):
            # A "price on request" row must never read €0 on the dashboard — null a
            # stray 0 so it shows "on request" (Samuel: never €0). A genuinely stated
            # €0.00 keeps its own "Zero price retained" semantics elsewhere.
            if not any(isinstance(f, str) and 'Zero price retained' in f for f in flags):
                for _pk in ('price_per_unit', 'price_per_case',
                            'price_per_unit_eur', 'price_per_case_eur'):
                    if offer_dict.get(_pk) in (0, 0.0):
                        offer_dict[_pk] = None
            return True
        # No price, but keep the offer when it carries another commercial signal
        # — a quantity, an EAN or a supplier reference. Suppliers legitimately
        # list stock "price on request" (Samuel ElixeMarket: Havana 3,200 cs,
        # Bacardi 1,000 cs, JD & Coca-Cola 3,000 cs — all price-less). Benchmarks
        # and Best Prices already exclude null/zero prices, so these appear in
        # the catalogue without skewing any comparison.
        has_qty = any(offer_dict.get(k) for k in (
            'quantity_case', 'quantity_bottles', 'quantity_pallets', 'quantity_ftl'))
        has_ref = bool(offer_dict.get('ean_code') or offer_dict.get('supplier_reference')
                       or offer_dict.get('product_reference'))
        if has_qty or has_ref:
            # Price is absent (not an explicit €0.00 — that path returned above):
            # store null, never 0.0, so the dashboard shows "on request".
            for _pk in ('price_per_unit', 'price_per_case',
                        'price_per_unit_eur', 'price_per_case_eur'):
                if offer_dict.get(_pk) in (0, 0.0):
                    offer_dict[_pk] = None
            if isinstance(flags, list):
                _m = 'Price on request — no price stated in source'
                if _m not in flags:
                    flags.append(_m)
                    offer_dict['error_flags'] = flags
            offer_dict['needs_manual_review'] = True
            return True
        logger.debug(f"Offer {name} rejected: no price and no quantity/reference.")
        return False

    return True


def _safe_float(value, default=None):
    if value is None or value == "" or value == "Not Found":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _is_placeholder_supplier(value, source_filename) -> bool:
    """A supplier that is really the source filename or a test marker, not a
    trading partner. Bulk-uploaded catalogs (stock lists / price lists) with no
    per-line supplier were landing with a filename-stem value like
    "STOCK_LIST_PERFUMES" / "W39_Designer" / "Cosmetics_18_06", or a test label
    like "L'Oreal Luxury Test". Those must never be stored as a supplier.

    Signals (kept conservative so real names are never rejected):
      - the value ends with a "test" token;
      - the value contains an underscore AND its tokens are a subset of the
        source filename's tokens — real supplier names use spaces, not
        underscores, so this targets the mechanical filename-stem placeholders
        without catching e.g. "FBC Trades" from "fbc_trades_email.txt".
    """
    if not value:
        return False
    v = str(value).strip()
    low = v.lower()
    if low == "test" or low.endswith(" test"):
        return True
    if "_" in v and source_filename:
        stem = os.path.splitext(os.path.basename(str(source_filename)))[0]
        toks = lambda s: {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t}
        vt, ft = toks(v), toks(stem)
        if vt and vt <= ft:
            return True
    return False


# A trailing offer-reference glued onto the supplier name, e.g.
# "Cedar Bay QA Trading — ref. QA-20261107-W07". The supplier field must hold
# ONLY the trading name; the reference belongs in supplier_reference
# (Samuel Cedar Bay, 2026-11-07).
_SUPPLIER_REF_SPLIT_RE = re.compile(
    r'\s*[—–\-|,;:]*\s*(?:ref(?:erence)?|our\s*ref|offer\s*ref)\.?\s*[:.#]?\s*'
    r'([A-Za-z0-9][A-Za-z0-9/_\-]{2,})\s*$', re.IGNORECASE)


def _supplier_name_and_ref(raw):
    """('Cedar Bay QA Trading — ref. QA-20261107-W07') ->
    ('Cedar Bay QA Trading', 'QA-20261107-W07'). Conservative: splits only a
    trailing ref-marker followed by a token that contains a digit, so an
    ordinary trading name is never truncated."""
    if not raw:
        return (raw, None)
    s = str(raw).strip()
    m = _SUPPLIER_REF_SPLIT_RE.search(s)
    if m and re.search(r'\d', m.group(1)):
        name = s[:m.start()].strip(' .,-—–|;:')
        if name:
            return (name, m.group(1).strip(' .,-'))
    return (s, None)


def _resolve_supplier_name(ai_name, payload) -> str:
    """Return the best available supplier name.

    Priority:
      1. payload.supplier_name  — Make extracted this from email body / WhatsApp text
      2. ai_name                — AI extracted from attached file (PDF/Excel)
      3. "Not Found"            — sender info is NOT used (sender ≠ supplier)

    A candidate that is really the source filename or a test marker is skipped
    (see _is_placeholder_supplier) so a catalog filename never poses as a supplier.
    A trailing offer reference is stripped (see _supplier_name_and_ref).
    """
    _missing = [None, "", "Not Found"]
    src = getattr(payload, 'source_filename', None)
    pname = getattr(payload, 'supplier_name', None)
    if pname not in _missing and not _is_placeholder_supplier(pname, src):
        return _supplier_name_and_ref(str(pname))[0]
    if ai_name not in _missing and not _is_placeholder_supplier(ai_name, src):
        return _supplier_name_and_ref(str(ai_name))[0]
    # Fallback: derive the supplier from the message body / subject at PROCESS
    # time. Covers (a) reprocessed jobs whose saved payload predates improved
    # detection, and (b) forwarded emails where the original supplier sits in
    # the body ("De: FBC Trades <…>") — distinct from the person forwarding.
    try:
        from core.ingest_service import detect_supplier_from_text
        derived = detect_supplier_from_text(getattr(payload, 'text_body', None),
                                            getattr(payload, 'subject', None))
        if derived and not _is_placeholder_supplier(derived, src):
            return _supplier_name_and_ref(str(derived))[0]
    except Exception:
        pass
    return "Not Found"


def _resolve_supplier_ref(ai_name, payload):
    """The offer reference glued onto the supplier string, taken from the SAME
    source _resolve_supplier_name used for the name. Needed because the name is
    cleaned (ref stripped) before the ref-storage step runs, so by then the
    reference is gone from the resolved name (Samuel Cedar Bay: the body carries
    "Cedar Bay QA Trading — ref. QA-20261107-W07"). Returns None if there is none."""
    _missing = [None, "", "Not Found"]
    src = getattr(payload, 'source_filename', None)
    pname = getattr(payload, 'supplier_name', None)
    if pname not in _missing and not _is_placeholder_supplier(pname, src):
        return _supplier_name_and_ref(str(pname))[1]
    if ai_name not in _missing and not _is_placeholder_supplier(ai_name, src):
        return _supplier_name_and_ref(str(ai_name))[1]
    try:
        from core.ingest_service import detect_supplier_from_text
        derived = detect_supplier_from_text(getattr(payload, 'text_body', None),
                                            getattr(payload, 'subject', None))
        if derived and not _is_placeholder_supplier(derived, src):
            _r = _supplier_name_and_ref(str(derived))[1]
            if _r:
                return _r
    except Exception:
        pass
    # Fallback: the reference may sit in the body text beside the supplier line
    # even when the LLM already extracted a clean supplier_name without it
    # (Samuel Cedar Bay WhatsApp: "Offer from Cedar Bay QA Trading — ref.
    # QA-20261107-W07"). Conservative: a "ref" marker followed by a token with a digit.
    try:
        body = getattr(payload, 'text_body', None) or ''
        m = re.search(
            r'(?:^|[—–\-|,;:(\s])(?:ref(?:erence)?|our\s*ref|offer\s*ref)\.?\s*[:.#]?\s*'
            r'([A-Za-z0-9][A-Za-z0-9/_\-]{2,})', body, re.IGNORECASE)
        if m and re.search(r'\d', m.group(1)):
            return m.group(1).strip(' .,-')
    except Exception:
        pass
    return None


def _resolve_supplier_email(payload, ai_email=None) -> str:
    """Return the best available supplier email.

    Priority:
      1. payload.supplier_email — Make extracted this from email/WA metadata
      2. ai_email                — AI extracted from attached file (PDF/Excel)
      3. "Not Found"
    """
    _missing = [None, "", "Not Found"]

    def _ok(e):
        # Reject reserved test/documentation domains (RFC 2606/6761): a
        # verification@loxo.test or test@loreal.example sender is not a supplier.
        return e not in _missing and not re.search(r"\.(test|example|invalid|localhost)$",
                                                    str(e).strip().lower())

    pemail = getattr(payload, 'supplier_email', None)
    if _ok(pemail):
        return str(pemail)
    if _ok(ai_email):
        return str(ai_email)
    return "Not Found"


# Approximate fixed fallback exchange rates (used only when fx_rate is absent)
_FALLBACK_FX = {
    "USD": 0.92,   # 1 USD ≈ 0.92 EUR
    "GBP": 1.17,   # 1 GBP ≈ 1.17 EUR
    "CHF": 1.05,   # 1 CHF ≈ 1.05 EUR
    "EUR": 1.0,
}

# Currency symbol → ISO code mapping
_CURRENCY_SYMBOLS = {
    "€": "EUR",
    "$": "USD",
    "£": "GBP",
    "₣": "CHF",
}


_PRODUCT_LINE_RE = re.compile(
    r"(?mi)^\s*(?:Product|Product\s+name|Item|Description|Désignation|Designation)"
    r"\s*[:\-]\s*(.+?)\s*$"
)


def _preserve_raw_identity(safe_data: dict) -> None:
    """R-16 (Samuel 2026-10-30, extended 2026-10-31): product_name_raw must
    hold the EXACT source description — not the LLM's post-cleanup
    product_name. Samuel reviewed v3 and found that B05's raw dropped the
    "Foundation" suffix and B09's dropped the "Test" prefix because we were
    copying the LLM's already-normalised product_name into _raw.

    Strategy: parse the "Product:" line directly out of _raw_row_text (the
    unmodified source block, threaded in by openai_client.extract_offer).
    That line is literally what the supplier wrote. We only fall back to
    the LLM's product_name when the raw block has no "Product:" marker
    (e.g. Excel rows where columns were split out before extraction).

    brand_raw stays on the LLM-pass behaviour because the source files
    don't have a dedicated Brand: line — the brand is inside the product
    name, so the LLM's brand split is already our best source of truth.
    """
    raw = safe_data.get("_raw_row_text") or ""
    source_line = None
    if raw:
        m = _PRODUCT_LINE_RE.search(raw)
        if m:
            source_line = m.group(1).strip()
            # Collapse internal whitespace runs so a tab inside the raw
            # block doesn't make the stored value look doctored.
            source_line = re.sub(r"\s+", " ", source_line)
    if source_line:
        # Overwrite even if a prior pass filled it — the source line is the
        # authoritative raw text, by Samuel's acceptance criterion.
        safe_data["product_name_raw"] = source_line
    elif not safe_data.get("product_name_raw"):
        pn = safe_data.get("product_name")
        if pn and pn != "Not Found":
            safe_data["product_name_raw"] = pn
    if not safe_data.get("brand_raw"):
        br = safe_data.get("brand")
        if br and br != "Not Found":
            safe_data["brand_raw"] = br


def _drop_misleading_multi_incoterm_flag(safe_data: dict) -> None:
    """Samuel 2026-10-30: a fresh Jameson test rendered a "multiple
    incoterms detected — row duplicated" flag even though the stored row
    carries one incoterm. GPT occasionally attaches that flag to a Rule-7
    compound-location row (one incoterm, several locations) because the
    source line listed many places. The fix is a server-side guard: if
    the raw source row contains at most one distinct incoterm token, the
    "multiple incoterms" flag is a misread and we drop it. The row
    still keeps every other flag the LLM emitted.
    """
    raw = (safe_data.get("_raw_row_text") or "").upper()
    flags = safe_data.get("error_flags") or []
    if not isinstance(flags, list) or not flags:
        return
    distinct = set(_INCOTERM_RE.findall(raw))
    distinct = {t.upper() for t in distinct}
    if len(distinct) >= 2:
        return

    def _is_multi_incoterm_flag(f) -> bool:
        if not isinstance(f, str):
            return False
        fl = f.lower()
        if "incoterm" not in fl:
            return False
        return any(w in fl for w in ("multiple", "several", "mixed",
                                     "many", "various"))

    kept = [f for f in flags if not _is_multi_incoterm_flag(f)]
    if len(kept) != len(flags):
        safe_data["error_flags"] = kept


def _snap_duplicated_row_per_unit(safe_data: dict) -> None:
    """When a row carries the Rule-6 "multiple incoterms detected — row
    duplicated" flag, Rule 6 says all sibling rows must be identical except
    for incoterm/location — which includes price. GPT occasionally emits
    `price_per_unit` with a stray trailing digit on one sibling (7.754999…)
    and a cleanly-rounded one on another (7.755), which the dashboard then
    renders as "€7.75" vs "€7.76" even though both carry the SAME
    price_per_case. Snap price_per_unit = price_per_case / units_per_case
    so every sibling of a split-by-incoterm group displays identically.
    (Samuel 2026-10-27 — two Jameson rows €7.75 vs €7.76, same €46.53/case.)
    """
    flags = safe_data.get("error_flags") or []
    if not isinstance(flags, list):
        return
    if not any(isinstance(f, str) and "multiple incoterms detected" in f and
               "row duplicated" in f for f in flags):
        return
    pc = safe_data.get("price_per_case")
    upc = safe_data.get("units_per_case")
    try:
        pc_f = float(pc) if pc not in (None, "") else None
        upc_f = float(upc) if upc not in (None, "") else None
    except (TypeError, ValueError):
        return
    if pc_f and upc_f and upc_f > 0:
        safe_data["price_per_unit"] = round(pc_f / upc_f, 4)


def _normalize_currency_and_prices(safe_data: dict) -> dict:
    """Detect currency from sign characters and populate EUR price fields.

    Logic:
    1. If currency is already a known non-EUR code (USD/GBP/…), convert to EUR.
    2. If currency still looks like a symbol (€/$), resolve to ISO code.
    3. Populate price_per_unit_eur and price_per_case_eur:
       - If currency == EUR: EUR fields = native price fields.
       - Otherwise: EUR = native * fx_rate (using AI-provided fx_rate or fallback)
    """
    currency = (safe_data.get('currency') or 'EUR').strip()

    # Resolve symbol to ISO code
    if currency in _CURRENCY_SYMBOLS:
        currency = _CURRENCY_SYMBOLS[currency]

    # Normalise to uppercase
    currency = currency.upper()
    if currency not in _FALLBACK_FX:
        currency = 'EUR'  # Unknown symbols → default EUR

    safe_data['currency'] = currency

    # Determine fx_rate: use AI-provided rate, or if it's 1.0/invalid, use fallback
    ai_fx = safe_data.get('fx_rate') or 1.0
    try:
        ai_fx = float(ai_fx)
    except (TypeError, ValueError):
        ai_fx = 1.0

    # If AI didn't provide a meaningful rate for non-EUR, use table fallback.
    # Record the PROVENANCE of whichever rate wins so Samuel (R-09) can
    # reconcile EUR amounts against a known FX source: either the source
    # document (ai_fx != 1.0 on a non-EUR row) or the static fallback table.
    if currency == 'EUR':
        fx = 1.0
        fx_source = 'identity-EUR'
    elif ai_fx != 1.0:
        fx = ai_fx
        fx_source = 'source-document'
    else:
        # Samuel 2026-11-11: no more silent static rate. Fetch the ECB daily
        # reference rate for the offer's date, record it + its effective date +
        # provider. A static approximation is used ONLY if the live lookup
        # fails, and is flagged so the EUR figure is never shown as exact.
        from core.fx_rates import get_eur_rate
        _on = safe_data.get('offer_date') or safe_data.get('date_received')
        fx, _fx_eff, fx_source, _fx_fallback = get_eur_rate(currency, _on)
        if _fx_eff and not safe_data.get('fx_date'):
            safe_data['fx_date'] = _fx_eff
        if _fx_fallback:
            _fl = safe_data.get('error_flags')
            if isinstance(_fl, list):
                _m = (f'FX {currency}→EUR is an approximate static fallback '
                      f'(live ECB rate unavailable) — EUR figures indicative, not exact')
                if _m not in _fl:
                    _fl.append(_m)
    safe_data['fx_rate'] = fx
    # Only overwrite fx_source if the pipeline hasn't already set one upstream
    # (keeps future ECB-daily or dated-rate injections intact).
    if not safe_data.get('fx_source'):
        safe_data['fx_source'] = fx_source
    # fx_date: left to upstream (source document) or null. Not fabricated here;
    # Samuel R-06 explicitly forbids inventing dates.

    pu = safe_data.get('price_per_unit')
    pc = safe_data.get('price_per_case')
    upc = safe_data.get('units_per_case')

    def _conv(v):
        if v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return round(v * fx, 4) if currency != 'EUR' else round(v, 4)

    safe_data['price_per_unit_eur'] = _conv(pu)
    safe_data['price_per_case_eur'] = _conv(pc)

    # ─── EUR INTERNAL ARITHMETIC CONSISTENCY ────────────────────────────
    # Samuel 2026-10-29: a USD row with ppc=323, ppu=26.9167 (= 323/12
    # rounded), fx=0.92 was giving ppc_eur=297.16 but ppu_eur=24.7634.
    # Inside EUR: 297.16/12 = 24.76333 → 24.7633, so the stored EUR
    # columns drifted 0.0001 against each other because ppu_eur chained
    # TWO roundings (ppc/units → 4dp, then ×fx → 4dp again).
    # When:
    #   (a) ppu looks derived (within 0.001 of ppc/units — i.e. the LLM
    #       just divided the case price), and
    #   (b) a positive ppc_eur and a sane units_per_case exist,
    # recompute ppu_eur from the EUR case price in one rounding step so
    # the two EUR columns are mathematically consistent. Explicit per-
    # unit source values (where ppu ≠ ppc/units — e.g. a case discount)
    # are left as the straight fx conversion so the explicit source
    # price is preserved verbatim.
    try:
        _pu_f = float(pu) if pu is not None else None
        _pc_f = float(pc) if pc is not None else None
        _upc_f = float(upc) if upc is not None else None
        _ppc_eur = safe_data.get('price_per_case_eur')
        if (_pu_f is not None and _pc_f is not None and _upc_f and _upc_f > 1
                and _ppc_eur and _ppc_eur > 0):
            _derived_pu_native = _pc_f / _upc_f
            # If LLM-emitted ppu is within rounding tolerance of ppc/units,
            # the per-unit figure wasn't quoted — it was computed. Keep EUR
            # internally consistent by deriving from ppc_eur in one step.
            if abs(_pu_f - _derived_pu_native) <= 0.001:
                _new_ppu_eur = round(float(_ppc_eur) / _upc_f, 4)
                _old_ppu_eur = safe_data['price_per_unit_eur']
                if _new_ppu_eur != _old_ppu_eur:
                    safe_data['price_per_unit_eur'] = _new_ppu_eur
                    logger.info(
                        f"[_normalize_currency_and_prices] EUR consistency: "
                        f"price_per_unit_eur {_old_ppu_eur} → {_new_ppu_eur} "
                        f"(derived from {_ppc_eur} / {_upc_f}; single-step "
                        f"rounding; ppu was derived in native currency)"
                    )
    except (TypeError, ValueError) as _eur_e:
        logger.warning(f"[_normalize_currency_and_prices] EUR consistency skipped: {_eur_e}")
    # ──────────────────────────────────────────────────────────────────

    return safe_data


# ─── incoterm / location separation ───────────────────────────────────────────
_VALID_INCOTERMS = {"EXW", "FOB", "CIF", "CFR", "CNF", "DAP", "DDP", "CPT",
                    "FCA", "DAT", "DPU", "FAS", "CIP"}
_LOAD_TYPES = {"FTL", "LTL", "FCL", "LCL"}
_INCO_PREFIX_RE = re.compile(
    r"^\s*(EXW|EXWORKS?|EX|CNF|CFR|CIF|FOB|DAP|DDP|CPT|FCA|CIP|DPU|FAS|DAT)\b[\s:.\-]*(.*)$",
    re.IGNORECASE)


def _normalize_incoterm_location(safe_data: dict) -> None:
    """Keep incoterm, load-type and location in their own fields (Samuel FBC):
      * FTL/LTL/FCL/LCL are LOAD types, not incoterms — never store them as the
        incoterm (FTL also implies a full-truck quantity).
      * a location that carries an incoterm prefix ("CNF Riga", "Ex Loen",
        "EXW Loendersloot") is split into incoterm + place.
      * "Ex"/"Exworks" normalise to EXW.
    """
    inco = str(safe_data.get("incoterm") or "").strip()
    inco_u = inco.upper()
    if inco_u in _LOAD_TYPES:
        if inco_u == "FTL" and not safe_data.get("quantity_ftl"):
            safe_data["quantity_ftl"] = 1
        safe_data["incoterm"] = None
        inco_u = ""
    elif inco_u in ("EX", "EXWORK", "EXWORKS"):
        safe_data["incoterm"] = "EXW"
        inco_u = "EXW"

    loc = str(safe_data.get("location") or "").strip()
    if loc:
        m = _INCO_PREFIX_RE.match(loc)
        if m:
            tok = m.group(1).upper()
            tok = "EXW" if tok in ("EX", "EXWORK", "EXWORKS") else tok
            if not safe_data.get("incoterm"):
                safe_data["incoterm"] = tok
            rest = m.group(2).strip(" .,-")
            safe_data["location"] = rest or None


# ─── Round-4 deterministic offer-shape fixes (Samuel 2026-11-11) ────────────
_R4_INCO = r'EXW|EXWORKS?|DAP|DDP|CIF|CFR|CNF|FOB|FCA|CPT|CIP|DPU|FAS'
# A place is 1–3 Capitalised tokens. The leading letter is forced case-SENSITIVE
# with a scoped inline flag (?-i:…) so that under the pattern's overall
# re.IGNORECASE a following lowercase word ("in around 6 weeks") is NOT swallowed
# into the location.
_R4_PLACE = r'(?-i:[A-ZÀ-Ý])[\wÀ-ÿ.\-]*(?:\s+(?-i:[A-ZÀ-Ý])[\wÀ-ÿ.\-]*){0,2}'
# "EXW Bordeaux or DAP Loendersloot" — two delivery terms, each its OWN place.
# Tight (requires an explicit disjunction) so a multi-product email's separate
# per-row terms never match and get cross-wired.
_R4_DISJ_RE = re.compile(
    r'\b(' + _R4_INCO + r')\s+(' + _R4_PLACE + r')\s*(?:\bor\b|/|,|&|\band\b)\s*'
    r'(' + _R4_INCO + r')\s+(' + _R4_PLACE + r')', re.IGNORECASE)
_R4_MOQ_FULLLOT_RE = re.compile(
    r'MOQ\b[^A-Za-z0-9]{0,4}(full\s+lot|entire\s+lot|whole\s+lot|full\s+load)', re.IGNORECASE)
_R4_MOQ_PALLETS_RE = re.compile(r'MOQ\b[^0-9A-Za-z]{0,6}(\d+(?:[.,]\d+)?)\s*pallets?', re.IGNORECASE)
_R4_DEPOSIT_RE = re.compile(
    r'deposit\s+confirmation\s+required|deposit\s+confirmation|deposit\s+required', re.IGNORECASE)
_R4_EXW_BASIS_RE = re.compile(
    r'unless\s+otherwise\s+stated|ex[\s\-]?works|\bex[\s\-](?:cellars?|warehouse)\b', re.IGNORECASE)
_R4_DELIVERED = {'DAP', 'DDP', 'CIF', 'CFR', 'CNF', 'CIP', 'CPT', 'DPU'}
# Samuel 2026-11-12: a single lot quoted on a PRIMARY priced delivery term with
# a SECONDARY "same stock can also be collected/delivered <INCO> <place>" option
# whose price the source says to confirm — the MIRROR of (2) where the priced
# term is the delivered one and the ex-works option is the unconfirmed
# alternative (Moët: "EUR 171.00/case, DAP Antwerp. Same stock can also be
# collected EXW Breda; please confirm whether the quoted price is the same for
# both options."). The LLM copies the primary price onto the secondary row → two
# priced rows for ONE lot. _R4_SAMELOT_ALT_RE captures the SECONDARY incoterm
# (the one after "also collected/delivered"); _R4_PRICE_UNCONFIRMED_RE is the
# explicit "price is not (yet) confirmed" cue that gates the blanking.
_R4_SAMELOT_ALT_RE = re.compile(
    r'\b(?:same|this)\s+(?:stock|lot|goods?|products?)\b.{0,80}?'
    r'(?:can\s+)?(?:also\s+)?(?:be\s+)?'
    r'(?:collected|delivered|picked\s*up|shipped|supplied|available|offered)\b'
    r'.{0,25}?\b(' + _R4_INCO + r')\b',
    re.IGNORECASE | re.DOTALL)
_R4_PRICE_UNCONFIRMED_RE = re.compile(
    r'confirm\s+whether\s+the\s+(?:quoted\s+)?price|'
    r'\bprice\b[^.\n]{0,40}\b(?:to\s+be\s+confirmed|tbc|not\s+confirmed|'
    r'same\s+for\s+both)\b|'
    r'please\s+confirm\b[^.\n]{0,40}\bprice\b',
    re.IGNORECASE)


def _block_for_ref(text: str, ref: str) -> str:
    """Return ONE reference's own block from the document text — from the
    reference token up to the next reference / 'REF' boundary. Lets the per-row
    MOQ / deposit scans work when the LLM product carries no `_raw_row_text`
    (attachment / vision extractions frequently don't), by giving each row the
    exact slice of source that belongs to it (Elixe 'MOQ · Full lot' / 'Deposit
    confirmation required' sit under their own OF-ref, Samuel 2026-11-11)."""
    ref = str(ref or '').strip()
    if not ref or not text or ref not in text:
        return ''
    i = text.index(ref)
    tail = text[i + len(ref):]
    m = re.search(r'(OF-\d{3,}-\d{3,}|ELIXEMARKET\s+REF|\bREF\b\s*[·:])',
                  tail, re.IGNORECASE)
    return ref + (tail[:m.start()] if m else tail[:600])


# ── Multi-currency retail reconciliation (CHELSEA HK, Samuel 2026-11-11) ──────
# HK perfume stock sheets quote BOTH an HKD and a USD price plus a stock Qty
# column. The LLM mis-maps them: the USD value lands in price_per_unit but
# labelled EUR with no conversion, the HKD value is mistaken for the quantity,
# and the real Qty is dropped. These parse the row text and set it right
# deterministically (memory: prefer deterministic reconciliation over
# extract_offer prompt nudges).
_MC_HKD_RE = re.compile(r'(?:price_hkd\s*=\s*|HKD\s*|HK\$\s*)([\d][\d,]*(?:\.\d+)?)', re.IGNORECASE)
_MC_USD_RE = re.compile(r'(?:price_usd\s*=\s*|USD\s*|US\$\s*)([\d][\d,]*(?:\.\d+)?)', re.IGNORECASE)
_MC_QTY_RE = re.compile(r'(?:qty|quantity|数量)\s*[=:]?\s*([\d][\d,]*)\b', re.IGNORECASE)


def _reconcile_multicurrency_retail(safe_data: dict) -> None:
    """A HK perfume/cosmetics stock sheet quotes BOTH an HKD and a USD price plus
    a Qty column; the LLM mis-maps them (USD value in price but labelled EUR, HKD
    mistaken for quantity, Qty dropped). USD is the tradeable price. Runs BEFORE
    FX normalisation so USD→EUR is converted by the ECB module.

    Currency + EUR conversion are DOCUMENT-level robust — they fire for every row
    of a dual-currency sheet, because the vision transcription does NOT reliably
    label or barcode every line (so a per-row-only approach left rows as EUR with
    no conversion). The HKD list-price note and the stock-Qty recovery stay
    best-effort from the row's own transcribed line."""
    raw = safe_data.get('_raw_row_text') or ''
    if not raw:
        return
    # Dual-currency HK sheet? (the chunk mentions BOTH HKD and USD somewhere)
    if not (_MC_HKD_RE.search(raw) and _MC_USD_RE.search(raw)):
        return

    def _num(s):
        try:
            return float(str(s).replace(',', ''))
        except (TypeError, ValueError):
            return None

    # Best-effort: isolate THIS row's own line (by barcode, else product name)
    # for the row's exact HKD/USD/Qty. _raw_row_text is the WHOLE image chunk
    # (openai_client sets _p.setdefault('_raw_row_text', chunk)).
    line = raw if '\n' not in raw else None
    if line is None:
        ean = re.sub(r'\s', '', str(safe_data.get('ean_code') or ''))
        if ean:
            for ln in raw.splitlines():
                if ean in re.sub(r'\s', '', ln):
                    line = ln
                    break
        if line is None:
            _tk = [t for t in re.findall(r'[A-Za-zÀ-ÿ]{4,}',
                                        str(safe_data.get('product_name') or ''))][:2]
            if _tk:
                for ln in raw.splitlines():
                    if all(t.lower() in ln.lower() for t in _tk):
                        line = ln
                        break

    usd = hkd = None
    if line is not None:
        _us = _MC_USD_RE.search(line)
        _hk = _MC_HKD_RE.search(line)
        usd = _num(_us.group(1)) if _us else None
        hkd = _num(_hk.group(1)) if _hk else None
    # Fall back to the value the LLM already placed in price_per_unit — on these
    # sheets that is reliably the USD column (prices verified 17/17 against source).
    if usd is None:
        usd = _safe_float(safe_data.get('price_per_unit'))
    if usd is None or usd <= 0:
        return

    # Currency + conversion — ALWAYS for a dual-currency-sheet row.
    safe_data['currency'] = 'USD'
    safe_data['price_per_unit'] = usd
    safe_data['price_per_case'] = None            # per-unit retail, no case price
    safe_data['price_per_unit_eur'] = None        # force re-conversion from USD
    safe_data['price_per_case_eur'] = None
    safe_data['fx_rate'] = None
    safe_data['fx_source'] = None

    # Clear a quantity the LLM lifted from the HKD list price (best-effort).
    if hkd is not None:
        for q in ('quantity_case', 'quantity_bottles', 'quantity_pallets', 'quantity_ftl'):
            v = _safe_float(safe_data.get(q))
            if v is not None and abs(v - hkd) < 0.5:
                safe_data[q] = None

    # Recover the stock Qty from the row's own line (best-effort).
    if line is not None:
        qm = _MC_QTY_RE.search(line)
        if qm:
            qv = _num(qm.group(1))
            if qv is not None and (hkd is None or abs(qv - hkd) >= 0.5) and abs(qv - usd) >= 0.5:
                safe_data['quantity_bottles'] = qv
                safe_data['quantity_case'] = None
                safe_data['quantity_unit'] = 'units'

    # Keep BOTH prices visible when we have the HKD (Samuel: 17 refs with HKD+USD).
    if hkd is not None:
        note = f"List price HKD {hkd:g} / USD {usd:g} (USD used for EUR comparison)"
        cc = safe_data.get('commercial_conditions')
        if not cc:
            safe_data['commercial_conditions'] = note
        elif 'HKD' not in str(cc):
            safe_data['commercial_conditions'] = f"{cc}; {note}"


def _propagate_dual_currency(offers: list, job_id: str) -> int:
    """If a job's perfume/cosmetics batch has CONFIRMED dual-currency HK rows
    (USD converted via ECB, or carrying a 'List price HKD' note), the whole
    supplier batch is a dual-currency sheet. A sibling perfume/cosmetic row that
    stayed EUR — because the vision model omitted the HKD/USD labels on a dense
    image (Samuel 2026-11-11, Etat Libre d'Orange 5×10 ml) — is set to USD and
    converted via ECB, so currency/EUR are consistent across all 17 CHELSEA refs.
    Updates BOTH the offers list and the DB; flags the row for review."""
    if not offers:
        return 0

    def _is_conf(o):
        if str(o.get('currency')).upper() != 'USD':
            return False
        cat = str(o.get('category_slug') or o.get('category') or '').lower()
        if 'perfume' not in cat and 'cosmetic' not in cat:
            return False
        return ('HKD' in str(o.get('commercial_conditions') or '')
                or 'ECB' in str(o.get('fx_source') or ''))

    if len([o for o in offers if _is_conf(o)]) < 2:
        return 0
    targets = []
    for o in offers:
        if str(o.get('currency')).upper() == 'USD':
            continue
        cat = str(o.get('category_slug') or o.get('category') or '').lower()
        if 'perfume' not in cat and 'cosmetic' not in cat:
            continue
        pu = _safe_float(o.get('price_per_unit'))
        if pu and pu > 0:
            targets.append(o)
    if not targets:
        return 0
    try:
        from core.fx_rates import get_eur_rate
    except Exception:
        return 0
    updated = 0
    try:
        db = get_session_factory()()
        try:
            for o in targets:
                pu = _safe_float(o.get('price_per_unit'))
                fx, _eff, _src, _fb = get_eur_rate(
                    'USD', o.get('offer_date') or o.get('date_received'))
                o['currency'] = 'USD'
                o['price_per_unit_eur'] = round(pu * fx, 4)
                o['price_per_case'] = None
                o['price_per_case_eur'] = None
                o['fx_rate'] = fx
                o['fx_source'] = _src
                if _eff and not o.get('fx_date'):
                    o['fx_date'] = _eff
                o['needs_manual_review'] = True
                _fl = o.get('error_flags')
                if isinstance(_fl, list):
                    _m = ("currency set to USD from the supplier's dual-currency "
                          "sheet — this image did not transcribe its own HKD/USD "
                          "labels; verify")
                    if _m not in _fl:
                        _fl.append(_m)
                db.query(OfferItemDB).filter(
                    OfferItemDB.job_id == job_id,
                    OfferItemDB.uid == o.get('uid')).update({
                        'currency': 'USD',
                        'price_per_unit_eur': o['price_per_unit_eur'],
                        'price_per_case': None,
                        'price_per_case_eur': None,
                        'fx_rate': fx,
                        'fx_source': _src,
                        'fx_date': o.get('fx_date'),
                        'needs_manual_review': True,
                    }, synchronize_session=False)
                updated += 1
            db.commit()
            logger.info(f"dual-currency propagate: {updated} sibling row(s) → USD "
                        f"for job {job_id}")
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"dual-currency propagate failed for {job_id}: {e}")
        return 0
    return updated


def _apply_round4_shape_fixes(safe_data: dict, body: str = "") -> None:
    """Samuel 2026-11-11 deterministic offer-shape fixes, keyed on the row's
    own source text (`_raw_row_text`):
      (1) pair each incoterm with its OWN location — Yamazaki "EXW Bordeaux or
          DAP Loendersloot" must not become "Bordeaux OR Loendersloot" on both;
      (2) qualify a delivered-term row that only carries the ex-works basis price
          (no separate delivered price was confirmed);
      (3) preserve a non-numeric MOQ ("Full lot") and a pallet MOQ verbatim;
      (4) carry a per-line deposit condition into commercial_conditions;
      (5) stop asserting "loose" when the source states no case configuration.
    """
    raw = safe_data.get('_raw_row_text') or ''
    # When the per-row text is missing/partial (common on PDF / image / XLSX
    # extractions — the offer arrives as an attachment, so payload.text_body is
    # empty and the LLM product dict carries no _raw_row_text), fall back to the
    # row's OWN block located in the document text by its reference. Without
    # this, the per-row MOQ / deposit scans silently no-op on exactly the
    # attachment offers Samuel sends (Elixe 'Full lot' / deposit).
    _ref = (safe_data.get('product_reference')
            or safe_data.get('supplier_reference') or '')
    _block = _block_for_ref(body, _ref) if (body and _ref) else ''
    row_src = (raw + "\n" + _block) if _block else raw
    flags = safe_data.get('error_flags')
    if not isinstance(flags, list):
        flags = []
        safe_data['error_flags'] = flags

    def _norm_ic(v):
        v = str(v or '').strip().upper()
        return 'EXW' if v.startswith('EX') else v

    # (1) incoterm → location pairing (explicit disjunction only)
    src_pairs = raw if _R4_DISJ_RE.search(raw) else (body or '')
    dm = _R4_DISJ_RE.search(src_pairs)
    if dm:
        pairs = {_norm_ic(dm.group(1)): dm.group(2).strip(' .,-'),
                 _norm_ic(dm.group(3)): dm.group(4).strip(' .,-')}
        row_ic = _norm_ic(safe_data.get('incoterm'))
        if row_ic in pairs and str(safe_data.get('location') or '') != pairs[row_ic]:
            safe_data['location'] = pairs[row_ic]
            _m = (f"location paired to incoterm {row_ic} → {pairs[row_ic]} "
                  f"(source states one place per delivery term)")
            if _m not in flags:
                flags.append(_m)

    # (1b) lot quantity stated as "N btls/bottles <BRAND>" that the LLM dropped
    # (Yamazaki "11 btls YAMAZAKI 25YO" came back with no quantity on either
    # delivery-term row). Scoped: the brand token must FOLLOW the count, so a
    # bare "480 bottles available" elsewhere never leaks in.
    if not any(safe_data.get(q) for q in
               ('quantity_bottles', 'quantity_case', 'quantity_pallets', 'quantity_ftl')):
        _bt = (safe_data.get('brand') or '').split()
        if _bt:
            _qm = re.search(r'\b(\d{1,5})\s*(?:btls?|bottles?)\s+' + re.escape(_bt[0]),
                            body or '', re.IGNORECASE)
            if _qm:
                safe_data['quantity_bottles'] = float(_qm.group(1))
                if not safe_data.get('quantity_unit'):
                    safe_data['quantity_unit'] = 'bottles'

    # (2) delivered-term row carrying only the ex-works basis price
    row_ic2 = _norm_ic(safe_data.get('incoterm'))
    if row_ic2 in _R4_DELIVERED and _R4_EXW_BASIS_RE.search(f"{raw}\n{body or ''}"):
        _m = (f"price shown is the ex-works basis — no separate {row_ic2} "
              f"delivered price was confirmed in the source; delivered price TBC")
        if _m not in flags:
            flags.append(_m)
            safe_data['needs_manual_review'] = True
        # Samuel 2026-11-12: when this delivered term shares ONE lot with an
        # ex-works term via an explicit disjunction (Yamazaki "EXW Bordeaux or
        # DAP Loendersloot"), it is the SAME stock and its delivered price is NOT
        # confirmed. So the app must not present a quantity or a price as
        # confirmed for this row — blank both (the lot quantity and the firm
        # price stay on the ex-works row); incoterm + location remain so the
        # delivery option is still visible. The disjunction may live in the
        # row's own text OR (single-offer attachment like the Yamazaki PDF,
        # whose row carries no reference block) in the document body — mirror
        # section (1)'s fallback. Blank ONLY when this row's delivered incoterm
        # is one named side of an EXW-vs-this-term disjunction, so a standalone
        # confirmed delivered price (e.g. NE-410 Moet DAP) is never touched.
        _disj_src = raw if _R4_DISJ_RE.search(raw) else (body or '')
        _dm2 = _R4_DISJ_RE.search(_disj_src)
        _sides = {_norm_ic(_dm2.group(1)), _norm_ic(_dm2.group(3))} if _dm2 else set()
        if row_ic2 in _sides and 'EXW' in _sides:
            for _pf in ('price_per_unit', 'price_per_case',
                        'price_per_unit_eur', 'price_per_case_eur'):
                safe_data[_pf] = None
            for _qf in ('quantity_bottles', 'quantity_case',
                        'quantity_pallets', 'quantity_ftl'):
                safe_data[_qf] = None
            _note = (f"Same lot as the ex-works row (not additional stock); "
                     f"{row_ic2} delivered price not confirmed (ex-works basis "
                     f"only, TBC).")
            cc = safe_data.get('commercial_conditions')
            if not cc:
                safe_data['commercial_conditions'] = _note
            elif 'not confirmed' not in str(cc).lower():
                safe_data['commercial_conditions'] = f"{cc}; {_note}"

    # (2b) same-lot SECONDARY delivery option with an unconfirmed price — the
    # MIRROR of (2). Here the PRIMARY priced term is the delivered one and the
    # ex-works "can also be collected" option is the unconfirmed alternative
    # (Moët: DAP Antwerp priced / EXW Breda TBC). There is no explicit "A or B"
    # disjunction, so (1)/(2) don't fire. Scope the scan to THIS product's own
    # paragraph (located by brand) so an unrelated same-incoterm row is never
    # touched, and require BOTH an explicit same-lot "also collected/delivered
    # <INCO>" cue AND a "price not confirmed / confirm whether the price is the
    # same" cue. Blank the SECONDARY row's qty+price and flag TBC, exactly as the
    # Yamazaki shared-lot row — so one lot is never double-counted and the
    # secondary price is never shown as confirmed.
    _row_ic3 = _norm_ic(safe_data.get('incoterm'))
    _brandtok = str(safe_data.get('brand') or '').strip()
    _pblock = ''
    if body and _brandtok and len(_brandtok) >= 3:
        _bm = re.search(re.escape(_brandtok), body, re.IGNORECASE)
        if _bm:
            _seg = body[_bm.start():]
            _nx = re.search(r'\r?\n\s*\r?\n', _seg)
            _pblock = _seg[:_nx.start()] if _nx else _seg[:400]
    _altm = _R4_SAMELOT_ALT_RE.search(_pblock) if _pblock else None
    if (_altm and _R4_PRICE_UNCONFIRMED_RE.search(_pblock)
            and _row_ic3 and _row_ic3 == _norm_ic(_altm.group(1))):
        for _pf in ('price_per_unit', 'price_per_case',
                    'price_per_unit_eur', 'price_per_case_eur'):
            safe_data[_pf] = None
        for _qf in ('quantity_bottles', 'quantity_case',
                    'quantity_pallets', 'quantity_ftl'):
            safe_data[_qf] = None
        safe_data['needs_manual_review'] = True
        _fl = f"same lot — {_row_ic3} price not confirmed (TBC)"
        if _fl not in flags:
            flags.append(_fl)
        _note2 = (f"Same lot as the priced row (not additional stock); "
                  f"{_row_ic3} price not confirmed — the source asks to confirm "
                  f"whether the quoted price applies to both delivery options "
                  f"(TBC).")
        cc = safe_data.get('commercial_conditions')
        if not cc:
            safe_data['commercial_conditions'] = _note2
        elif 'not confirmed' not in str(cc).lower():
            safe_data['commercial_conditions'] = f"{cc}; {_note2}"

    # (3) non-numeric MOQ "Full lot" + pallet MOQ, verbatim
    if row_src and not safe_data.get('moq_text') and _R4_MOQ_FULLLOT_RE.search(row_src):
        safe_data['moq_text'] = 'Full lot'
    if row_src and not safe_data.get('moq_pallets'):
        mpl = _R4_MOQ_PALLETS_RE.search(row_src)
        if mpl:
            try:
                safe_data['moq_pallets'] = float(mpl.group(1).replace(',', '.'))
                safe_data['moq_unit'] = 'pallets'
                _m = f"MOQ {mpl.group(1)} pallets (from source)"
                if _m not in flags:
                    flags.append(_m)
            except ValueError:
                pass

    # (4) per-line deposit condition → commercial_conditions
    if row_src and _R4_DEPOSIT_RE.search(row_src):
        _dep = 'Deposit confirmation required'
        cc = safe_data.get('commercial_conditions')
        if not cc:
            safe_data['commercial_conditions'] = _dep
        elif _dep.lower() not in str(cc).lower():
            safe_data['commercial_conditions'] = f"{cc}; {_dep}"

    # (5) "loose" → honest "case configuration not stated"
    upc = _safe_float(safe_data.get('units_per_case'))
    # Fall back to the document text when the row carries no per-row text (an
    # attachment-only offer like the Yamazaki PDF), so the honest "case
    # configuration not stated" rewrite still reaches the 'loose pack' flag.
    _loose_src = row_src or (body or '')
    if _loose_src and upc in (None, 1.0) and not re.search(
            r'\d+\s*[x×]\s*\d|/\s*cs|per\s*case|\bcases?\b|\bcs\b', _loose_src, re.IGNORECASE):
        changed = False
        newflags = []
        for f in flags:
            if isinstance(f, str) and 'loose pack' in f.lower():
                newflags.append(re.sub(r'loose pack', 'case configuration not stated',
                                       f, flags=re.IGNORECASE))
                changed = True
            else:
                newflags.append(f)
        if changed:
            safe_data['error_flags'] = newflags


# A full personal name ("Bob Curtis") vs a company token.
_R4_PERSON_RE = re.compile(r"^[A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ'\-]+){1,2}$")
_R4_CO_SUFFIX_RE = re.compile(
    r'\b(ltd|llc|inc|gmbh|srl|sarl|b\.?v|n\.?v|s\.?a|a\.?g|co|company|trading|group|'
    r'import|export|wholesale|spirits|wines?|beverages?|desk|broker|market|estate)\b',
    re.IGNORECASE)


def _company_from_domain(company, body, supplier_email):
    """Upgrade an abbreviated company ("T&T") to the full name found in the body
    signature whose alphanumeric form matches the supplier email's domain root
    ("terroirettradition" ← "Terroir & Tradition"). Returns the fuller name or
    None."""
    if not (body and supplier_email and '@' in str(supplier_email)):
        return None
    try:
        domain_root = str(supplier_email).split('@')[-1].split('.')[0].lower()
    except Exception:
        return None
    if len(domain_root) < 4:
        return None
    for line in str(body).splitlines():
        ln = line.strip().lstrip('>').strip()
        if not ln or len(ln) > 60:
            continue
        low = ln.lower()
        if re.sub(r'[^a-z0-9]', '', low.replace('&', 'et')) == domain_root or \
           re.sub(r'[^a-z0-9]', '', low.replace('&', 'and')) == domain_root or \
           re.sub(r'[^a-z0-9]', '', low) == domain_root:
            return ln.strip(' .,-')
    return None


def _split_company_contact(name, body=None, supplier_email=None):
    """'T&T - Bob Curtis' → ('Terroir & Tradition', 'Bob Curtis'): split a
    "Company - Person" supplier string into the trading company and the human
    contact, and upgrade an abbreviated company from the body signature. The
    supplier field must hold ONLY the company (Samuel 2026-11-11, Yamazaki)."""
    if not name:
        return (name, None)
    company, contact = str(name).strip(), None
    m = re.match(r'^(.*?)\s*[—–|/]\s*(.+)$|^(.*?\S)\s+-\s+(.+)$', company)
    if m:
        left = (m.group(1) if m.group(1) is not None else m.group(3) or '').strip()
        right = (m.group(2) if m.group(2) is not None else m.group(4) or '').strip()
        if right and _R4_PERSON_RE.match(right) and not _R4_CO_SUFFIX_RE.search(right):
            company, contact = left, right
        elif left and _R4_PERSON_RE.match(left) and not _R4_CO_SUFFIX_RE.search(left):
            company, contact = right, left
    upgraded = _company_from_domain(company, body, supplier_email)
    if upgraded:
        company = upgraded
    return (company or name, contact)


def _dedup_fragments_in_place(offers: list, job_id: str) -> int:
    """Drop an LLM over-split fragment: a no-quantity row that duplicates a
    quantity-bearing twin (same brand/product/incoterm/location/price). Yamazaki
    'one lot, two delivery terms' came back as THREE rows — EXW(no qty), EXW(11),
    DAP(11); the no-qty EXW is the fragment (Samuel 2026-11-11). Removes it from
    BOTH the offers list and the DB."""
    if not offers:
        return 0

    def _hasqty(d):
        return any(d.get(f) for f in ('quantity_bottles', 'quantity_case',
                                      'quantity_pallets', 'quantity_ftl'))

    def _sig(d):
        return ((d.get('brand') or '').strip().lower(),
                (d.get('product_name') or '').strip().lower(),
                (d.get('incoterm') or '').strip().upper(),
                (d.get('location') or '').strip().lower(),
                d.get('price_per_unit'), d.get('price_per_case'))

    with_qty = {_sig(d) for d in offers if _hasqty(d)}
    frag_uids = {d.get('uid') for d in offers
                 if (not _hasqty(d)) and _sig(d) in with_qty}
    if not frag_uids:
        return 0
    try:
        db = get_session_factory()()
        try:
            db.query(OfferItemDB).filter(
                OfferItemDB.job_id == job_id,
                OfferItemDB.uid.in_(frag_uids)).delete(synchronize_session=False)
            sf = db.query(SourceFileDB).filter(SourceFileDB.job_id == job_id).first()
            if sf and sf.product_count:
                sf.product_count = max(0, sf.product_count - len(frag_uids))
            db.commit()
            logger.info(f"dedup: dropped {len(frag_uids)} no-quantity fragment "
                        f"row(s) for job {job_id}")
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"fragment dedup DB delete failed for {job_id}: {e}")
        return 0
    offers[:] = [d for d in offers if d.get('uid') not in frag_uids]
    return len(frag_uids)


_GLUED_AT_RE = re.compile(r'@\s*(EUR|USD|GBP|CHF|€|\$|£)\s*([\d]+(?:[.,]\d+)?)',
                          re.IGNORECASE)
_NON_NAME_RE = re.compile(
    r'^\s*(?:\d{1,3}$|T[12]\b|EXW|DAP|DDP|CIF|CFR|CNF|FOB|FCA|CPT|CIP|DPU|FAS|MOQ|'
    r'price\b|from\b|best\b|ref\b|offer\b|supplier\b|stock\b|North\b)', re.IGNORECASE)


def _recover_dropped_at_priced(offers: list, source_text: str, job_id: str) -> int:
    """Recover a glued '@ <CUR> <num>' ambiguous-basis line the LLM dropped —
    Samuel 2026-11-11, Alder NE-410 item 07 'Beefeater London Dry Gin … @ EUR 79'
    (6 of 7 rows extracted). The '@ CUR num' format is distinctive; we only add a
    row when NO extracted row already shares a name token, so an already-captured
    '@'-priced product (ATV list, Yamazaki) is never duplicated. Kept with null
    prices + 'EUR 79 quoted — basis unspecified', like the ATV Beefeater."""
    if not source_text or not offers:
        return 0
    lines = [l.rstrip() for l in str(source_text).splitlines()]
    seen = set()
    for d in offers:
        for v in (d.get('brand'), d.get('product_name')):
            for t in re.findall(r'[a-z0-9]+', (v or '').lower()):
                if len(t) > 2:
                    seen.add(t)
    tmpl = offers[0]
    base_uid = str(tmpl.get('uid') or 'row').rsplit('_', 1)[0]
    added = 0
    for idx, ln in enumerate(lines):
        m = _GLUED_AT_RE.search(ln)
        if not m:
            continue
        cur = m.group(1).upper()
        cur = _CURRENCY_SYMBOLS.get(m.group(1), cur)
        try:
            amt = float(m.group(2).replace(',', '.'))
        except ValueError:
            continue
        if amt <= 0:
            continue
        # Guard (Samuel 2026-11-11, Yamazaki): if the '@'-priced line ITSELF
        # names a product already extracted ("11 btls YAMAZAKI 25YO @ EUR
        # 3995/btl"), this is NOT a dropped row — skip, so a leading quantity
        # token ("11 btls") can never make us scrape a phantom from the
        # surrounding text ("Bordeaux or DAP Loendersloot …").
        _line_toks = [t for t in re.findall(r'[a-z0-9]+', ln[:m.start()].lower())
                      if len(t) > 2]
        if any(t in seen for t in _line_toks):
            continue
        # Candidate name: this line before '@' (minus size/customs tokens), else
        # the nearest preceding name-like line (handles "07 / Name / 70cl @ EUR").
        name = re.sub(r'\b\d+\s*(?:cl|ml|l|cs|btls?|bottles?|pcs?|pieces?|nrf|rf|gbx|ngbx)\b.*$',
                      '', ln[:m.start()], flags=re.IGNORECASE).strip(' |·-')
        if not re.search(r'[A-Za-z]{3}', name):
            for j in range(idx - 1, max(-1, idx - 4), -1):
                cand = lines[j].strip()
                if not cand or _NON_NAME_RE.match(cand) or _GLUED_AT_RE.search(cand):
                    continue
                if re.search(r'[A-Za-z]{3}', cand):
                    name = re.sub(r'\b\d+\s*(?:cl|ml|l)\b.*$', '', cand,
                                  flags=re.IGNORECASE).strip(' |·-')
                    break
        toks = [t for t in re.findall(r'[a-z0-9]+', name.lower()) if len(t) > 2]
        if not toks or any(t in seen for t in toks):
            continue
        brand = name.split()[0]
        prod = name[len(brand):].strip() or name
        vol = None
        vm = re.search(r'(\d+(?:\.\d+)?)\s*(cl|ml|l)\b',
                       ln + ' ' + (lines[idx - 1] if idx > 0 else ''), re.IGNORECASE)
        if vm:
            v = float(vm.group(1)); u = vm.group(2).lower()
            vol = v * 10 if u == 'cl' else v * 1000 if u == 'l' else v
        od = dict(tmpl)
        od.update(
            uid=f"{base_uid}_at{added}", brand=brand, brand_raw=brand,
            product_name=prod, product_name_raw=name, product_key=None,
            unit_volume_ml=vol, units_per_case=None,
            price_per_unit=None, price_per_case=None,
            price_per_unit_eur=None, price_per_case_eur=None,
            currency=cur, fx_rate=1.0, fx_source=None, fx_date=None,
            incoterm=None, location=None,
            quantity_case=None, quantity_bottles=None, quantity_pallets=None, quantity_ftl=None,
            quantity_unit=None, moq_cases=None, moq_bottles=None, moq_pallets=None,
            moq_ftl=None, moq_text=None, offer_date=None,
            needs_manual_review=True,
            error_flags=[f"{cur} {amt:g} quoted — basis unspecified; excluded from price comparison",
                         "price basis ambiguous — unit/case basis not stated in source"])
        try:
            save_offer_to_db(od, job_id)
            offers.append(od)
            seen.update(toks)
            added += 1
            logger.info(f"recovered dropped @-priced row '{name}' ({cur} {amt:g}) "
                        f"for job {job_id}")
        except Exception as e:
            logger.warning(f"recover @-priced '{name}' failed: {e}")
    return added


_NUM_MARK_RE = re.compile(r'^\s*(0[1-9]|[1-9]|[1-4]\d)\s*$')
_NUM_PRICE_RE = re.compile(
    r'(EUR|USD|GBP|CHF|€|\$|£)\s*([\d][\d.,]*)\s*(?:per|/)\s*'
    r'(bottle|btl|case|cs|piece|pcs?|unit|pallet)', re.IGNORECASE)
_NUM_INCO_RE = re.compile(
    r'\b(EXW|EXWORKS?|DAP|DDP|CIF|CFR|CNF|FOB|FCA|CPT|CIP|DPU|FAS)\b\s+'
    r'([A-ZÀ-Ý][\w.\-]*)', re.IGNORECASE)
_NUM_VOL_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*(cl|ml|l)\b', re.IGNORECASE)
_NUM_AVAIL_RE = re.compile(
    r'(\d[\d,]*)\s*(bottles?|btls?|cases?|cs|pieces?|pcs|pallets?|ftl)\s+available',
    re.IGNORECASE)
_NUM_MOQ_RE = re.compile(
    r'MOQ\s*(\d[\d,]*)\s*(bottles?|btls?|cases?|cs|pieces?|pcs|pallets?|ftl)?',
    re.IGNORECASE)


def _recover_dropped_numbered_items(offers: list, source_text: str, job_id: str) -> int:
    """Recover a NUMBERED stock-circular item the LLM dropped (NE-410 item 06
    'L'Oréal Paris Revitalift Day Cream … EUR 7.80 per piece' — a lone cosmetic
    in a spirits list is consistently skipped). Fires ONLY on a clearly-numbered
    circular (≥3 standalone 'NN' markers) and ONLY for an item no extracted row
    already covers; builds a properly-priced row from the block's own text. The
    '@ <CUR> <num>' items are left to _recover_dropped_at_priced (no 'per basis'
    here, so they are skipped)."""
    if not source_text or not offers:
        return 0
    lines = [l.rstrip() for l in str(source_text).splitlines()]
    marks = [i for i, ln in enumerate(lines) if _NUM_MARK_RE.match(ln)]
    if len(marks) < 3:
        return 0
    seen = set()
    for d in offers:
        for v in (d.get('brand'), d.get('product_name')):
            for t in re.findall(r'[a-zà-ÿ0-9]+', (v or '').lower()):
                if len(t) > 2:
                    seen.add(t)
    tmpl = offers[0]
    base_uid = str(tmpl.get('uid') or 'row').rsplit('_', 1)[0]
    bounds = marks + [len(lines)]
    added = 0
    for k, mi in enumerate(marks):
        blk = [b for b in lines[mi + 1:bounds[k + 1]] if b.strip()]
        if not blk:
            continue
        name = blk[0].strip()
        toks = [t for t in re.findall(r'[a-zà-ÿ0-9]+', name.lower()) if len(t) > 2]
        if not toks or any(t in seen for t in toks):
            continue
        bt = "\n".join(blk)
        pm = _NUM_PRICE_RE.search(bt)
        if not pm:
            continue  # no clear "<CUR> <num> per <basis>" price → not ours
        cur = _CURRENCY_SYMBOLS.get(pm.group(1), pm.group(1).upper())
        try:
            amt = float(pm.group(2).replace(',', '.'))
        except ValueError:
            continue
        if amt <= 0:
            continue
        basis = pm.group(3).lower()
        per_case = basis in ('case', 'cs')
        im = _NUM_INCO_RE.search(bt)
        inco = ('EXW' if im and im.group(1).upper().startswith('EX')
                else (im.group(1).upper() if im else None))
        loc = im.group(2).strip(' .,-') if im else None
        vol = None
        vm = _NUM_VOL_RE.search(bt)
        if vm:
            v = float(vm.group(1).replace(',', '.')); u = vm.group(2).lower()
            vol = v * 10 if u == 'cl' else v * 1000 if u == 'l' else v
        brand = name.split()[0]
        prod = name[len(brand):].strip() or name
        is_eur = (cur == 'EUR')
        od = dict(tmpl)
        od.update(
            uid=f"{base_uid}_num{added}", brand=brand, brand_raw=brand,
            product_name=prod, product_name_raw=name, product_key=None,
            unit_volume_ml=vol, units_per_case=None,
            price_per_unit=None if per_case else amt,
            price_per_case=amt if per_case else None,
            price_per_unit_eur=(amt if (is_eur and not per_case) else None),
            price_per_case_eur=(amt if (is_eur and per_case) else None),
            currency=cur, fx_rate=1.0 if is_eur else None,
            fx_source='identity-EUR' if is_eur else None, fx_date=None,
            incoterm=inco, location=loc,
            quantity_case=None, quantity_bottles=None, quantity_pallets=None,
            quantity_ftl=None, quantity_unit=None,
            moq_cases=None, moq_bottles=None, moq_pallets=None, moq_ftl=None,
            moq_text=None, offer_date=None, needs_manual_review=True,
            error_flags=[f"row recovered from numbered source item — the extractor "
                         f"had dropped it; {cur} {amt:g} per {basis} "
                         f"{(inco or '')} {(loc or '')}".strip()])
        am = _NUM_AVAIL_RE.search(bt)
        if am:
            qv = float(am.group(1).replace(',', '')); u = am.group(2).lower()
            if u.startswith(('bottle', 'btl')):
                od['quantity_bottles'] = qv; od['quantity_unit'] = 'bottles'
            elif u.startswith(('case', 'cs')):
                od['quantity_case'] = qv; od['quantity_unit'] = 'cases'
            elif u.startswith(('piece', 'pcs')):
                od['quantity_bottles'] = qv; od['quantity_unit'] = 'units'
            elif u.startswith('pallet'):
                od['quantity_pallets'] = qv; od['quantity_unit'] = 'pallets'
        mm = _NUM_MOQ_RE.search(bt)
        if mm:
            try:
                mv = float(mm.group(1).replace(',', ''))
            except ValueError:
                mv = None
            if mv is not None:
                u = (mm.group(2) or '').lower()
                if u.startswith(('case', 'cs')):
                    od['moq_cases'] = mv
                elif u.startswith('pallet'):
                    od['moq_pallets'] = mv; od['moq_unit'] = 'pallets'
                elif u.startswith(('piece', 'pcs', 'bottle', 'btl')):
                    od['moq_bottles'] = mv
        try:
            save_offer_to_db(od, job_id)
            offers.append(od)
            seen.update(toks)
            added += 1
            logger.info(f"recovered dropped numbered item '{name}' "
                        f"({cur} {amt:g}/{basis}) for job {job_id}")
        except Exception as e:
            logger.warning(f"recover numbered '{name}' failed: {e}")
    return added


_PALLET_PER_CASE_RE = re.compile(
    r'(\d+(?:[.,]\d+)?)\s*(?:pallets?|plt|pll)\b[\s,;:]*'
    r'(?:per|x|×|/|of|\*|each|with|containing|=)?[\s,;:]*'
    r'(\d+(?:[.,]\d+)?)\s*(?:cs\b|cases?|cartons?|ctns?)', re.IGNORECASE)

# Per-option bottle prices in a pallet-scale quote, e.g.
# "EUR 4.76 per bottle EXW Germany" / "EUR 4.80 per bottle DAP Riga".
# Lets us preserve the QUOTED bottle price EXACTLY (not the LLM's round-tripped
# value) and attach it to the row with the matching incoterm (Samuel Cedar Bay).
_BOTTLE_OPTION_RE = re.compile(
    r'(?:eur|usd|gbp|chf|€|\$|£)\s*([\d]+(?:[.,]\d+)?)\s*'
    r'(?:per\s*bottle|/\s*b(?:tl|ottle)\b|per\s*btl\b)\s*'
    r'(exworks?|exw|dap|ddp|cif|cfr|cnf|fob|fca|cpt|cip|dpu|fas|ex)\b',
    re.IGNORECASE)


def _fix_pallet_per_case_phrasing(safe_data: dict) -> None:
    """'30 pallets per 112cs' = 30 pallets × 112 CASES-per-pallet = 3360 cases.
    The LLM read '112cs' as units_per_case and swapped it with the pallet count
    (Samuel Edo). Parse the source text and set the quantity fields correctly:
    the cases-per-pallet figure is NEVER units_per_case, and a bare price in a
    pallet-scale quote (no '/cs' in the source) is per BOTTLE, not per case."""
    raw = safe_data.get('_raw_row_text') or ''
    m = _PALLET_PER_CASE_RE.search(raw)
    if not m:
        return
    try:
        pallets = float(m.group(1).replace(',', '.'))
        cpp = float(m.group(2).replace(',', '.'))
    except (TypeError, ValueError):
        return
    if pallets <= 0 or cpp <= 0:
        return
    safe_data['quantity_pallets'] = pallets
    safe_data['cases_per_pallet'] = cpp
    safe_data['quantity_case'] = round(pallets * cpp, 4)
    safe_data['quantity_unit'] = 'cases'
    upc = _safe_float(safe_data.get('units_per_case'))
    if upc is not None and (upc == cpp or upc > 24):
        safe_data['units_per_case'] = None  # 112 was cases/pallet, not units/case

    # ── Price: the bottles-per-case is unknown here, so a CASE price cannot be
    # derived. Preserve the quoted BOTTLE price exactly and leave the case price
    # blank (Samuel Cedar Bay: "preserve the quoted bottle prices exactly and
    # leave both case prices blank when the bottles per case is unknown").
    _has_case_marker = re.search(
        r'/\s*cs|/\s*case|per\s*case|p/?cs|eur/cs|usd/cs', raw, re.IGNORECASE)
    # Exact per-option bottle prices from the source, keyed by incoterm.
    _opts: dict = {}
    for _mo in _BOTTLE_OPTION_RE.finditer(raw):
        _pv = _safe_float(_mo.group(1).replace(',', '.'))
        _ic = _mo.group(2).upper()
        _ic = 'EXW' if _ic in ('EX', 'EXWORK', 'EXWORKS') else _ic
        if _pv is not None and _ic:
            _opts.setdefault(_ic, _pv)
    _row_inco = str(safe_data.get('incoterm') or '').strip().upper()
    _row_inco = 'EXW' if _row_inco in ('EX', 'EXWORK', 'EXWORKS') else _row_inco
    if _opts and _row_inco in _opts and not _has_case_marker:
        # Authoritative: the quoted bottle price for THIS row's incoterm.
        safe_data['price_per_unit'] = _opts[_row_inco]
        safe_data['price_per_case'] = None
        safe_data['price_per_unit_eur'] = None   # recomputed by currency pass
        safe_data['price_per_case_eur'] = None
    elif not _has_case_marker:
        # Bare price with no per-case marker → it is per bottle. Move a value the
        # LLM parked in price_per_case to price_per_unit (only if unit is empty),
        # then clear the un-derivable case price.
        ppc = _safe_float(safe_data.get('price_per_case'))
        ppu = _safe_float(safe_data.get('price_per_unit'))
        if ppu is None and ppc is not None:
            safe_data['price_per_unit'] = ppc
        safe_data['price_per_case'] = None
        safe_data['price_per_case_eur'] = None
        safe_data['price_per_unit_eur'] = None
    flags = safe_data.get('error_flags')
    if isinstance(flags, list):
        _m = f'Quantity: {pallets:g} pallets × {cpp:g} cases/pallet = {pallets * cpp:g} cases'
        if _m not in flags:
            flags.append(_m)


def _backfill_numeric_brand(safe_data: dict) -> None:
    """A numeric brand (1800 Tequila, 818) is often missed — the LLM leaves
    brand 'Not Found' with the digits on the product name. When brand is absent
    and the product name starts with a numeric token, promote it to brand. Safe:
    word-brands like 'Grey Goose' never start with a digit."""
    brand = (safe_data.get('brand') or '').strip()
    if brand and brand.lower() not in ('not found', 'unknown'):
        return
    name = (safe_data.get('product_name') or '').strip()
    # (a) the numeric token is still at the start of the product name
    m = re.match(r'^(\d{2,4})\s+(\S.*)$', name)
    if m:
        safe_data['brand'] = m.group(1)
        safe_data['product_name'] = m.group(2).strip()
        return
    # (b) the LLM dropped the numeric brand from the name — recover it from the
    #     source: "1800 Reserva Blanco" → brand "1800". The number must be
    #     immediately followed by the product's first word (so a quantity like
    #     "2007 cs" never matches — it's followed by a unit, not the name).
    first_word = name.split()[0] if name else ''
    if first_word:
        m2 = re.search(r'(?<![\d.])(\d{2,4})\s+' + re.escape(first_word),
                       safe_data.get('_raw_row_text') or '')
        if m2:
            safe_data['brand'] = m2.group(1)


# ─── MOQ / QUANTITY normalization ──────────────────────────────────────────────
def _normalize_moq_and_quantity(safe_data: dict, error_flags: list) -> dict:
    """
    Convert AI-extracted MOQ and quantity to CASES only, using units_per_case.
    Preserves original unit; flags conversions. See RULE 0.3 in openai_client.py.
    """
    upc = safe_data.get('units_per_case')
    try:
        upc = float(upc) if upc not in (None, 0, "0", "") else None
    except (TypeError, ValueError):
        upc = None

    def _to_float(v):
        try:
            return float(v) if v not in (None, "", "Not Found") else None
        except (TypeError, ValueError):
            return None

    # ── MOQ ─────────────────────────────────────────────────────────────
    moq_cases = _to_float(safe_data.get('moq_cases'))
    moq_bottles = _to_float(safe_data.get('moq_bottles'))
    moq_unit = (safe_data.get('moq_unit') or '').lower().strip()

    if moq_unit == 'bottles' and moq_bottles:
        if upc:
            converted = round(moq_bottles / upc, 2)
            safe_data['moq_cases'] = converted
            error_flags.append('MOQ converted from bottles to cases')
        else:
            safe_data['moq_cases'] = None
            error_flags.append(f'MOQ in bottles ({int(moq_bottles)}), cases unknown (no units_per_case)')
    elif moq_unit == 'pallets' and moq_cases is None:
        # rare — leave null, flag
        error_flags.append('MOQ in pallets, cases unknown')
    elif moq_cases is not None and moq_cases > 10000 and upc and upc >= 2:
        # sanity check: values >10k with a small case pack suggest the LLM stored bottles under moq_cases
        # only trigger if moq_unit was NOT explicitly cases (defensive)
        if moq_unit not in ('cases', 'case', 'cs'):
            converted = round(moq_cases / upc, 2)
            error_flags.append(f'MOQ auto-normalized {int(moq_cases)}→{converted} cases (looked like bottles)')
            safe_data['moq_bottles'] = safe_data.get('moq_bottles') or moq_cases
            safe_data['moq_cases'] = converted

    safe_data['moq_bottles'] = _to_float(safe_data.get('moq_bottles'))

    # Stale-flag cleanup: the LLM sometimes attaches "MOQ in bottles, not
    # cases" even when the source expresses MOQ in pallets or FTL (A10, A11
    # in the acceptance pack). After we read `moq_unit`, drop that flag if
    # moq_unit disagrees — otherwise the drawer reads as a defect it isn't.
    if moq_unit and moq_unit not in ("bottles", "bottle"):
        before = list(error_flags)
        # Mutate the caller's list in place rather than rebinding — subsequent
        # appends in this function (FTL quantity, pallet notes, etc.) must land
        # on the same list and reach safe_data['error_flags'] below.
        error_flags[:] = [f for f in before
                          if not (isinstance(f, str) and "MOQ in bottles" in f)]
        if len(error_flags) != len(before):
            logger.info(
                f"[_normalize_moq_and_quantity] dropped stale MOQ-in-bottles "
                f"flag(s) because moq_unit = {moq_unit!r}"
            )

    # ── quantity_case ───────────────────────────────────────────────────
    qty = _to_float(safe_data.get('quantity_case'))
    qty_unit = (safe_data.get('quantity_unit') or '').lower().strip()

    if qty_unit == 'ftl':
        # Samuel 2026-10-31 (A11): keep the source FTL count persistent so
        # the drawer can show "1 FTL available" even though the case
        # equivalent is intentionally null. Parse the count out of the raw
        # row text ("1 FTL available", "FTL × 1") when the LLM gave a
        # numeric quantity_case before we null it; default to 1 when the
        # supplier wrote "1 FTL" or just "FTL" with no explicit count.
        # Honour a count already set upstream (e.g. the structured XLSX path
        # parsed "1 FTL" into quantity_ftl) before recomputing — otherwise a
        # row with no _raw_row_text would null it (Samuel E07-09).
        ftl_count = _to_float(safe_data.get('quantity_ftl'))
        if ftl_count is None and qty is not None and qty > 0:
            ftl_count = qty
        if ftl_count is None:
            raw = safe_data.get('_raw_row_text') or ''
            m = re.search(r'(\d+(?:\.\d+)?)\s*(?:FTL|full\s*truck\s*load)\b',
                          raw, re.IGNORECASE)
            if m:
                try:
                    ftl_count = float(m.group(1))
                except (TypeError, ValueError):
                    ftl_count = None
            elif re.search(r'\bFTL\b', raw, re.IGNORECASE):
                ftl_count = 1.0
        safe_data['quantity_ftl'] = ftl_count
        safe_data['quantity_case'] = None
        if ftl_count:
            error_flags.append(
                f'Quantity: {int(ftl_count) if ftl_count == int(ftl_count) else ftl_count} '
                f'FTL (Full Truck Load, case equivalent unknown)'
            )
        else:
            error_flags.append('Quantity: FTL (Full Truck Load)')
    elif qty_unit in ('pallets', 'pallet', 'plt'):
        # Source expressed availability in pallets (A10: "2 pallets ... each
        # pallet contains 80 cases"). Rule 0.3 says `quantity_case` must
        # ALWAYS store cases, so when cases_per_pallet is known we convert.
        # Three paths:
        #   (a) LLM already converted → qty % cpp == 0 and qty > cpp.
        #   (b) LLM left qty as the pallet count (A10 → qty=2) → multiply.
        #   (c) Parse pallet count out of _raw_row_text when LLM gave null.
        cpp = _to_float(safe_data.get('cases_per_pallet'))
        pallets = None
        if cpp and cpp > 0 and qty is not None and qty > 0:
            if qty <= cpp:
                pallets = qty
                safe_data['quantity_case'] = qty * cpp
            else:
                # LLM already converted pallets → cases (qty > cpp): derive
                # the pallet count by dividing so the drawer can still show
                # "2 plt × 80 cs/plt" instead of only "160 cs".
                try:
                    _p = qty / cpp
                    if _p > 0 and abs(_p - round(_p)) < 1e-6:
                        pallets = _p
                except (TypeError, ZeroDivisionError):
                    pallets = None
        else:
            raw = safe_data.get('_raw_row_text') or ''
            m = re.search(r'(\d+(?:\.\d+)?)\s*(?:pallets?|plt)\b',
                          raw, re.IGNORECASE)
            if m and cpp and cpp > 0:
                try:
                    pallets = float(m.group(1))
                    safe_data['quantity_case'] = pallets * cpp
                except (TypeError, ValueError):
                    pass
        if pallets:
            safe_data['_quantity_pallets'] = pallets
            safe_data['quantity_pallets'] = pallets
    elif qty_unit in ('bottles', 'bottle', 'btls',
                       'pieces', 'piece', 'pcs'):
        # Rule 0.3 is explicit: `quantity_case` ALWAYS stores the number of
        # cases. For loose packs (upc 0/1) the case equivalent is null; the
        # source piece / bottle count still has to reach the drawer. Samuel
        # 2026-10-31 (A07/B05/H02 NARS "100 pieces"): the LLM sometimes
        # emits quantity_case=null + a flag "quantity_case not explicitly
        # stated" even though the source has "Available quantity: 100
        # pieces". Fall back to parsing the raw block when the LLM dropped
        # the number.
        if qty is None:
            raw = safe_data.get('_raw_row_text') or ''
            m = re.search(
                r'Available\s+quantity\s*:\s*(\d+(?:[.,]\d+)?)\s*'
                r'(?:pieces|piece|pcs|bottles|bottle|btls)\b',
                raw, re.IGNORECASE
            )
            if m:
                try:
                    qty = float(m.group(1).replace(',', '.'))
                except (TypeError, ValueError):
                    qty = None
        if qty is None:
            # Nothing to persist; keep columns null and skip the flag.
            pass
        elif not upc or upc in (0, 1):
            # Loose pack: persist the source count on quantity_bottles so
            # the drawer renders "100 pieces" instead of "—".
            safe_data['quantity_case'] = None
            safe_data['_quantity_bottles'] = qty
            safe_data['quantity_bottles'] = qty
            # Normalise the stored unit for consistent downstream handling.
            if qty_unit in ('pieces', 'piece', 'pcs'):
                safe_data['quantity_unit'] = 'pieces'
            label = 'pieces' if qty_unit in ('pieces', 'piece', 'pcs') else 'bottles'
            error_flags.append(
                f'Quantity in {label} ({int(qty)}), loose pack — case qty N/A'
            )
        else:
            # Case pack is known: the LLM's quantity_case is already in
            # cases, so compute the source bottle count for the drawer.
            try:
                safe_data['quantity_bottles'] = float(qty) * float(upc)
            except (TypeError, ValueError):
                pass

    # Second-chance pallet-count: when the LLM already converted pallets to
    # cases (quantity_unit='cases', quantity_case=160 for A10) the raw block
    # still contains "N pallets". Fish out the pallet count so the drawer
    # can render the source phrase. Only fires when quantity_pallets is still
    # null (no double-write) and the row looks pallet-y.
    if safe_data.get('quantity_pallets') in (None, 0, 0.0):
        raw_block = safe_data.get('_raw_row_text') or ''
        cpp = _to_float(safe_data.get('cases_per_pallet'))
        qc_final = _to_float(safe_data.get('quantity_case'))
        if raw_block and cpp and cpp > 0 and qc_final and qc_final > 0:
            m = re.search(r'(\d+(?:\.\d+)?)\s*pallets?\b', raw_block, re.IGNORECASE)
            if m:
                try:
                    _plt = float(m.group(1))
                    if _plt * cpp == qc_final:
                        safe_data['quantity_pallets'] = _plt
                except (TypeError, ValueError):
                    pass

    # Final sync: safe_data['error_flags'] must reflect every flag this function
    # produced (FTL quantity, bottle-pack notes, pallet notes). The caller's
    # list and safe_data may be the same list; _assign unconditionally_ so both
    # paths land in storage even when safe_data arrived with a stale value.
    safe_data['error_flags'] = list(error_flags)
    return safe_data


# ─── offer_date parsing ────────────────────────────────────────────────────────
_FRENCH_MONTHS = {
    'janvier': 1, 'février': 2, 'fevrier': 2, 'mars': 3, 'avril': 4, 'mai': 5, 'juin': 6,
    'juillet': 7, 'août': 8, 'aout': 8, 'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12, 'decembre': 12,
}


def _parse_offer_date(raw) -> Optional[datetime]:
    """
    Parse the AI-extracted offer_date into a datetime. Handles:
      - ISO 8601 (YYYY-MM-DD or full)
      - French "11 mars 2026" / "11 mars 2026 à 14:40"
      - English "March 11, 2026" / "11 March 2026"
    Returns None on failure so the caller can fall back to utcnow.
    """
    if raw in (None, "", "Not Found"):
        return None
    if isinstance(raw, datetime):
        return raw
    s = str(raw).strip()
    # Strip common French time suffix
    s = re.sub(r'\s+à\s+\d{1,2}[:h]\d{2}.*$', '', s, flags=re.IGNORECASE)

    # Try ISO
    for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
        try:
            return datetime.strptime(s[:len(fmt) + 4], fmt) if fmt.endswith('S') else datetime.strptime(s[:10], fmt)
        except (ValueError, TypeError):
            pass

    # French: "11 mars 2026"
    m = re.match(r'(\d{1,2})\s+([a-zA-Zûéèçîôàâäïüö]+)\s+(\d{4})', s)
    if m:
        day, month_name, year = m.groups()
        mo = _FRENCH_MONTHS.get(month_name.lower())
        if mo:
            try:
                return datetime(int(year), mo, int(day))
            except ValueError:
                pass

    # English "March 11, 2026" / "11 March 2026"
    for fmt in ('%B %d, %Y', '%d %B %Y', '%d %b %Y', '%b %d, %Y'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass

    return None


_FILENAME_MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
    **_FRENCH_MONTHS,
}


def _date_from_filename(filename: str, fallback_year: int) -> Optional[datetime]:
    """
    Derive the document's own date from the source filename when the offer
    itself carries no parseable date (typical for stock lists / price lists,
    e.g. "STOCK LIST PERFUMES 13.04.xlsx", "W39_Designer_Offer_23.09.xlsx",
    "Cosmetics_Offer_09.06.2026.xlsx", "BACARDI MARTINI - OFFER (AUGUST 2026).xlsx",
    "FBC Trades - Perfumes stock list - 8 June 2026.pdf").

    This is the price's real age for the buyer, so it takes priority over the
    ingest date.  Returns None when no date can be read from the name, so the
    caller falls back to the received/ingest date.  A date that parses to the
    future (bad year inference) is rejected.
    """
    if not filename:
        return None
    name = str(filename)
    now = datetime.utcnow()

    def _ok(dt: Optional[datetime]) -> Optional[datetime]:
        # Reject future dates (e.g. a DD.MM whose inferred year overshoots).
        return dt if (dt is not None and dt <= now) else None

    # 0. ISO YYYY-MM-DD / YYYY.MM.DD / YYYY_MM_DD — unambiguous, and MUST win
    #    over the D.M patterns below (which would read 2026-11-05 as 11 May and
    #    stamp every row with the wrong offer date — Samuel QA 2026-11-05).
    m = re.search(r'(?<!\d)(\d{4})[.\-_](\d{1,2})[.\-_](\d{1,2})(?!\d)', name)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        try:
            return _ok(datetime(y, mo, d))
        except ValueError:
            pass

    # 1. D.M.YYYY or D-M-YYYY (e.g. 09.06.2026, 18.9.2026)
    m = re.search(r'(?<!\d)(\d{1,2})[.\-](\d{1,2})[.\-](\d{4})(?!\d)', name)
    if m:
        d, mo, y = (int(x) for x in m.groups())
        try:
            return _ok(datetime(y, mo, d))
        except ValueError:
            pass

    # 2. "8 June 2026" / "8 juin 2026"
    m = re.search(r'(?<!\d)(\d{1,2})\s+([A-Za-zûéèçîôàâäïüö]+)\s+(\d{4})', name)
    if m:
        d, mon, y = m.group(1), m.group(2).lower(), m.group(3)
        mo = _FILENAME_MONTHS.get(mon)
        if mo:
            try:
                return _ok(datetime(int(y), mo, int(d)))
            except ValueError:
                pass

    # 3. "August 2026" / "April 2026" (month + year, no day → first of month)
    m = re.search(r'([A-Za-zûéèçîôàâäïüö]{3,})\s+(\d{4})', name)
    if m:
        mo = _FILENAME_MONTHS.get(m.group(1).lower())
        if mo:
            try:
                return _ok(datetime(int(m.group(2)), mo, 1))
            except ValueError:
                pass

    # 4. D.M with no year (e.g. 13.04, 23.09) → infer the fallback year, and
    #    step back a year if that lands in the future.
    m = re.search(r'(?<!\d)(\d{1,2})[.\-](\d{1,2})(?!\d)', name)
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
        for y in (fallback_year, fallback_year - 1):
            try:
                cand = datetime(y, mo, d)
            except ValueError:
                break
            if cand <= now:
                return cand

    return None


def _resolve_offer_date(raw, source_filename: str, received: datetime) -> Optional[datetime]:
    """
    Resolve the stored offer_date, in priority order:
      1. a real date parsed from the document/header (AI-extracted),
      2. the date encoded in the source filename (stock lists / price lists),
      3. None — Samuel R-06 explicitly forbids inventing a commercial offer
         date. date_received remains available on the row as the arrival
         time, and the caller must emit a review flag so the dashboard
         labels this row as "offer date unknown — using received date for
         freshness display" instead of showing a fabricated commercial date.
    """
    parsed = _parse_offer_date(raw)
    if parsed is not None:
        return parsed
    from_name = _date_from_filename(source_filename or "", (received or datetime.utcnow()).year)
    if from_name is not None:
        return from_name
    return None


def _offer_date_corroborated(parsed: Optional[datetime], raw_row_text: str,
                             received: Optional[datetime]) -> bool:
    """Is an LLM-extracted offer_date actually supported by the source?

    Samuel 2026-11-11: the Jim Beam White Bourbon row carried offer_date
    2026-03-11 (→ "8 months old") though nothing in the source stated any date —
    the model invented it, and every sibling row from the same email had no
    offer_date. A commercial offer date must trace to the source, so we accept it
    only when it is (a) within 2 days of receipt — the ordinary same-day forward,
    or (b) literally present, in a common textual form, in the row's own source
    text. Otherwise it is an invention and is dropped (freshness then falls back
    to the received date, labelled)."""
    if parsed is None:
        return True
    try:
        if received and abs((parsed.date() - received.date()).days) <= 2:
            return True
    except Exception:
        pass
    raw = raw_row_text or ""
    if not raw:
        return False
    y, m, d = parsed.year, parsed.month, parsed.day
    # Numeric forms (ISO + day/month/year in either order, several separators).
    numeric = {
        f"{y:04d}-{m:02d}-{d:02d}", f"{y:04d}/{m:02d}/{d:02d}",
        f"{d:02d}/{m:02d}/{y}", f"{d}/{m}/{y}",
        f"{d:02d}.{m:02d}.{y}", f"{d}.{m}.{y}",
        f"{d:02d}-{m:02d}-{y}", f"{d}-{m}-{y}",
        f"{m:02d}/{d:02d}/{y}", f"{m}/{d}/{y}",
    }
    if any(tok in raw for tok in numeric):
        return True
    # Month-name forms (English + French): "11 March", "March 11", "11 mars".
    low = raw.lower()
    months_en = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
                 'august', 'september', 'october', 'november', 'december']
    fr_names = [name for name, num in _FRENCH_MONTHS.items() if num == m]
    names = [months_en[m - 1]] + fr_names
    for nm in names:
        if f"{d} {nm}" in low or f"{nm} {d}" in low:
            return True
    return False


def _offer_date_or_flag(raw, source_filename: str, received: datetime,
                       safe_data: dict) -> Optional[datetime]:
    """Client §4 (2026-11-06): the COMMERCIAL offer_date is only ever a date the
    offer itself stated. A date read from the source FILENAME is NOT a
    commercial offer date — it is returned separately (see the `document_date`
    keys set on safe_data) and must never drive expiry, alerts or history.

    Returns the commercial offer_date (parsed from the offer, else None) and, as
    a side effect, writes `document_date` / `document_date_source` onto
    safe_data plus the appropriate labelled review flag.
    """
    commercial = _parse_offer_date(raw)
    # Reject an invented offer_date not supported by the source (Samuel Jim Beam).
    if commercial is not None and not _offer_date_corroborated(
            commercial, safe_data.get('_raw_row_text') or '', received):
        try:
            _fl = safe_data.get('error_flags') or []
            if isinstance(_fl, list):
                _msg = (f"offer_date {commercial.date().isoformat()} was not stated "
                        f"anywhere in the source — dropped as unverified; freshness "
                        f"shown from the received date")
                if _msg not in _fl:
                    _fl.append(_msg)
                safe_data['error_flags'] = _fl
        except Exception:
            pass
        commercial = None
    from_name = _date_from_filename(source_filename or "",
                                    (received or datetime.utcnow()).year)
    try:
        flags = safe_data.get('error_flags') or []
        if not isinstance(flags, list):
            flags = []

        # Filename-derived date → store separately with provenance, never as
        # the commercial date.
        if from_name is not None:
            safe_data['document_date'] = from_name
            # VARCHAR(255) column — truncate the provenance label defensively.
            safe_data['document_date_source'] = (
                f"source filename: {os.path.basename(str(source_filename))}")[:255]
        else:
            safe_data['document_date'] = None
            safe_data['document_date_source'] = None

        if commercial is None:
            if from_name is not None:
                # Martell VS: blank per-offer date, but the QA filename encodes
                # one. Keep the commercial date UNKNOWN; surface the filename
                # date only as a provenance-labelled document date.
                _msg = ("offer date not stated in the offer — commercial offer "
                        f"date kept unknown. A date ({from_name.date().isoformat()}) "
                        "was read from the source filename and is stored "
                        "separately as the document date (provenance: filename); "
                        "it is NOT used as the commercial date, for expiry or for "
                        "alerts. Freshness shown from the received date, labelled.")
            else:
                _msg = ("offer date not stated in source — stored as unknown "
                        "(freshness displayed from date_received as a labelled "
                        "fallback, never as a commercial offer date)")
            if _msg not in flags:
                flags.append(_msg)
            safe_data['needs_manual_review'] = True
        safe_data['error_flags'] = flags
    except Exception:
        pass
    return commercial


# ─── commercial conditions (client §6.3) ────────────────────────────────────
_WEEKDAYS = {
    'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4,
    'saturday': 5, 'sunday': 6,
    'lundi': 0, 'mardi': 1, 'mercredi': 2, 'jeudi': 3, 'vendredi': 4,
    'samedi': 5, 'dimanche': 6,
}


def _resolve_relative_validity(raw: str, anchor: Optional[datetime]):
    """Resolve a relative validity term ("tomorrow", "today", a weekday) against
    a RELIABLE anchor — the offer's own stated date, else the original message
    received date — never the reprocessing date (client §6.3). Returns
    (iso_date | None, basis_note, uncertain_bool). When no reliable anchor is
    available the wording is preserved and uncertainty flagged."""
    s = (raw or "").strip().lower().rstrip('.')
    if not s:
        return None, "no validity term", True
    # An explicit date in the term wins and needs no anchor.
    explicit = _parse_offer_date(raw)
    if explicit is not None:
        return explicit.date().isoformat(), "explicit date stated in the offer", False
    if anchor is None:
        return None, "no reliable original message date available", True
    base = anchor.date().isoformat()
    if s in ('tomorrow', 'the next day', 'next day', 'demain'):
        return (anchor + timedelta(days=1)).date().isoformat(), \
            f"anchored to message date {base} + 1 day", False
    if s in ('today', 'end of day', 'eod', "aujourd'hui", 'this day'):
        return anchor.date().isoformat(), f"anchored to message date {base}", False
    for name, idx in _WEEKDAYS.items():
        if re.search(rf'\b{name}\b', s):
            delta = (idx - anchor.weekday()) % 7
            delta = delta or 7  # a bare weekday means the NEXT one
            return (anchor + timedelta(days=delta)).date().isoformat(), \
                f"anchored to message date {base} → next {name.title()}", False
    return None, "relative validity term not understood", True


# An explicitly stated offer/original date in the body, e.g.
# "Original offer date: 7 November 2026." — the reliable anchor for relative
# validity ("valid till tomorrow"), so it never depends on the reprocessing
# date (Samuel rule #1, Cedar Bay 2026-11-07).
_STATED_OFFER_DATE_RE = re.compile(
    r'(?:original\s+offer\s+date|offer\s+date|date\s+of\s+offer|'
    r'offer\s+issued(?:\s+on)?|issued\s+on|dated)\s*[:\-]?\s*'
    r'(\d{1,2}\s+[A-Za-zûéèçîôàâäïüö]+\s+\d{4}'
    r'|\d{4}-\d{2}-\d{2}'
    r'|\d{1,2}[./]\d{1,2}[./]\d{2,4}'
    r'|[A-Za-z]+\s+\d{1,2},?\s+\d{4})',
    re.IGNORECASE)


def _parse_stated_offer_date(text: str) -> Optional[datetime]:
    """Return the offer/original date stated in the body text, or None."""
    if not text:
        return None
    m = _STATED_OFFER_DATE_RE.search(str(text))
    if not m:
        return None
    return _parse_offer_date(m.group(1))


def _extract_commercial_conditions(text: str, anchor: Optional[datetime]):
    """Client §6.3 — capture offer-level commercial conditions verbatim
    (payment, PO requirement, bonded-warehouse delivery, validity) and resolve
    relative validity against a reliable anchor. Returns
    (conditions_text | None, resolved_valid_until | None, flags:list)."""
    if not text:
        return None, None, []
    t = str(text)
    found, flags = [], []
    resolved_valid_until = None

    # Validity — captured verbatim, then resolved when possible.
    mv = re.search(r'valid(?:ity)?\s+(?:is\s+)?(?:till|until|thru|through|up to)\s+'
                   r'([^\.,;\n]+)', t, re.IGNORECASE)
    if mv:
        raw_validity = mv.group(1).strip()
        found.append(f"Validity: valid until {raw_validity}")
        resolved, basis, uncertain = _resolve_relative_validity(raw_validity, anchor)
        if resolved is not None:
            resolved_valid_until = resolved
            flags.append(f"validity '{raw_validity}' resolved to {resolved} ({basis})")
        else:
            flags.append(f"validity '{raw_validity}' preserved verbatim — not "
                         f"resolved to a date ({basis}); no expiry computed")
            # uncertainty must be visible, not silently dropped
            if uncertain:
                pass

    # PO requirement
    if re.search(r'immediate\s+p\.?\s?o\.?|p\.?\s?o\.?\s+required|'
                 r'required\s+immediate\s+p\.?\s?o\.?|purchase\s+order\s+required',
                 t, re.IGNORECASE):
        found.append("Immediate PO required")

    # Payment terms
    if re.search(r'\bT\.?\s?T\.?\b|telegraphic\s+transfer|wire\s+transfer|'
                 r'bank\s+transfer', t, re.IGNORECASE):
        found.append("Payment: TT (bank transfer)")
    if re.search(r'\bL\.?\s?/?\s?C\b|letter\s+of\s+credit', t, re.IGNORECASE):
        found.append("Payment: L/C (letter of credit)")
    if re.search(r'\b(?:cash\s+in\s+advance|prepayment|advance\s+payment|'
                 r'payment\s+in\s+advance)\b', t, re.IGNORECASE):
        found.append("Payment: in advance")

    # Delivery / customs condition
    if re.search(r'excise[-\s]?bonded|bonded\s+warehouse|under\s+bond|in\s+bond|'
                 r'delivery\s+to\s+excise', t, re.IGNORECASE):
        found.append("Delivery to an excise-bonded warehouse")

    if not found:
        return None, resolved_valid_until, flags
    return "; ".join(found), resolved_valid_until, flags


def _apply_offer_level(safe_data: dict, conditions, cond_valid_until,
                       cond_flags, source_type, source_flags) -> None:
    """Stamp offer-level fields (client §6.3 conditions, §2 source provenance)
    onto a per-product safe_data before the OfferItem is built. Relative
    validity only fills valid_until when the offer did not state an explicit
    one. All condition / source notes are merged into error_flags so nothing
    is silently dropped."""
    if conditions:
        safe_data['commercial_conditions'] = conditions
    if source_type:
        # VARCHAR(255) column — truncate, never let a long value fail the save.
        safe_data['source_type'] = str(source_type)[:255]
    if cond_valid_until and not safe_data.get('valid_until'):
        safe_data['valid_until'] = cond_valid_until
    flags = safe_data.get('error_flags') or []
    if not isinstance(flags, list):
        flags = []
    for f in list(cond_flags or []) + list(source_flags or []):
        if f not in flags:
            flags.append(f)
    safe_data['error_flags'] = flags


def _apply_offer_defaults(data: dict) -> dict:
    """
    Replace AI's 'Not Found' with the proper default value according to the OfferItem schema.
    For optional fields with default None, 'Not Found' becomes None.
    For fields with a non‑None default (e.g. "Bottle", "EUR"), that default is applied.
    """
    defaults = {
        # fields with non‑None defaults (keep 'Not Found' only if it is the default)
        'product_name': "Not Found",
        'product_key': "Not Found",
        'brand': "Not Found",
        'packaging': "Bottle",
        'packaging_raw': "bottle",
        'refillable_status': "",
        'currency': "EUR",
        'incoterm': "Not Found",
        'location': "Not Found",
        'lead_time': "Not Found",
        'source_channel': "",
        'source_message_id': "",
        'source_filename': "",
        # client §4 — label language must come from the source, never a blanket
        # "EN" default (every row was reading EN regardless of the document).
        'label_language': None,
        'processing_version': "1.0.0",  # will be overridden in OfferItem

        # optional fields (should become None when missing)
        'category': None,
        'sub_category': None,
        'bottle_or_can_type': None,
        'cases_per_pallet': None,
        'quantity_case': None,
        'quantity_unit': None,
        'gift_box': None,
        # Multi-category (Phase 3 M1)
        'range_name': None,
        'category_slug': None,
        'perfume_format': None,
        'retail_state': None,
        'gender': None,
        'product_type': None,
        'shade': None,
        'size_weight_g': None,
        'alcohol_percent': None,
        'origin_country': None,
        'supplier_country': "",
        'moq_cases': None,
        'valid_until': None,
        'best_before_date': None,
        'vintage': None,
        'supplier_name': None,
        'supplier_email': None,
        'sender_name': None,
        'sender_email': None,
        'supplier_reference': None,
        'custom_status': None,
        'ean_code': None,
        'product_reference': None,

        # fx and price fields (already handled by _safe_float, but we keep them here for completeness)
        'fx_date': None,
        'fx_source': None,
        'fx_rate': 1.0,
        'price_per_unit': 0,
        'price_per_unit_eur': 0,
        'price_per_case': 0,
        'price_per_case_eur': 0,
        'unit_volume_ml': 0,
        'units_per_case': 0,
    }

    for field, default in defaults.items():
        if field in data:
            if data[field] == "Not Found" or data[field] is None:
                data[field] = default
    return data


async def process_offer(payload, job_id: str):
    try:
        uid = str(uuid.uuid4())
        extracted_data = {}
        all_products = []
        valid_count = 0

        redis_manager.set_job_status(job_id, "processing")

        # Idempotent (re)processing — Samuel 2026-11-12. A big file whose task is
        # hard-killed mid-import (OOM / Azure container recycle on the B1 box) is
        # redelivered by the broker (acks_late + visibility_timeout) and re-runs
        # under the SAME job_id. Without this, the re-run APPENDED a second copy
        # of every row — W46 reached 646 rows for a 367-row file and burned a
        # second full extraction. Clear any rows this job wrote on a previous
        # attempt so the redelivery rewrites ONE clean set (never duplicates),
        # and reset the source-file counters it rolls up. First run: a no-op.
        try:
            from core.database import get_session_factory
            from models.offer_item import OfferItemDB
            from models.source_file import SourceFileDB
            _idb = get_session_factory()()
            try:
                _prior = (_idb.query(OfferItemDB)
                              .filter(OfferItemDB.job_id == job_id)
                              .delete(synchronize_session=False))
                _sf = (_idb.query(SourceFileDB)
                           .filter(SourceFileDB.job_id == job_id).first())
                if _sf:
                    _sf.product_count = 0
                    _sf.imported_row_count = 0
                    _sf.import_incomplete = False
                _idb.commit()
                if _prior:
                    logger.warning(
                        f"Idempotent re-process: cleared {_prior} row(s) from a "
                        f"previous attempt of job {job_id} before re-writing "
                        f"(redelivery/retry of an interrupted import).")
            finally:
                _idb.close()
        except Exception as _ce:
            logger.warning(f"Idempotent pre-clear skipped for job {job_id}: {_ce}")

        logger.info(f"Processing job {job_id} for supplier: {payload.supplier_name}")
        logger.info(f"Text body length: {len(payload.text_body) if payload.text_body else 0}")
        logger.info(f"Attachment count: {len(payload.attachments) if payload.attachments else 0}")

        # Client §2/§7 — track the content extraction actually ran on, so the
        # stored source is honest (never a signature logo) and nothing is
        # silently dropped.
        from core.ingest_service import is_signature_attachment
        _channel = (payload.source_channel or "").lower()
        _body_label = ("whatsapp message" if "whatsapp" in _channel or _channel == "wa"
                       else "email body")
        _extracted_sources = []    # human labels of content that yielded products
        _skipped_signatures = []   # signature/decorative images ignored
        _empty_attachments = []    # readable attachments that produced no rows

        if payload.text_body:
            try:
                extracted_data = await extract_offer(payload.text_body or "")
                logger.info(f"Text extraction completed for job {job_id}")
                if isinstance(extracted_data, dict) and 'products' in extracted_data:
                    products_list = extracted_data['products']
                    logger.info(f"Found {len(products_list)} products in text")
                    all_products.extend(products_list)
                else:
                    logger.info("Using extracted_data as single product")
                    all_products.append(extracted_data)
                if all_products:
                    _extracted_sources.append(_body_label)

            except OpenAICreditsExhausted:
                raise
            except Exception as e:
                logger.error(f"Error in text extraction for job {job_id}: {e}")

        if payload.attachments:
            for attachment in payload.attachments:
                try:
                    file_name = attachment.fileName

                    # Client §2/§7 — a signature / decorative image is NOT an
                    # offer source. Skip it before extraction (no wasted vision
                    # call, no phantom source) and record that it was ignored.
                    # BUT the name-based signature filter exists for EMAIL inline
                    # decorations (Outlook "image001.gif" logos). A WhatsApp photo
                    # offer arrives with NO filename and is given the generic
                    # fallback name "image.jpg", which matches that filter — so
                    # every caption-less WhatsApp photo (a core offer channel) was
                    # being dropped before vision (0 OpenAI calls, no source row,
                    # DONE with 0 products — Samuel's TRADOR image, 2026-11-12).
                    # WhatsApp media is a deliberately-sent offer, never an inline
                    # signature: never signature-skip it.
                    _is_wa = _channel in ("whatsapp", "wa")
                    if not _is_wa and is_signature_attachment(attachment):
                        logger.info(
                            f"Skipping signature/decorative image (not an offer "
                            f"source): {file_name}")
                        _skipped_signatures.append(file_name)
                        continue

                    logger.info(f"Processing attachment: {file_name}")

                    file_ext = file_name.split('.')[-1].lower() if '.' in file_name else ''
                    file_bytes = await resolve_attachment_bytes(attachment)

                    if not file_bytes:
                        logger.warning(f"Could not parse file data for {file_name}")
                        _empty_attachments.append(file_name)
                        continue

                    # Durable archive (never-lose-files, Samuel 2026-11-11):
                    # persist the raw bytes to Postgres. /home on this App Service
                    # is ephemeral and is wiped on every restart/deploy, so a
                    # binary attachment saved only there is lost (this is why the
                    # CHELSEA images / Alder PDF could not be recovered). The DB
                    # copy survives. Best-effort — never breaks extraction.
                    try:
                        from core.raw_blob import save_blob
                        save_blob(job_id, file_name, attachment.contentType, file_bytes)
                    except Exception as _be:
                        logger.warning(f"raw_blob archive skipped for {file_name}: {_be}")

                    with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=f'.{file_ext}' if file_ext else '.bin'
                    ) as tmp_file:
                        tmp_file.write(file_bytes)
                        tmp_file_path = tmp_file.name

                    try:
                        file_extracted = await extract_from_file(
                            tmp_file_path,
                            attachment.contentType
                        )

                        # Client §2/§7 — record whether this (readable, non-
                        # signature) attachment actually produced offer content.
                        _produced = False
                        if file_extracted:
                            if ('products' in file_extracted
                                    and isinstance(file_extracted['products'], list)):
                                _produced = len(file_extracted['products']) > 0
                            else:
                                _produced = any(
                                    v not in (None, "", [])
                                    for v in file_extracted.values())
                        if _produced:
                            _extracted_sources.append(f"attachment: {file_name}")
                        else:
                            _empty_attachments.append(file_name)

                        if file_extracted:
                            logger.info(f"File extraction completed for {file_name}")

                            # Capture the raw document text so the
                            # deterministic-defaults pass later can parse
                            # "EXW Rotterdam / EUR / FBC Trades" from the
                            # actual PDF / XLSX content rather than the
                            # (usually-empty) email body on an
                            # attachment-only ingest.
                            if file_extracted.get('source_text'):
                                extracted_data.setdefault('_source_texts', []).append(
                                    file_extracted['source_text']
                                )
                            # Ground-truth source-row count (from the
                            # extractor's barcode regex on the raw PDF /
                            # row count on XLSX). Written to
                            # source_files.expected_row_count so the
                            # import_incomplete flag reflects
                            # source-vs-imported, not LLM-vs-imported.
                            if file_extracted.get('source_row_estimate') is not None:
                                extracted_data['_source_row_estimate'] = \
                                    file_extracted['source_row_estimate']

                            # Check if this is multiple products from Excel
                            if 'products' in file_extracted and isinstance(file_extracted['products'], list):
                                logger.info(f"Found {len(file_extracted['products'])} products in Excel file")
                                all_products.extend(file_extracted['products'])
                                # Copy document-level metadata extracted from
                                # the XLSX header rows (offer_date, incoterm,
                                # location, currency, supplier_name) as
                                # fallbacks — only if not already set by text
                                # extraction. These are absent from product_data
                                # dicts so merged_data picks them up from here.
                                for _mf in ('offer_date', 'incoterm', 'location', 'currency', 'supplier_name'):
                                    _fv = file_extracted.get(_mf)
                                    if _fv and not extracted_data.get(_mf):
                                        extracted_data[_mf] = _fv
                                        logger.info(f"[process_offer] XLSX metadata fallback: {_mf}={_fv!r}")
                            else:
                                # Single product extraction (old format)
                                for key, value in file_extracted.items():
                                    if value not in [None, "", []]:
                                        extracted_data[key] = value

                    finally:
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass

                except OpenAICreditsExhausted:
                    raise
                except Exception as e:
                    logger.error(f"Error processing attachment {attachment.fileName}: {e}")
                    continue

        # ── Client §6.3 — offer-level commercial conditions (computed once) ──
        # Scan the real offer content (email/WhatsApp body + any document text)
        # for validity / PO / payment / bonded-warehouse terms, preserve them
        # verbatim, and resolve relative validity ("valid till tomorrow")
        # against a RELIABLE anchor — the offer's own date if stated, else the
        # original message received time — never the reprocessing date.
        _cond_source_text = (payload.text_body or "")
        for _dt in (extracted_data.get('_source_texts') or []):
            _cond_source_text += "\n" + str(_dt)
        # Prefer an offer date EXPLICITLY STATED in the body ("Original offer
        # date: 7 November 2026") over the LLM's document-level offer_date — the
        # stated date is the authoritative original date and must never be
        # displaced by the reprocessing/received date (Samuel rule #1).
        _cond_anchor = (_parse_stated_offer_date(_cond_source_text)
                        or _parse_offer_date(extracted_data.get('offer_date'))
                        or _received_at(payload))
        _commercial_conditions, _cond_valid_until, _cond_flags = \
            _extract_commercial_conditions(_cond_source_text, _cond_anchor)
        # Client §2 — a durable copy of the raw offer content, stored on the
        # SourceFileDB so the source survives Redis payload expiry (capped to
        # keep the row sane).
        _durable_source_text = (_cond_source_text or "").strip()[:200_000] or None

        # ── Client §2/§7 — source provenance + nothing silently dropped ──────
        _seen_src, _job_source_type_parts = set(), []
        for _s in _extracted_sources:
            if _s not in _seen_src:
                _seen_src.add(_s)
                _job_source_type_parts.append(_s)
        _job_source_type = "; ".join(_job_source_type_parts) or None
        _source_flags = []
        for _sig in _skipped_signatures:
            _source_flags.append(
                f"ignored signature/decorative image (not an offer source): {_sig}")
        for _emp in _empty_attachments:
            _source_flags.append(
                f"attachment '{_emp}' produced no readable offer content — "
                f"flagged for manual review, not silently dropped")

        # Deduplicate products extracted from multiple sources (text body + attachments,
        # or the same product listed twice). Key on normalised name + volume + price.
        if all_products:
            # EAN-first dedup: two product dicts that share brand +
            # product_name + unit_volume_ml + ean_code + price +
            # retail_state are the SAME SKU. retail_state is included
            # because the same EAN can legitimately appear at different
            # retail states (e.g. Clean EU vs Deco) as distinct line
            # items — matches admin._canon_key. Perfume_format, shade,
            # and product_type are still NOT in the key: a null-vs-
            # populated mismatch for those does not create a new SKU.
            def _s(v):
                return str(v).strip().lower() if v is not None else ""
            def _canon_key(p):
                # Collapse ONLY true extractor duplicates (same product emitted
                # twice, or from body + attachment). Two rows that differ in any
                # commercial field below are DISTINCT offers and must both
                # survive — a supplier legitimately lists the same SKU at a
                # different warehouse, incoterm, price, quantity, date, lot or
                # customs status (Samuel's "Cas école" dedup test: 3 Jameson at
                # Loendersloot/Revera/Mentrex, 3 Blackberry at different dates).
                # Mirrors the display fingerprint in api/records.get_records.
                return (
                    _s(p.get("brand")), _s(p.get("product_name")),
                    p.get("unit_volume_ml") or 0, p.get("units_per_case") or 0,
                    (p.get("ean_code") or "").strip().lstrip("0"),
                    p.get("price_per_unit") or 0, p.get("price_per_case") or 0,
                    _s(p.get("currency")), _s(p.get("retail_state")),
                    _s(p.get("supplier_reference")), _s(p.get("product_reference")),
                    _s(p.get("incoterm")), _s(p.get("location")),
                    p.get("quantity_case") or 0, p.get("quantity_bottles") or 0,
                    p.get("quantity_pallets") or 0, _s(p.get("quantity_unit")),
                    _s(p.get("offer_date")), _s(p.get("valid_until")),
                    _s(p.get("custom_status")), _s(p.get("vintage")),
                )
            def _completeness(p):
                return sum(1 for v in (p.get("gender"), p.get("retail_state"),
                                       p.get("shade"), p.get("perfume_format"),
                                       p.get("product_type"))
                           if v and str(v).strip())
            kept = {}
            for p in all_products:
                k = _canon_key(p)
                prev = kept.get(k)
                if prev is None or _completeness(p) > _completeness(prev):
                    if prev is not None:
                        logger.info(f"Dedup: replacing '{prev.get('product_name')}' with more-complete copy")
                    kept[k] = p
                else:
                    logger.info(f"Dedup: dropping duplicate product '{p.get('product_name')}'")
            deduped = list(kept.values())
            if len(deduped) != len(all_products):
                logger.info(f"Dedup: {len(all_products)} → {len(deduped)} products")
            all_products = deduped

            # ── Deterministic backend rules (see core/deterministic_rules.py) ───
            # Any fact that's derivable from the document header (currency,
            # incoterm) should NOT depend on the LLM. If the source says
            # "Net price (EUR)" every row gets currency=EUR regardless of
            # whether the model captured it correctly.
            from core.deterministic_rules import apply_deterministic_defaults
            header_text = (payload.text_body or "")
            if payload.attachments:
                header_text = f"Subject: {payload.subject or ''}\nFrom: {payload.sender_email or ''}\n\n" + header_text
            # Append the raw document text(s) so header detection sees
            # "EXW Rotterdam (EUR)", "FBC Trades", etc. — not just the
            # surrounding email metadata.
            for _doc_text in (extracted_data.get('_source_texts') or []):
                header_text = header_text + "\n\n" + _doc_text
            all_products, drule_corrections = apply_deterministic_defaults(
                all_products, header_text,
                source_filename=payload.source_filename,
                sender_email=payload.sender_email,
            )
            if drule_corrections:
                logger.info(f"Deterministic rules applied {len(drule_corrections)} corrections:")
                for c in drule_corrections[:10]:
                    logger.info(f"  - {c}")

        # Create offers
        offers = []

        if all_products:
            logger.info(f"Processing {len(all_products)} products from Excel for job {job_id}")

            for idx, product_data in enumerate(all_products):
                try:
                    merged_data = {**extracted_data, **product_data}
                    # Re-apply pre-scan XLSX metadata when the LLM returned a
                    # sentinel — "Not Found" from product_data would otherwise
                    # overwrite the header-row values we extracted before batching.
                    # EXCEPT incoterm/location on a bare '@'-price row that the
                    # source-authoritative reconcile deliberately cleared (price
                    # basis ambiguous, e.g. Beefeater 70cl @€75): that row has NO
                    # delivery terms of its own, so an offer-level incoterm/location
                    # bled from another supplier's line ("ex NL bond") must NOT be
                    # back-filled onto it (Samuel 2026-11-10 round 3).
                    _recon_ambiguous = any(
                        isinstance(f, str) and 'basis ambiguous' in f
                        for f in (product_data.get('error_flags') or []))
                    for _mf in ('offer_date', 'incoterm', 'location', 'currency', 'supplier_name'):
                        if _recon_ambiguous and _mf in ('incoterm', 'location'):
                            continue
                        if merged_data.get(_mf) in (None, '', 'Not Found', 'nan', 'NaN') and extracted_data.get(_mf) not in (None, '', 'Not Found', 'nan', 'NaN'):
                            merged_data[_mf] = extracted_data[_mf]

                    if merged_data.get('product_key') in [None, '', 'Not Found']:
                        logger.warning(
                            f"Row {idx} skipped — product_key not found: "
                            f"name={merged_data.get('product_name')!r}"
                        )
                        continue

                    logger.debug(
                        f"Row {idx} raw data: {merged_data.get('product_name')} | {merged_data.get('brand')}")

                    _dp_brand, _dp_name = depollute_identity(
                        merged_data.get('brand'), merged_data.get('product_name'))
                    # client §4 — rebuild the identity key from the FINAL
                    # (depolluted) brand/name + size + real pack, so the key
                    # never carries a stale pre-depollute name, a bare "VSOP"
                    # (no brand), or a logistics pack like "112CS".
                    from core.normalization import build_product_key
                    _pkey = build_product_key(_dp_brand, _dp_name,
                                              merged_data.get('unit_volume_ml'),
                                              merged_data.get('units_per_case'))
                    safe_data = {
                        'product_name': _dp_name or "Not Found",
                        'product_key': _pkey or merged_data.get('product_key') or "Not Found",
                        'brand': _dp_brand or "Not Found",
                        # Phase 3 M2 — raw source-row text threaded through from
                        # core.openai_client.extract_from_file so the customs
                        # backfill (see _backfill_customs_from_text) has a
                        # third fallback when GPT-4o cleaned the token out
                        # of brand/product_name without populating
                        # custom_status.
                        '_raw_row_text': merged_data.get('_raw_row_text'),
                        'range_name': merged_data.get('range_name'),
                        'category': merged_data.get('category'),
                        'sub_category': merged_data.get('sub_category'),
                        # Multi-category (Phase 3 M1) — MUST be here or PDF/Excel
                        # batch products land in the DB with None on every one
                        # of these fields even when the LLM emitted them and
                        # apply_deterministic_defaults set category_slug.
                        'category_slug': merged_data.get('category_slug'),
                        'perfume_format': merged_data.get('perfume_format'),
                        'retail_state': merged_data.get('retail_state'),
                        'gender': merged_data.get('gender'),
                        'product_type': merged_data.get('product_type'),
                        'shade': merged_data.get('shade'),
                        'size_weight_g': merged_data.get('size_weight_g'),
                        'packaging': merged_data.get('packaging') or "Bottle",
                        'packaging_raw': merged_data.get('packaging_raw') or "bottle",
                        'bottle_or_can_type': merged_data.get('bottle_or_can_type'),
                        'unit_volume_ml': _safe_float(merged_data.get('unit_volume_ml')),
                        'units_per_case': _safe_float(merged_data.get('units_per_case')),
                        'cases_per_pallet': merged_data.get('cases_per_pallet'),
                        'quantity_case': merged_data.get('quantity_case'),
                        # client §2 — carry the original-unit quantity through
                        # to the DB row (was dropped here, so "3 pallets" /
                        # "9600 bottles" / "24 pieces" showed blank).
                        'quantity_bottles': merged_data.get('quantity_bottles'),
                        'quantity_pallets': merged_data.get('quantity_pallets'),
                        'quantity_ftl': merged_data.get('quantity_ftl'),
                        'gift_box': merged_data.get('gift_box'),
                        'refillable_status': merged_data.get('refillable_status') or "",
                        'currency': merged_data.get('currency') or "EUR",
                        'price_per_unit': _safe_float(merged_data.get('price_per_unit')),
                        'price_per_unit_eur': _safe_float(merged_data.get('price_per_unit_eur')),
                        'price_per_case': _safe_float(merged_data.get('price_per_case')),
                        'price_per_case_eur': _safe_float(merged_data.get('price_per_case_eur')),
                        'fx_rate': _safe_float(merged_data.get('fx_rate'), 1.0),
                        'fx_date': merged_data.get('fx_date'),
                        'fx_source': merged_data.get('fx_source'),
                        'alcohol_percent': merged_data.get('alcohol_percent'),
                        'origin_country': merged_data.get('origin_country'),
                        'supplier_country': merged_data.get('supplier_country') or "",
                        'incoterm': merged_data.get('incoterm') or "Not Found",
                        'location': merged_data.get('location') or "Not Found",
                        'lead_time': merged_data.get('lead_time') or "Not Found",
                        'moq_cases': merged_data.get('moq_cases'),
                        'moq_bottles': merged_data.get('moq_bottles'),
                        'moq_pallets': merged_data.get('moq_pallets'),
                        'moq_ftl': merged_data.get('moq_ftl'),
                        'moq_unit': merged_data.get('moq_unit'),
                        'quantity_unit': merged_data.get('quantity_unit'),
                        'min_order_quantity_case': merged_data.get('min_order_quantity_case'),
                        'port': merged_data.get('port') or "Not Found",
                        'valid_until': merged_data.get('valid_until'),
                        'best_before_date': merged_data.get('best_before_date'),
                        'vintage': merged_data.get('vintage'),
                        'supplier_reference': merged_data.get('supplier_reference'),
                        'ean_code': merged_data.get('ean_code'),
                        # client §4 — only when the source stated it (no "EN" default)
                        'label_language': merged_data.get('label_language') if merged_data.get('label_language') not in (None, '', 'Not Found') else None,
                        'product_reference': merged_data.get('product_reference'),
                        'custom_status': merged_data.get('custom_status') if merged_data.get('custom_status') not in [None, '', 'Not Found'] else None,
                        'supplier_name': merged_data.get('supplier_name'),
                        'supplier_email': merged_data.get('supplier_email'),  # AI extracted email
                        'error_flags': merged_data.get('error_flags', []),
                        # Phase 3 M2 — carry needs_manual_review from clean_product_data.
                        # clean_product_data sets this to True when an EAN fails
                        # check-digit validation (unfixable), or any other
                        # genuinely-ambiguous condition. Without it here the
                        # OfferItem construction below silently resets review=False.
                        'needs_manual_review': bool(merged_data.get('needs_manual_review', False)),
                    }

                    # Convert numeric fields (except those already handled by _safe_float)
                    numeric_fields = ['cases_per_pallet', 'quantity_case', 'moq_cases', 'min_order_quantity_case']
                    for field in numeric_fields:
                        if safe_data[field] is not None:
                            try:
                                safe_data[field] = float(safe_data[field])
                            except (ValueError, TypeError):
                                safe_data[field] = None

                    safe_data['alcohol_percent'] = (
                        float(str(safe_data['alcohol_percent']).replace('%', '').strip())
                        if safe_data.get('alcohol_percent') not in [None, '', 'Not Found']
                        else None
                    )

                    # Apply OfferItem defaults (convert "Not Found" to None or the proper default)
                    safe_data = _apply_offer_defaults(safe_data)

                    # ── Phase 3 M2 — customs_status fallback from raw text ─
                    # Case-school rows B08/B11/B12/B32 carry T1/T2 inside the
                    # raw description. GPT-4o sometimes leaves custom_status
                    # empty and the token sitting in brand or product_name.
                    # Extract it at ingest time (same logic as the admin
                    # backfill endpoint) so the structured column is populated
                    # on the row's first write.
                    _backfill_customs_from_text(safe_data)

                    # R-16 (Samuel 2026-10-30): keep the LLM's original
                    # product name + brand immutably on every row.
                    _preserve_raw_identity(safe_data)

                    # Compound-location preservation guard (Rule 7). Flags
                    # rows where the raw source listed ≥2 alternative
                    # locations and GPT kept only one.
                    _detect_dropped_alt_locations(safe_data)

                    # Drop the misleading "multiple incoterms detected"
                    # flag when the raw row has at most one incoterm —
                    # Samuel's Jameson fresh-test regression.
                    _drop_misleading_multi_incoterm_flag(safe_data)

                    # R-07 supersession — parse "replaces our earlier A01"
                    # before persistence so save_offer_to_db has the field.
                    _parse_supersedes_ref(safe_data)

                    # ── Currency normalisation & EUR conversion ──────────────
                    _snap_duplicated_row_per_unit(safe_data)
                    _fix_pallet_per_case_phrasing(safe_data)
                    _backfill_numeric_brand(safe_data)
                    _reconcile_multicurrency_retail(safe_data)
                    safe_data = _normalize_currency_and_prices(safe_data)
                    _normalize_incoterm_location(safe_data)

                    # ── MOQ / quantity unit-aware normalization (RULE 0.3) ───
                    _flag_bucket = list(safe_data.get('error_flags') or [])
                    safe_data = _normalize_moq_and_quantity(safe_data, _flag_bucket)
                    safe_data['error_flags'] = _flag_bucket

                    # Round-4 deterministic offer-shape fixes (Samuel 2026-11-11).
                    _apply_round4_shape_fixes(safe_data, _cond_source_text or '')

                    # client §5 — MOQ unit follows the SOURCE. When the source
                    # stated the MOQ in cases (moq_cases populated) and no other
                    # unit was captured, record "cases" so the display never
                    # inherits the product's stock/price unit (a loose item's
                    # "10 cs" must not become "10 pieces").
                    if safe_data.get('moq_cases') is not None and not safe_data.get('moq_unit'):
                        safe_data['moq_unit'] = 'cases'

                    # ── Parse units_per_case & unit_volume_ml from packaging ──
                    _pkg_str = str(safe_data.get('packaging') or '')
                    _pkg_m = re.search(
                        r'(\d+)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(cl|ml|l)',
                        _pkg_str, re.IGNORECASE
                    )
                    if _pkg_m:
                        _pu = int(_pkg_m.group(1))
                        _pv = float(_pkg_m.group(2).replace(',', '.'))
                        _pu_unit = _pkg_m.group(3).lower()
                        _cpp_now = _safe_float(safe_data.get('cases_per_pallet'))
                        safe_data['unit_volume_ml'] = (
                            _pv * 10 if _pu_unit == 'cl' else
                            _pv * 1000 if _pu_unit == 'l' else _pv
                        )
                        # Canonical display form — store raw LLM string first
                        if not safe_data.get('packaging_raw'):
                            safe_data['packaging_raw'] = _pkg_str
                        # When the "pack count" equals the cases-per-pallet, the
                        # LLM built packaging like "112x70cl" out of "112cs"
                        # (cases/pallet), not a real pack size (Samuel Edo). Take
                        # only the bottle volume; leave units_per_case as the
                        # pallet parser set it (null) AND do not fake a
                        # "112/case" pack label.
                        if not (_cpp_now and float(_pu) == _cpp_now):
                            safe_data['units_per_case'] = float(_pu)
                            safe_data['packaging'] = f"{_pu}/case"
                        else:
                            safe_data['packaging'] = (
                                f"{_pkg_m.group(2).replace(',', '.')}{_pu_unit}")
                        logger.debug(
                            f"Row {idx}: packaging '{_pkg_str}' → "
                            f"units_per_case={safe_data.get('units_per_case')}, "
                            f"unit_volume_ml={safe_data['unit_volume_ml']}"
                        )

                    # client §4 — rebuild the identity key from the FINAL
                    # corrected pack so a logistics value the LLM mistook for a
                    # case pack ("112cs" → units_per_case=112) never survives in
                    # the product_key (Edo JW: ..._112). build_product_key omits
                    # the pack token when units_per_case is null.
                    _final_key = build_product_key(
                        safe_data.get('brand'), safe_data.get('product_name'),
                        safe_data.get('unit_volume_ml'),
                        safe_data.get('units_per_case'))
                    if _final_key:
                        safe_data['product_key'] = _final_key

                    _apply_offer_level(
                        safe_data, _commercial_conditions, _cond_valid_until,
                        _cond_flags, _job_source_type, _source_flags)

                    _sup_name  = safe_data.get('supplier_name')
                    _sup_email = _resolve_supplier_email(payload, safe_data.get('supplier_email'))
                    # A reference glued onto the supplier name belongs in
                    # supplier_reference (Cedar Bay — ref. QA-20261107-W07).
                    if not safe_data.get('supplier_reference'):
                        _sr = _resolve_supplier_ref(_sup_name, payload)
                        if _sr:
                            safe_data['supplier_reference'] = _sr
                    # Split "Company - Person" into company + contact (Yamazaki).
                    _sup_company, _sup_contact = _split_company_contact(
                        _resolve_supplier_name(_sup_name, payload),
                        _cond_source_text, _sup_email)

                    offer = OfferItem(
                        uid=f"{uid}_{idx}",
                        product_name=safe_data['product_name'],
                        product_name_raw=safe_data.get('product_name_raw'),
                        product_key=safe_data['product_key'],
                        brand=safe_data['brand'],
                        brand_raw=safe_data.get('brand_raw'),
                        category=safe_data['category'],
                        sub_category=safe_data['sub_category'],
                        packaging=safe_data['packaging'],
                        packaging_raw=safe_data['packaging_raw'],
                        bottle_or_can_type=safe_data['bottle_or_can_type'],
                        unit_volume_ml=safe_data['unit_volume_ml'],
                        units_per_case=safe_data['units_per_case'],
                        cases_per_pallet=safe_data['cases_per_pallet'],
                        quantity_case=safe_data['quantity_case'],
                        quantity_unit=safe_data.get('quantity_unit'),
                        quantity_bottles=safe_data.get('quantity_bottles'),
                        quantity_pallets=safe_data.get('quantity_pallets'),
                        quantity_ftl=safe_data.get('quantity_ftl'),
                        range_name=safe_data.get('range_name'),
                        category_slug=safe_data.get('category_slug'),
                        perfume_format=safe_data.get('perfume_format'),
                        retail_state=safe_data.get('retail_state'),
                        gender=safe_data.get('gender'),
                        product_type=safe_data.get('product_type'),
                        shade=safe_data.get('shade'),
                        size_weight_g=safe_data.get('size_weight_g'),
                        gift_box=safe_data['gift_box'],
                        refillable_status=safe_data['refillable_status'],
                        currency=safe_data['currency'],
                        price_per_unit=safe_data['price_per_unit'],
                        price_per_unit_eur=safe_data['price_per_unit_eur'],
                        price_per_case=safe_data['price_per_case'],
                        price_per_case_eur=safe_data['price_per_case_eur'],
                        fx_rate=safe_data['fx_rate'],
                        fx_date=safe_data['fx_date'],
                        fx_source=safe_data.get('fx_source'),
                        alcohol_percent=safe_data['alcohol_percent'],
                        origin_country=safe_data['origin_country'],
                        supplier_country=safe_data['supplier_country'],
                        incoterm=safe_data['incoterm'],
                        location=safe_data['location'],
                        lead_time=safe_data['lead_time'],
                        moq_cases=safe_data['moq_cases'],
                        moq_unit=safe_data.get('moq_unit'),
                        moq_bottles=safe_data.get('moq_bottles'),
                        moq_pallets=safe_data.get('moq_pallets'),
                        moq_ftl=safe_data.get('moq_ftl'),
                        moq_text=safe_data.get('moq_text'),
                        valid_until=safe_data['valid_until'],
                        offer_date=_offer_date_or_flag(
                            merged_data.get('offer_date'),
                            payload.source_filename or "",
                            datetime.utcnow(), safe_data),
                        document_date=safe_data.get('document_date'),
                        document_date_source=safe_data.get('document_date_source'),
                        commercial_conditions=safe_data.get('commercial_conditions'),
                        source_type=safe_data.get('source_type'),
                        date_received=_received_at(payload),
                        best_before_date=safe_data['best_before_date'],
                        vintage=safe_data['vintage'],
                        supplier_name=_sup_company,
                        contact_name=_sup_contact,
                        supplier_email=_sup_email,
                        sender_name=payload.sender_name,
                        sender_email=payload.sender_email,
                        supplier_reference=safe_data['supplier_reference'],
                        source_channel=payload.source_channel or "",
                        source_message_id=payload.source_message_id or "",
                        source_filename=payload.source_filename or "",
                        attachment_filenames=[att.fileName for att in
                                              payload.attachments] if payload.attachments else [],
                        attachment_count=len(payload.attachments) if payload.attachments else 0,
                        confidence_score=0.95,
                        # Pulled from safe_data (threaded from clean_product_data).
                        # Rows that failed unfixable-EAN or any other genuine
                        # review condition arrive here with review=True; must
                        # not be hardcoded False.
                        needs_manual_review=safe_data.get('needs_manual_review', False),
                        error_flags=safe_data['error_flags'] if isinstance(safe_data.get('error_flags'), list) else [],
                        custom_status=safe_data['custom_status'],
                        processing_version="2.0.0",
                        ean_code=safe_data['ean_code'],
                        label_language=safe_data['label_language'],
                        product_reference=safe_data['product_reference'],
                        supersedes_ref=safe_data.get('supersedes_ref'),
                    )

                    offer_dict = offer.model_dump(mode='json')
                    # Phase 3 M1: thread content-hash + source-row estimate
                    # so save_offer_to_db can persist them on the SourceFileDB.
                    offer_dict["_content_hash"] = getattr(payload, "content_hash", None)
                    offer_dict["_expected_row_count"] = extracted_data.get("_source_row_estimate") or len(all_products)
                    offer_dict["_source_text"] = _durable_source_text

                    if not is_valid_offer(offer_dict):
                        logger.info(
                            f"Skipping invalid offer at row {idx}: "
                            f"name={offer_dict.get('product_name')!r}, "
                            f"price_unit={offer_dict.get('price_per_unit')}, "
                            f"price_case={offer_dict.get('price_per_case')}"
                        )
                        continue

                    # ── Loose-unit €/case cleanup ───────────────────────────
                    # If no case pack was declared (units_per_case null / 0 / 1),
                    # a per-case price is meaningless — either a display artifact
                    # or (R-14 bug) a copy of the per-bottle price. Null it so the
                    # dashboard shows "—" or "N/A (loose)" instead.
                    upc = offer_dict.get("units_per_case")
                    if not upc or upc in (0, 0.0, 1, 1.0):
                        # NARS retest bug: a loose pack whose price_per_case was a COPY of
                        # the per-bottle price was passing through. Treat loose-pack case
                        # prices as unavailable — UNLESS the source genuinely quoted a
                        # per-case price for a product ordered BY THE CASE (Samuel ATV:
                        # Cody Vody "3456 cases @ €23.40 per case"), where the bottles-per-
                        # case is simply unknown. That is a real case price the trader buys
                        # at, not an artifact, so it must survive.
                        _ppc = _safe_float(offer_dict.get("price_per_case"))
                        _ppu = _safe_float(offer_dict.get("price_per_unit"))
                        _by_case = (offer_dict.get("quantity_unit") or "").strip().lower() in ("cases", "case")
                        _is_copy = (_ppc is not None and _ppu is not None and _ppu > 0
                                    and abs(_ppc - _ppu) <= 0.01)
                        _genuine_case = bool(_by_case and _ppc and not _is_copy)
                        if not _genuine_case:
                            offer_dict["price_per_case"] = None
                            offer_dict["price_per_case_eur"] = None
                            if upc in (0, 0.0):
                                offer_dict["units_per_case"] = None
                            # Also null quantity_case if it was passed through as bottle count
                            # (defensive — the processor's _normalize_quantity_units should have
                            # already caught this, but belt-and-braces for direct DB writes).
                            if (offer_dict.get("quantity_unit") or "").lower() == "bottles":
                                offer_dict["quantity_case"] = None
                        elif offer_dict.get("price_per_unit") in (0, 0.0):
                            # Genuine per-case price with no derivable per-bottle price:
                            # show the unit price as blank, never €0 (Samuel: Cody Vody
                            # must not read "€0/unit").
                            offer_dict["price_per_unit"] = None
                            offer_dict["price_per_unit_eur"] = None

                    offers.append(offer_dict)
                    valid_count += 1

                    # ── Persist to database ──────────────────────────────────
                    save_offer_to_db(offer_dict, job_id)

                    logger.info(f"Writing product row directly to Excel: {offer_dict['product_name']}")
                    write_offer_row(offer_dict)

                except Exception as e:
                    error_trace = traceback.format_exc()
                    logger.error(f"Error creating offer for product {idx}: {e}\n{error_trace}")
                    continue

        else:
            logger.info(f"No Excel products found, using single product extraction for job {job_id}")
            try:
                _dp_brand, _dp_name = depollute_identity(
                    extracted_data.get('brand'), extracted_data.get('product_name'))
                # client §4 — rebuild identity key from final brand/name + size
                # + real pack (see batch-path note above).
                from core.normalization import build_product_key
                _pkey = build_product_key(_dp_brand, _dp_name,
                                          extracted_data.get('unit_volume_ml'),
                                          extracted_data.get('units_per_case'))
                safe_data = {
                    'product_name': _dp_name or "Not Found",
                    'product_key': _pkey or extracted_data.get('product_key') or "Not Found",
                    'brand': _dp_brand or "Not Found",
                    # Phase 3 M2 — raw source-row text (see batch-path note above).
                    '_raw_row_text': extracted_data.get('_raw_row_text'),
                    'category': extracted_data.get('category'),
                    'sub_category': extracted_data.get('sub_category'),
                    'packaging': extracted_data.get('packaging') or "Bottle",
                    'packaging_raw': extracted_data.get('packaging_raw') or "bottle",
                    'bottle_or_can_type': extracted_data.get('bottle_or_can_type'),
                    'unit_volume_ml': _safe_float(extracted_data.get('unit_volume_ml')),
                    'units_per_case': _safe_float(extracted_data.get('units_per_case')),
                    'cases_per_pallet': extracted_data.get('cases_per_pallet'),
                    'quantity_case': extracted_data.get('quantity_case'),
                    # client §2 — preserve original-unit quantity (no conversion)
                    'quantity_bottles': extracted_data.get('quantity_bottles'),
                    'quantity_pallets': extracted_data.get('quantity_pallets'),
                    'quantity_ftl': extracted_data.get('quantity_ftl'),
                    'gift_box': extracted_data.get('gift_box'),
                    'refillable_status': extracted_data.get('refillable_status') or "",
                    'currency': extracted_data.get('currency') or "EUR",
                    'price_per_unit': _safe_float(extracted_data.get('price_per_unit')),
                    'price_per_unit_eur': _safe_float(extracted_data.get('price_per_unit_eur')),
                    'price_per_case': _safe_float(extracted_data.get('price_per_case')),
                    'price_per_case_eur': _safe_float(extracted_data.get('price_per_case_eur')),
                    'fx_rate': _safe_float(extracted_data.get('fx_rate'), 1.0),
                    'fx_date': extracted_data.get('fx_date'),
                    'fx_source': extracted_data.get('fx_source'),
                    'alcohol_percent': extracted_data.get('alcohol_percent'),
                    'origin_country': extracted_data.get('origin_country'),
                    'supplier_country': extracted_data.get('supplier_country') or "",
                    'incoterm': extracted_data.get('incoterm') or "Not Found",
                    'location': extracted_data.get('location') or "Not Found",
                    'lead_time': extracted_data.get('lead_time') or "Not Found",
                    'moq_cases': extracted_data.get('moq_cases'),
                    'moq_bottles': extracted_data.get('moq_bottles'),
                    'moq_pallets': extracted_data.get('moq_pallets'),
                    'moq_ftl': extracted_data.get('moq_ftl'),
                    'moq_unit': extracted_data.get('moq_unit'),
                    'quantity_unit': extracted_data.get('quantity_unit'),
                    'min_order_quantity_case': extracted_data.get('min_order_quantity_case'),
                    'port': extracted_data.get('port') or "Not Found",
                    'valid_until': extracted_data.get('valid_until'),
                    'best_before_date': extracted_data.get('best_before_date'),
                    'vintage': extracted_data.get('vintage'),
                    'supplier_reference': extracted_data.get('supplier_reference'),
                    'ean_code': extracted_data.get('ean_code'),
                    # client §4 — only when the source stated it (no "EN" default)
                    'label_language': extracted_data.get('label_language') if extracted_data.get('label_language') not in (None, '', 'Not Found') else None,
                    'product_reference': extracted_data.get('product_reference'),
                    'custom_status': extracted_data.get('custom_status') if extracted_data.get('custom_status') not in [None, '', 'Not Found'] else None,
                    'supplier_name': extracted_data.get('supplier_name'),
                    'supplier_email': extracted_data.get('supplier_email'),  # AI extracted email
                    'error_flags': extracted_data.get('error_flags', []),
                    # Phase 3 M2 — carry needs_manual_review from clean_product_data.
                    # Same reason as the batch-path safe_data above: without it
                    # the OfferItem construction silently resets review=False.
                    'needs_manual_review': bool(extracted_data.get('needs_manual_review', False)),
                }

                # Convert numeric fields
                numeric_fields = ['cases_per_pallet', 'quantity_case', 'moq_cases', 'min_order_quantity_case']
                for field in numeric_fields:
                    if safe_data[field] is not None:
                        try:
                            safe_data[field] = float(safe_data[field])
                        except (ValueError, TypeError):
                            safe_data[field] = None

                safe_data['alcohol_percent'] = (
                    float(str(safe_data['alcohol_percent']).replace('%', '').strip())
                    if safe_data.get('alcohol_percent') not in [None, '', 'Not Found']
                    else None
                )

                safe_data = _apply_offer_defaults(safe_data)

                # ── Phase 3 M2 — customs_status fallback from raw text ─
                # Mirrors the batch path above; see note there.
                _backfill_customs_from_text(safe_data)
                _preserve_raw_identity(safe_data)
                _detect_dropped_alt_locations(safe_data)
                _drop_misleading_multi_incoterm_flag(safe_data)
                _parse_supersedes_ref(safe_data)

                _fix_pallet_per_case_phrasing(safe_data)
                _backfill_numeric_brand(safe_data)
                _reconcile_multicurrency_retail(safe_data)
                safe_data = _normalize_currency_and_prices(safe_data)
                _normalize_incoterm_location(safe_data)

                # client §4 — a cases/pallet figure the LLM mistook for a case
                # pack must not survive in units_per_case or the identity key.
                # Precise signal only (units_per_case == cases_per_pallet); never
                # a blanket size cap, which would wrongly null legitimate large
                # packs like the 120-count 50 ml miniatures.
                _upc = _safe_float(safe_data.get('units_per_case'))
                _cpp = _safe_float(safe_data.get('cases_per_pallet'))
                if _upc is not None and _cpp and _upc == _cpp:
                    safe_data['units_per_case'] = None
                # client §5 — MOQ unit follows the source (see batch path note).
                if safe_data.get('moq_cases') is not None and not safe_data.get('moq_unit'):
                    safe_data['moq_unit'] = 'cases'
                # Round-4 deterministic offer-shape fixes (Samuel 2026-11-11).
                _apply_round4_shape_fixes(safe_data, _cond_source_text or '')
                _final_key = build_product_key(
                    safe_data.get('brand'), safe_data.get('product_name'),
                    safe_data.get('unit_volume_ml'),
                    safe_data.get('units_per_case'))
                if _final_key:
                    safe_data['product_key'] = _final_key

                _apply_offer_level(
                    safe_data, _commercial_conditions, _cond_valid_until,
                    _cond_flags, _job_source_type, _source_flags)

                _sup_name  = safe_data.get('supplier_name')
                _sup_email = _resolve_supplier_email(payload, safe_data.get('supplier_email'))
                # A reference glued onto the supplier name belongs in
                # supplier_reference (Cedar Bay — ref. QA-20261107-W07).
                if not safe_data.get('supplier_reference'):
                    _sr = _resolve_supplier_ref(_sup_name, payload)
                    if _sr:
                        safe_data['supplier_reference'] = _sr
                # Split "Company - Person" into company + contact (Yamazaki).
                _sup_company, _sup_contact = _split_company_contact(
                    _resolve_supplier_name(_sup_name, payload),
                    _cond_source_text, _sup_email)

                offer = OfferItem(
                    uid=uid,
                    product_name=safe_data['product_name'],
                    product_name_raw=safe_data.get('product_name_raw'),
                    product_key=safe_data['product_key'],
                    brand=safe_data['brand'],
                    brand_raw=safe_data.get('brand_raw'),
                    category=safe_data['category'],
                    sub_category=safe_data['sub_category'],
                    packaging=safe_data['packaging'],
                    packaging_raw=safe_data['packaging_raw'],
                    bottle_or_can_type=safe_data['bottle_or_can_type'],
                    unit_volume_ml=safe_data['unit_volume_ml'],
                    units_per_case=safe_data['units_per_case'],
                    cases_per_pallet=safe_data['cases_per_pallet'],
                    quantity_case=safe_data['quantity_case'],
                    quantity_unit=safe_data.get('quantity_unit'),
                    quantity_bottles=safe_data.get('quantity_bottles'),
                    quantity_pallets=safe_data.get('quantity_pallets'),
                    quantity_ftl=safe_data.get('quantity_ftl'),
                    gift_box=safe_data['gift_box'],
                    refillable_status=safe_data['refillable_status'],
                    currency=safe_data['currency'],
                    price_per_unit=safe_data['price_per_unit'],
                    price_per_unit_eur=safe_data['price_per_unit_eur'],
                    price_per_case=safe_data['price_per_case'],
                    price_per_case_eur=safe_data['price_per_case_eur'],
                    fx_rate=safe_data['fx_rate'],
                    fx_date=safe_data['fx_date'],
                    fx_source=safe_data.get('fx_source'),
                    alcohol_percent=safe_data.get('alcohol_percent'),
                    origin_country=safe_data['origin_country'],
                    supplier_country=safe_data['supplier_country'],
                    incoterm=safe_data['incoterm'],
                    location=safe_data['location'],
                    lead_time=safe_data['lead_time'],
                    moq_cases=safe_data['moq_cases'],
                    moq_unit=safe_data.get('moq_unit'),
                    moq_bottles=safe_data.get('moq_bottles'),
                    moq_pallets=safe_data.get('moq_pallets'),
                    moq_ftl=safe_data.get('moq_ftl'),
                    moq_text=safe_data.get('moq_text'),
                    valid_until=safe_data['valid_until'],
                    offer_date=_offer_date_or_flag(
                        extracted_data.get('offer_date'),
                        payload.source_filename or "",
                        datetime.utcnow(), safe_data),
                    document_date=safe_data.get('document_date'),
                    document_date_source=safe_data.get('document_date_source'),
                    commercial_conditions=safe_data.get('commercial_conditions'),
                    source_type=safe_data.get('source_type'),
                    date_received=_received_at(payload),
                    best_before_date=safe_data['best_before_date'],
                    vintage=safe_data['vintage'],
                    supplier_name=_sup_company,
                    contact_name=_sup_contact,
                    supplier_email=_sup_email,
                    sender_name=payload.sender_name,
                    sender_email=payload.sender_email,
                    supplier_reference=safe_data['supplier_reference'],
                    source_channel=payload.source_channel or "",
                    source_message_id=payload.source_message_id or "",
                    source_filename=payload.source_filename or "",
                    attachment_filenames=[att.fileName for att in payload.attachments] if payload.attachments else [],
                    attachment_count=len(payload.attachments) if payload.attachments else 0,
                    confidence_score=0.95,
                    # Pulled from safe_data (threaded from clean_product_data).
                    needs_manual_review=safe_data.get('needs_manual_review', False),
                    error_flags=safe_data['error_flags'] if isinstance(safe_data.get('error_flags'), list) else [],
                    custom_status=safe_data['custom_status'],
                    processing_version="2.0.0",
                    ean_code=safe_data['ean_code'],
                    label_language=safe_data['label_language'],
                    product_reference=safe_data['product_reference'],
                    supersedes_ref=safe_data.get('supersedes_ref'),
                )

                offer_dict = offer.model_dump(mode='json')
                # Phase 3 M1: thread content-hash + source-row estimate
                offer_dict["_content_hash"] = getattr(payload, "content_hash", None)
                offer_dict["_expected_row_count"] = extracted_data.get("_source_row_estimate") or len(all_products)
                offer_dict["_source_text"] = _durable_source_text

                # Loose-unit €/case cleanup — see multi-product path above.
                upc = offer_dict.get("units_per_case")
                if not upc or upc in (0, 0.0, 1, 1.0):
                    if offer_dict.get("price_per_case") in (0, 0.0):
                        offer_dict["price_per_case"] = None
                    if offer_dict.get("price_per_case_eur") in (0, 0.0):
                        offer_dict["price_per_case_eur"] = None
                    if upc in (0, 0.0):
                        offer_dict["units_per_case"] = None

                if not is_valid_offer(offer_dict):
                    logger.info(
                        f"Skipping invalid single offer: "
                        f"name={offer_dict.get('product_name')!r}, "
                        f"price_unit={offer_dict.get('price_per_unit')}, "
                        f"price_case={offer_dict.get('price_per_case')}"
                    )
                else:
                    offers.append(offer_dict)

                    # ── Persist to database ──────────────────────────────────
                    save_offer_to_db(offer_dict, job_id)

                    logger.info(f"Writing single offer row directly to Excel: {offer_dict['product_name']}")
                    write_offer_row(offer_dict)

            except Exception as e:
                error_trace = traceback.format_exc()
                logger.error(f"Error creating offer item for job {job_id}: {e}\n{error_trace}")

                result_data = {
                    "job_id": job_id,
                    "status": "partial_success",
                    "error": str(e),
                    "error_trace": error_trace,
                    "extracted_data": extracted_data,
                    "source_data": {
                        "supplier_name": payload.supplier_name,
                        "supplier_email": payload.supplier_email,
                        "source_channel": payload.source_channel,
                        "text_body_preview": payload.text_body[:200] if payload.text_body else None,
                        "attachments": [att.fileName for att in payload.attachments] if payload.attachments else []
                    }
                }

                redis_manager.set_job_result(job_id, result_data)
                redis_manager.set_job_status(job_id, "done")
                logger.info(f"Job {job_id} marked as done with partial success")
                return

        # ── Round-4 source-completeness pass (Samuel 2026-11-11), once per job ──
        # (1) drop LLM over-split no-quantity fragments (Yamazaki 3 rows → 2);
        # (2) recover a glued "@ <CUR> <num>" ambiguous line the LLM dropped
        #     (Alder Beefeater @ EUR 79). Both operate on the whole source.
        try:
            _dedup_fragments_in_place(offers, job_id)
            # Prefer the FULL document text (body + extracted attachment text) so
            # the Beefeater '@ EUR 79' line in an attachment-only PDF (NE-410,
            # where payload.text_body is empty) is still available to recover.
            _recov_src = (_cond_source_text
                          or getattr(payload, 'text_body', None)
                          or (offers[0].get('_source_text') if offers else None) or '')
            _recover_dropped_at_priced(offers, _recov_src, job_id)
            # (3) recover a dropped NUMBERED circular item (NE-410 item 06
            #     L'Oréal, a lone cosmetic the extractor skips in a spirits list).
            _recover_dropped_numbered_items(offers, _recov_src, job_id)
            # (4) propagate USD across a dual-currency perfume batch (CHELSEA) so a
            #     dense image that transcribed without HKD/USD labels still shows
            #     the supplier's currency, not a stray EUR with no conversion.
            _propagate_dual_currency(offers, job_id)
        except Exception as _cp_exc:
            logger.warning(f"completeness pass failed for {job_id}: {_cp_exc}")

        # Store results
        result_data = {
            "job_id": job_id,
            "status": "done",
            "total_products": len(offers),
            "products": offers,
            "source_info": {
                "supplier": payload.supplier_name,
                "attachments": len(payload.attachments) if payload.attachments else 0,
                "file_processed": len(all_products) > 0,
                # Diagnostic: so we can see what the extractor's barcode
                # ground-truth said without needing server logs.
                "source_row_estimate": extracted_data.get("_source_row_estimate"),
                "all_products_count": len(all_products),
                # Client §2/§7 — exactly what we extracted from, what signature
                # images were ignored, and any readable attachment that gave
                # nothing. Visible without digging into server logs.
                "source_type": _job_source_type,
                "ignored_signature_images": _skipped_signatures,
                "empty_attachments": _empty_attachments,
            }
        }

        redis_manager.set_job_result(job_id, result_data)
        redis_manager.set_job_status(job_id, "done")
        logger.info(f"Successfully processed {len(offers)} products for job {job_id}")

        # Client §2/§7 — a job that produced NO offer rows must be visibly
        # flagged, never silently dropped. Record why (signatures ignored /
        # attachments unreadable) on the intake feed.
        try:
            from core.intake_log import update_intake
            if len(offers) == 0:
                _why = []
                if _skipped_signatures:
                    _why.append("only signature/decorative image(s): "
                                + ", ".join(_skipped_signatures))
                if _empty_attachments:
                    _why.append("unreadable/empty attachment(s): "
                                + ", ".join(_empty_attachments))
                if not _why:
                    _why.append("no offer content could be extracted from the "
                                "message body or attachments")
                logger.warning(f"Job {job_id} produced 0 offers — flagged: "
                               + "; ".join(_why))
                update_intake(job_id, status="no_offer_content", products=0,
                              error="No offer extracted — " + "; ".join(_why))
            else:
                update_intake(job_id, status="done", products=len(offers))
        except Exception as _ie:
            logger.warning(f"intake finalize failed for {job_id}: {_ie}")

        # R-12: persist any new trading events the full offer_items table
        # now implies, so /api/events keeps the full chain (A01 €25 → B01
        # €24 → C01 €23.50) rather than just the latest snapshot. The
        # unique index on (peer_group_id, current_uid, previous_uid) makes
        # this additive and idempotent.
        try:
            from core.database import SessionLocal
            from core.event_writer import record_price_events
            with SessionLocal() as _ev_db:
                _new = record_price_events(_ev_db)
                logger.info(f"R-12 events: {len(_new)} new event(s) persisted after job {job_id}")
        except Exception as _ev_err:
            logger.warning(f"R-12 event retention failed for job {job_id}: {_ev_err}")

    except OpenAICreditsExhausted:
        # Out of OpenAI credits — do NOT mark this job done/failed. Let it
        # propagate so the worker parks it and auto-resumes when credits return.
        raise
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Critical error processing job {job_id}: {e}\n{error_trace}")

        error_result = {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "error_trace": error_trace,
            "timestamp": datetime.utcnow().isoformat(),
            "payload_info": {
                "supplier_name": payload.supplier_name if hasattr(payload, 'supplier_name') else None,
                "has_text": bool(payload.text_body) if hasattr(payload, 'text_body') else None,
                "attachment_count": len(payload.attachments) if hasattr(payload, 'attachments') else None
            }
        }
        redis_manager.set_job_result(job_id, error_result)
        redis_manager.set_job_status(job_id, "failed")