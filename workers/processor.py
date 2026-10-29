import uuid
import re
import json
import os
import traceback
import tempfile
import logging
from datetime import datetime
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
                )
                db.add(source_file)
                db.flush()  # get the id without committing yet
                logger.info(f"DB: created SourceFileDB for job {job_id}, file='{source_filename}'")
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
                alcohol_percent=offer_dict.get("alcohol_percent"),
                origin_country=offer_dict.get("origin_country"),
                supplier_country=offer_dict.get("supplier_country"),
                incoterm=offer_dict.get("incoterm"),
                location=offer_dict.get("location"),
                lead_time=offer_dict.get("lead_time"),
                moq_cases=offer_dict.get("moq_cases"),
                valid_until=offer_dict.get("valid_until"),
                best_before_date=offer_dict.get("best_before_date"),
                vintage=offer_dict.get("vintage"),
                ean_code=offer_dict.get("ean_code"),
                label_language=offer_dict.get("label_language"),
                product_reference=offer_dict.get("product_reference"),
                supplier_name=offer_dict.get("supplier_name"),
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
                date_received=offer_dict.get("date_received"),
            )
            db.add(row)

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


_INCOTERM_RE = re.compile(r"\b(EXW|DAP|DDP|FOB|CIF|CFR|CPT|FCA|DAT|FAS|EXWORK)\b", re.IGNORECASE)
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
                isinstance(f, str) and 'Zero price retained' in f for f in flags):
            return True
        logger.debug(f"Offer {name} rejected: No price data found.")
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


def _resolve_supplier_name(ai_name, payload) -> str:
    """Return the best available supplier name.

    Priority:
      1. payload.supplier_name  — Make extracted this from email body / WhatsApp text
      2. ai_name                — AI extracted from attached file (PDF/Excel)
      3. "Not Found"            — sender info is NOT used (sender ≠ supplier)

    A candidate that is really the source filename or a test marker is skipped
    (see _is_placeholder_supplier) so a catalog filename never poses as a supplier.
    """
    _missing = [None, "", "Not Found"]
    src = getattr(payload, 'source_filename', None)
    pname = getattr(payload, 'supplier_name', None)
    if pname not in _missing and not _is_placeholder_supplier(pname, src):
        return str(pname)
    if ai_name not in _missing and not _is_placeholder_supplier(ai_name, src):
        return str(ai_name)
    return "Not Found"


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

    # If AI didn't provide a meaningful rate for non-EUR, use table fallback
    fx = ai_fx if (currency == 'EUR' or ai_fx != 1.0) else _FALLBACK_FX.get(currency, 1.0)
    safe_data['fx_rate'] = fx

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

    # ── quantity_case ───────────────────────────────────────────────────
    qty = _to_float(safe_data.get('quantity_case'))
    qty_unit = (safe_data.get('quantity_unit') or '').lower().strip()

    if qty_unit == 'ftl':
        safe_data['quantity_case'] = None
        error_flags.append('Quantity: FTL (Full Truck Load)')
    elif qty_unit == 'bottles' and qty:
        if upc and upc > 1:
            converted = round(qty / upc, 2)
            safe_data['quantity_case'] = converted
            error_flags.append(f'Quantity converted from {int(qty)} bottles to {converted} cases')
        else:
            # Loose or unknown pack: cases is NOT the bottle count.
            # Null quantity_case and keep raw bottles in moq_bottles for traceability
            # so the UI shows "20 btls (loose)" rather than "20 cases".
            safe_data['quantity_case'] = None
            safe_data['_quantity_bottles'] = qty
            if not upc or upc in (0, 1):
                error_flags.append(f'Quantity in bottles ({int(qty)}), loose pack — case qty N/A')
            else:
                error_flags.append(f'Quantity in bottles ({int(qty)}), cases unknown (no units_per_case)')

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


def _resolve_offer_date(raw, source_filename: str, received: datetime) -> datetime:
    """
    Resolve the stored offer_date, in priority order:
      1. a real date parsed from the document/header (AI-extracted),
      2. the date encoded in the source filename (stock lists / price lists),
      3. the received/ingest date — offers arrive and are ingested together,
         so arrival is the logical offer date when nothing else is known.
    Always returns a date, so no row ever shows an age with no offer date.
    """
    parsed = _parse_offer_date(raw)
    if parsed is not None:
        return parsed
    from_name = _date_from_filename(source_filename or "", (received or datetime.utcnow()).year)
    if from_name is not None:
        return from_name
    return received or datetime.utcnow()


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
        'label_language': "EN",
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

        logger.info(f"Processing job {job_id} for supplier: {payload.supplier_name}")
        logger.info(f"Text body length: {len(payload.text_body) if payload.text_body else 0}")
        logger.info(f"Attachment count: {len(payload.attachments) if payload.attachments else 0}")

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

            except Exception as e:
                logger.error(f"Error in text extraction for job {job_id}: {e}")

        if payload.attachments:
            for attachment in payload.attachments:
                try:
                    file_name = attachment.fileName
                    logger.info(f"Processing attachment: {file_name}")

                    file_ext = file_name.split('.')[-1].lower() if '.' in file_name else ''
                    file_bytes = await resolve_attachment_bytes(attachment)

                    if not file_bytes:
                        logger.warning(f"Could not parse file data for {file_name}")
                        continue

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

                except Exception as e:
                    logger.error(f"Error processing attachment {attachment.fileName}: {e}")
                    continue

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
            def _canon_key(p):
                return (
                    (p.get("brand") or "").strip().lower(),
                    (p.get("product_name") or "").strip().lower(),
                    p.get("unit_volume_ml") or 0,
                    (p.get("ean_code") or "").strip().lstrip("0"),
                    p.get("price_per_unit") or 0,
                    (p.get("retail_state") or "").strip().lower(),
                    # Supplier reference distinguishes same-EAN/same-price line
                    # items with different conditions (e.g. "Clean EU" vs "Deco").
                    (p.get("supplier_reference") or "").strip().lower(),
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
                    for _mf in ('offer_date', 'incoterm', 'location', 'currency', 'supplier_name'):
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
                    safe_data = {
                        'product_name': _dp_name or "Not Found",
                        'product_key': merged_data.get('product_key') or "Not Found",
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
                        'gift_box': merged_data.get('gift_box'),
                        'refillable_status': merged_data.get('refillable_status') or "",
                        'currency': merged_data.get('currency') or "EUR",
                        'price_per_unit': _safe_float(merged_data.get('price_per_unit')),
                        'price_per_unit_eur': _safe_float(merged_data.get('price_per_unit_eur')),
                        'price_per_case': _safe_float(merged_data.get('price_per_case')),
                        'price_per_case_eur': _safe_float(merged_data.get('price_per_case_eur')),
                        'fx_rate': _safe_float(merged_data.get('fx_rate'), 1.0),
                        'fx_date': merged_data.get('fx_date'),
                        'alcohol_percent': merged_data.get('alcohol_percent'),
                        'origin_country': merged_data.get('origin_country'),
                        'supplier_country': merged_data.get('supplier_country') or "",
                        'incoterm': merged_data.get('incoterm') or "Not Found",
                        'location': merged_data.get('location') or "Not Found",
                        'lead_time': merged_data.get('lead_time') or "Not Found",
                        'moq_cases': merged_data.get('moq_cases'),
                        'moq_bottles': merged_data.get('moq_bottles'),
                        'moq_unit': merged_data.get('moq_unit'),
                        'quantity_unit': merged_data.get('quantity_unit'),
                        'min_order_quantity_case': merged_data.get('min_order_quantity_case'),
                        'port': merged_data.get('port') or "Not Found",
                        'valid_until': merged_data.get('valid_until'),
                        'best_before_date': merged_data.get('best_before_date'),
                        'vintage': merged_data.get('vintage'),
                        'supplier_reference': merged_data.get('supplier_reference'),
                        'ean_code': merged_data.get('ean_code'),
                        'label_language': merged_data.get('label_language') or "EN",
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

                    # Compound-location preservation guard (Rule 7). Flags
                    # rows where the raw source listed ≥2 alternative
                    # locations and GPT kept only one.
                    _detect_dropped_alt_locations(safe_data)

                    # ── Currency normalisation & EUR conversion ──────────────
                    _snap_duplicated_row_per_unit(safe_data)
                    safe_data = _normalize_currency_and_prices(safe_data)

                    # ── MOQ / quantity unit-aware normalization (RULE 0.3) ───
                    _flag_bucket = list(safe_data.get('error_flags') or [])
                    safe_data = _normalize_moq_and_quantity(safe_data, _flag_bucket)
                    safe_data['error_flags'] = _flag_bucket

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
                        safe_data['units_per_case'] = float(_pu)
                        safe_data['unit_volume_ml'] = (
                            _pv * 10 if _pu_unit == 'cl' else
                            _pv * 1000 if _pu_unit == 'l' else _pv
                        )
                        # Canonical display form — store raw LLM string first
                        if not safe_data.get('packaging_raw'):
                            safe_data['packaging_raw'] = _pkg_str
                        safe_data['packaging'] = f"{_pu}/case"
                        logger.debug(
                            f"Row {idx}: packaging '{_pkg_str}' → "
                            f"units_per_case={_pu}, unit_volume_ml={safe_data['unit_volume_ml']}"
                        )

                    _sup_name  = safe_data.get('supplier_name')
                    _sup_email = _resolve_supplier_email(payload, safe_data.get('supplier_email'))

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
                        alcohol_percent=safe_data['alcohol_percent'],
                        origin_country=safe_data['origin_country'],
                        supplier_country=safe_data['supplier_country'],
                        incoterm=safe_data['incoterm'],
                        location=safe_data['location'],
                        lead_time=safe_data['lead_time'],
                        moq_cases=safe_data['moq_cases'],
                        valid_until=safe_data['valid_until'],
                        offer_date=_resolve_offer_date(
                            merged_data.get('offer_date'),
                            payload.source_filename or "",
                            datetime.utcnow()),
                        date_received=datetime.utcnow(),
                        best_before_date=safe_data['best_before_date'],
                        vintage=safe_data['vintage'],
                        supplier_name=_resolve_supplier_name(_sup_name, payload),
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
                    )

                    offer_dict = offer.model_dump(mode='json')
                    # Phase 3 M1: thread content-hash + source-row estimate
                    # so save_offer_to_db can persist them on the SourceFileDB.
                    offer_dict["_content_hash"] = getattr(payload, "content_hash", None)
                    offer_dict["_expected_row_count"] = extracted_data.get("_source_row_estimate") or len(all_products)

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
                        # NARS retest bug: loose pack with price_per_case == price_per_unit
                        # was passing through. Treat all loose-pack case prices as unavailable.
                        offer_dict["price_per_case"] = None
                        offer_dict["price_per_case_eur"] = None
                        if upc in (0, 0.0):
                            offer_dict["units_per_case"] = None
                        # Also null quantity_case if it was passed through as bottle count
                        # (defensive — the processor's _normalize_quantity_units should have
                        # already caught this, but belt-and-braces for direct DB writes).
                        if (offer_dict.get("quantity_unit") or "").lower() == "bottles":
                            offer_dict["quantity_case"] = None

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
                safe_data = {
                    'product_name': _dp_name or "Not Found",
                    'product_key': extracted_data.get('product_key') or "Not Found",
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
                    'gift_box': extracted_data.get('gift_box'),
                    'refillable_status': extracted_data.get('refillable_status') or "",
                    'currency': extracted_data.get('currency') or "EUR",
                    'price_per_unit': _safe_float(extracted_data.get('price_per_unit')),
                    'price_per_unit_eur': _safe_float(extracted_data.get('price_per_unit_eur')),
                    'price_per_case': _safe_float(extracted_data.get('price_per_case')),
                    'price_per_case_eur': _safe_float(extracted_data.get('price_per_case_eur')),
                    'fx_rate': _safe_float(extracted_data.get('fx_rate'), 1.0),
                    'fx_date': extracted_data.get('fx_date'),
                    'alcohol_percent': extracted_data.get('alcohol_percent'),
                    'origin_country': extracted_data.get('origin_country'),
                    'supplier_country': extracted_data.get('supplier_country') or "",
                    'incoterm': extracted_data.get('incoterm') or "Not Found",
                    'location': extracted_data.get('location') or "Not Found",
                    'lead_time': extracted_data.get('lead_time') or "Not Found",
                    'moq_cases': extracted_data.get('moq_cases'),
                    'moq_bottles': extracted_data.get('moq_bottles'),
                    'moq_unit': extracted_data.get('moq_unit'),
                    'quantity_unit': extracted_data.get('quantity_unit'),
                    'min_order_quantity_case': extracted_data.get('min_order_quantity_case'),
                    'port': extracted_data.get('port') or "Not Found",
                    'valid_until': extracted_data.get('valid_until'),
                    'best_before_date': extracted_data.get('best_before_date'),
                    'vintage': extracted_data.get('vintage'),
                    'supplier_reference': extracted_data.get('supplier_reference'),
                    'ean_code': extracted_data.get('ean_code'),
                    'label_language': extracted_data.get('label_language') or "EN",
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
                _detect_dropped_alt_locations(safe_data)

                safe_data = _normalize_currency_and_prices(safe_data)

                _sup_name  = safe_data.get('supplier_name')
                _sup_email = _resolve_supplier_email(payload, safe_data.get('supplier_email'))

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
                    gift_box=safe_data['gift_box'],
                    refillable_status=safe_data['refillable_status'],
                    currency=safe_data['currency'],
                    price_per_unit=safe_data['price_per_unit'],
                    price_per_unit_eur=safe_data['price_per_unit_eur'],
                    price_per_case=safe_data['price_per_case'],
                    price_per_case_eur=safe_data['price_per_case_eur'],
                    fx_rate=safe_data['fx_rate'],
                    fx_date=safe_data['fx_date'],
                    alcohol_percent=safe_data.get('alcohol_percent'),
                    origin_country=safe_data['origin_country'],
                    supplier_country=safe_data['supplier_country'],
                    incoterm=safe_data['incoterm'],
                    location=safe_data['location'],
                    lead_time=safe_data['lead_time'],
                    moq_cases=safe_data['moq_cases'],
                    valid_until=safe_data['valid_until'],
                    offer_date=_resolve_offer_date(
                        extracted_data.get('offer_date'),
                        payload.source_filename or "",
                        datetime.utcnow()),
                    date_received=datetime.utcnow(),
                    best_before_date=safe_data['best_before_date'],
                    vintage=safe_data['vintage'],
                    supplier_name=_resolve_supplier_name(_sup_name, payload),
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
                )

                offer_dict = offer.model_dump(mode='json')
                # Phase 3 M1: thread content-hash + source-row estimate
                offer_dict["_content_hash"] = getattr(payload, "content_hash", None)
                offer_dict["_expected_row_count"] = extracted_data.get("_source_row_estimate") or len(all_products)

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
            }
        }

        redis_manager.set_job_result(job_id, result_data)
        redis_manager.set_job_status(job_id, "done")
        logger.info(f"Successfully processed {len(offers)} products for job {job_id}")

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