"""
Unit tests for the extraction post-processing helpers. Ground truth:
- MOQ in bottles → cases with proper conversion + flag
- offer_date parsing across French / ISO / English formats
- null vs zero preservation
- is_valid_offer rejection of fake rows

These do NOT hit the network or OpenAI — they exercise the pure functions.
Add fixture-based end-to-end tests later once we can mock the LLM output.
"""
import sys
import os
from datetime import datetime

# Allow "python -m pytest tests/" from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from types import SimpleNamespace
from workers.processor import (
    _parse_offer_date,
    _date_from_filename,
    _resolve_offer_date,
    _is_placeholder_supplier,
    _resolve_supplier_name,
    _resolve_supplier_email,
    _normalize_moq_and_quantity,
    _safe_float,
    is_valid_offer,
)


# ─── _parse_offer_date ────────────────────────────────────────────────────────

class TestParseOfferDate:
    def test_french_full(self):
        assert _parse_offer_date("11 mars 2026 à 14:40") == datetime(2026, 3, 11)

    def test_french_date_only(self):
        assert _parse_offer_date("12 mars 2026") == datetime(2026, 3, 12)

    def test_iso_date(self):
        assert _parse_offer_date("2026-03-11") == datetime(2026, 3, 11)

    def test_iso_datetime(self):
        assert _parse_offer_date("2026-03-11 14:56:11") == datetime(2026, 3, 11, 14, 56, 11)

    def test_english_long(self):
        assert _parse_offer_date("March 11, 2026") == datetime(2026, 3, 11)

    def test_english_dmy(self):
        assert _parse_offer_date("11 March 2026") == datetime(2026, 3, 11)

    def test_english_short_month(self):
        assert _parse_offer_date("11 Mar 2026") == datetime(2026, 3, 11)

    @pytest.mark.parametrize("junk", [None, "", "Not Found", "unknown", "next Tuesday"])
    def test_returns_none_on_junk(self, junk):
        assert _parse_offer_date(junk) is None

    def test_passthrough_datetime(self):
        d = datetime(2026, 3, 11, 10, 0)
        assert _parse_offer_date(d) == d


# ─── _date_from_filename ──────────────────────────────────────────────────────

class TestDateFromFilename:
    def test_day_month_year_dotted(self):
        # "09.06.2026" → 9 June 2026
        assert _date_from_filename("Cosmetics_Offer_09.06.2026.xlsx", 2026) == datetime(2026, 6, 9)

    def test_day_month_year_single_digit_month(self):
        # "18.9.2026" → 18 Sept 2026
        assert _date_from_filename("hot_offer_18.9.2026xls.xlsx", 2026) == datetime(2026, 9, 18)

    def test_day_month_no_year_infers_year(self):
        # "13.04" with no year → 13 April of the fallback year
        assert _date_from_filename("STOCK LIST PERFUMES 13.04.xlsx", 2026) == datetime(2026, 4, 13)

    def test_day_month_no_year_steps_back_when_future(self):
        # A DD.MM that would land in the future for the fallback year uses the prior year
        d = _date_from_filename("W39_Designer_Offer_23.09.xlsx", 2027)
        assert d == datetime(2026, 9, 23)

    def test_day_month_name_year(self):
        assert _date_from_filename("FBC Trades - stock list - 8 June 2026.pdf", 2026) == datetime(2026, 6, 8)

    def test_month_name_year_only(self):
        # Month + year, no day → first of the month
        assert _date_from_filename("BACARDI MARTINI - OFFER (AUGUST 2026).xlsx", 2026) == datetime(2026, 8, 1)

    def test_no_date_in_name(self):
        assert _date_from_filename("LOREAL_LUX.xlsx", 2026) is None
        assert _date_from_filename("SCOTCH WHISKY MALTS OFFER EX HOLLAND.pdf", 2026) is None

    def test_empty(self):
        assert _date_from_filename("", 2026) is None


# ─── _resolve_offer_date (priority: document → filename → arrival) ─────────────

class TestResolveOfferDate:
    RECV = datetime(2026, 10, 8)

    def test_parsed_document_date_wins(self):
        # A real header date beats both the filename and the arrival date
        assert _resolve_offer_date("2026-03-11", "Offer_09.06.2026.xlsx", self.RECV) == datetime(2026, 3, 11)

    def test_filename_date_used_when_no_header_date(self):
        # No header date → the filename's document date is the price's true age
        assert _resolve_offer_date("Not Found", "STOCK LIST PERFUMES 13.04.xlsx", self.RECV) == datetime(2026, 4, 13)

    def test_returns_none_when_source_and_filename_have_no_date(self):
        # Samuel acceptance v1 (R-06): "do not invent an offer date".
        # When neither the source nor the filename supplies a date, the
        # resolver must return None. The caller is responsible for raising
        # the "offer_date unknown" review flag — this helper never
        # fabricates a commercial date.
        assert _resolve_offer_date("Not Found", "LOREAL_LUX.xlsx", self.RECV) is None

    def test_returns_none_when_everything_unknown(self):
        # Samuel acceptance v1 (R-06): no fabrication at all. None allows
        # the DB column to stay null so freshness is displayed from
        # date_received with an explicit "offer_date unknown" label.
        assert _resolve_offer_date(None, "", self.RECV) is None


# ─── supplier placeholder guard (a filename / test marker is not a supplier) ───

class TestPlaceholderSupplier:
    def test_filename_stem_placeholders_detected(self):
        assert _is_placeholder_supplier("W39_Designer", "W39_Designer_Offer_23.09.xlsx")
        assert _is_placeholder_supplier("STOCK_LIST_PERFUMES", "STOCK LIST PERFUMES 13.04.xlsx")
        assert _is_placeholder_supplier("Cosmetics_18_06", "Cosmetics_Offer_18.06.xlsx")

    def test_test_marker_detected(self):
        assert _is_placeholder_supplier("L'Oreal Luxury Test", "LOREAL_LUX.xlsx")
        assert _is_placeholder_supplier("Test", "anything.xlsx")

    def test_real_supplier_not_flagged(self):
        # Real names (spaces, not underscores) must never be rejected, even when
        # the file is named after them.
        assert not _is_placeholder_supplier("FBC Trades", "fbc_trades_email.txt")
        assert not _is_placeholder_supplier("Balkan Luxury Supply Ltd.", "Cosmetics_Offer_09.06.2026.xlsx")

    def test_resolver_skips_placeholder_to_not_found(self):
        pl = SimpleNamespace(source_filename="W39_Designer_Offer_23.09.xlsx", supplier_name="W39_Designer")
        assert _resolve_supplier_name(None, pl) == "Not Found"

    def test_resolver_keeps_real_name(self):
        pl = SimpleNamespace(source_filename="fbc_trades_email.txt", supplier_name="FBC Trades")
        assert _resolve_supplier_name(None, pl) == "FBC Trades"

    def test_email_rejects_reserved_test_domains(self):
        assert _resolve_supplier_email(SimpleNamespace(supplier_email="verification@loxo.test")) == "Not Found"
        assert _resolve_supplier_email(SimpleNamespace(supplier_email="test@loreal.example")) == "Not Found"
        assert _resolve_supplier_email(SimpleNamespace(supplier_email="sales@balkan.com")) == "sales@balkan.com"


# ─── _normalize_moq_and_quantity ──────────────────────────────────────────────

class TestMoqNormalization:
    def test_bottles_to_cases_with_upc(self):
        """Grey Goose case: MOQ 45,240 bottles, 6 per case → 7540 cases."""
        flags = []
        data = {"moq_cases": 45240, "moq_bottles": 45240, "moq_unit": "bottles",
                "units_per_case": 6.0, "quantity_case": None, "quantity_unit": None}
        r = _normalize_moq_and_quantity(data, flags)
        assert r["moq_cases"] == 7540.0
        assert "MOQ converted from bottles to cases" in flags

    def test_bottles_without_upc_leaves_null(self):
        flags = []
        data = {"moq_cases": None, "moq_bottles": 100, "moq_unit": "bottles",
                "units_per_case": None, "quantity_case": None, "quantity_unit": None}
        r = _normalize_moq_and_quantity(data, flags)
        assert r["moq_cases"] is None
        assert any("cases unknown" in f for f in flags)

    def test_cases_unit_left_alone(self):
        flags = []
        data = {"moq_cases": 500, "moq_bottles": None, "moq_unit": "cases",
                "units_per_case": 6.0}
        r = _normalize_moq_and_quantity(data, flags)
        assert r["moq_cases"] == 500
        assert flags == []

    def test_auto_normalize_when_unit_missing(self):
        """Client-reported case: 45,240 in moq_cases with no unit → treat as bottles."""
        flags = []
        data = {"moq_cases": 45240, "moq_bottles": None, "moq_unit": None,
                "units_per_case": 6.0}
        r = _normalize_moq_and_quantity(data, flags)
        assert r["moq_cases"] == 7540.0
        assert any("auto-normalized" in f for f in flags)

    def test_quantity_bottles_to_cases(self):
        """FBC Trades Johnnie Walker: 9600 btls / 6 per case → 1600 cases."""
        flags = []
        data = {"moq_cases": None, "quantity_case": 9600, "quantity_unit": "bottles",
                "units_per_case": 6.0}
        r = _normalize_moq_and_quantity(data, flags)
        assert r["quantity_case"] == 1600.0
        assert any("bottles to" in f for f in flags)

    def test_quantity_ftl_is_null_with_flag(self):
        flags = []
        data = {"moq_cases": None, "quantity_case": None, "quantity_unit": "ftl",
                "units_per_case": 6.0}
        r = _normalize_moq_and_quantity(data, flags)
        assert r["quantity_case"] is None
        assert any("FTL" in f for f in flags)


# ─── _safe_float ──────────────────────────────────────────────────────────────

class TestSafeFloat:
    def test_none_returns_default_none(self):
        assert _safe_float(None) is None

    def test_empty_string_returns_none(self):
        assert _safe_float("") is None

    def test_not_found_returns_none(self):
        assert _safe_float("Not Found") is None

    def test_zero_preserved(self):
        """Zero must NOT be treated as missing — could be a free sample."""
        assert _safe_float(0) == 0.0
        assert _safe_float("0") == 0.0

    def test_valid_float(self):
        assert _safe_float(16.00) == 16.00
        assert _safe_float("16.00") == 16.00

    def test_junk_returns_default(self):
        assert _safe_float("abc") is None
        assert _safe_float("abc", 99.0) == 99.0


# ─── is_valid_offer (rejects placeholder rows) ────────────────────────────────

class TestIsValidOffer:
    def test_valid_row(self):
        assert is_valid_offer({
            "product_name": "Grey Goose Original",
            "price_per_unit": 16.00,
            "price_per_case": None,
        })

    def test_zero_price_case_still_valid_if_unit_is_priced(self):
        assert is_valid_offer({
            "product_name": "Grey Goose Original",
            "price_per_unit": 16.00,
            "price_per_case": 0,
        })

    @pytest.mark.parametrize("bad_name", ["Row 12", "row 5", "Missing Row 3", "MISSING ROW 20", "Not Found", "", None])
    def test_placeholder_names_rejected(self, bad_name):
        assert not is_valid_offer({
            "product_name": bad_name,
            "price_per_unit": 16.00,
        })

    def test_both_prices_zero_rejected(self):
        assert not is_valid_offer({
            "product_name": "Real Product",
            "price_per_unit": 0,
            "price_per_case": 0,
        })
