"""Phase 3 acceptance — v3 regression tests (Samuel 2026-10-30).

Each test locks in one of Samuel's six v2-review observations against
the code paths the fix changed. These exercise the deterministic rules
and normalization helpers directly with LLM-shaped payloads so the
regression lands in CI even when the live extractor is not available.
"""
from __future__ import annotations

from core.category_classifier import (
    detect_category,
    detect_row_category,
    WINES_SPIRITS,
    PERFUMES,
    COSMETICS,
)
from core.deterministic_rules import apply_deterministic_defaults
from workers.processor import (
    _drop_misleading_multi_incoterm_flag,
    _normalize_moq_and_quantity,
    _preserve_raw_identity,
)


# ───────────────────────── 1. Mixed-category rows ──────────────────────
class TestRowLevelCategoryBackstop:
    """Samuel point 5 — NARS / YSL in an otherwise-spirits document must
    be classified as cosmetics / perfumes, not inherit the document
    default of wines_spirits."""

    DOC_TEXT = (
        "SYNTHETIC ACCEPTANCE DATA — ISOLATED TEST ENVIRONMENT ONLY\n"
        "Supplier: Acceptance Alpha\n"
        "Commercial offer date: 2026-10-27\n"
        "Valid until: 2027-10-31\n\n"
        "Hennessy VS, Johnnie Walker Red Label, Jameson Irish Whiskey,\n"
        "Bacardi Carta Blanca, Jack Daniel's Old No. 7, Campari,\n"
        "Grey Goose, Martell VSOP\n"
    )

    def test_doc_level_detects_wines_spirits(self):
        assert detect_category(text=self.DOC_TEXT) == WINES_SPIRITS

    def test_row_backstop_catches_nars_as_cosmetics(self):
        rows = [
            {
                "supplier_reference": "A07",
                "brand": "NARS",
                "product_name": "Punjab Medium 1 Foundation",
                "_raw_row_text":
                    "Supplier reference: A07\nProduct: NARS Punjab Medium 1 "
                    "Foundation\nPackaging: single 30ml\nPrice: EUR 20/piece\n"
                    "Available quantity: 100 pieces\nCommercial terms: EXW "
                    "Rotterdam; customs status unspecified\nEAN: 2000000000022\n"
                    "Cosmetics; foundation; shade Punjab Medium 1; retail; "
                    "no case packaging.\n",
                "category_slug": None,
                "price_per_unit": 20,
            },
        ]
        rows_fixed, corrections = apply_deterministic_defaults(
            rows, self.DOC_TEXT, source_filename="01_Alpha.txt")
        assert rows_fixed[0]["category_slug"] == COSMETICS, (
            f"got {rows_fixed[0]['category_slug']}; corrections={corrections}"
        )

    def test_row_backstop_catches_ysl_as_perfumes(self):
        rows = [{
            "supplier_reference": "A08",
            "brand": "Yves Saint Laurent",
            "product_name": "Test Lumiere",
            "_raw_row_text":
                "Supplier reference: A08\nProduct: Yves Saint Laurent "
                "Test Lumiere\nPackaging: 6x50ml\nPrice: EUR 480/case\n"
                "Available quantity: 120 cases\nCommercial terms: EXW "
                "Rotterdam; customs status unspecified\nEAN: 2000000000039\n"
                "Perfumes; EDP; women; retail.\n",
            "category_slug": None,
            "price_per_case": 480,
        }]
        rows_fixed, _ = apply_deterministic_defaults(
            rows, self.DOC_TEXT, source_filename="01_Alpha.txt")
        assert rows_fixed[0]["category_slug"] == PERFUMES

    def test_spirit_row_still_inherits_doc_default(self):
        rows = [{
            "supplier_reference": "A01",
            "brand": "Hennessy",
            "product_name": "VS",
            "_raw_row_text": "Hennessy VS 12x1000ml EUR 300/case",
            "category_slug": None,
            "price_per_case": 300,
        }]
        rows_fixed, _ = apply_deterministic_defaults(
            rows, self.DOC_TEXT, source_filename="01_Alpha.txt")
        assert rows_fixed[0]["category_slug"] == WINES_SPIRITS

    def test_detect_row_category_returns_none_on_empty(self):
        assert detect_row_category("") is None
        assert detect_row_category(None) is None


# ───────────────────────── 2. Pallet → case conversion ──────────────────
class TestPalletsToCases:
    """Samuel point 2 — A10 ships "2 pallets ... each pallet 80 cases";
    quantity_case must store 160 cases, not blank."""

    def test_pallet_count_converts_when_llm_returns_small_number(self):
        safe = {
            "quantity_case": 2,
            "quantity_unit": "pallets",
            "cases_per_pallet": 80,
            "_raw_row_text": "Available quantity: 2 pallets",
            "units_per_case": 6,
            "moq_cases": None,
            "moq_unit": "pallets",
        }
        out = _normalize_moq_and_quantity(safe, [])
        assert out["quantity_case"] == 160
        assert out.get("_quantity_pallets") == 2

    def test_already_in_cases_is_trusted(self):
        safe = {
            "quantity_case": 160,
            "quantity_unit": "pallets",
            "cases_per_pallet": 80,
            "_raw_row_text": "Available quantity: 2 pallets, 80 cases per pallet",
            "units_per_case": 6,
            "moq_cases": None,
            "moq_unit": "pallets",
        }
        out = _normalize_moq_and_quantity(safe, [])
        # 160 > cpp (80), already in cases: leave untouched
        assert out["quantity_case"] == 160

    def test_parse_pallet_count_from_raw_when_llm_blank(self):
        safe = {
            "quantity_case": None,
            "quantity_unit": "pallets",
            "cases_per_pallet": 80,
            "_raw_row_text":
                "Available quantity: 2 pallets. Each pallet contains 80 cases.",
            "units_per_case": 6,
            "moq_cases": None,
            "moq_unit": "pallets",
        }
        out = _normalize_moq_and_quantity(safe, [])
        assert out["quantity_case"] == 160

    def test_ftl_still_blanks_quantity(self):
        safe = {
            "quantity_case": None,
            "quantity_unit": "ftl",
            "cases_per_pallet": None,
            "_raw_row_text": "Available quantity: 1 FTL",
            "units_per_case": 6,
            "moq_cases": None,
            "moq_unit": "ftl",
            "error_flags": [],
        }
        flags: list = safe["error_flags"]
        out = _normalize_moq_and_quantity(safe, flags)
        assert out["quantity_case"] is None
        all_flags = list(flags) + list(out.get("error_flags") or [])
        assert any("FTL" in f for f in all_flags)


# ───────────────── 3. Multi-incoterm flag suppression ──────────────────
class TestMultiIncotermFlagSuppression:
    """Samuel point 6 — A03 lists one incoterm (EXW) at three locations;
    the "multiple incoterms detected" flag is a misread and must drop."""

    def test_a03_single_incoterm_three_locations_drops_flag(self):
        safe = {
            "_raw_row_text":
                "Commercial terms: EXW Loendersloot OR Revera OR Mentrex; T1",
            "error_flags": [
                "Multiple incoterms detected — row duplicated per unit",
                "Rule 7: compound location split into siblings",
            ],
        }
        _drop_misleading_multi_incoterm_flag(safe)
        assert all("incoterm" not in f.lower() or
                   not any(w in f.lower() for w in ("multiple", "several",
                                                     "mixed", "many"))
                   for f in safe["error_flags"])
        assert "Rule 7: compound location split into siblings" in safe["error_flags"]

    def test_two_distinct_incoterms_keep_flag(self):
        safe = {
            "_raw_row_text": "Terms: EXW Rotterdam OR DAP Loendersloot",
            "error_flags": ["Multiple incoterms detected — row duplicated per unit"],
        }
        _drop_misleading_multi_incoterm_flag(safe)
        # Two distinct incoterms → flag is NOT misleading, kept
        assert any("multiple incoterms" in f.lower()
                   for f in safe["error_flags"])

    def test_variant_wording_also_dropped(self):
        safe = {
            "_raw_row_text": "Terms: EXW Rotterdam; T2",
            "error_flags": ["Several incoterms detected on this row"],
        }
        _drop_misleading_multi_incoterm_flag(safe)
        assert safe["error_flags"] == []


# ───────────────── 4. Raw identity preservation ────────────────────────
class TestRawIdentityPreserved:
    """Samuel point 4 — product_name_raw must survive even when the LLM
    skipped the field; falls back to product_name on every row."""

    def test_blank_raw_falls_back_to_product_name(self):
        safe = {
            "product_name": "Grey Goose",
            "brand": "Grey Goose",
            "product_name_raw": "",
            "brand_raw": None,
        }
        _preserve_raw_identity(safe)
        assert safe["product_name_raw"] == "Grey Goose"
        assert safe["brand_raw"] == "Grey Goose"

    def test_existing_raw_wins(self):
        safe = {
            "product_name": "Hennessy VS (bottle)",
            "product_name_raw": "Hennessy VS",   # pre-normalization text
            "brand": "Hennessy",
            "brand_raw": "Hennessy",
        }
        _preserve_raw_identity(safe)
        assert safe["product_name_raw"] == "Hennessy VS"

    def test_not_found_sentinel_is_not_promoted(self):
        safe = {
            "product_name": "Not Found",
            "brand": "Not Found",
            "product_name_raw": "",
            "brand_raw": "",
        }
        _preserve_raw_identity(safe)
        # Sentinel must not back-fill a raw field
        assert not safe.get("product_name_raw")
        assert not safe.get("brand_raw")


# ───────────────── 5. MOQ-unit distinction (A12) ───────────────────────
class TestMoqUnitDistinction:
    """Samuel point 6b — A12: available stock 120 cases, MOQ 12 bottles
    (= 2 cases). moq_unit stays 'bottles' and moq_bottles is populated,
    so the UI can render both representations distinctly from stock."""

    def test_a12_bottle_moq_normalizes_to_cases_but_keeps_source(self):
        safe = {
            "quantity_case": 120,
            "quantity_unit": "cases",
            "units_per_case": 6,
            "cases_per_pallet": None,
            "moq_cases": None,
            "moq_bottles": 12,
            "moq_unit": "bottles",
        }
        out = _normalize_moq_and_quantity(safe, [])
        assert out["quantity_case"] == 120, "stock untouched"
        assert out["moq_cases"] == 2, "12 btls ÷ 6 upc = 2 cs"
        assert out["moq_unit"] == "bottles", "source unit retained"
        assert out["moq_bottles"] == 12, "source quantity retained"
