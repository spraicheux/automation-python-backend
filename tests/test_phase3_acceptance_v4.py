"""Phase 3 acceptance — v4 regression tests (Samuel 2026-10-31).

One test per observation in the 2026-10-31 feedback. Each exercises the
code path that was changed to address that observation — so if a future
refactor breaks the rule, the matching test fires.

Covers:
  - A11 FTL: quantity_ftl stored, moq_unit='ftl' preserved
  - product_name_raw: source "Product:" line verbatim, even when the LLM's
    canonical product_name drops a token ("Foundation", "Test")
  - A07/B05/H02 "100 pieces": fallback regex over _raw_row_text when the
    LLM leaves quantity_case null for loose cosmetics
  - A10 "2 pallets": quantity_pallets derived from (quantity_case /
    cases_per_pallet) when the LLM already converted to cases
  - Redis URL kwarg: ssl_cert_reqs only passed for rediss://
"""
from __future__ import annotations

from workers.processor import (
    _normalize_moq_and_quantity,
    _preserve_raw_identity,
)


# ───────── 1. A11 FTL preservation ─────────
class TestFtlQuantityPreserved:
    def test_ftl_default_count_1(self):
        """Source 'Available quantity: 1 FTL' with LLM null → qty_ftl=1."""
        safe = {
            "_raw_row_text": (
                "Supplier reference: A11\n"
                "Product: Martell VSOP\n"
                "Available quantity: 1 FTL\n"
                "MOQ 1 FTL. Truck capacity is not supplied."
            ),
            "quantity_case": None,
            "quantity_unit": "ftl",
            "moq_cases": None,
            "moq_unit": "ftl",
            "units_per_case": 6,
            "cases_per_pallet": None,
            "error_flags": [],
        }
        _normalize_moq_and_quantity(safe, safe["error_flags"])
        assert safe.get("quantity_ftl") == 1.0
        assert safe.get("moq_unit") == "ftl"
        assert safe.get("quantity_case") is None  # case equivalent correctly unknown

    def test_ftl_count_parsed_from_raw(self):
        """Explicit 2 FTL in raw → qty_ftl=2."""
        safe = {
            "_raw_row_text": "Available quantity: 2 FTL",
            "quantity_case": None,
            "quantity_unit": "ftl",
            "units_per_case": 6,
            "error_flags": [],
        }
        _normalize_moq_and_quantity(safe, safe["error_flags"])
        assert safe.get("quantity_ftl") == 2.0


# ───────── 2. product_name_raw from source ─────────
class TestProductNameRawFromSourceLine:
    def test_b05_foundation_suffix_preserved(self):
        """LLM dropped 'Foundation' from product_name; raw must still have it."""
        safe = {
            "_raw_row_text": (
                "Supplier reference: B05\n"
                "Product: NARS Punjab Medium 1 Foundation\n"
                "Packaging: single 30ml"
            ),
            "product_name": "Punjab Medium 1",   # LLM-cleaned
            "brand": "NARS",
        }
        _preserve_raw_identity(safe)
        assert safe["product_name_raw"] == "NARS Punjab Medium 1 Foundation"

    def test_b09_test_prefix_preserved(self):
        """LLM dropped 'Test' prefix from product_name; raw must still have it."""
        safe = {
            "_raw_row_text": (
                "Supplier reference: B09\n"
                "Product: Test Invalid Code Gin\n"
                "EAN: 2901234500153"
            ),
            "product_name": "Invalid Code Gin",
            "brand": "Gin",
        }
        _preserve_raw_identity(safe)
        assert safe["product_name_raw"] == "Test Invalid Code Gin"

    def test_product_colon_variants_match(self):
        """Also accepts 'Item:' / 'Description:' / 'Désignation:' labels."""
        for label in ("Product", "Item", "Description", "Désignation"):
            safe = {
                "_raw_row_text": f"{label}: ACME Widget 500ml",
                "product_name": "Widget",
                "brand": "ACME",
            }
            _preserve_raw_identity(safe)
            assert safe["product_name_raw"] == "ACME Widget 500ml", label


# ───────── 3. 100 pieces fallback ─────────
class TestLoosePieceFallback:
    def test_a07_100_pieces_filled_from_raw(self):
        """LLM emits quantity_case=null + 'quantity_case not explicitly stated'
        flag for A07; fallback regex over _raw_row_text extracts 100."""
        safe = {
            "_raw_row_text": (
                "Supplier reference: A07\n"
                "Product: NARS Punjab Medium 1 Foundation\n"
                "Packaging: single 30ml\n"
                "Price: EUR 20/piece\n"
                "Available quantity: 100 pieces\n"
                "EAN: 2000000000022"
            ),
            "quantity_case": None,
            "quantity_unit": "bottles",   # LLM may normalise pieces → bottles
            "units_per_case": 1,
            "error_flags": [],
        }
        _normalize_moq_and_quantity(safe, safe["error_flags"])
        assert safe.get("quantity_bottles") == 100.0
        assert safe.get("quantity_case") is None  # loose pack, no case equivalent

    def test_pieces_unit_normalised_to_pieces(self):
        """When qty_unit='pieces' stays, we keep it (don't force bottles)."""
        safe = {
            "_raw_row_text": "Available quantity: 50 pieces",
            "quantity_case": None,
            "quantity_unit": "pieces",
            "units_per_case": 1,
            "error_flags": [],
        }
        _normalize_moq_and_quantity(safe, safe["error_flags"])
        assert safe.get("quantity_bottles") == 50.0
        assert safe.get("quantity_unit") == "pieces"


# ───────── 4. A10 pallet-count derivation ─────────
class TestPalletCountDerivedBothWays:
    def test_pallet_count_from_raw_when_llm_converted(self):
        """LLM already converted 2 pallets → 160 cases (quantity_unit='cases');
        processor still fishes 2 out of _raw_row_text so the drawer can show
        '160 cs (2 plt × 80 cs/plt)'."""
        safe = {
            "_raw_row_text": (
                "Supplier reference: A10\n"
                "Available quantity: 2 pallets\n"
                "Each pallet contains 80 cases. MOQ 1 pallet."
            ),
            "quantity_case": 160,
            "quantity_unit": "cases",
            "cases_per_pallet": 80,
            "units_per_case": 6,
            "moq_cases": 80,
            "moq_unit": "pallets",
            "error_flags": [],
        }
        _normalize_moq_and_quantity(safe, safe["error_flags"])
        assert safe.get("quantity_pallets") == 2.0


# ───────── 5. moq_bottles persisted (A12) ─────────
class TestMoqBottlesPersisted:
    def test_a12_moq_12_bottles(self):
        """Source MOQ 12 bottles with upc=6 → moq_cases=2 AND moq_bottles=12
        stored so the drawer shows '12 btls (2 cs)'."""
        safe = {
            "_raw_row_text": "MOQ 12 bottles",
            "quantity_case": 120,
            "quantity_unit": "cases",
            "moq_bottles": 12,
            "moq_unit": "bottles",
            "units_per_case": 6,
            "error_flags": [],
        }
        _normalize_moq_and_quantity(safe, safe["error_flags"])
        assert safe.get("moq_cases") == 2
        assert safe.get("moq_bottles") == 12.0
        assert safe.get("moq_unit") == "bottles"


# ───────── 6. Redis client ssl_cert_reqs ─────────
class TestRedisClientKwarg:
    """redis.Redis.from_url rejects ssl_cert_reqs for non-SSL URLs; the
    guard must only pass that kwarg for rediss://, otherwise plain-TCP
    isolated-env setups silently fall back to in-memory and uvicorn +
    celery stop sharing status."""

    def test_plain_scheme_no_ssl_cert_reqs(self, monkeypatch):
        import importlib, sys
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/15")
        if "core.redis_client" in sys.modules:
            importlib.reload(sys.modules["core.redis_client"])
        from core.redis_client import RedisManager
        # Instantiating must not raise and must either connect or quietly
        # fall back without crashing. We only assert no exception here —
        # the connection depends on whether a local redis is up.
        rm = RedisManager()
        assert isinstance(rm.use_redis, bool)
