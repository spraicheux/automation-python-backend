"""
R-14 Quantity integrity — the NARS retest bug.

Locks in: 20 bottles on a loose pack MUST NOT become quantity_case=20.
When a case pack is known, quantity_case = bottles / units_per_case.
When no case pack is known, quantity_case is None (not the bottle count).
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from workers.processor import _normalize_moq_and_quantity


def _run(**fields):
    """Thin wrapper so each test passes just the fields it cares about."""
    data = {
        'quantity_case': fields.get('qty'),
        'quantity_unit': fields.get('unit'),
        'units_per_case': fields.get('upc'),
        'moq_cases': fields.get('moq_cases'),
        'moq_bottles': fields.get('moq_bottles'),
        'moq_unit': fields.get('moq_unit'),
    }
    flags = []
    out = _normalize_moq_and_quantity(data, flags)
    return out, flags


class TestLoosePackBottleQuantity:
    """The exact NARS retest regression — 20 bottles on a loose pack must
    NOT persist as quantity_case = 20."""

    def test_loose_pack_bottles_null_quantity_case(self):
        out, flags = _run(qty=20, unit='bottles', upc=None)
        assert out['quantity_case'] is None, (
            "R-14 regression: 20 bottles on loose pack became quantity_case=20"
        )
        assert any('loose' in f.lower() or 'cases unknown' in f.lower() for f in flags)

    def test_loose_pack_upc_1_null_quantity_case(self):
        out, flags = _run(qty=20, unit='bottles', upc=1)
        assert out['quantity_case'] is None

    def test_loose_pack_upc_0_null_quantity_case(self):
        out, flags = _run(qty=20, unit='bottles', upc=0)
        assert out['quantity_case'] is None


class TestBottlesToCasesPassthrough:
    """Rule 0.3 contract (Samuel 2026-10-29 acceptance): the LLM is
    required to emit `quantity_case` ALREADY expressed in cases, even
    when the source stated availability in bottles. In that case
    `quantity_unit` remains "bottles" for traceability but the value
    in `quantity_case` is the case count. _normalize_moq_and_quantity
    must trust that value and NOT re-divide by units_per_case
    (pre-Rule-0.3 pipelines double-converted, corrupting quantity —
    e.g. Johnnie Walker Red Label's "9600 bottles / 6" would land
    at 266.67 cases instead of 1600)."""

    def test_12_pack_passthrough(self):
        # LLM pre-converted: "240 btls on 12-pack" → quantity_case=20.
        out, flags = _run(qty=20, unit='bottles', upc=12)
        assert out['quantity_case'] == 20
        assert not any('converted from' in f.lower() for f in flags)

    def test_6_pack_passthrough(self):
        out, flags = _run(qty=20, unit='bottles', upc=6)
        assert out['quantity_case'] == 20

    def test_non_integer_case_count_passthrough(self):
        # Already-converted non-integer case count (e.g. 1.25 cases)
        # must pass through unchanged.
        out, flags = _run(qty=1.25, unit='bottles', upc=12)
        assert out['quantity_case'] == 1.25


class TestCaseUnitPassthrough:
    """When the extractor says 'cases', do nothing — the value is already in cases."""

    def test_cases_unchanged(self):
        out, flags = _run(qty=20, unit='cases', upc=12)
        assert out['quantity_case'] == 20
        assert not any('converted' in f for f in flags)


class TestFtl:
    """FTL (Full Truck Load) is not a numeric cases quantity."""

    def test_ftl_nulls(self):
        out, flags = _run(qty=1, unit='ftl', upc=12)
        assert out['quantity_case'] is None
        assert any('FTL' in f for f in flags)


class TestConversionFlagContract:
    """Legacy contract test. The pipeline no longer re-converts quantity
    (Rule 0.3: the LLM does it), so the conversion-flag string is now
    emitted by the extraction prompt rather than by _normalize_moq_and_
    quantity. The classification contract still holds (any flag that
    reaches the DB starting with 'Quantity converted from' is info, not
    review) — the test for its presence belongs at the extraction level,
    not here."""

    def test_loose_pack_still_emits_no_conversion_claim(self):
        # No case pack → the pipeline never claims a conversion happened.
        _out, flags = _run(qty=20, unit='bottles', upc=None)
        assert not any(f.lower().startswith("quantity converted from") for f in flags)
