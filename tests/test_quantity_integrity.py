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


class TestBottlesToCasesConversion:
    """When a real case pack exists, convert correctly."""

    def test_12_pack_converts(self):
        out, flags = _run(qty=240, unit='bottles', upc=12)
        assert out['quantity_case'] == 20.0
        assert any('converted from 240 bottles' in f for f in flags)

    def test_6_pack_converts(self):
        out, flags = _run(qty=120, unit='bottles', upc=6)
        assert out['quantity_case'] == 20.0

    def test_odd_division_rounds(self):
        out, flags = _run(qty=15, unit='bottles', upc=12)
        # 15 / 12 = 1.25
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
    """R-14 (Samuel's acceptance blocker): a VALID bottles->cases conversion is
    an informational audit note, not a REVIEW case. The flag text is a contract
    shared by three places — core.openai_client._INFO_PREFIXES, api.admin
    INFO_PATTERNS, and the dashboard classifyFlags INFO_FLAG_PREFIXES — all of
    which key on the exact prefix 'Quantity converted from'. If the flag text
    ever drifts from that prefix, the row silently reverts to a false REVIEW
    badge (the Johnnie Walker Red Label symptom). Lock the prefix here."""

    CONVERSION_PREFIX = "Quantity converted from"

    def test_valid_conversion_flag_uses_contract_prefix(self):
        out, flags = _run(qty=9600, unit='bottles', upc=6)
        # 9600 bottles / 6 per case = 1600 cases — the exact JW Red Label case.
        assert out['quantity_case'] == 1600.0
        conv = [f for f in flags if f.startswith(self.CONVERSION_PREFIX)]
        assert conv, (
            f"conversion flag must start with {self.CONVERSION_PREFIX!r} so the "
            f"info-classification in every layer keeps matching; got {flags!r}"
        )

    def test_loose_pack_does_not_emit_conversion_prefix(self):
        # No case pack -> no 'converted to cases' claim (would be a lie).
        _out, flags = _run(qty=20, unit='bottles', upc=None)
        assert not any(f.startswith(self.CONVERSION_PREFIX) for f in flags)
