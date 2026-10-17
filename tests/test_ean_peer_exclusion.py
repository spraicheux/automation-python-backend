"""EAN identity integrity (REQ 01 / R-04, and REQ 10/11 peer identity).

Samuel's finding: an invalid-check-digit EAN still retrieved a record, and
checksum exclusion from identity matching was "not proven". These tests prove,
in CI with no network/DB, that:

  1. ean_key() accepts only well-formed GS1 codes (8/12/13/14 digits with a
     valid check digit) and returns "" for everything else — including the
     exact invalid EAN from the report (2901234500153).
  2. An invalid EAN never contributes to peer identity: two rows that share
     the same *invalid* EAN but differ in attributes do NOT collapse into one
     peer group via that EAN.
  3. A valid EAN folds consistently so genuine peers still align.
"""
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from core.normalization import ean_key, peer_group_id  # noqa: E402


class TestEanCheckDigit:
    def test_report_invalid_ean_rejected(self):
        # The exact invalid EAN Samuel flagged on the NUXE record.
        assert ean_key("2901234500153") == ""

    def test_report_valid_ean_accepted(self):
        # The valid NARS EAN from the same report.
        assert ean_key("2901234500084") == "2901234500084"

    def test_bad_checksum_rejected(self):
        assert ean_key("3348901234567") == ""  # wrong check digit

    def test_wrong_length_rejected(self):
        assert ean_key("12345") == ""
        assert ean_key("notanean") == ""

    def test_punctuation_is_tolerated_for_valid_code(self):
        # A valid code written with noise still folds to bare digits.
        assert ean_key("EAN 2901234500084") == "2901234500084"


class TestInvalidEanNeverJoinsPeerGroup:
    """Two rows sharing only an INVALID ean (and differing otherwise) must not
    be pulled into the same peer group through that ean."""

    def test_invalid_ean_does_not_merge_distinct_products(self):
        bad = "2901234500153"
        a = peer_group_id("NUXE", "Reve de Miel", unit_volume_ml=40,
                          incoterm="EXW", location="Rotterdam",
                          ean_code=bad, category_slug="cosmetics")
        b = peer_group_id("Clinique", "Moisture Surge", unit_volume_ml=50,
                          incoterm="EXW", location="Rotterdam",
                          ean_code=bad, category_slug="cosmetics")
        assert a != b, "invalid EAN must not merge two different products"

    def test_invalid_ean_peer_key_equals_no_ean_peer_key(self):
        # Since the invalid EAN is dropped, the peer key must be identical to
        # the same row computed with no EAN at all — proving the bad code is
        # simply ignored for identity.
        kw = dict(unit_volume_ml=40, incoterm="EXW", location="Rotterdam",
                  category_slug="cosmetics")
        with_bad = peer_group_id("NUXE", "Reve de Miel", ean_code="2901234500153", **kw)
        without = peer_group_id("NUXE", "Reve de Miel", ean_code=None, **kw)
        assert with_bad == without

    def test_valid_ean_folds_consistently(self):
        kw = dict(unit_volume_ml=40, incoterm="EXW", location="Rotterdam",
                  category_slug="cosmetics")
        one = peer_group_id("NARS", "Prism Veil", ean_code="2901234500084", **kw)
        two = peer_group_id("NARS", "Prism Veil", ean_code="2901234500084", **kw)
        assert one == two and one


class TestLooseAndSinglePackMerge:
    """R-07/R-10: a loose pack (units_per_case None) and a single-unit pack
    (units_per_case 1) are the same physical product and MUST land in the same
    peer group — otherwise same-EAN offers split and a historical offer drops
    out of Best Prices (the NARS 30ml case Samuel flagged)."""

    def _nars(self, upc):
        return peer_group_id("NARS", "Prism Veil Foundation", unit_volume_ml=30,
                             units_per_case=upc, incoterm="EXW", location="Rotterdam",
                             category_slug="cosmetics", retail_state="retail",
                             shade="Deauville", ean_code="2901234500084")

    def test_none_and_one_merge(self):
        assert self._nars(None) == self._nars(1)
        assert self._nars(0) == self._nars(1)

    def test_multi_pack_stays_distinct(self):
        assert self._nars(6) != self._nars(1)
        assert self._nars(12) != self._nars(6)
