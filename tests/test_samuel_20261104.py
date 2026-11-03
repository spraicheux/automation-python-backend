"""Regression tests for Samuel's 2026-11-04 production QA findings.

1. Spaced-volume search: "50 ml" must parse the same as "50ml" (the raw
   search string was split on whitespace, separating "50" and "ml").
2. Derived unit-price precision: the EUR unit price of a derived per-bottle
   figure must be ppc_eur / units in ONE rounding step, never
   round(ppc/units, 2) * fx (double-rounding the native unit first).
"""
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from api.records import _parse_search_tokens  # noqa: E402


class TestSpacedVolumeSearch:
    def test_50ml_variants_all_parse_to_50(self):
        for q in ["50ml", "50 ml", "5cl", "5 cl",
                  "jack daniel's honey 50 ml", "Jack Daniel's Honey 5 cl"]:
            _, vol, _, _, _ = _parse_search_tokens(q)
            assert vol == 50, (q, vol)

    def test_other_volumes_unaffected(self):
        assert _parse_search_tokens("750ml")[1] == 750
        assert _parse_search_tokens("750 ml")[1] == 750
        assert _parse_search_tokens("1 L")[1] == 1000
        assert _parse_search_tokens("1L")[1] == 1000
        # pack token still works
        assert _parse_search_tokens("Hennessy 12x70cl")[1] == 700

    def test_spaced_volume_narrows_not_widens(self):
        # "honey 50 ml" keeps the text term AND sets volume 50 (so the search
        # filters to the 50ml row, not 750ml/1L).
        text, vol, _, _, _ = _parse_search_tokens("honey 50 ml")
        assert vol == 50 and "honey" in text

    def test_words_ending_in_unit_not_mangled(self):
        # "label" / "gift" must not be treated as a volume unit.
        text, vol, _, _, _ = _parse_search_tokens("5 label gift")
        assert vol is None and "label" in text


class TestDerivedUnitPricePrecision:
    def _eur(self, ppc, ppu_llm, upc, fx=0.92):
        from core.openai_client import clean_product_data
        from workers.processor import _normalize_currency_and_prices
        prod = {"product_name": "X", "brand": "X", "currency": "USD",
                "fx_rate": fx, "price_per_case": ppc,
                "price_per_unit": ppu_llm, "units_per_case": upc}
        cleaned = clean_product_data(dict(prod))
        after = _normalize_currency_and_prices(dict(cleaned))
        return after["price_per_unit_eur"], after["price_per_case_eur"]

    def test_bacardi_limon(self):
        # USD 92/case, 12/case, fx 0.92. ppu_eur must be 84.64/12 = 7.0533,
        # NOT round(92/12,2)*0.92 = 7.0564.
        u, c = self._eur(92, 7.67, 12)
        assert c == 84.64
        assert u == 7.0533, u

    def test_grey_goose(self):
        u, _ = self._eur(112, 18.67, 6)
        assert u == 17.1733, u          # NOT 17.1764

    def test_jd_honey_50ml(self):
        u, _ = self._eur(77, 0.64, 120)
        assert u == 0.5903, u           # NOT 0.5888

    def test_explicit_discount_unit_price_preserved(self):
        # When the source quotes a per-bottle price that is NOT ppc/units
        # (a genuine case discount), the explicit unit is preserved (converted
        # straight), not overwritten with ppc/units.
        # USD 120/case, 6/case => ppc/units = 20; but supplier quotes 18/bottle.
        u, c = self._eur(120, 18.0, 6)
        assert c == round(120 * 0.92, 4)         # 110.4
        assert u == round(18.0 * 0.92, 4)        # 16.56 — explicit unit kept


class TestPeerKeyIncludesCustomStatus:
    """Samuel 2026-11-04: an EAN-less T1/T2 row blanked its 12M-Low because
    to_dict's peer_group_id omitted custom_status while get_benchmarks included
    it (JW Red Label 24-Mar "no older historical" despite an identical 16-Mar)."""

    def _session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from core.database import Base
        from models.offer_item import OfferItemDB
        from models.source_file import SourceFileDB  # noqa: F401
        eng = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(eng)
        return sessionmaker(bind=eng)(), OfferItemDB

    def _pg(self, cs):
        db, OfferItemDB = self._session()
        r = OfferItemDB(uid=f"j_{cs}", job_id="t", brand="Johnnie Walker",
                        product_name="Red Label", ean_code=None, unit_volume_ml=1000,
                        units_per_case=6, incoterm="DAP", location="Loendersloot",
                        custom_status=cs, price_per_unit_eur=6.7)
        db.add(r); db.commit()
        return r.to_dict()["peer_group_id"]

    def test_ean_less_peer_key_includes_custom_status(self):
        assert "|t2" in self._pg("T2")

    def test_t1_and_t2_are_distinct_peers(self):
        assert self._pg("T1") != self._pg("T2")


class TestAuthorisedLocationSplitNotesAreInformational:
    """Samuel 2026-11-04: Jameson's authorised compound-location split notes
    documented an approved transformation but showed under Needs review."""

    def _session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from core.database import Base
        from models.offer_item import OfferItemDB
        from models.source_file import SourceFileDB  # noqa: F401
        eng = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(eng)
        return sessionmaker(bind=eng)(), OfferItemDB

    def test_location_split_notes_not_review_worthy(self):
        import json
        from api.admin import sync_review_flags
        db, OfferItemDB = self._session()
        db.add(OfferItemDB(
            uid="jm1", job_id="t", brand="Jameson", product_name="Irish Whiskey",
            category_slug="wines_spirits", needs_manual_review=True,
            error_flags=json.dumps([
                "compound location (Loendersloot, Revera, Mentrex) split into 3 rows (Samuel 2026-10-28, authorised)",
                "alternative-location row backfilled from source line Exwork Loendersloot or Revera or Mentrex (Samuel 2026-10-28, authorised)",
                "source lists 3 alternative locations (Loendersloot, Revera, Mentrex); only Loendersloot persisted",
                "sub_category inferred from brand name",
            ])))
        db.commit()
        sync_review_flags(category_slug="wines_spirits", apply=True, db=db)
        assert not db.query(OfferItemDB).filter_by(uid="jm1").first().needs_manual_review
