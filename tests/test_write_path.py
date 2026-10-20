"""Write-dependent behaviours, run in CI against in-memory SQLite (no network,
no production). These cover the items Samuel asked to see demonstrated:

  - REQ 08  Expired offers: an expired offer is RETAINED in the Best Prices
            group (history) but is NEVER the current headline.
  - REQ 15  Manual create / edit persistence: creating an offer persists every
            field (including the category-specific ones), and editing it
            persists the change and logs a correction.

They exercise the real endpoint functions (api.records.get_best_prices,
api.offers_write.create_offer / edit_offer) directly against a throwaway DB.
"""
import os
import sys
import pathlib
import asyncio
from datetime import datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("OPENAI_API_KEY", "x")

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from core.database import Base  # noqa: E402
from models.offer_item import OfferItemDB  # noqa: E402
from models.source_file import SourceFileDB  # noqa: E402
from api.records import get_best_prices, get_benchmarks, get_records  # noqa: E402
from api.offers_write import create_offer, edit_offer, get_offer, OfferCreate, OfferPatch  # noqa: E402
from fastapi import HTTPException  # noqa: E402

_ = SourceFileDB  # keep import so create_all builds the FK target table


def _session():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)          # builds every table, incl. offer_corrections
    return sessionmaker(bind=eng)()


def _offer(db, uid, price_eur, valid_until, supplier):
    """One row in a single, well-qualified peer group (same identity + EAN)."""
    db.add(OfferItemDB(
        uid=uid, job_id="t", source_file_id=None,
        brand="Belvedere", product_name="Pure Vodka",
        unit_volume_ml=700, units_per_case=6,
        incoterm="EXW", location="Rotterdam", ean_code="5901041000956",
        category_slug="wines_spirits",
        price_per_unit=price_eur, price_per_unit_eur=price_eur, currency="EUR",
        supplier_name=supplier, valid_until=valid_until,
        offer_date=datetime.utcnow() - timedelta(days=10),
        created_at=datetime.utcnow(),
    ))
    db.commit()


def _bp(db):
    return asyncio.run(get_best_prices(
        skip=0, limit=24, search=None, category=None,
        sub_category=None, category_slug=None, db=db))


class TestExpiredOfferHandling:
    """REQ 08 — expired offers stay in history but never win the headline."""

    def test_expired_retained_but_not_headline(self):
        db = _session()
        _offer(db, "active", 50.0, valid_until="2099-12-31", supplier="SupplierA")   # active, dearer
        _offer(db, "expired", 40.0, valid_until="2020-01-01", supplier="SupplierB")  # cheaper but EXPIRED
        groups = _bp(db)["groups"]
        assert len(groups) == 1, groups
        peer = groups[0]["peers"][0]          # one peer: EXW / Rotterdam
        # The cheaper offer is expired → it must NOT be the headline…
        assert peer["best_uid"] == "active", peer
        assert peer["has_active_offers"] is True
        # …but it must still be present in the group (history), flagged expired.
        by_uid = {o["uid"]: o for o in peer["offers"]}
        assert by_uid["expired"]["is_expired"] is True
        assert by_uid["active"]["is_expired"] is False

    def test_all_expired_group_has_no_active(self):
        db = _session()
        _offer(db, "e1", 40.0, valid_until="2020-01-01", supplier="SupplierA")
        _offer(db, "e2", 45.0, valid_until="2019-06-30", supplier="SupplierB")
        peer = _bp(db)["groups"][0]["peers"][0]
        assert peer["has_active_offers"] is False
        assert all(o["is_expired"] for o in peer["offers"])


class TestManualCreateEditPersistence:
    """REQ 15 — create/edit persist every field, incl. category-specific ones."""

    def test_create_persists_all_fields(self):
        db = _session()
        res = create_offer(OfferCreate(
            product_name="Black Opium", brand="YSL", category="perfumes",
            category_slug="perfumes", unit_volume_ml=90, units_per_case=1,
            perfume_format="EDP", retail_state="tester", gender="women",
            custom_status="T2", currency="EUR", price_per_unit=52.0,
            supplier_name="SupplierX", incoterm="EXW", location="Rotterdam",
        ), db=db)
        uid = res["uid"]
        row = db.query(OfferItemDB).filter(OfferItemDB.uid == uid).first()
        assert row is not None
        assert row.product_name == "Black Opium" and row.brand == "YSL"
        assert row.perfume_format == "EDP" and row.retail_state == "tester"
        assert row.gender == "women" and row.custom_status == "T2"
        assert row.price_per_unit_eur == 52.0
        assert row.source_channel == "manual"      # traceability, not a test label

    def test_edit_persists_and_logs_correction(self):
        db = _session()
        uid = create_offer(OfferCreate(
            product_name="Black Opium", brand="YSL", category_slug="perfumes",
            retail_state="tester", currency="EUR", price_per_unit=52.0,
        ), db=db)["uid"]
        # correct the price and the retail state
        out = edit_offer(uid, OfferPatch(fields={"price_per_unit": 49.5, "retail_state": "retail"}), db=db)
        assert set(out["changed"]) == {"price_per_unit", "retail_state"}
        row = db.query(OfferItemDB).filter(OfferItemDB.uid == uid).first()
        assert row.price_per_unit == 49.5 and row.price_per_unit_eur == 49.5
        assert row.retail_state == "retail"

    def test_edit_rejects_unknown_field(self):
        db = _session()
        uid = create_offer(OfferCreate(product_name="X", brand="Y", price_per_unit=1.0), db=db)["uid"]
        out = edit_offer(uid, OfferPatch(fields={"not_a_real_column": "boom"}), db=db)
        assert out["changed"] == []


class TestServerSideFilters:
    """REQ: the customs-status and supplier filters must apply to the WHOLE
    dataset server-side, not just the loaded page — the bug was that selecting
    T1/T2 (or keeping a supplier when clearing the category) returned nothing
    because matching rows were past the first page."""

    def _rows(self, db):
        # 150 no-status rows (fill the "first page"), then the T2/supplier rows
        for n in range(150):
            db.add(OfferItemDB(uid=f"n{n}", job_id="t", brand="Bulk", product_name=f"P{n}",
                               price_per_unit_eur=10.0, supplier_name="MassSupplier",
                               created_at=datetime.utcnow()))
        db.add(OfferItemDB(uid="t2a", job_id="t", brand="Hennessy", product_name="VS",
                           custom_status="T2", supplier_name="FBC Trades",
                           price_per_unit_eur=40.0, created_at=datetime.utcnow() - timedelta(days=1)))
        db.add(OfferItemDB(uid="t1a", job_id="t", brand="Glenlivet", product_name="12YO",
                           custom_status="T1", supplier_name="FBC Trades",
                           price_per_unit_eur=30.0, created_at=datetime.utcnow() - timedelta(days=2)))
        db.commit()

    def _recs(self, db, **kw):
        base = dict(skip=0, limit=100, source_file_id=None, search=None, category=None,
                    sub_category=None, category_slug=None, custom_status=None, supplier=None, db=db)
        base.update(kw)
        return asyncio.run(get_records(**base))

    def test_t2_filter_alone_finds_deep_rows(self):
        db = _session(); self._rows(db)
        out = self._recs(db, custom_status="T2")
        assert out["total"] == 1
        assert out["records"][0]["uid"] == "t2a"

    def test_supplier_filter_alone_finds_deep_rows(self):
        db = _session(); self._rows(db)
        out = self._recs(db, supplier="FBC Trades")
        assert out["total"] == 2
        assert {r["uid"] for r in out["records"]} == {"t2a", "t1a"}

    def test_none_customs_filter(self):
        db = _session(); self._rows(db)
        out = self._recs(db, custom_status="__none__")
        assert out["total"] == 150   # the T1/T2 rows excluded


class TestExactOfferLookup:
    """REQ 12 / P1 — the drawer must open the EXACT offer by uid (alerts and
    benchmark links). GET /api/offers/{uid} returns one record, no dedup/scan."""

    def test_get_exact_offer(self):
        db = _session()
        for uid, price in (("a", 10.0), ("b", 20.0), ("c", 30.0)):
            db.add(OfferItemDB(uid=uid, job_id="t", brand="B", product_name="P",
                               price_per_unit_eur=price, created_at=datetime.utcnow()))
        db.commit()
        assert get_offer("b", db=db)["uid"] == "b"
        assert get_offer("c", db=db)["price_per_unit_eur"] == 30.0

    def test_missing_offer_404(self):
        db = _session()
        try:
            get_offer("nope", db=db)
            assert False, "expected 404"
        except HTTPException as e:
            assert e.status_code == 404


class TestBenchmarkExcludesZeroPrice:
    """P1 — a €0.00 / unusable-price row must NOT drag the 12M LOW to 0 (which
    produced +Infinity% on Tanqueray Rangpur)."""

    def test_zero_price_not_in_benchmark(self):
        db = _session()
        # one real offer + one zero-price "unusable" row in the same peer
        db.add(OfferItemDB(uid="real", job_id="t", brand="Tanqueray", product_name="Rangpur",
                           unit_volume_ml=1000, units_per_case=6, incoterm="EXW", location="NL",
                           ean_code="5000289021234", category_slug="wines_spirits",
                           price_per_unit=18.0, price_per_unit_eur=18.0,
                           created_at=datetime.utcnow() - timedelta(days=5)))
        db.add(OfferItemDB(uid="zero", job_id="t", brand="Tanqueray", product_name="Rangpur",
                           unit_volume_ml=1000, units_per_case=6, incoterm="EXW", location="NL",
                           ean_code="5000289021234", category_slug="wines_spirits",
                           price_per_unit=0.0, price_per_unit_eur=0.0,
                           created_at=datetime.utcnow() - timedelta(days=5)))
        db.commit()
        bm = asyncio.run(get_benchmarks(window="12M", db=db))
        vals = next(iter(bm["peers"].values()))
        assert vals["low_eur"] == 18.0        # not 0.0
        assert vals["samples"] == 1           # the zero-price row is excluded


class TestBestPricesSameEanDifferentName:
    """R-10/R-11/P1 — two suppliers of the SAME EAN must both appear in Best
    Prices even when they format the product name differently (Mugler EAN
    3614274085518: 'Angel Fantasm' vs 'Angel Fantasm Tester')."""

    def test_same_ean_two_suppliers_grouped(self):
        db = _session()
        for uid, name, sup, gender in (
            ("a", "Angel Fantasm Tester", "Balkan Luxury Supply Ltd.", "women"),
            ("b", "Angel Fantasm", "L'Oréal Luxury", None)):
            db.add(OfferItemDB(uid=uid, job_id="t", brand="Mugler", product_name=name,
                               unit_volume_ml=100, units_per_case=1, incoterm="EXW", location="NL",
                               ean_code="3614274085518", category="Perfumes", category_slug="perfumes",
                               perfume_format="EDP", retail_state="tester", gender=gender,
                               price_per_unit=43.0, price_per_unit_eur=43.0,
                               supplier_name=sup, offer_date=datetime.utcnow() - timedelta(days=25),
                               created_at=datetime.utcnow()))
        db.commit()
        groups = _bp(db)["groups"]
        # one product group, and its peer shows BOTH suppliers
        assert len(groups) == 1, groups
        suppliers = {o["supplier_name"] for g in groups for p in g["peers"] for o in p["offers"]}
        assert suppliers == {"Balkan Luxury Supply Ltd.", "L'Oréal Luxury"}


class TestSimilarRowsKeepDistinctIdentity:
    """Samuel's wrong-record navigation finding. Two near-identical offers (same
    supplier/product/price/pack, different source file + date — e.g. the same
    standing offer received twice) must each keep their OWN uid in the data the
    UI binds to, and an equal-priced peer must NOT be presented as the other's
    lower historical benchmark (which is what made the '12M LOW' link jump to
    the other record and look like wrong-record navigation)."""

    def _pair(self, db):
        # same peer (EAN + incoterm + location), same price, two source files/dates
        for uid, sfid, date in (("jw-pdf", "fileA", "2026-03-16"),
                                ("jw-email", "fileB", "2026-03-24")):
            db.add(OfferItemDB(
                uid=uid, job_id="t", source_file_id=sfid,
                brand="Johnnie Walker", product_name="Red Label",
                unit_volume_ml=1000, units_per_case=6, incoterm="DAP",
                location="Loendersloot", ean_code="5000267023656",
                category_slug="wines_spirits",
                price_per_unit=6.70, price_per_unit_eur=6.70, currency="EUR",
                supplier_name="FBC Trades", quantity_case=1600, custom_status="T2",
                offer_date=datetime.strptime(date, "%Y-%m-%d"), created_at=datetime.utcnow(),
            ))
        db.commit()

    def test_both_rows_tracked_as_distinct_records(self):
        db = _session()
        self._pair(db)
        # Both rows land in one peer, but as TWO distinct uids (low + second-low),
        # never collapsed into one — so each All Offers row binds to its own record.
        bm = asyncio.run(get_benchmarks(window="12M", db=db))
        assert len(bm["peers"]) == 1
        vals = next(iter(bm["peers"].values()))
        assert vals["samples"] == 2
        assert {vals["low_uid"], vals["second_low_uid"]} == {"jw-pdf", "jw-email"}
        assert vals["low_uid"] != vals["second_low_uid"]

    def test_equal_priced_peer_is_not_a_lower_benchmark(self):
        db = _session()
        self._pair(db)
        bm = asyncio.run(get_benchmarks(window="12M", db=db))
        vals = next(iter(bm["peers"].values()))
        # low == second-low (both €6.70) → no genuine prior low below the current
        # price, so the UI must not render a navigable '12M LOW' link to the twin.
        assert vals["low_eur"] == 6.70
        assert vals["second_low_eur"] == 6.70
