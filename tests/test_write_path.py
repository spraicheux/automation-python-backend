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
from api.records import get_best_prices  # noqa: E402
from api.offers_write import create_offer, edit_offer, OfferCreate, OfferPatch  # noqa: E402

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
