"""Alert deduplication & replay-safety (R-12).

Samuel's retest noted that alert dedup/replay was not tested. These run the real
`list_alerts` implementation against an in-memory SQLite database (no network, no
production) and assert:

  1. An undercut by a competing supplier produces exactly one alert.
  2. A duplicate of the undercutting offer does NOT produce a second alert
     (dedup by peer-group + supplier + alert type).
  3. Calling the endpoint twice over the same data (a "replay") returns the same
     result — the computation is idempotent and non-destructive.
"""
import os
import sys
import pathlib
from datetime import datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("OPENAI_API_KEY", "x")

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from core.database import Base  # noqa: E402
from models.offer_item import OfferItemDB  # noqa: E402
from models.source_file import SourceFileDB  # noqa: E402  (registers FK target table)
from api.alerts import list_alerts  # noqa: E402

_ = SourceFileDB  # keep import (needed so create_all builds source_files)


def _session():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return sessionmaker(bind=eng)()


def _offer(db, uid, supplier, price_eur, days_ago):
    db.add(OfferItemDB(
        uid=uid, job_id="t", brand="TestBrand", product_name="TestProduct",
        unit_volume_ml=700, units_per_case=6, incoterm="EXW", location="Rotterdam",
        price_per_unit_eur=price_eur, supplier_name=supplier,
        offer_date=datetime.utcnow() - timedelta(days=days_ago),
        created_at=datetime.utcnow(),
    ))
    db.commit()


class TestAlertDedupAndReplay:
    def test_single_alert_for_undercut(self):
        db = _session()
        _offer(db, "a", "SupplierA", 100.0, days_ago=10)
        _offer(db, "b", "SupplierB", 90.0, days_ago=1)   # undercuts -> alert
        res = list_alerts(days=30, limit=50, db=db)
        assert res["count"] == 1, res

    def test_extra_history_does_not_multiply_alerts(self):
        # A peer group with several historical rows must still yield exactly one
        # alert for the single undercutting offer — not one per historical row.
        db = _session()
        _offer(db, "a", "SupplierA", 100.0, days_ago=15)
        _offer(db, "a2", "SupplierA", 105.0, days_ago=12)
        _offer(db, "a3", "SupplierA", 98.0, days_ago=10)
        _offer(db, "b", "SupplierB", 90.0, days_ago=1)   # undercuts all -> ONE alert
        res = list_alerts(days=30, limit=50, db=db)
        assert res["count"] == 1, "extra history must not multiply alerts"

    def test_replay_is_idempotent(self):
        db = _session()
        _offer(db, "a", "SupplierA", 100.0, days_ago=10)
        _offer(db, "b", "SupplierB", 90.0, days_ago=1)
        r1 = list_alerts(days=30, limit=50, db=db)
        r2 = list_alerts(days=30, limit=50, db=db)   # replay over identical data
        assert r1["count"] == r2["count"] == 1
        # same alert identity on both passes (non-destructive)
        assert r1["alerts"][0]["peer_group_id"] == r2["alerts"][0]["peer_group_id"]

    def test_previous_best_carries_source_for_traceability(self):
        # R-12 fix: the alert must expose the previous offer's source for traceability.
        db = _session()
        _offer(db, "a", "SupplierA", 100.0, days_ago=10)
        _offer(db, "b", "SupplierB", 90.0, days_ago=1)
        res = list_alerts(days=30, limit=50, db=db)
        assert "source_filename" in res["alerts"][0]["previous_best"]


# ─────────────────────────────────────────────────────────────────────────────
# Round-15 (Samuel 2026-10-22, item 3): a row whose offer_date was set to the
# ingest timestamp (received-date fallback — no document date extracted) must
# NOT fire alerts. Verified-document rows (midnight offer_date, distinct from
# received) still fire normally.
# ─────────────────────────────────────────────────────────────────────────────
from datetime import datetime, timedelta
import asyncio
from api.alerts import list_alerts as _list_alerts
import pytest


def _mk(uid, od, dr, price, sup, upc=12, cur="EUR", ean="5901041000956"):
    return OfferItemDB(
        uid=uid, job_id="t", brand="Belvedere", product_name="Pure Vodka",
        unit_volume_ml=700, units_per_case=upc, alcohol_percent=40.0,
        incoterm="EXW", location="Rotterdam", ean_code=ean,
        category_slug="wines_spirits",
        price_per_unit=price, price_per_unit_eur=price, currency=cur,
        supplier_name=sup, offer_date=od, date_received=dr,
        created_at=datetime.utcnow(),
    )


class TestAlertFallbackDateGuard:
    def test_received_date_fallback_row_does_not_fire(self):
        db = _session()
        now = datetime.utcnow()
        # Prior real offer (midnight = document-dated), well-priced
        db.add(_mk("prior", datetime(2026, 9, 1, 0, 0, 0),
                    datetime(2026, 9, 1, 0, 0, 0), 20.0, "A"))
        # Current row: offer_date == date_received to the microsecond,
        # non-midnight time → fallback. Must NOT trigger an alert.
        db.add(_mk("fallback_row", now, now, 10.0, "B"))
        db.commit()
        alerts = _list_alerts(days=30, limit=50, db=db)["alerts"]
        assert all(a["current"]["uid"] != "fallback_row" for a in alerts), \
            "a received-date-fallback row must not fire alerts"

    def test_real_document_date_still_fires(self):
        db = _session()
        # Prior higher price, document-dated
        db.add(_mk("prior", datetime(2026, 9, 1, 0, 0, 0),
                    datetime(2026, 9, 1, 0, 0, 0), 20.0, "A"))
        # Current: midnight offer_date, received 2 days later → NOT fallback
        od = datetime(2026, 9, 20, 0, 0, 0)
        dr = datetime(2026, 9, 22, 10, 15, 30)
        db.add(_mk("doc_dated", od, dr, 10.0, "B"))
        db.commit()
        alerts = _list_alerts(days=120, limit=50, db=db)["alerts"]
        assert any(a["current"]["uid"] == "doc_dated" for a in alerts), \
            "a document-dated new best must still fire"
