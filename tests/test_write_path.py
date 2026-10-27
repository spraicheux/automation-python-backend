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

    # Production runs on Postgres, which has translate(); the search path folds
    # accents with it (core.records._fold_col). In-memory SQLite lacks the
    # function, so register a Postgres-compatible shim for the CI harness only
    # (chars in `frm` with no counterpart in `to` are deleted).
    from sqlalchemy import event

    @event.listens_for(eng, "connect")
    def _register_translate(dbapi_conn, _rec):  # noqa: ANN001
        import re as _re
        def _translate(s, frm, to):
            if s is None:
                return None
            return "".join((to[frm.index(c)] if frm.index(c) < len(to) else "")
                            if c in frm else c for c in s)
        def _regexp_replace(s, pat, repl, _flags=""):
            if s is None:
                return None
            # Postgres regexp_replace with 'g' flag → re.sub (unlimited).
            return _re.sub(pat, repl, str(s))
        dbapi_conn.create_function("translate", 3, _translate)
        dbapi_conn.create_function("regexp_replace", 4, _regexp_replace)

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


# ─────────────────────────────────────────────────────────────────────────────
# Round-10 retest (Samuel 2026-10-21): search tokens, EAN peer merge, current
# offer at a single-supplier incoterm, review exclusion, ABV guard, Other pill.
# ─────────────────────────────────────────────────────────────────────────────

def _rec(db, **kw):
    """Insert one offer row; sensible defaults for the fields these tests ignore."""
    base = dict(
        job_id="t", source_file_id=None, currency="EUR",
        offer_date=datetime.utcnow() - timedelta(days=10),
        created_at=datetime.utcnow(),
    )
    base.update(kw)
    db.add(OfferItemDB(**base))
    db.commit()


def _records(db, search=None, category_slug=None):
    return asyncio.run(get_records(
        skip=0, limit=100, source_file_id=None, search=search, category=None,
        sub_category=None, category_slug=category_slug,
        custom_status=None, supplier=None, db=db))


class TestSearchTokens:
    """REQ 01/03/05 — reference, pack and year tokens are searchable."""

    def test_product_reference_is_searchable(self):
        db = _session()
        _rec(db, uid="m1", brand="Mugler", product_name="Angel Fantasm",
             unit_volume_ml=100, product_reference="40746", supplier_reference=None,
             price_per_unit_eur=43, category_slug="perfumes")
        # product_reference, not supplier_reference, carries the code the trader sees.
        assert _records(db, search="40746")["total"] == 1

    def test_pack_token_matches_pack_and_volume(self):
        db = _session()
        _rec(db, uid="jd12", brand="Jack Daniel's", product_name="Old No.7",
             unit_volume_ml=700, units_per_case=12, price_per_unit_eur=20)
        _rec(db, uid="jd6", brand="Jack Daniel's", product_name="Old No.7",
             unit_volume_ml=700, units_per_case=6, price_per_unit_eur=20)
        # "12x70cl" → volume 700 AND pack 12 → only the 12-pack row.
        r = _records(db, search="Jack 12x70cl")
        assert r["total"] == 1
        assert r["records"][0]["uid"] == "jd12"
        # A pack that does not exist returns 0 (correct, not a crash).
        assert _records(db, search="Jack 24x70cl")["total"] == 0

    def test_year_token_matches_vintage(self):
        db = _session()
        _rec(db, uid="mo25", brand="Moutai", product_name="Feitian",
             unit_volume_ml=500, vintage="2025", price_per_unit_eur=140)
        _rec(db, uid="mo24", brand="Moutai", product_name="Feitian",
             unit_volume_ml=500, vintage="2024", price_per_unit_eur=140)
        r = _records(db, search="Moutai Feitian 2025")
        assert r["total"] == 1 and r["records"][0]["uid"] == "mo25"

    def test_alt_locations_detected_from_raw_row_text(self):
        """Samuel 2026-10-27 Jameson: GPT dropped "Revera" and "Mentrex"
        from "Exwork Loendersloot or Revera or Mentrex" and persisted only
        Loendersloot. The _detect_dropped_alt_locations guard scans the
        raw row text, finds the alternatives, and adds an explicit
        'source lists N alternative locations …' review flag so no drop
        goes silent."""
        from workers.processor import _detect_dropped_alt_locations
        d = {
            "location": "Loendersloot",
            "_raw_row_text": "Jameson Irish Whiskey 6x70cl 40% T1, CODED@ €46.53/cs. Exwork Loendersloot or Revera or Mentrex.",
            "error_flags": ["sub_category inferred from brand name"],
        }
        _detect_dropped_alt_locations(d)
        joined = " || ".join(d["error_flags"])
        assert "alternative locations" in joined
        assert "Revera" in joined and "Mentrex" in joined and "Loendersloot" in joined
        assert "only Loendersloot persisted" in joined
        # Idempotent — second call must not duplicate the flag.
        _detect_dropped_alt_locations(d)
        assert sum("alternative locations" in f for f in d["error_flags"]) == 1

    def test_alt_locations_not_flagged_when_compound_already_stored(self):
        """If GPT already followed Rule 7 and stored "A / B / C" as a
        single compound location, don't double-flag the row."""
        from workers.processor import _detect_dropped_alt_locations
        d = {
            "location": "Loendersloot / Revera / Mentrex",
            "_raw_row_text": "Exwork Loendersloot or Revera or Mentrex",
            "error_flags": [],
        }
        _detect_dropped_alt_locations(d)
        assert d["error_flags"] == []

    def test_alt_locations_slash_separator(self):
        from workers.processor import _detect_dropped_alt_locations
        d = {
            "location": "Amsterdam",
            "_raw_row_text": "DAP Amsterdam / Rotterdam / Antwerp – 12x70cl",
            "error_flags": [],
        }
        _detect_dropped_alt_locations(d)
        assert any("alternative locations" in f and "Rotterdam" in f and "Antwerp" in f
                   for f in d["error_flags"])

    def test_alt_locations_prompt_rule_covers_or(self):
        """Guard the Rule 7 prompt text so a future edit cannot regress
        the "or" / three-location example that drove this fix."""
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent / "core" / "openai_client.py"
        src = p.read_text()
        # The Rule 7 block must mention "or" among the separators and must
        # carry the Loendersloot / Revera / Mentrex worked example.
        assert "RULE 7 — COMPOUND LOCATION" in src
        assert "\"or\"" in src.split("RULE 7")[1].split("RULE 8")[0]
        assert "Loendersloot / Revera / Mentrex" in src

    def test_duplicated_rows_snap_to_same_per_unit_price(self):
        """Samuel 2026-10-27: two Jameson rows on the same source file,
        both carrying the Rule-6 "multiple incoterms detected — row
        duplicated" flag, with the SAME €46.53/case but €7.75 vs €7.76 per
        bottle. Rule 6 says sibling rows must be identical except for
        incoterm/location — so snap price_per_unit = price_per_case /
        units_per_case before persistence to kill the display drift."""
        from workers.processor import _snap_duplicated_row_per_unit
        a = {"price_per_case": 46.53, "units_per_case": 6,
             "price_per_unit": 7.754,
             "error_flags": ["multiple incoterms detected — row duplicated"]}
        b = {"price_per_case": 46.53, "units_per_case": 6,
             "price_per_unit": 7.756,
             "error_flags": ["multiple incoterms detected — row duplicated"]}
        _snap_duplicated_row_per_unit(a)
        _snap_duplicated_row_per_unit(b)
        assert a["price_per_unit"] == b["price_per_unit"] == round(46.53 / 6, 4)
        # Row WITHOUT the duplicate flag is left alone (we don't want to
        # silently rewrite prices on rows the extractor didn't mark as split).
        c = {"price_per_case": 46.53, "units_per_case": 6,
             "price_per_unit": 7.90,  # legitimate per-unit list price
             "error_flags": ["supplier_name not found"]}
        _snap_duplicated_row_per_unit(c)
        assert c["price_per_unit"] == 7.90

    def test_ean_search_matches_when_stored_has_spaces(self):
        """Samuel 2026-10-27: Paco Rabanne EAN 3349668515714 was visible in
        the drawer/alert but the search returned 0. Root cause: the stored
        ean_code sometimes carries spaces or an 'EAN ' prefix the way the
        supplier wrote it, so a bare-digits trader paste didn't ilike-match.
        Fix: strip non-digits on both sides when the term looks like an EAN."""
        db = _session()
        # Stored with spaces (how a supplier might format it on the source line).
        _rec(db, uid="pr-spaces", brand="Paco Rabanne", product_name="1 Million",
             unit_volume_ml=100, ean_code="3 349668 515714",
             category_slug="perfumes", price_per_unit_eur=89)
        # Stored with an "EAN " prefix.
        _rec(db, uid="pr-prefix", brand="Paco Rabanne", product_name="Invictus",
             unit_volume_ml=100, ean_code="EAN 3349668579716",
             category_slug="perfumes", price_per_unit_eur=79)
        # Bare digits (control — must still match).
        _rec(db, uid="pr-bare",   brand="Paco Rabanne", product_name="Pure XS",
             unit_volume_ml=100, ean_code="3349668572014",
             category_slug="perfumes", price_per_unit_eur=74)
        # Every one of these finds exactly the right row.
        r1 = _records(db, search="3349668515714")
        assert r1["total"] == 1 and r1["records"][0]["uid"] == "pr-spaces"
        r2 = _records(db, search="3349668579716")
        assert r2["total"] == 1 and r2["records"][0]["uid"] == "pr-prefix"
        r3 = _records(db, search="3349668572014")
        assert r3["total"] == 1 and r3["records"][0]["uid"] == "pr-bare"
        # A short digit sequence (< 8 digits) must NOT trigger the EAN path,
        # or every 2-digit code would scan the whole table.
        assert _records(db, search="1234")["total"] == 0

    def test_ean_search_accepts_label_prefix(self):
        """Samuel 2026-10-28 retest: traders paste "EAN 3349668515714" with
        the label still attached. Before this fix "EAN" became an AND-filter
        text term that no row matched, so the search returned 0. Strip the
        leading EAN/GTIN/Barcode label before tokenising so any of these
        pasted formats find the row."""
        db = _session()
        _rec(db, uid="pr-labelled", brand="Paco Rabanne", product_name="Invictus",
             unit_volume_ml=100, ean_code="3349668515714",
             category_slug="perfumes", price_per_unit_eur=79)
        for q in ("EAN 3349668515714", "ean 3349668515714",
                  "EAN: 3349668515714", "GTIN 3349668515714",
                  "Barcode: 3349668515714", "3349668515714"):
            r = _records(db, search=q)
            assert r["total"] == 1 and r["records"][0]["uid"] == "pr-labelled", q


class TestBestPricesSameEanPeer:
    """REQ 11 — same valid EAN + terms = ONE peer group, both suppliers."""

    def test_same_ean_different_name_is_one_peer(self):
        db = _session()
        ean = "3614274085518"  # GS1-valid
        _rec(db, uid="balkan", brand="Mugler", product_name="Angel Fantasm Tester",
             unit_volume_ml=100, ean_code=ean, incoterm="EXW", location="Nl",
             category="Perfumes", category_slug="perfumes", perfume_format="EDP",
             retail_state="tester", gender="women", range_name="Mugler Angel Fantasm",
             price_per_unit_eur=43, supplier_name="Balkan Luxury Supply Ltd.")
        _rec(db, uid="loreal", brand="Mugler", product_name="Angel Fantasm",
             unit_volume_ml=100, ean_code=ean, incoterm="EXW", location="Nl",
             category="Perfumes", category_slug="perfumes", perfume_format="EDP",
             retail_state="tester", price_per_unit_eur=43, supplier_name="L'Oreal Luxury")
        groups = _bp(db)["groups"]
        assert len(groups) == 1
        g = groups[0]
        assert g["supplier_count"] == 2
        # The whole point: ONE peer group holding both suppliers, not two
        # one-supplier peers (the pre-fix behaviour).
        assert len(g["peers"]) == 1
        assert g["peers"][0]["supplier_count"] == 2


class TestBestPricesMultiIncotermSupplier:
    """REQ 07 — a current offer at a single-supplier incoterm is retained."""

    def test_single_incoterm_supplier_not_dropped(self):
        db = _session()
        common = dict(brand="Hennessy", product_name="VS", unit_volume_ml=1000,
                      units_per_case=12, alcohol_percent=40.0,
                      category="Wines & Spirits", category_slug="wines_spirits")
        # 2 suppliers EXW (historical) + 1 supplier DAP (current). Before the
        # fix the DAP supplier was dropped because DAP had only one supplier.
        _rec(db, uid="elixe", incoterm="EXW", location="Loendersloot",
             price_per_unit_eur=24.77, supplier_name="ElixeMarket",
             offer_date=datetime.utcnow() - timedelta(days=210), **common)
        _rec(db, uid="fbc", incoterm="EXW", location="Loendersloot",
             price_per_unit_eur=26.50, supplier_name="FBC Trades",
             offer_date=datetime.utcnow() - timedelta(days=210), **common)
        _rec(db, uid="urd", incoterm="DAP", location="Loendersloot",
             price_per_unit_eur=27.09, supplier_name="URD Exports",
             offer_date=datetime.utcnow() - timedelta(days=5), **common)
        groups = _bp(db)["groups"]
        assert len(groups) == 1
        peers = groups[0]["peers"]
        incoterms = {p["incoterm"] for p in peers}
        assert "DAP" in incoterms and "EXW" in incoterms
        dap = next(p for p in peers if p["incoterm"] == "DAP")
        assert dap["best_supplier"] == "URD Exports"

    def test_headline_marks_stale_history(self):
        db = _session()
        common = dict(brand="Hennessy", product_name="VS", unit_volume_ml=1000,
                      units_per_case=12, alcohol_percent=40.0, incoterm="EXW",
                      location="Loendersloot", category="Wines & Spirits",
                      category_slug="wines_spirits")
        _rec(db, uid="h1", price_per_unit_eur=24.77, supplier_name="ElixeMarket",
             offer_date=datetime.utcnow() - timedelta(days=210), **common)
        _rec(db, uid="h2", price_per_unit_eur=26.50, supplier_name="FBC Trades",
             offer_date=datetime.utcnow() - timedelta(days=210), **common)
        g = _bp(db)["groups"][0]
        # Headline comes from a 7-month-old offer → flagged stale/historical.
        assert g["headline_is_stale"] is True
        assert g["headline_age_days"] > 120


class TestBenchmarkExcludesReview:
    """P1 — a needs_manual_review row never sets the benchmark low."""

    def test_review_row_excluded_from_low(self):
        db = _session()
        common = dict(brand="Hennessy", product_name="VSOP", unit_volume_ml=1000,
                      units_per_case=12, incoterm="EXW", location="Loendersloot",
                      category_slug="wines_spirits",
                      offer_date=datetime.utcnow() - timedelta(days=10))
        _rec(db, uid="good", price_per_unit_eur=25.0, supplier_name="A",
             needs_manual_review=False, **common)
        _rec(db, uid="anomaly", price_per_unit_eur=3.33, supplier_name="ALCODIS",
             needs_manual_review=True, **common)
        bm = asyncio.run(get_benchmarks(window="12M", db=db))
        vals = next(iter(bm["peers"].values()))
        assert vals["low_eur"] == 25.0           # not 3.33
        assert vals["samples"] == 1              # the review row is excluded entirely


class TestOtherCategoryFilter:
    """REQ 04 — the 'Other' pill isolates offers with no trade category."""

    def test_none_slug_returns_only_uncategorised(self):
        db = _session()
        _rec(db, uid="ws", brand="X", product_name="A", category_slug="wines_spirits",
             price_per_unit_eur=10)
        _rec(db, uid="other1", brand="Y", product_name="B", category_slug=None,
             price_per_unit_eur=10)
        _rec(db, uid="other2", brand="Z", product_name="C", category_slug=None,
             price_per_unit_eur=10)
        assert _records(db, category_slug="__none__")["total"] == 2
        assert _records(db, category_slug="wines_spirits")["total"] == 1


class TestAbvSanityGuard:
    """P1 — impossible/implausible ABV is quarantined at ingest."""

    def test_abv_guard(self):
        from core.openai_client import clean_product_data
        impossible = clean_product_data({"product_name": "Elyx", "brand": "Absolut",
                                          "alcohol_percent": 423})
        assert impossible["alcohol_percent"] is None
        assert impossible["needs_manual_review"] is True
        implausible = clean_product_data({"product_name": "Rye", "brand": "Sazerac",
                                          "alcohol_percent": 90})
        assert implausible["alcohol_percent"] == 90       # kept (could be real overproof)
        assert implausible["needs_manual_review"] is True
        ok = clean_product_data({"product_name": "VS", "brand": "Hennessy",
                                 "alcohol_percent": 40})
        assert ok["alcohol_percent"] == 40
        assert ok["needs_manual_review"] is False


class TestCustomsStatusPeerSplit:
    """Samuel R-02/R-11 — a T1 (in-bond) and a T2 (duty-paid) offer of the SAME
    EAN are materially different cost bases and must NOT share a peer group,
    while two unspecified-status rows still peer together."""

    def _common(self):
        return dict(brand="Hennessy", product_name="VS", unit_volume_ml=700,
                    units_per_case=6, alcohol_percent=40.0, ean_code="5901041000956",
                    incoterm="EXW", location="Rotterdam", category="Wines & Spirits",
                    category_slug="wines_spirits")

    def test_t1_and_t2_same_ean_do_not_group(self):
        from core.normalization import peer_group_id_from_ean
        ean = "3614274085518"  # GS1-valid
        assert peer_group_id_from_ean(ean, "EXW", "Rotterdam", "T1") \
            != peer_group_id_from_ean(ean, "EXW", "Rotterdam", "T2")
        # unspecified folds to the same key regardless of None vs ""
        assert peer_group_id_from_ean(ean, "EXW", "Rotterdam", None) \
            == peer_group_id_from_ean(ean, "EXW", "Rotterdam", "")

    def test_best_prices_splits_t1_from_t2(self):
        db = _session()
        _rec(db, uid="t1", custom_status="T1", price_per_unit_eur=25.0,
             supplier_name="A", **self._common())
        _rec(db, uid="t2", custom_status="T2", price_per_unit_eur=26.0,
             supplier_name="B", **self._common())
        groups = _bp(db)["groups"]
        assert len(groups) == 1                      # one physical SKU card
        peers = groups[0]["peers"]
        assert len(peers) == 2                       # but T1 and T2 are separate peers
        assert all(p["supplier_count"] == 1 for p in peers)

    def test_same_status_still_groups(self):
        db = _session()
        _rec(db, uid="a", custom_status="T2", price_per_unit_eur=25.0,
             supplier_name="A", **self._common())
        _rec(db, uid="b", custom_status="T2", price_per_unit_eur=26.0,
             supplier_name="B", **self._common())
        peers = _bp(db)["groups"][0]["peers"]
        assert len(peers) == 1 and peers[0]["supplier_count"] == 2


class TestEanRepairDoesNotInventData:
    """Samuel case-school B15: an invalid 13-digit EAN whose first 12 digits
    happen to form a check-valid UPC-A must NOT be silently repaired to the
    shorter code (would invent a different product). Must be nulled + REVIEW."""

    def test_invalid_13digit_not_repaired_to_upca(self):
        from core.openai_client import clean_product_data, _repair_ean
        # Samuel's test code: 13 digits, bad check, first 12 happen to be a
        # valid UPC-A (290123450015). The old repair-trim would accept it.
        repaired, _ = _repair_ean("2901234500153")
        assert repaired is None, "invalid 13-digit EAN must not be shortened to a UPC-A"
        r = clean_product_data({"brand": "QA", "product_name": "X",
                                "ean_code": "2901234500153", "price_per_unit": 1.0,
                                "currency": "EUR"})
        assert r["ean_code"] is None
        assert r["needs_manual_review"] is True
        assert any("ean_code invalid" in f for f in r["error_flags"])


class TestZeroPriceRetained:
    """Samuel case-school B31: a stated €0.00 is a signal to retain + REVIEW,
    not drop silently. (The schema-field loop coerces 0→None, so the guard
    must detect zero on the RAW input dict and tag the row explicitly.)"""

    def test_zero_price_kept_and_flagged(self):
        from core.openai_client import clean_product_data
        r = clean_product_data({"brand": "QA Zero", "product_name": "QA Zero Price",
                                "unit_volume_ml": 700, "units_per_case": 6,
                                "price_per_unit": 0, "currency": "EUR"})
        assert r["price_per_unit"] == 0
        assert r["needs_manual_review"] is True
        assert any("Zero price retained" in f for f in r["error_flags"])

    def test_non_zero_price_unaffected(self):
        from core.openai_client import clean_product_data
        r = clean_product_data({"brand": "Hennessy", "product_name": "VS",
                                "unit_volume_ml": 1000, "units_per_case": 12,
                                "price_per_unit": 24.77, "currency": "EUR"})
        assert r["price_per_unit"] == 24.77
        assert r["needs_manual_review"] is False


class TestCrossCategoryEanConflict:
    """Samuel 2026-10-25: Paco Rabanne EAN 3349668515714 is classified as
    both Perfumes (W39) and Cosmetics (Cosmetics_Offer…). get_records now
    tags any row whose valid EAN appears in multiple category_slugs with a
    `cross_category_ean_conflict` list so the UI can surface it."""

    def test_cross_category_ean_flagged(self):
        db = _session()
        ean = "3349668515714"  # GS1-valid in prod
        _rec(db, uid="p1", brand="Paco Rabanne", product_name="Invictus ASL",
             unit_volume_ml=100, ean_code=ean, category="Perfumes",
             category_slug="perfumes", price_per_unit_eur=27.5,
             supplier_name="Balkan Luxury Supply Ltd.")
        _rec(db, uid="p2", brand="Paco Rabanne", product_name="Invictus (M) ASL",
             unit_volume_ml=100, ean_code=ean, category="Cosmetics",
             category_slug="cosmetics", price_per_unit_eur=28.9,
             supplier_name="Balkan Luxury Supply Ltd.")
        _rec(db, uid="other", brand="Hennessy", product_name="VS",
             unit_volume_ml=1000, ean_code="3614274085518",
             category="Wines & Spirits", category_slug="wines_spirits",
             price_per_unit_eur=24.77, supplier_name="FBC Trades")
        r = _records(db)
        by_uid = {x["uid"]: x for x in r["records"]}
        assert "cross_category_ean_conflict" in by_uid["p1"]
        assert sorted(by_uid["p1"]["cross_category_ean_conflict"]) == ["cosmetics", "perfumes"]
        assert "cross_category_ean_conflict" in by_uid["p2"]
        # unrelated row with its own EAN is NOT flagged
        assert "cross_category_ean_conflict" not in by_uid["other"]


class TestIngestCustomsBackfill:
    """Samuel 2026-10-26 case-school real-pipeline run: fixture rows B08,
    B11, B12 and B32 carry T1/T2 inside the raw description and the GPT-4o
    extraction left them in brand or product_name with custom_status blank.
    workers.processor._backfill_customs_from_text closes that gap at ingest
    time (same logic as api.admin.retest_backfill_customs_status). The
    token is extracted to custom_status, stripped from brand/product_name,
    and the pre-strip values are preserved in brand_raw / product_name_raw."""

    def test_token_in_product_name_extracted_at_ingest(self):
        from workers.processor import _backfill_customs_from_text
        safe_data = {
            "brand": "Hennessy",
            "product_name": "VS 0.7L T2 12x0.7L",
            "custom_status": None,
        }
        _backfill_customs_from_text(safe_data)
        assert safe_data["custom_status"] == "T2"
        assert "T2" not in safe_data["product_name"]
        assert safe_data["product_name_raw"] == "VS 0.7L T2 12x0.7L"
        assert safe_data.get("brand_raw") is None

    def test_token_in_brand_extracted_at_ingest(self):
        from workers.processor import _backfill_customs_from_text
        safe_data = {
            "brand": "Campari T2",
            "product_name": "Bitter 100cl",
            "custom_status": None,
        }
        _backfill_customs_from_text(safe_data)
        assert safe_data["custom_status"] == "T2"
        assert safe_data["brand"] == "Campari"
        assert safe_data["brand_raw"] == "Campari T2"

    def test_t1_recognised_separately_from_t2(self):
        from workers.processor import _backfill_customs_from_text
        safe_data = {
            "brand": "Moutai",
            "product_name": "Feitian 2025 500ml 53% 6x500ml T1 Caja",
            "custom_status": None,
        }
        _backfill_customs_from_text(safe_data)
        assert safe_data["custom_status"] == "T1"
        assert "T1" not in safe_data["product_name"]

    def test_already_populated_status_is_not_overwritten(self):
        from workers.processor import _backfill_customs_from_text
        safe_data = {
            "brand": "Bacardi",
            "product_name": "Carta Oro 1L T1",
            "custom_status": "T2",   # explicitly set elsewhere; must not change
        }
        _backfill_customs_from_text(safe_data)
        assert safe_data["custom_status"] == "T2"
        # Also must not strip the token in that case — the original value stays.
        assert safe_data["product_name"] == "Carta Oro 1L T1"

    def test_no_token_leaves_fields_untouched(self):
        from workers.processor import _backfill_customs_from_text
        safe_data = {
            "brand": "Belvedere",
            "product_name": "Pure 700ml",
            "custom_status": None,
        }
        _backfill_customs_from_text(safe_data)
        assert safe_data["custom_status"] is None
        assert safe_data["brand"] == "Belvedere"
        assert safe_data["product_name"] == "Pure 700ml"
        assert "brand_raw" not in safe_data or safe_data["brand_raw"] is None
        assert "product_name_raw" not in safe_data or safe_data["product_name_raw"] is None

    def test_not_a_t1t2_lookalike(self):
        """'T200' / 'T1000' / 'ST1' / '2T1' must NOT be mistaken for a token."""
        from workers.processor import _backfill_customs_from_text
        for name in ("Model T200", "T1000 Terminator", "Sabre ST1", "2T1 Mixer"):
            safe_data = {"brand": "", "product_name": name, "custom_status": None}
            _backfill_customs_from_text(safe_data)
            assert safe_data["custom_status"] is None, name
            assert safe_data["product_name"] == name, name

    def test_token_in_raw_row_text_only(self):
        """Case-school B01/B02/B04/B06/B08/B11/B12/B32: GPT-4o cleanly
        strips the T-token from product_name before returning, so neither
        brand nor product_name carries it anymore. The raw source-row text
        threaded through from core.openai_client.extract_from_file still
        has it — the fallback must pull it from there."""
        from workers.processor import _backfill_customs_from_text
        safe_data = {
            "brand": "Hennessy",
            "product_name": "VS",   # GPT-4o cleaned it out of here
            "custom_status": None,
            "_raw_row_text": "Supplier Reference=HEN-QA-1001 | Source Description (raw)=Hennessy VS 1L T2 12 x 1L",
        }
        _backfill_customs_from_text(safe_data)
        assert safe_data["custom_status"] == "T2"
        # The raw-row-text source does NOT get written back to any stored
        # field — brand and product_name stay as GPT-4o cleaned them.
        assert safe_data["brand"] == "Hennessy"
        assert safe_data["product_name"] == "VS"
        # No *_raw preservation when the token wasn't in brand/name.
        assert safe_data.get("brand_raw") is None
        assert safe_data.get("product_name_raw") is None

    def test_raw_row_text_not_consulted_when_already_set(self):
        """custom_status already populated by GPT-4o or an earlier pass
        must not be overwritten even when the raw row still carries a
        different token."""
        from workers.processor import _backfill_customs_from_text
        safe_data = {
            "brand": "Bacardi",
            "product_name": "Carta Oro",
            "custom_status": "T1",
            "_raw_row_text": "Source Description (raw)=Bacardi Carta Oro 1L T2 6 x 1L",
        }
        _backfill_customs_from_text(safe_data)
        assert safe_data["custom_status"] == "T1"
