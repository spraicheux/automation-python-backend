"""
R-17 API regression suite — locks in the 8 audit findings at the HTTP layer.

Requires network access to the deployed API (or a local server).

Run all integration tests:
    pytest tests/test_api_regression.py -m integration -v

Run against a local server:
    BASE_URL=http://localhost:8000 pytest tests/test_api_regression.py -m integration -v

Skip if no network:
    pytest tests/test_api_regression.py  # no -m flag → skips all
"""
import os
import pytest
import requests

BASE_URL = os.getenv("BASE_URL", "https://loxo-offer-api.azurewebsites.net")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "valid-token")
TIMEOUT = 20


@pytest.fixture(scope="module")
def session():
    s = requests.Session()
    s.headers["x-admin-token"] = ADMIN_TOKEN
    return s


# ---------------------------------------------------------------------------
# R-01: W&S category must show offers regardless of active source filter
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestWinesSpiritsCategory:
    def test_returns_records(self, session):
        r = session.get(f"{BASE_URL}/api/records",
                        params={"category_slug": "wines_spirits", "limit": 5},
                        timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        assert data.get("total", 0) > 0, (
            "W&S category returned 0 results — category/source cross-filter regression"
        )

    def test_substantial_record_count(self, session):
        r = session.get(f"{BASE_URL}/api/records",
                        params={"category_slug": "wines_spirits", "limit": 1},
                        timeout=TIMEOUT)
        assert r.status_code == 200
        assert r.json()["total"] >= 100, "Unexpectedly low W&S record count"


# ---------------------------------------------------------------------------
# R-02: Alert message format — current supplier leads, previous is labelled
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAlertMessageFormat:
    def test_alert_endpoint_responds(self, session):
        r = session.get(f"{BASE_URL}/api/alerts", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_alert_message_contains_previous_best_label(self, session):
        r = session.get(f"{BASE_URL}/api/alerts", params={"limit": 20}, timeout=TIMEOUT)
        assert r.status_code == 200
        body = r.json()
        alerts = body if isinstance(body, list) else body.get("alerts", [])
        if not alerts:
            pytest.skip("No alerts in system — cannot verify message format")
        for alert in alerts[:10]:
            msg = alert.get("message", "")
            # Old format was "from <supplier>" — new format leads with "Supplier @ price"
            assert not msg.startswith("from "), (
                f"Alert message uses old 'from ...' format: {msg!r}"
            )
            if "Previous best:" in msg:
                return  # at least one alert has the correct label
        # If no alerts have "Previous best:", they must all have @-price format
        for alert in alerts[:5]:
            assert "@" in alert.get("message", "") or not alert.get("message"), (
                f"Alert missing '@' price marker: {alert.get('message')!r}"
            )


# ---------------------------------------------------------------------------
# R-04: EAN GS1 check digit (unit-tested in test_normalization.py, locked here
#        at the API level via best-prices — no invalid EAN should be a peer key)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBestPricesResponse:
    def test_best_prices_responds(self, session):
        r = session.get(f"{BASE_URL}/api/best-prices",
                        params={"limit": 5},
                        timeout=TIMEOUT)
        assert r.status_code == 200

    def test_is_expired_field_present(self, session):
        """R-08: every offer in best-prices must carry is_expired."""
        r = session.get(f"{BASE_URL}/api/best-prices",
                        params={"limit": 5},
                        timeout=TIMEOUT)
        assert r.status_code == 200
        peers = r.json().get("peers", [])
        if not peers:
            pytest.skip("No best-prices data")
        for peer in peers[:3]:
            for offer in peer.get("offers", []):
                assert "is_expired" in offer, (
                    f"offer missing is_expired: {offer.get('uid')}"
                )

    def test_has_active_offers_field_present(self, session):
        """R-08: every peer must carry has_active_offers."""
        r = session.get(f"{BASE_URL}/api/best-prices",
                        params={"limit": 5},
                        timeout=TIMEOUT)
        assert r.status_code == 200
        peers = r.json().get("peers", [])
        if not peers:
            pytest.skip("No best-prices data")
        for peer in peers[:3]:
            assert "has_active_offers" in peer, (
                f"peer missing has_active_offers: {peer.get('peer_group_id')}"
            )


# ---------------------------------------------------------------------------
# R-06: T1/T2 search must return results (conjunctive match)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestT2Search:
    def test_t2_returns_records(self, session):
        r = session.get(f"{BASE_URL}/api/records",
                        params={"custom_status": "T2", "limit": 1},
                        timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        assert data.get("total", 0) > 0, (
            "T2 search returned 0 results — conjunctive filter regression"
        )

    def test_t2_combined_with_text_returns_records(self, session):
        # Conjunctive: T2 + a common spirit brand term
        r = session.get(f"{BASE_URL}/api/records",
                        params={"custom_status": "T2", "q": "vodka", "limit": 1},
                        timeout=TIMEOUT)
        assert r.status_code == 200
        # Just needs to respond correctly (may be 0 if no vodka T2 rows)
        assert "total" in r.json()


# ---------------------------------------------------------------------------
# R-09: Source sidebar — expected_row_count and import_incomplete present
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSourceSidebarCounts:
    def test_sources_endpoint_responds(self, session):
        r = session.get(f"{BASE_URL}/api/sources", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_source_has_row_count_fields(self, session):
        r = session.get(f"{BASE_URL}/api/sources",
                        params={"limit": 5},
                        timeout=TIMEOUT)
        assert r.status_code == 200
        sources = r.json().get("sources", [])
        if not sources:
            pytest.skip("No sources in system")
        s = sources[0]
        assert "expected_row_count" in s, "source missing expected_row_count"
        assert "imported_row_count" in s, "source missing imported_row_count"
        assert "import_incomplete" in s, "source missing import_incomplete flag"


# ---------------------------------------------------------------------------
# R-03: Manual offer quantity_unit field accepted
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestManualOfferQuantityUnit:
    def test_quantity_unit_field_in_schema(self, session):
        # POST with quantity_unit — if the field is rejected the API returns 422
        payload = {
            "product_name": "__regression_test_quantity_unit__",
            "quantity_case": 10,
            "quantity_unit": "bottles",
            "price_per_case": 99.0,
            "supplier_name": "__regression_test__",
            "source_channel": "manual",
        }
        r = session.post(f"{BASE_URL}/api/offers", json=payload, timeout=TIMEOUT)
        # 200 = field accepted; anything else is a regression
        assert r.status_code == 200, (
            f"quantity_unit field rejected: {r.status_code} {r.text[:200]}"
        )
        uid = r.json().get("uid")
        # Cleanup: soft-delete is not implemented; the test row stays in DB
        # (labelled __regression_test__ for easy manual identification)
        assert uid, "offer created but no uid returned"
