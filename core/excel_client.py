"""
core/excel_client.py

Writes offer rows directly to a Microsoft 365 / OneDrive Excel workbook via
Microsoft Graph, using DELEGATED auth (same pattern Make uses):

- One-time interactive login via device-code flow (`python -m core.excel_client login`)
- MSAL refresh-token cache persisted in Redis under `MSAL_CACHE_KEY`
- Every write calls `acquire_token_silent()` which auto-refreshes the access token
- Graph is called as the signed-in user against their `/me/drive`

Works for both personal Microsoft accounts (spraicheux@gmail.com) and work/school.

Env vars required:
    MS_CLIENT_ID          — public client (application) ID from Entra app reg
    EXCEL_WORKBOOK_PATH   — e.g. "/Automation/output file.xlsx"
    EXCEL_WORKSHEET       — e.g. "Data"

Optional:
    MS_AUTHORITY          — default https://login.microsoftonline.com/common
    EXCEL_TABLE_NAME      — pin to a specific Excel Table; else auto-detect / auto-create
    MSAL_CACHE_KEY        — Redis key for the serialized token cache (default: msal:cache:onedrive)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import msal
import requests

from core.redis_client import redis_manager

logger = logging.getLogger(__name__)

GRAPH_ROOT = "https://graph.microsoft.com/v1.0"
SCOPES = ["Files.ReadWrite"]  # MSAL adds openid, profile, offline_access
DEFAULT_AUTHORITY = "https://login.microsoftonline.com/common"
DEFAULT_CACHE_KEY = "msal:cache:onedrive"


class ExcelClientError(RuntimeError):
    pass


# ─── Token cache backed by Redis ──────────────────────────────────────────────

def _redis():
    """Direct redis handle for cache reads/writes (bypasses RedisManager helpers)."""
    if redis_manager.use_redis and redis_manager.client:
        return redis_manager.client
    return None


def _load_cache() -> msal.SerializableTokenCache:
    cache = msal.SerializableTokenCache()
    key = os.getenv("MSAL_CACHE_KEY", DEFAULT_CACHE_KEY)
    r = _redis()
    if r:
        blob = r.get(key)
        if blob:
            cache.deserialize(blob)
    else:
        # Fallback: local file for dev without redis
        path = os.path.expanduser("~/.msal_cache.json")
        if os.path.exists(path):
            with open(path) as f:
                cache.deserialize(f.read())
    return cache


def _save_cache(cache: msal.SerializableTokenCache) -> None:
    if not cache.has_state_changed:
        return
    key = os.getenv("MSAL_CACHE_KEY", DEFAULT_CACHE_KEY)
    blob = cache.serialize()
    r = _redis()
    if r:
        r.set(key, blob)
    else:
        path = os.path.expanduser("~/.msal_cache.json")
        with open(path, "w") as f:
            f.write(blob)


# ─── Excel client ─────────────────────────────────────────────────────────────

class ExcelClient:
    def __init__(self) -> None:
        self.client_id = os.getenv("MS_CLIENT_ID")
        self.authority = os.getenv("MS_AUTHORITY", DEFAULT_AUTHORITY)
        self.workbook_path = os.getenv("EXCEL_WORKBOOK_PATH", "/Automation/output file.xlsx")
        self.worksheet = os.getenv("EXCEL_WORKSHEET", "Data")
        self.table_name = os.getenv("EXCEL_TABLE_NAME") or None

        self._columns: Optional[List[str]] = None
        self._resolved_table: Optional[str] = None
        self._lock = threading.Lock()
        self._session = requests.Session()

    # -- auth ------------------------------------------------------------------

    def _app(self, cache: msal.SerializableTokenCache) -> msal.PublicClientApplication:
        if not self.client_id:
            raise ExcelClientError("MS_CLIENT_ID is not set.")
        return msal.PublicClientApplication(
            self.client_id,
            authority=self.authority,
            token_cache=cache,
        )

    def get_access_token(self) -> str:
        with self._lock:
            cache = _load_cache()
            app = self._app(cache)
            accounts = app.get_accounts()
            if not accounts:
                raise ExcelClientError(
                    "No signed-in account. Run:  python -m core.excel_client login"
                )
            result = app.acquire_token_silent(SCOPES, account=accounts[0])
            _save_cache(cache)
            if not result or "access_token" not in result:
                raise ExcelClientError(
                    f"Silent token acquisition failed: {result}. "
                    "Re-run:  python -m core.excel_client login"
                )
            return result["access_token"]

    def login_device_flow(self) -> Dict[str, Any]:
        """Interactive one-time login. Prints URL + code, waits for user to sign in."""
        cache = _load_cache()
        app = self._app(cache)
        flow = app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            raise ExcelClientError(f"Device flow init failed: {flow}")
        print("\n" + "=" * 60)
        print(f"Open this URL in a browser: {flow['verification_uri']}")
        print(f"Enter this code:            {flow['user_code']}")
        print("Sign in with the Microsoft account that owns the Excel workbook.")
        print("=" * 60 + "\n")
        result = app.acquire_token_by_device_flow(flow)  # blocks
        _save_cache(cache)
        if "access_token" not in result:
            raise ExcelClientError(f"Login failed: {result}")
        return result

    # -- graph plumbing --------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.get_access_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _workbook_url(self) -> str:
        encoded_path = quote(self.workbook_path.lstrip("/"))
        return f"{GRAPH_ROOT}/me/drive/root:/{encoded_path}:/workbook"

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        r = self._session.request(method, url, headers=self._headers(), timeout=30, **kwargs)
        if not r.ok:
            raise ExcelClientError(f"Graph {method} {url} → {r.status_code} {r.text[:500]}")
        return r

    def _ensure_table(self) -> str:
        if self._resolved_table:
            return self._resolved_table
        wb = self._workbook_url()
        ws_seg = f"/worksheets('{quote(self.worksheet)}')"

        if self.table_name:
            self._resolved_table = self.table_name
            return self._resolved_table

        # Find first table on this worksheet
        r = self._request("GET", f"{wb}{ws_seg}/tables")
        tables = r.json().get("value", [])
        if tables:
            self._resolved_table = tables[0]["name"]
            logger.info(f"Excel: using existing table '{self._resolved_table}' on '{self.worksheet}'")
            return self._resolved_table

        # No table — create one from the used range so row 1 becomes the header
        used = self._request("GET", f"{wb}{ws_seg}/usedRange(valuesOnly=true)").json()
        address = used.get("address")
        if not address:
            raise ExcelClientError(
                f"Sheet '{self.worksheet}' appears empty. Add a header row before writing."
            )
        create = self._request(
            "POST",
            f"{wb}{ws_seg}/tables/add",
            json={"address": address, "hasHeaders": True},
        ).json()
        self._resolved_table = create["name"]
        logger.info(f"Excel: created table '{self._resolved_table}' from {address}")
        return self._resolved_table

    def _load_columns(self) -> List[str]:
        if self._columns:
            return self._columns
        table = self._ensure_table()
        wb = self._workbook_url()
        r = self._request("GET", f"{wb}/tables('{quote(table)}')/columns?$select=name")
        self._columns = [c["name"] for c in r.json().get("value", [])]
        if not self._columns:
            raise ExcelClientError(f"Table '{table}' has no columns.")
        logger.info(f"Excel: loaded {len(self._columns)} columns from '{table}'")
        return self._columns

    @staticmethod
    def _to_cell(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return ", ".join("" if v is None else str(v) for v in value)
        return str(value)

    def add_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        table = self._ensure_table()
        cols = self._load_columns()
        values = [[self._to_cell(row.get(c)) for c in cols] for row in rows]
        wb = self._workbook_url()
        self._request(
            "POST",
            f"{wb}/tables('{quote(table)}')/rows/add",
            json={"values": values},
        )
        logger.info(f"Excel: appended {len(values)} row(s) to '{table}'")

    def add_row(self, row: Dict[str, Any]) -> None:
        self.add_rows([row])


_client: Optional[ExcelClient] = None


def get_excel_client() -> ExcelClient:
    global _client
    if _client is None:
        _client = ExcelClient()
    return _client


def write_offer_row(offer_dict: Dict[str, Any]) -> bool:
    try:
        get_excel_client().add_row(offer_dict)
        return True
    except ExcelClientError as e:
        logger.error(f"Excel write failed: {e}")
        return False
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Excel write crashed: {e}")
        return False


# ─── CLI: `python -m core.excel_client login | whoami | test` ────────────────

def _cli() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    if len(sys.argv) < 2:
        print("Usage: python -m core.excel_client [login|whoami|test]")
        return 1
    cmd = sys.argv[1]
    c = ExcelClient()

    if cmd == "login":
        c.login_device_flow()
        print("✓ Login successful. Refresh token cached in Redis.")
        return 0

    if cmd == "whoami":
        tok = c.get_access_token()
        r = requests.get(
            f"{GRAPH_ROOT}/me",
            headers={"Authorization": f"Bearer {tok}"},
            timeout=15,
        )
        print(json.dumps(r.json(), indent=2))
        return 0 if r.ok else 2

    if cmd == "test":
        wb = c._workbook_url()
        r = requests.get(wb, headers=c._headers(), timeout=15)
        print(f"Workbook probe: {r.status_code}")
        print(r.text[:600])
        return 0 if r.ok else 2

    print(f"Unknown command: {cmd}")
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
