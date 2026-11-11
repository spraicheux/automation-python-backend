"""core/fx_rates.py

Live EUR foreign-exchange rates from the ECB (via the Frankfurter API), cached
per day in Redis.

Why (Samuel 2026-11-11, Jim Beam Apple): the pipeline stamped every non-EUR
offer with a hardcoded static rate — USD = 0.92, no provider, no effective date
(fx_source read "fallback-table-static-USD-0.92"). A buyer could not tell what
rate was applied, when it was effective, or whether it was a real market rate.
Since the converted EUR price feeds Best Prices, benchmarks and alerts, a fixed
rate silently biases all of them.

This module fetches the ECB daily reference rate for the offer's date (or the
latest published), records the rate + its ECB effective date + the provider, and
falls back to an approximate static rate ONLY when the live lookup fails — always
flagged so the EUR figure is never presented as exact.
"""
from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Approximate fallbacks — used ONLY when the live ECB lookup fails. Every use is
# flagged (see fx_source "fallback-static-approx-*" and the row error_flag).
_STATIC = {"USD": 0.92, "GBP": 1.17, "CHF": 1.05, "EUR": 1.0}

_PROVIDER = "ECB (Frankfurter)"
_BASE = "https://api.frankfurter.dev/v1"
_TIMEOUT = 4.0
_CACHE_TTL = 36 * 3600  # a daily rate is stable; refetch next day


def _cache_get(key: str):
    try:
        from core.redis_client import redis_manager
        if redis_manager.use_redis and redis_manager.client:
            v = redis_manager.client.get(key)
            if v:
                return json.loads(v)
    except Exception:
        return None
    return None


def _cache_set(key: str, val: dict, ttl: int = _CACHE_TTL) -> None:
    try:
        from core.redis_client import redis_manager
        if redis_manager.use_redis and redis_manager.client:
            redis_manager.client.set(key, json.dumps(val), ex=ttl)
    except Exception:
        pass


def get_eur_rate(currency: Optional[str], on_date=None) -> Tuple[float, Optional[str], str, bool]:
    """Return (rate_to_EUR, effective_date_iso|None, source_label, is_fallback).

    rate_to_EUR is 'how many EUR for 1 unit of `currency`'. `on_date` may be a
    datetime/date or an ISO string; the ECB reference rate for that date is used
    (Frankfurter returns the nearest prior business day when markets were shut)."""
    cur = (currency or "EUR").upper()
    if cur == "EUR":
        return (1.0, None, "identity-EUR", False)

    datekey = None
    if on_date:
        try:
            datekey = on_date.date().isoformat() if hasattr(on_date, "date") else str(on_date)[:10]
        except Exception:
            datekey = None

    ckey = f"fx:EUR:{cur}:{datekey or 'latest'}"
    cached = _cache_get(ckey)
    if cached:
        return (cached["rate"], cached.get("date"), cached.get("source"),
                bool(cached.get("fallback", False)))

    try:
        import httpx
        # Use the always-available 'latest' endpoint (the current ECB reference
        # rate, with its real effective date). A date-specific query is avoided:
        # the ECB only publishes up to the latest business day, so asking for a
        # same-day/future date can 404 and needlessly trip the fallback.
        url = f"{_BASE}/latest?from={cur}&to=EUR"
        resp = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        j = resp.json()
        rate = float(j["rates"]["EUR"])
        eff = j.get("date")  # ECB reference date actually used
        out = {"rate": round(rate, 6), "date": eff, "source": _PROVIDER, "fallback": False}
        _cache_set(ckey, out)
        return (out["rate"], eff, _PROVIDER, False)
    except Exception as e:
        logger.warning(f"fx live lookup failed for {cur} ({e}); using static approx")
        rate = _STATIC.get(cur, 1.0)
        out = {"rate": rate, "date": None,
               "source": f"fallback-static-approx-{cur}", "fallback": True}
        # Cache the failure briefly so a flaky endpoint isn't hit on every row.
        _cache_set(ckey, out, ttl=3600)
        return (rate, None, out["source"], True)
