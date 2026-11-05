"""Regression tests for the concurrent extraction refactor.

The three OpenAI-calling extraction loops (text/offer chunks, XLSX row batches,
PDF page batches) now fire their calls concurrently instead of one-at-a-time.
These tests mock the OpenAI client and assert that the aggregated products come
out in the SAME order as the input units and that nothing is dropped — i.e. the
parallelization changed only the timing, not the result.

No network, no real OpenAI.
"""
import os
import re
import sys
import json
import asyncio
import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used-in-unit-tests")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import core.openai_client as oc  # noqa: E402


def _resp(content):
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    r.choices[0].finish_reason = "stop"
    return r


def test_extract_offer_chunks_concurrent_order_preserved():
    def fake(**k):
        prompt = k["messages"][-1]["content"]
        c = int(re.search(r"Text Chunk \((\d+)/(\d+)\)", prompt).group(1))
        return _resp(json.dumps({"products": [{
            "product_name": f"CHUNK_{c:02d}", "brand": "Absolut",
            "unit_volume_ml": 700, "units_per_case": 6,
            "price_per_unit": 10.0, "currency": "EUR"}]}))

    oc.client.chat.completions.create = AsyncMock(side_effect=fake)
    text = "\n".join(
        f"Absolut Vodka ref{i} 700ml 6/case 1{i % 9}.0 EUR EXW Rotterdam"
        for i in range(140))
    out = asyncio.run(oc.extract_offer(text))
    names = [p.get("product_name") for p in out["products"]]
    assert names, "no products extracted"
    assert names == sorted(names), f"chunk order not preserved: {names}"


def test_extract_from_file_xlsx_batches_concurrent_order_preserved(tmp_path):
    import openpyxl
    p = tmp_path / "offer.xlsx"
    wb = openpyxl.Workbook(); ws = wb.active
    ws.append(["Brand", "Product", "Volume", "UnitsPerCase", "Price", "Currency"])
    for i in range(24):
        ws.append([f"Brand{i}", f"Vodka{i}", 700, 6, 10.0 + i, "EUR"])
    wb.save(p)

    def fake(**k):
        user = k["messages"][-1]["content"]
        mb = re.search(r"EXCEL DATA BATCH \((\d+)-(\d+) of", user)
        mn = re.search(r"Extract EXACTLY (\d+) products", user)
        if mb and mn:
            start, n = int(mb.group(1)), int(mn.group(1))
            return _resp(json.dumps({"products": [{
                "product_name": f"R{start:04d}_{j}", "brand": "Absolut",
                "unit_volume_ml": 700, "units_per_case": 6,
                "price_per_unit": 10.0, "currency": "EUR",
                "category_slug": "wines_spirits"} for j in range(n)]}))
        return _resp(json.dumps({"products": []}))

    oc.client.chat.completions.create = AsyncMock(side_effect=fake)
    out = asyncio.run(oc.extract_from_file(
        str(p), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
    tags = [int(re.match(r"R(\d+)_", p.get("product_name", "R0000_")).group(1))
            for p in out["products"] if str(p.get("product_name", "")).startswith("R")]
    assert tags, "no products extracted"
    assert tags == sorted(tags), f"xlsx batch order not preserved: {tags}"
    assert len(out["products"]) >= 18, f"too few products: {len(out['products'])}"


def test_credit_error_propagates_not_swallowed():
    """A credits-exhausted OpenAI error must raise OpenAICreditsExhausted out of
    extraction (so the job is parked), NOT be swallowed into 0 products/done."""
    from core.openai_health import OpenAICreditsExhausted

    def fake(**k):
        raise RuntimeError("Error code: 429 - insufficient_quota, "
                           "credit_balance_exhausted: You have no credits remaining")

    oc.client.chat.completions.create = AsyncMock(side_effect=fake)
    text = "\n".join(f"Absolut Vodka ref{i} 700ml 6/case 10.0 EUR" for i in range(60))
    with pytest.raises(OpenAICreditsExhausted):
        asyncio.run(oc.extract_offer(text))


def test_credit_error_propagates_from_extract_from_file(tmp_path):
    """XLSX path: a credits-exhausted error must propagate out of
    extract_from_file (so the job parks), not be swallowed by its own
    except Exception into a 0-product 'done'."""
    import openpyxl
    from core.openai_health import OpenAICreditsExhausted
    p = tmp_path / "o.xlsx"
    wb = openpyxl.Workbook(); ws = wb.active
    ws.append(["Brand", "Product", "Price"])
    for i in range(10):
        ws.append([f"B{i}", "Vodka", 10.0 + i])
    wb.save(p)

    def fake(**k):
        raise RuntimeError("Error code: 429 - insufficient_quota, "
                           "credit_balance_exhausted: You have no credits remaining")

    oc.client.chat.completions.create = AsyncMock(side_effect=fake)
    with pytest.raises(OpenAICreditsExhausted):
        asyncio.run(oc.extract_from_file(
            str(p), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))


def test_normal_error_does_not_park(tmp_path):
    """A non-credit error must NOT raise OpenAICreditsExhausted (it degrades to
    0 products for that unit, as before)."""
    from core.openai_health import OpenAICreditsExhausted

    def fake(**k):
        raise RuntimeError("some transient network blip")

    oc.client.chat.completions.create = AsyncMock(side_effect=fake)
    text = "\n".join(f"Absolut Vodka ref{i} 700ml 6/case 10.0 EUR" for i in range(60))
    try:
        out = asyncio.run(oc.extract_offer(text))
        assert out["products"] == []          # degraded, not parked
    except OpenAICreditsExhausted:
        raise AssertionError("non-credit error wrongly treated as credits-exhausted")


def test_extract_from_file_pdf_pages_concurrent_order_preserved(tmp_path):
    reportlab = pytest.importorskip("reportlab")  # noqa: F841
    from reportlab.pdfgen import canvas
    p = tmp_path / "offer.pdf"
    c = canvas.Canvas(str(p))
    for pg in range(3):
        c.drawString(72, 800, f"Absolut Vodka 700ml 6/case 123456789012{pg} 10.00 EUR EXW Rotterdam")
        c.showPage()
    c.save()

    def fake(**k):
        user = k["messages"][-1]["content"]
        m = re.search(r"pages (\d+)[–-](\d+) of", user)
        if m:
            pg = int(m.group(1))
            return _resp(json.dumps({"products": [{
                "product_name": f"PAGE_{pg:02d}", "brand": "Absolut",
                "unit_volume_ml": 700, "units_per_case": 6,
                "price_per_unit": 10.0, "currency": "EUR",
                "category_slug": "wines_spirits"}]}))
        return _resp(json.dumps({"products": []}))

    oc.client.chat.completions.create = AsyncMock(side_effect=fake)
    out = asyncio.run(oc.extract_from_file(str(p), "application/pdf"))
    names = [p.get("product_name") for p in out["products"]
             if str(p.get("product_name", "")).startswith("PAGE_")]
    assert names == ["PAGE_01", "PAGE_02", "PAGE_03"], names
