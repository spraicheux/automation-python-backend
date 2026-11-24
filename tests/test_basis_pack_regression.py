"""Regression tests for Samuel's 2026-11-24 round — the "green must be trustworthy"
contract. Every case uses DIFFERENT products, quantities and wording from the files
that exposed the bug, so the test locks the RULE, not a phrasing.

Each assertion FAILED on the pre-recon40 code and PASSES now. The old (wrong) value
is noted beside each check.

Invariants under test:
  1. A pack size or MOQ on its own is NEVER taken as the price basis (text path).
  2. An explicit basis word IS honoured ("/case", "/set") — per set is per unit,
     never multiplied into a fabricated case price.
  3. A gift set / grams pack never becomes a unit count ("3 × 40 g" ≠ 120).
  4. A lot CODE is never read as the quantity ("Lot BX88: 40 cs" → 40, not 88).
  5. "N pallets × M cases" is a real case count (N×M).
  6. GREEN-TRUST GUARDRAIL: a currency with no price, a sold-out line, a
     case≠unit×pack conflict, and a currency-awaiting note all go YELLOW, not green.
"""
import os
import tempfile

from openpyxl import Workbook

from core.xlsx_structured import (extract_structured_products, _parse_pack,
                                  _parse_quantity_cell)
from core.openai_client import clean_product_data
import core.offer_reconcile as rec


# ── 1 + 2: text-path basis (new wording, no "basis not stated" disclaimer) ──────
def test_bare_amount_is_basis_unconfirmed():
    # "900 cases Northwind Reserve 12/70/40 at 55 GBP" — no per-bottle/per-case word.
    # OLD: 55 assumed per case, GBP 55/case + GBP 4.58/btl, GREEN.
    src = "900 cases Northwind Reserve 12/70/40 at 55 GBP ex-Hull on the floor"
    o = {"brand": "Northwind", "product_name": "Reserve", "units_per_case": 12.0,
         "price_per_case": 55.0, "price_per_unit": 4.58, "currency": "GBP",
         "quantity_case": 900.0, "uid": "t1"}
    rec.flag_absent_price_basis([o], src)
    assert o["price_per_case"] is None and o["price_per_unit"] is None
    assert o["needs_manual_review"] is True
    assert "as quoted: GBP 55" in (o["commercial_conditions"] or "")


def test_explicit_per_case_stays_green():
    # "EUR 10,80 per case" — basis IS stated → keep the price, no flag.
    src = "Southpaw Pilsner 24 x 33cl, 200 cases, EUR 10,80 per case, EXW NL"
    o = {"brand": "Southpaw", "product_name": "Pilsner", "units_per_case": 24.0,
         "price_per_case": 10.8, "price_per_unit": 0.45, "currency": "EUR",
         "quantity_case": 200.0, "uid": "t2"}
    n = rec.flag_absent_price_basis([o], src)
    assert n == 0 and o["price_per_case"] == 10.8 and not o.get("needs_manual_review")


# ── 3: gift set / grams never become a unit count ───────────────────────────────
def test_parse_pack_gift_set_and_grams():
    assert _parse_pack("Gift set of 2 × 50 g") is None      # OLD: 100 (2×50)
    assert _parse_pack("18 × 30 g") == 18.0                 # grams stripped, count kept
    assert _parse_pack("24 × 500 ml cans") == 24.0


# ── 4 + 5: quantity cell — lot code and pallet math ─────────────────────────────
def test_quantity_cell_lot_code_and_pallets():
    assert _parse_quantity_cell("Lot BX88: 40 cs + 3 loose")[0] == 40.0   # OLD: 88
    assert _parse_quantity_cell("6 pallets × 50 cases")[0] == 300.0       # OLD: 6
    assert _parse_quantity_cell("Total 100 cases on floor")[0] == 100.0


# ── 6: end-to-end guardrail on a freeform workbook with NEW products ────────────
def _freeform_wb(path):
    wb = Workbook(); ws = wb.active; ws.title = "Desk"
    ws.append(["NORTHWIND TRADING BV", "Circular NT-99", "2026-11-24"])
    ws.append(["Ref. / lot", "Description", "Packing", "Warehouse position",
               "Quotation", "Delivery", "Desk notes"])
    ws.append(["T01", "Aurora salt candle", "Gift set of 2 × 50 g", "25 sets",
               "EUR 30 / set", "FCA Ghent", "Sold as complete sets"])
    ws.append(["T02", "Borealis Lager", "24 × 500 ml cans", "Lot BX88: 40 cs + 3 loose",
               "EUR 12 / case", "EXW NL", "T2"])
    ws.append(["T03", "Cedar Tonic", "12 × 250 ml glass", "6 pallets × 50 cases",
               "EUR 9 / cs", "EXW Breda", "Net goods"])
    ws.append(["T04", "Delta Gin", "6 × 700 ml", "20 cases", "0.05 / can",
               "EXW NL", "Settlement currency awaiting confirmation"])
    ws.append(["T05", "Echo Rum", "6 × 1 L", "Sold out", "GBP 5 / btl",
               "EXW Hull", "Closed parcel; retained for reference"])
    ws.append(["T06", "Foxtrot Vermouth", "6 × 1 L", "50 cases",
               "EUR 30 / case; EUR 8 / btl", "EXW Breda", "Both figures received; desk checking"])
    wb.save(path)


def _by_ref(path):
    rows = extract_structured_products(path) or []
    out = {}
    for r in rows:
        out[r.get("product_reference")] = clean_product_data(dict(r))
    return out


def test_guardrail_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "northwind.xlsx")
        _freeform_wb(p)
        rows = _by_ref(p)

    # T01 gift set — per set, NO fabricated case price. OLD: EUR 3,000/case (30×100).
    assert rows["T01"]["price_per_unit"] == 30.0
    assert rows["T01"]["price_per_case"] is None

    # T02 lot code not read as qty, /case honoured. OLD: qc 88.
    assert rows["T02"]["quantity_case"] == 40.0
    assert rows["T02"]["price_per_case"] == 12.0

    # T03 pallet math + "/cs" per case. OLD: 6 pallets, price per unit.
    assert rows["T03"]["quantity_case"] == 300.0
    assert rows["T03"]["price_per_case"] == 9.0

    # T04 currency awaiting → unconfirmed + flagged (never EUR). OLD: EUR 1.20/case, green.
    assert rows["T04"]["currency"] in (None, "Not Found")
    assert rows["T04"]["needs_manual_review"] is True

    # T05 sold out → no stock, flagged. OLD: green GBP 5 offer with stock.
    assert not any(rows["T05"].get(k) for k in
                   ("quantity_case", "quantity_bottles", "quantity_pallets"))
    assert rows["T05"]["needs_manual_review"] is True

    # T06 conflicting prices (30 ≠ 8×6) → flagged. OLD: green.
    assert rows["T06"]["needs_manual_review"] is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} regression tests passed")
