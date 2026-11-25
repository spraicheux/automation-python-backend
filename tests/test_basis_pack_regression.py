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
                                  _parse_quantity_cell, _header_date,
                                  _scan_header_offer_date, scan_workbook_offer_date)
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
    # T07: stock unit NOT labelled — price unit (piece) is clear, stock unit is not.
    ws.append(["T07", "Helix Body Mist", "100 ml spray", "Balance: 200",
               "USD 7 / piece", "EXW Utrecht", "Warehouse did not label quantity unit"])
    # T08: bare price cell, but the same-row note states currency AND per-case basis.
    ws.append(["T08", "Icarus Soda", "12 × 250 ml cans", "90 cases",
               "6.00", "EXW NL", "Net rate EUR 0.50 per can × 12 cans; cell is in EUR/case"])
    # T09: multi-pack-of-packs — price is per 6-pack, must be labelled.
    ws.append(["T09", "Juno Lager", "4 wrapped sixes × 330 ml", "30 cases",
               "EUR 8 / case", "EXW NL", "Wrapped format"])
    wb.save(path)


def _by_ref(path):
    rows = extract_structured_products(path) or []
    out = {}
    for r in rows:
        c = clean_product_data(dict(r))
        key = (c.get("product_reference") or r.get("supplier_reference")
               or c.get("supplier_reference") or c.get("product_name"))
        out[key] = c
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

    # T07 stock unit not labelled → price per piece preserved, stock flagged, NOT
    # assumed "cases". OLD: 200 cases, green, eligible (Alder S03 class).
    assert rows["T07"]["price_per_unit"] == 7.0
    assert rows["T07"]["needs_manual_review"] is True
    assert "unit" in " ".join(rows["T07"].get("error_flags") or []).lower()

    # T08 same-row note gives currency + per-case basis → EUR 6/case, green.
    # OLD: currency & basis "missing", flagged (Alder W03 class).
    assert rows["T08"]["price_per_case"] == 6.0
    assert rows["T08"]["currency"] == "EUR"
    assert not rows["T08"].get("needs_manual_review")

    # T09 "4 wrapped sixes" = 24 cans → the per-unit price is per CAN (not per
    # six-pack), and the six-pack price is noted. OLD: units_per_case 4, €2/unit.
    assert rows["T09"]["units_per_case"] == 24
    assert round(rows["T09"]["price_per_unit"], 2) == 0.33
    assert "6-pack" in (rows["T09"].get("commercial_conditions") or "")


def _grey_wb(path, price_header, data_rows):
    # A "dock list" whose columns are named differently (Samuel 2026-11-24, Grey
    # Inlet): Supplier ref / Article / Presentation / Balance / <price> / Delivery
    # / remarks. Must route to the structured path (where the guardrails live).
    wb = Workbook(); ws = wb.active; ws.title = "Dock"
    ws.append(["NORTHWIND MERCHANTS BV", "NM-1", "2026-11-24"])
    ws.append(["Supplier ref", "Article", "Presentation", "Balance",
               price_header, "Delivery", "Desk / warehouse remarks"])
    for row in data_rows:
        ws.append(row)
    wb.save(path)


def _by_name(path):
    out = {}
    for r in (extract_structured_products(path) or []):
        c = clean_product_data(dict(r))
        full = f"{c.get('brand') or ''} {c.get('product_name') or ''}".strip()
        out[full] = c
        out[(c.get("product_name") or "").strip()] = c
    return out


def test_grey_balance_and_net_eur_heading():
    # 'Balance' → quantity; 'Net EUR per case' heading states the currency.
    # OLD: unrecognised columns → LLM path → quantity & currency lost, green.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.xlsx")
        _grey_wb(p, "Net EUR per case", [
            ["G01", "Harbor Ale", "24 × 330 ml cans", "40 cases", "10.00", "EXW NL", "On floor"],
            ["G02", "Quay Soda", "24 × 250 ml cans", "3 pallets of 50 cases", "8.40", "FCA Hull", "Min one pallet"],
        ])
        rows = _by_name(p)
    assert rows["Harbor Ale"]["quantity_case"] == 40.0
    assert rows["Harbor Ale"]["price_per_case"] == 10.0
    assert rows["Harbor Ale"]["currency"] == "EUR"
    assert rows["Quay Soda"]["quantity_case"] == 150.0   # 3 pallets × 50


def test_grey_ambiguous_dollar_xlsx():
    # '$20 … Dollar denomination to be confirmed: Canada or US' → currency
    # unconfirmed, NO EUR, flagged. OLD: assumed USD, FX-converted, green.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "b.xlsx")
        _grey_wb(p, "Quotation", [
            ["G03", "Nova Scent EDP", "50 ml", "12 pieces", "$20 each", "EXW Hull",
             "Dollar denomination to be confirmed: Canada or US"],
        ])
        rows = _by_name(p)
    assert rows["Nova Scent EDP"]["currency"] in (None, "Not Found")
    assert rows["Nova Scent EDP"]["needs_manual_review"] is True
    assert not rows["Nova Scent EDP"].get("price_per_unit_eur")


def test_header_date_helper():
    # Real date forms a circular's title row uses — all recognised.
    assert _header_date("2026-11-24") == "2026-11-24"
    assert _header_date("24 November 2026") == "2026-11-24"
    assert _header_date("November 24, 2026") == "2026-11-24"
    assert _header_date("24 novembre 2026") == "2026-11-24"
    assert _header_date("24/11/2026") == "2026-11-24"      # European day/month
    import datetime as _dt
    assert _header_date(_dt.datetime(2026, 11, 24, 9, 30)) == "2026-11-24"
    assert _header_date(_dt.date(2026, 11, 24)) == "2026-11-24"
    # NOT dates — a pack spec, a lone year, a bare number, a plain count.
    assert _header_date("12/70/40") is None               # pack spec, not a date
    assert _header_date("2026") is None                    # lone year
    assert _header_date("Circular NM-1") is None
    assert _header_date(40) is None                        # a quantity, not a date


def test_header_offer_date_from_dated_header():
    # A dated title row, NO per-row offer_date column → every row inherits the
    # circular's date, read deterministically. OLD: offer date "unknown" because
    # the structured parser only looked at per-row columns (Samuel Arden Crest).
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dated.xlsx")
        _grey_wb(p, "Net EUR per case", [
            ["H01", "Pier Lager", "24 × 330 ml cans", "40 cases", "10.00", "EXW NL", "On floor"],
        ])
        rows = _by_name(p)
    # _grey_wb writes a title row "NORTHWIND MERCHANTS BV | NM-1 | 2026-11-24".
    assert rows["Pier Lager"]["offer_date"] == "2026-11-24"
    assert rows["Pier Lager"].get("_offer_date_trusted") is True


def test_header_valid_until_is_not_offer_date():
    # A header that carries ONLY a "Valid until" date must not be mistaken for
    # the offer date — an expiry is not when the offer was made.
    from openpyxl import Workbook
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "vu.xlsx")
        wb = Workbook(); ws = wb.active; ws.title = "Dock"
        ws.append(["NORTHWIND MERCHANTS BV", "Valid until", "2026-12-31"])
        ws.append(["Supplier ref", "Article", "Presentation", "Balance",
                   "Net EUR per case", "Delivery", "Desk / warehouse remarks"])
        ws.append(["V01", "Cove Ale", "24 × 330 ml cans", "40 cases", "10.00",
                   "EXW NL", "On floor"])
        wb.save(p)
        rows = _by_name(p)
    # The only header date is a validity date → offer date stays unknown, NOT 12-31.
    assert rows["Cove Ale"].get("offer_date") in (None, "Not Found")


def test_workbook_offer_date_for_fallback_path():
    # The LLM-fallback path (unfamiliar headers) must still read the circular's
    # dated title row — parity with the structured reader (Samuel 2026-11-25).
    from openpyxl import Workbook
    with tempfile.TemporaryDirectory() as d:
        # A free title row with the date in words, then unfamiliar headers + data.
        p = os.path.join(d, "fallback.xlsx")
        wb = Workbook(); ws = wb.active; ws.title = "Note"
        ws.append(["Harbor & Vale Traders — stock note — 25 November 2026"])
        ws.append(["Line", "What's in it", "How it's packed", "On hand", "Figure"])
        ws.append(["1", "Totton Pale Ale", "24 x 330ml", "60 cases", "EUR 11.00 per case"])
        wb.save(p)
        # Unfamiliar headers → structured reader declines (so the LLM path is used)…
        assert extract_structured_products(p) is None
        # …but the workbook scanner still reads the dated title row.
        assert scan_workbook_offer_date(p) == "2026-11-25"

    # A title row carrying ONLY a validity date must not be read as the offer date.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "vu.xlsx")
        wb = Workbook(); ws = wb.active; ws.title = "Note"
        ws.append(["Harbor & Vale Traders", "Valid until", "2026-12-31"])
        ws.append(["Line", "What's in it", "How it's packed", "On hand", "Figure"])
        ws.append(["1", "Totton Pale Ale", "24 x 330ml", "60 cases", "EUR 11.00 per case"])
        wb.save(p)
        assert scan_workbook_offer_date(p) is None


def test_review_gate_is_reader_agnostic():
    # The enforcement gate runs on STORED ROWS + source text, so it flags a bad
    # row no matter which reader produced it (structured, LLM-fallback, PDF,
    # email). This is the fix for Samuel 2026-11-25: the validation must apply to
    # every row before it can be shown confirmed or enter Best Prices.
    from core.field_check import rows_to_flag

    # (a) Ambiguous currency — the Arden class, as the LLM fallback would store it
    # (it guessed USD and FX-converted). Source says the denomination is unsettled
    # → flag AND neutralise: drop the EUR, keep the amount as quoted.
    src = ("Arden Crest EDP perfume, 12 pieces, $24 each. "
           "Dollar denomination to be confirmed: Canada or US.")
    row = {"brand": "Arden", "product_name": "Crest EDP", "currency": "USD",
           "price_per_unit": 24.0, "price_per_unit_eur": 21.94, "quantity_bottles": 12.0}
    f = rows_to_flag([row], src)
    assert 0 in f and f[0]["neutralize_currency"] is True

    # (b) A clearly-supported EUR row (source states "per case" + EUR) is NOT
    # flagged — valid information is retained, zero false positive.
    src_ok = "Pier Lager 24 x 330 ml cans, 40 cases, EUR 10.00 per case, EXW NL"
    ok = {"brand": "Pier", "product_name": "Lager", "currency": "EUR",
          "price_per_case": 10.0, "price_per_unit": 0.42, "units_per_case": 24.0,
          "price_per_case_eur": 10.0, "quantity_case": 40.0,
          "commercial_conditions": "EUR 10.00 per case"}
    assert rows_to_flag([ok], src_ok) == {}

    # (c) A price with no currency is unconfirmable regardless of reader → flagged.
    assert 0 in rows_to_flag([{"brand": "Foo", "product_name": "Bar",
                               "currency": None, "price_per_unit": 5.0}], "")

    # (d) An amount with a currency but NO basis word in the source → basis was
    # assumed → flagged (the per-case/per-bottle split can't be trusted).
    src_nb = "Cobble Stout, 300 cases, GBP 55 ex Hull"
    nb = {"brand": "Cobble", "product_name": "Stout", "currency": "GBP",
          "price_per_case": 55.0, "price_per_unit": 2.29, "units_per_case": 24.0,
          "quantity_case": 300.0}
    assert 0 in rows_to_flag([nb], src_nb)


# ── recon50: the 2026-11-25 live-test item-level rules (new wording each) ───────
def test_superseded_current_price_wins():
    # "the earlier 80 is withdrawn; current price EUR 92 per case" — stored the 80.
    src = "L4 Harrow Dark Ale 6x500ml, 30 cases EXW NL. Current price EUR 92 per case; the earlier 80 is withdrawn."
    o = {"brand": "Harrow", "product_name": "Dark Ale", "price_per_case": 80.0,
         "price_per_unit": 13.33, "units_per_case": 6.0, "currency": "EUR",
         "price_per_case_eur": 80.0, "price_per_unit_eur": 13.33}
    rec.apply_superseded_price([o], src)
    assert o["price_per_case"] == 92.0            # OLD: 80 (withdrawn) stayed

def test_retail_unit_price_not_a_case():
    # "EUR 18 per gift set … not a case price" — stored 3/case + EUR 54/case.
    src = "G7 Larkhill Candle Trio, three candles per set. 20 sets, EUR 18 per gift set FCA NL. Not a case price."
    o = {"brand": "Larkhill", "product_name": "Candle Trio", "price_per_unit": 18.0,
         "price_per_case": 54.0, "units_per_case": 3.0, "currency": "EUR",
         "error_flags": ["price ambiguous — assumed per case"]}
    rec.honor_retail_unit_price([o], src)
    assert o["price_per_case"] is None and o["units_per_case"] is None
    assert o["quantity_unit"] == "sets"
    assert not any("assumed per case" in str(f).lower() for f in o["error_flags"])

def test_retail_piece_price_recovered_from_bare_decimal():
    # price blanked by the extractor; a bare decimal among integer qty/weights is
    # the per-piece amount ("… | 40 | 7.5 | …"), recovered only on a retail line.
    src = "R2 | Dunmere Soap 120 g |  | 40 | 7.5 | FCA NL | Price is EUR per piece; stock is pieces."
    o = {"brand": "Dunmere", "product_name": "Soap 120 g", "price_per_unit": None,
         "price_per_case": None, "currency": "EUR"}
    rec.honor_retail_unit_price([o], src)
    assert o["price_per_unit"] == 7.5 and o["quantity_unit"] == "pieces"

def test_no_available_stock_flagged():
    # "all reserved, no stock available for new orders" → review + excluded.
    src = "K9 Ferncross Lager 24x330, 15 cases EUR 11 per case EXW NL. All reserved, no stock available for new orders."
    o = {"brand": "Ferncross", "product_name": "Lager", "price_per_case": 11.0,
         "needs_manual_review": False}
    rec.flag_no_available_stock([o], src)
    assert o["needs_manual_review"] is True

def test_supplier_read_from_supplier_line():
    # supplier is the "Supplier:" line, not the title/"Not Found".
    src = ("WESTMARCH CIRCULAR — offer dated 2 December 2026\n"
           "Supplier: Westmarch Trading BV | Circular WM-12 | Subject to unsold\n"
           "M1 Oakhill Gin 6x700, 10 cases EUR 60 per case EXW NL")
    offs = [{"brand": "Oakhill", "product_name": "Gin", "supplier_name": "Not Found"},
            {"brand": "Oakhill", "product_name": "Gin 2",
             "supplier_name": "WESTMARCH CIRCULAR — offer dated 2 December 2026"}]
    rec.fill_supplier_from_source(offs, src)
    assert all(o["supplier_name"] == "Westmarch Trading BV" for o in offs)

def test_unconfirmed_basis_and_unit_neutralised():
    # source says the desk hasn't confirmed what the price covers and the warehouse
    # hasn't identified what the balance counts → prices cleared, unit cleared, flag.
    src = ("Q3 Thornbury Gin 700ml, 6 per case. Balance 210. Desk figure GBP 4.20. "
           "The warehouse hasn't identified what the balance counts, and the desk "
           "hasn't confirmed what the price covers.")
    o = {"brand": "Thornbury", "product_name": "Gin", "price_per_unit": 4.2,
         "price_per_case": 25.2, "price_per_unit_eur": 5.0, "price_per_case_eur": 30.0,
         "currency": "GBP", "quantity_unit": "cases"}
    rec.flag_unconfirmed_basis_unit([o], src)
    assert o["price_per_case"] is None and o["price_per_unit"] is None
    assert o["quantity_unit"] is None and o["needs_manual_review"] is True

def test_customs_status_not_inherited_when_negated():
    # a line that NAMES T2 only to forbid inheriting it must not keep T2.
    src = "P5 Oakspan Bitter 12x500, 25 cases EUR 8 per case EXW NL. Customs pending; do not assume T2 from P1."
    o = {"brand": "Oakspan", "product_name": "Bitter", "custom_status": "T2"}
    rec.clear_unsourced_custom_status([o], src)
    assert o["custom_status"] is None

def test_buy_request_line_detected():
    from core.inquiry import is_strong_buy_line
    assert is_strong_buy_line("L9 BUY REQUEST: want Elm Ridge Rum, 40 cases around EUR 50/case. We do not have this stock.")
    # a plain sell line stays an offer
    assert not is_strong_buy_line("L8 Elm Ridge Rum 6x700, 40 cases EUR 50 per case EXW NL.")

def test_amount_pass_skips_duplicate_named_lots():
    # two same-named lots → the unique-token gate finds no single line, so an
    # amount-changing pass must leave BOTH untouched (never cross-assign).
    src = ("A1 Vale Pils 24x330, 30 cases EUR 9 per case EXW NL. Current price EUR 9.\n"
           "A2 Vale Pils 24x330, 20 cases EUR 10 per case DAP FR. Current price EUR 10.")
    offs = [{"brand": "Vale", "product_name": "Pils", "price_per_case": 9.0,
             "units_per_case": 24.0, "currency": "EUR"},
            {"brand": "Vale", "product_name": "Pils", "price_per_case": 10.0,
             "units_per_case": 24.0, "currency": "EUR"}]
    rec.apply_superseded_price(offs, src)
    assert [o["price_per_case"] for o in offs] == [9.0, 10.0]


# ── recon51: Grey Inlet GI482 reconciliation findings (new wording) ─────────────
def test_same_parcel_alternative_counted_once():
    # one parcel quoted a second way ("same LOT allocation … alternative for the
    # whole parcel") must merge to one row, not double-count the stock.
    src = ("H1 Oak Spire Rye 6x700, lot OS12: 20 cases, USD 20 per bottle EXW NL.\n"
           "H1B Oak Spire Rye 6x700, same OS12 allocation, EUR 130 per case, "
           "alternative for the whole parcel only, delivered FR.")
    offs = [{"brand": "Oak Spire", "product_name": "Rye", "price_per_case": 120.0,
             "price_per_unit": 20.0, "currency": "USD", "quantity_case": 20.0,
             "incoterm": "EXW", "location": "NL", "uid": "h1"},
            {"brand": "Oak Spire", "product_name": "Rye", "price_per_case": 130.0,
             "currency": "EUR", "quantity_case": None, "location": "FR", "uid": "h1b"}]
    dropped = rec.merge_same_stock_restatement(offs, src)
    assert len(dropped) == 1 and len(offs) == 1

def test_buying_for_a_customer_is_a_request():
    from core.inquiry import is_strong_buy_line
    assert is_strong_buy_line("Z1 Oak Spire Rye, 100 cases 6.24 EXW NL. Buying for a customer. Target only; please offer.")
    assert not is_strong_buy_line("Z2 Oak Spire Rye, 48 cases 10.56 EXW NL. On the floor.")

def test_closed_parcel_no_balance_flagged_dup_safe():
    # two same-named lots: the CLOSED one ("no balance") is flagged; the OPEN one
    # (distinguished only by its price/qty) is left alone.
    src = ("V7 Pine Rock Dry Gin 6x700, 27 cases USD 72 per case EXW NL, T2 applies.\n"
           "V8 Pine Rock Dry Gin 6x700, 13 cases USD 2 per case EXW NL. "
           "Prior parcel closed, no balance; retained for accounts.")
    closed = {"brand": "Pine Rock", "product_name": "Dry Gin", "price_per_case": 2.0,
              "currency": "USD", "quantity_case": 13.0, "needs_manual_review": False}
    openp = {"brand": "Pine Rock", "product_name": "Dry Gin", "price_per_case": 72.0,
             "currency": "USD", "quantity_case": 27.0, "needs_manual_review": False}
    rec.flag_no_available_stock([closed, openp], src)
    assert closed["needs_manual_review"] is True
    assert openp["needs_manual_review"] is False


# ── recon52: Samuel 2026-11-25 second live-test round (new wording) ──────────────
def test_basis_must_be_next_to_price_not_a_pack_spec():
    # "GBP 70" with only a PACK 'per case' earlier on the line is NOT a per-case
    # price — flag it (Estuary D05 Cairn 'GBP 66' was taken as per-case).
    src = "Oakmoor Dry Gin 700 ml 40% 6 bottles per case 20 cases GBP 70 EXW NL ready next week"
    o = {"brand": "Oakmoor", "product_name": "Dry Gin", "price_per_case": 70.0,
         "price_per_unit": 11.67, "units_per_case": 6.0, "currency": "GBP",
         "quantity_case": 20.0, "uid": "o"}
    rec.flag_absent_price_basis([o], src)
    assert o["price_per_case"] is None and o["needs_manual_review"] is True
    # a real 'per case' next to the price stays green
    ok = {"brand": "Birch", "product_name": "Ale", "price_per_case": 10.0,
          "price_per_unit": 0.42, "units_per_case": 24.0, "currency": "EUR",
          "quantity_case": 40.0, "uid": "ok"}
    rec.flag_absent_price_basis([ok], "Birch Ale 24 x 330 ml 40 cases EUR 10.00 per case EXW NL")
    assert ok["price_per_case"] == 10.0 and not ok.get("needs_manual_review")

def test_structured_row_basis_is_not_flagged():
    # a structured row's basis is its column header, not the row text — the
    # absent-basis pass must leave it alone (hybrid Copper Fen EUR 78/case).
    o = {"brand": "Copper", "product_name": "Fen Liqueur", "price_per_case": 78.0,
         "price_per_unit": 13.0, "units_per_case": 6.0, "currency": "EUR",
         "_structured_row": True, "uid": "s"}
    rec.flag_absent_price_basis([o], "N01 Copper Fen Liqueur 700 ml 30% 6 44 78 EXW Rotterdam T2")
    assert o["price_per_case"] == 78.0 and not o.get("needs_manual_review")

def test_lot_total_vs_breakdown_conflict():
    src = ("Harlow Pale Ale 24 cans per case Declared total 90 cases EUR 12 per case "
           "EXW NL BBD lots: 40 cases 30 Jun 2027 + 55 cases 31 Jul 2027.")
    o = {"brand": "Harlow", "product_name": "Pale Ale", "quantity_case": 90.0,
         "price_per_case": 12.0, "currency": "EUR", "uid": "h"}
    rec.flag_lot_total_conflict([o], src)
    assert o["needs_manual_review"] is True
    # a consistent total (40 + 50 = 90) is NOT flagged
    ok = {"brand": "Ridge", "product_name": "Lager", "quantity_case": 90.0, "uid": "r"}
    rec.flag_lot_total_conflict([ok], "Ridge Lager total 90 cases BBD lots: 40 cases + 50 cases")
    assert not ok.get("needs_manual_review")

def test_innerbox_qty_and_perpiece_price_recovered():
    # "3 inner boxes of 8 + 5 individual candles. GBP 7.10 each" → 29 pieces @ 7.10,
    # overriding a wrong stored per-unit price (W06 kept GBP 0.50).
    src = "Marlow Candle 180g, 3 inner boxes of 8 + 5 individual candles. GBP 7.10 each, FCA NL."
    o = {"brand": "Marlow", "product_name": "Candle", "price_per_unit": 0.5, "currency": "GBP", "uid": "m"}
    rec.fix_inner_box_from_text([o], src)
    assert o.get("quantity_bottles") == 29.0 and o.get("price_per_unit") == 7.1
    # reversed phrasing "inner boxes of 8 … 3 inners + 2 single … 7.40 per candle" → 26 @ 7.40
    src2 = "Harlow Candle 180 g Inner boxes of 8 candles 3 inners + 2 single candles GBP 7.40 per candle FCA NL"
    o2 = {"brand": "Harlow", "product_name": "Candle", "price_per_unit": None, "currency": "GBP", "uid": "h2"}
    rec.fix_inner_box_from_text([o2], src2)
    assert o2.get("quantity_bottles") == 26.0 and o2.get("price_per_unit") == 7.4


# ── recon53: columnar basis + retail/inner-box price survives absent-basis ───────
def test_columnar_price_header_is_not_basis_unconfirmed():
    from core.field_check import check_source_fidelity
    # a "Net EUR per case" column states the basis — a clean row must stay green.
    src = ("Reference Product Bottles per case Available cases Net EUR per case Terms\n"
           "N1 Harbour Stout 700 ml 6 44 78 EXW NL T2")
    issues = check_source_fidelity(
        [{"brand": "Harbour", "product_name": "Stout", "price_per_case": 78.0,
          "price_per_unit": 13.0, "currency": "EUR", "units_per_case": 6.0}], src)
    assert not any("price basis" in str(i).lower() for i in issues)

def test_retail_unit_price_survives_absent_basis_pass():
    # honor_retail sets a per-set price; the absent-basis pass must NOT clear it
    # ("per gift set" is a real basis / the row is marked unit-priced).
    src = "Larch Trio candle set, EUR 18 per gift set FCA NL. Not a case price."
    o = {"brand": "Larch", "product_name": "Trio", "price_per_unit": 18.0,
         "price_per_case": 54.0, "units_per_case": 3.0, "currency": "EUR", "uid": "l"}
    rec.honor_retail_unit_price([o], src)
    rec.flag_absent_price_basis([o], src)
    assert o["price_per_unit"] == 18.0 and o["price_per_case"] is None

def test_innerbox_price_survives_absent_basis_pass():
    src = "Harbour Candle Inner boxes of 8 candles 3 inners + 2 single candles GBP 7.40 per candle FCA NL"
    o = {"brand": "Harbour", "product_name": "Candle", "price_per_unit": None,
         "currency": "GBP", "uid": "h"}
    rec.fix_inner_box_from_text([o], src)
    rec.flag_absent_price_basis([o], src)
    assert o.get("quantity_bottles") == 26.0 and o.get("price_per_unit") == 7.4


# ── recon54: columnar price is supported, not a phantom ──────────────────────────
def test_columnar_price_is_not_a_phantom():
    from core.reconcile import reconcile_source_vs_rows
    # a "Net EUR per case" column carries the price as a bare number on the row —
    # the stored EUR 78 must NOT be reported as an invented/phantom value.
    src = ("Reference Product Bottles per case Available cases Net EUR per case Terms\n"
           "N1 Harbour Stout 700 ml 6 44 78 EXW NL T2")
    rc = reconcile_source_vs_rows(src, [{"brand": "Harbour", "product_name": "Stout",
        "currency": "EUR", "price_per_case": 78.0, "price_per_unit": 13.0,
        "price_per_case_eur": 78.0, "price_per_unit_eur": 13.0}])
    assert rc["phantom"] == []
    # but a free-text invented case price (no price column) IS still a phantom
    rc2 = reconcile_source_vs_rows(
        "W5 Cairn Thread Gin 700ml, 6 bottles/case. Desk figure GBP 3.60.",
        [{"brand": "Cairn", "product_name": "Thread Gin", "currency": "GBP",
          "price_per_case": 21.6, "price_per_unit": 3.6,
          "price_per_case_eur": 25.8, "price_per_unit_eur": 4.3}])
    assert any("cairn" in p.lower() for p in rc2["phantom"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} regression tests passed")
