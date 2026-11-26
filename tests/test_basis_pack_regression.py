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


# ── recon55: an explicit per-unit amount overrides a wrong stored value ──────────
def test_explicit_each_price_overrides_bled_value():
    # the extractor stored 0.52 (bled); the source says "$24 each" → use 24.
    src = "D4 Mere Lantern EDP 50 ml one retail bottle 12 pieces $24 each FCA NL dollar not agreed: CAD or USD."
    o = {"brand": "Mere Lantern", "product_name": "EDP", "price_per_unit": 0.52,
         "price_per_case": None, "currency": "USD", "quantity_bottles": 12.0,
         "quantity_unit": "pieces", "uid": "m"}
    rec.honor_retail_unit_price([o], src)
    assert o["price_per_unit"] == 24.0
    # a correct per-set price is left unchanged (no spurious override)
    o2 = {"brand": "Larch", "product_name": "Trio", "price_per_unit": 18.0,
          "price_per_case": 54.0, "units_per_case": 3.0, "currency": "EUR", "uid": "l"}
    rec.honor_retail_unit_price([o2], "Larch Trio EUR 18 per gift set FCA NL. Not a case price.")
    assert o2["price_per_unit"] == 18.0 and o2["price_per_case"] is None


# ── recon56: Samuel 2026-11-26 — twin rule, explicit basis, derived case price ───
def test_twin_never_overrides_a_source_confirmed_price():
    # a row whose OWN source confirms the basis (per-set price) must NOT be blanked
    # by a cross-source basis twin (a stale/duplicate import's unconfirmed copy).
    tmap = {rec._twin_key_fields("Acme", "Dune", None): {21.0}}
    o = {"brand": "Dune", "product_name": "Gift Set", "supplier_name": "Acme",
         "price_per_unit": 21.0, "price_per_case": None, "unit_volume_ml": None,
         "_unit_priced": True}
    rec.apply_basis_from_twin([o], tmap)
    assert o["price_per_unit"] == 21.0
    # a genuinely basis-ambiguous DERIVED row is still synced down
    o2 = {"brand": "Beef", "product_name": "Gin", "supplier_name": "Acme",
          "price_per_unit": 12.5, "price_per_case": 75.0, "units_per_case": 6.0,
          "unit_volume_ml": 700}
    rec.apply_basis_from_twin([o2], {rec._twin_key_fields("Acme", "Beef", 700): {75.0}})
    assert o2["price_per_case"] is None

def test_basis_clause_applies_without_adjacency():
    # an explicitly-applicable basis (column heading / note) counts even when not
    # next to the amount; a pack spec does not.
    assert rec._basis_stated_near_price("Reed Tonic 56 cases 6.24 EXW NL The figure is EUR per case: EUR 0.52 per bottle x 12")
    assert rec._basis_stated_near_price("Bracken Pils 4 wrapped six-packs per case 28 cases EUR 2.70 per six-pack EXW NL")
    assert not rec._basis_stated_near_price("Cairn Gin 6 bottles per case 17 cases GBP 66 EXW NL")

def test_derived_case_price_is_not_a_phantom():
    from core.reconcile import reconcile_source_vs_rows
    src = ("Reference Product Bottles per case Available cases Net EUR per case Terms\n"
           "D2 Bracken Pils 4 wrapped six-packs per case 28 cases EUR 2.70 per six-pack EXW NL")
    rc = reconcile_source_vs_rows(src, [{"brand": "Bracken", "product_name": "Pils",
        "currency": "EUR", "price_per_case": 10.8, "price_per_unit": 0.45,
        "units_per_case": 24.0, "price_per_case_eur": 10.8}])
    assert rc["phantom"] == []


# ── recon58: Samuel 2026-11-26 round 4 ──────────────────────────────────────────
def _moss_pair(exw_pc, dap_pc, parcel, extra=""):
    src = (f"D10 Moss Bridge Whisky 700ml 40% 6 bottles/case Parcel {parcel}: 19 cases "
           f"+ 5 loose bottles EUR {exw_pc} per case; EUR 20 per loose bottle EXW Ghent.\n"
           f"D10-L Moss Bridge Whisky same {parcel} parcel EUR {dap_pc} per case; EUR 21 "
           f"per loose bottle DAP Lille. Alternative delivery for the SAME stock. {extra}")
    offs = [
        {"brand": "Moss", "product_name": "Moss Bridge Whisky", "currency": "EUR",
         "price_per_case": exw_pc, "price_per_unit": 20.0, "quantity_case": 19.0,
         "quantity_bottles": 5.0, "units_per_case": 6.0, "incoterm": "EXW",
         "location": "Ghent", "uid": "exw", "error_flags": []},
        {"brand": "Moss", "product_name": "Moss Bridge Whisky", "currency": "EUR",
         "price_per_case": dap_pc, "price_per_unit": 21.0, "quantity_case": 19.0,
         "quantity_bottles": 5.0, "units_per_case": 6.0, "incoterm": "DAP",
         "location": "Lille", "uid": "dap",
         "error_flags": ["multiple incoterms detected — row duplicated"]},
    ]
    return offs, src

def test_dual_incoterm_same_stock_splits_into_two_comparable_rows():
    # Moss: EUR 120 EXW Ghent vs EUR 126 DAP Lille, same parcel MB71 → TWO rows,
    # each comparable in its own incoterm bucket, both green, stock counted once.
    offs, src = _moss_pair(120.0, 126.0, "MB71")
    rec.merge_same_stock_restatement(offs, src)
    rec.merge_alt_delivery_lots(offs)
    assert len(offs) == 2
    by_inc = {o["incoterm"]: o for o in offs}
    assert set(by_inc) == {"EXW", "DAP"}
    assert by_inc["EXW"]["price_per_case"] == 120.0
    assert by_inc["DAP"]["price_per_case"] == 126.0
    # stock counted once: only the EXW (kept) row carries quantity
    assert by_inc["EXW"]["quantity_case"] == 19.0
    assert by_inc["DAP"]["quantity_case"] is None
    # both green; the dual-delivery review marker is cleared
    assert by_inc["EXW"]["needs_manual_review"] is False
    assert by_inc["DAP"]["needs_manual_review"] is False
    assert all("multiple incoterms detected" not in f
               for f in by_inc["DAP"]["error_flags"])
    assert all("one stock pool" in (o.get("commercial_conditions") or "").lower()
               for o in offs)

def test_dual_incoterm_same_price_still_folds():
    # same stock, two incoterms, SAME price → one row + note (no spurious split).
    offs, src = _moss_pair(120.0, 120.0, "MB71")
    rec.merge_same_stock_restatement(offs, src)
    rec.merge_alt_delivery_lots(offs)
    assert len(offs) == 1

def test_whatsapp_one_stock_pool_splits():
    # North Fen W07: EUR 114 EXW vs EUR 120 DAP, "One stock pool, not two".
    src = ("W07 Moss Bridge Whisky 700ml 40%, six bottles/case. Parcel NF71: 11 cases "
           "and 2 loose bottles. EUR 114/case and EUR 19 per loose bottle EXW Ghent. "
           "Alternatively the SAME parcel delivered DAP Lille at EUR 120/case and EUR 20 "
           "per loose bottle. One stock pool, not two.")
    offs = [
        {"brand": "Moss", "product_name": "Moss Bridge Whisky", "currency": "EUR",
         "price_per_case": 114.0, "quantity_case": 11.0, "units_per_case": 6.0,
         "incoterm": "EXW", "location": "Ghent", "uid": "e", "error_flags": []},
        {"brand": "Moss", "product_name": "Moss Bridge Whisky", "currency": "EUR",
         "price_per_case": 120.0, "quantity_case": 11.0, "units_per_case": 6.0,
         "incoterm": "DAP", "location": "Lille", "uid": "d", "error_flags": []},
    ]
    rec.merge_same_stock_restatement(offs, src)
    rec.merge_alt_delivery_lots(offs)
    assert len(offs) == 2
    assert {o["incoterm"] for o in offs} == {"EXW", "DAP"}
    assert sum(1 for o in offs if o["quantity_case"]) == 1  # counted once

def test_unlabeled_balance_flags_and_blanks_quantity():
    # Plover: "360 … Warehouse balance has no unit label" + "USD 1.25 per piece".
    src = ("D09 Plover Lane Soap 100 g 12 pieces per carton 360 USD 1.25 per piece "
           "FCA Hull Warehouse balance has no unit label.")
    o = {"brand": "Plover", "product_name": "Plover Lane Soap", "currency": "USD",
         "price_per_unit": 1.25, "quantity_case": 30.0, "quantity_bottles": 360.0,
         "quantity_unit": "pieces", "incoterm": "FCA", "location": "Hull",
         "error_flags": ["currency normalized to EUR from the source line"]}
    offs = [o]
    rec.flag_unlabeled_quantity_balance(offs, src)
    rec.clear_contradictory_currency_flag(offs, src)
    assert o["price_per_unit"] == 1.25 and o["currency"] == "USD"  # price kept
    assert o["quantity_case"] is None and o["quantity_bottles"] is None
    assert o["needs_manual_review"] is True
    assert "no unit label" in (o["commercial_conditions"] or "").lower()
    # the contradictory "normalized to EUR" note is gone
    assert all("normalized to eur" not in f.lower() for f in o["error_flags"])

def test_clearly_labelled_quantity_not_flagged():
    # a line that states its unit ("17 cases") must NOT trip the unlabeled rule.
    src = "D05 Cairn Thread Gin 6 bottles per case 17 cases GBP 66 EXW Antwerp."
    o = {"brand": "Cairn", "product_name": "Cairn Thread Gin", "currency": "GBP",
         "quantity_case": 17.0, "incoterm": "EXW", "location": "Antwerp",
         "error_flags": []}
    offs = [o]
    n = rec.flag_unlabeled_quantity_balance(offs, src)
    assert n == 0 and o["quantity_case"] == 17.0

def test_cross_source_twin_needs_a_named_supplier():
    # Two different suppliers that both failed supplier attribution must NOT be
    # treated as twins (Samuel 2026-11-26: North Fen's quote must not reach into
    # Estuary). A price-less, basis-unstated "twin" with an EMPTY supplier is
    # ignored; the other file's derived price is left untouched.
    twin_map = {rec._twin_key_fields("", "Mere", 50.0): {24.0}}
    offs = [{"brand": "Mere", "product_name": "Mere Lantern EDP", "currency": "USD",
             "price_per_unit": 24.0, "unit_volume_ml": 50.0, "supplier_name": "",
             "needs_manual_review": False}]
    changed = rec.apply_basis_from_twin(offs, twin_map)
    assert changed == 0
    assert offs[0]["price_per_unit"] == 24.0 and offs[0]["currency"] == "USD"
    # the SAME named supplier still syncs (regression guard stays effective)
    twin_named = {rec._twin_key_fields("Acme BV", "Rae", 700.0): {120.0}}
    o2 = [{"brand": "Rae", "product_name": "Rae Gin", "currency": "EUR",
           "price_per_case": 120.0, "price_per_unit": 20.0, "units_per_case": 6.0,
           "unit_volume_ml": 700.0, "supplier_name": "Acme BV"}]
    assert rec.apply_basis_from_twin(o2, twin_named) == 1
    assert o2[0]["price_per_case"] is None

def test_nonproduct_header_row_dropped():
    # the circular banner read as a product (no price, no qty) is dropped; a real
    # price-on-request product (with a quantity) is kept.
    offs = [
        {"brand": "Not Found", "product_name": "Supplier Offer", "currency": "EUR",
         "product_reference": "EL-925", "uid": "hdr"},
        {"brand": "Sedge", "product_name": "Sedge Point Gin", "quantity_case": 15.0,
         "incoterm": "EXW", "location": "Antwerp", "uid": "sedge"},
    ]
    rec.drop_nonproduct_header_rows(offs, "")
    names = [o["product_name"] for o in offs]
    assert names == ["Sedge Point Gin"]

def test_header_row_dropped_when_named_after_supplier_company():
    # the LLM sometimes names the banner "Supplier <company>" (not "Supplier
    # Offer") — still a no-price, no-qty banner line and must be dropped.
    offs = [
        {"brand": "Not Found", "product_name": "Supplier Estuary Ledger Merchants BV",
         "currency": "EUR", "product_reference": "EL-925", "uid": "h1"},
        {"brand": None, "product_name": "Circular EL-925", "uid": "h2"},
        {"brand": "Copper", "product_name": "Fen Liqueur", "price_per_case": 78.0,
         "quantity_case": 44.0, "incoterm": "EXW", "uid": "real"},
    ]
    rec.drop_nonproduct_header_rows(offs, "")
    assert [o["product_name"] for o in offs] == ["Fen Liqueur"]

def test_backstop_dialect_reads_desk_sheet_refs():
    # the free-text 'stock circular' dialect must map the Desk columns so a
    # dropped line can be recovered deterministically.
    from core.xlsx_structured import _match_header, _FREETEXT_SYNONYMS
    assert _match_header("commercial figure", _FREETEXT_SYNONYMS) == "_price_value"
    assert _match_header("what is on the floor", _FREETEXT_SYNONYMS) == "product_name"
    assert _match_header("handover", _FREETEXT_SYNONYMS) == "incoterm"
    # OFF by default: a normal sheet never picks up this dialect
    assert _match_header("commercial figure") is None


def test_reconciler_matches_by_ticket_ref_not_only_price():
    # Samuel 2026-11-26: a present row reported "not imported" because its price
    # does not read literally in the source — a DERIVED case price (A02 wrapped:
    # EUR 2.82/six-pack × 4) and a SOLD-OUT price-less row (A14). Match by the
    # supplier's own ticket reference so neither is a false "missed".
    from core.reconcile import reconcile_source_vs_rows
    src = ("A02 Kestrel Strand Pils 4 wrapped six-packs / case 31 cases EUR 2.82 per six-pack EXW Breda\n"
           "A14 Quartz Meadow Vodka 12 bottles per case 0 available Former quote EUR 42/case All 62 cases SOLD\n")
    rows = [
        {"product_reference": "A02", "brand": "Kestrel", "product_name": "Strand Pils",
         "currency": "EUR", "price_per_case": 11.28, "price_per_unit": 0.47, "units_per_case": 24.0},
        {"product_reference": "A14", "brand": "Quartz", "product_name": "Meadow Vodka", "currency": "EUR"},
    ]
    r = reconcile_source_vs_rows(src, rows)
    assert r["missed"] == [], f"unexpected missed: {r['missed']}"
    # a genuinely dropped ref still shows missed (no row carries it)
    r2 = reconcile_source_vs_rows(src + "A99 Gone Product 10 cases EUR 5 per case\n", rows)
    assert any("A99" in m for m in r2["missed"])


def test_northwick_field_fixes():
    # Samuel 2026-11-26 Northwick NF1126 — the same bug classes on a new file.
    src = open(__file__).read()  # placeholder; passes use explicit src below
    def run(o, text):
        offs = [o]
        for fn in ("honor_retail_unit_price", "recover_loose_pieces_qty",
                   "recover_loose_bottles_addon", "flag_unlabeled_quantity_balance",
                   "fix_wrapped_pack_unit_price"):
            getattr(rec, fn)(offs, text)
        return o
    # W02 wrapped six-pack: per-can price, no false conflict
    o = run({"brand": "", "product_name": "Same Kestrel pils", "currency": "EUR",
             "price_per_unit": 2.7, "price_per_case": 10.8, "quantity_case": 27,
             "units_per_case": 24, "error_flags": ["case price 10.8 does not match unit 2.7 x pack 24"]},
            "W02 Same Kestrel pils as W01, wrapped in sixes: FOUR six-packs in a case. 27 cases, EUR 2.70 per six-pack EXW Breda.")
    assert o["price_per_unit"] == 0.45 and not o["needs_manual_review"]
    # W07 gift set priced per set (EUR 23/set), not per case
    o = run({"brand": "Mica", "product_name": "Garden Discovery Set", "currency": "EUR",
             "price_per_case": 23.0, "quantity_case": 13, "units_per_case": 0},
            "W07 Mica Garden Discovery Set: two 50ml sprays plus one 15ml spray in each intact set. 13 sets at EUR 23/set FCA Ghent. No outer cases.")
    assert o["price_per_unit"] == 23.0 and o["price_per_case"] is None and o["units_per_case"] is None
    # W04 / W20 per-bottle retail quantity
    o = run({"brand": "Linden", "product_name": "Vault EDP retail boxed", "currency": "USD", "price_per_unit": 26.0},
            "W04 Linden Vault EDP 50ml retail boxed: 15 individual bottles at USD 26 per bottle, FCA Hull.")
    assert o["quantity_bottles"] == 15.0
    # W05 unlabeled balance + basis kept as quoted
    o = run({"brand": "Slate", "product_name": "Orchard Gin", "currency": "GBP",
             "quantity_case": 275, "units_per_case": 6, "error_flags": ["Price on request — no price stated in source"]},
            "W05 Slate Orchard Gin 700ml 42%, packing 6 bottles/case. Balance 275. GBP 4.20. Neither the balance unit nor the price unit has been confirmed.")
    assert o["quantity_case"] is None and o["needs_manual_review"]
    assert "as quoted: gbp 4.20" in (o["commercial_conditions"] or "").lower()
    assert all("no price stated" not in f.lower() for f in o["error_flags"])
    # Rowan 9 cases + 5 loose (not the 54 = 9×6 double-count)
    o = run({"brand": "Rowan", "product_name": "Ferry Whisky", "currency": "EUR",
             "price_per_case": 126.0, "quantity_case": 9, "quantity_bottles": 54, "units_per_case": 6},
            "W08 Rowan Ferry Whisky 6/case. Parcel NF83: 9 cases plus 5 loose bottles. EUR 126/case EXW Ghent.")
    assert o["quantity_bottles"] == 5.0

def test_basis_unconfirmed_row_matched_by_as_quoted_amount():
    # Samuel 2026-11-26 W05 Slate: a basis-unconfirmed row carries its amount only
    # in "as quoted: GBP 4.20", and the source line's balance (275) is read as the
    # price — so the row must still join on the LITERAL quoted amount, not be
    # reported "not imported".
    from core.reconcile import reconcile_source_vs_rows
    src = ("W05 Slate Orchard Gin 700ml 42%, 6 bottles/case. Balance 275. GBP 4.20. "
           "Neither the balance unit nor the price unit has been confirmed.\n")
    row = {"brand": "Slate", "product_name": "Orchard Gin", "currency": "GBP",
           "price_per_unit": None, "price_per_case": None,
           "commercial_conditions": "as quoted: GBP 4.20 (price basis unconfirmed)"}
    assert reconcile_source_vs_rows(src, [row])["missed"] == []

def test_circular_id_mislabeled_as_product_reference_is_rederived():
    # Samuel 2026-11-26 W02: the LLM stored the circular number "NF1126" as the
    # row's reference instead of the ticket "W02", so the reconciler could not join
    # it. Re-derive the ticket from the row's source line.
    src = ("Northwick Fen Drinks BV — NF1126\nOffer issued 26 November 2026.\n"
           "W02 — Same Kestrel pils as W01, wrapped in sixes, 27 cases EUR 2.70 per six-pack.\n")
    o = {"brand": "", "product_name": "Same Kestrel pils", "price_per_case": 10.8,
         "quantity_case": 27, "product_reference": "NF1126"}
    rec.fix_mislabeled_product_reference([o], src)
    assert o["product_reference"] == "W02"

def test_bare_figure_price_not_a_phantom_but_invented_still_flagged():
    # Samuel 2026-11-26 W16 Tidal: "Desk figure 6.24. This figure is EUR per case"
    # — a correct price stated as a bare number must NOT be called a phantom; but a
    # price that appears nowhere in the source still must be.
    from core.reconcile import reconcile_source_vs_rows
    src = ("W16 Tidal Orchard Tonic 200ml, 12/case T2: 47 cases EXW Breda. "
           "Desk figure 6.24. This figure is EUR PER CASE, not per bottle.\n")
    good = {"brand": "Tidal", "product_name": "Orchard Tonic", "currency": "EUR",
            "price_per_case": 6.24, "price_per_unit": 0.52, "units_per_case": 12}
    assert reconcile_source_vs_rows(src, [good])["phantom"] == []
    bad = {"brand": "Ghost", "product_name": "Gin", "currency": "EUR", "price_per_case": 999.99}
    assert reconcile_source_vs_rows(src, [bad])["phantom"]  # invented → flagged

def test_pack_spec_not_read_as_retail_quantity():
    # "6 bottles/case" is a pack spec, never the available quantity
    o = {"brand": "Slate", "product_name": "Gin", "currency": "GBP", "units_per_case": 6}
    rec.recover_loose_pieces_qty([o], "Slate Gin 700ml, packing 6 bottles/case. Balance unclear.")
    assert o.get("quantity_bottles") is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} regression tests passed")
