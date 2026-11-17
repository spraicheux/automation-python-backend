"""Regression tests for the deterministic offer parsers + the persistence
contract they must satisfy. These lock in the fixes from Samuel's 2026-11-17/18
acceptance round so a future change can't silently re-break them:

  * FBC one-product-per-line "@ price" lists must NOT over-split into phantoms.
  * AQT WhatsApp pipe lists must keep a two-delivery-option line as ONE row.
  * EPIC flattened tables must recover every SKU.
  * Every deterministic row must survive clean_product_data WITH a product_key —
    the persistence loop drops any row without one (that was the silent 0-rows
    bug).
"""
from core.email_table import (parse_line_list_offer, parse_pipe_list_offer,
                              parse_flattened_price_table)
from core.openai_client import clean_product_data


FBC = """> Hello,
> Jack Daniels 12x70cl NRF FTL @ USD 112 per case, on floor, DAP Newcorp
> Bombay Sapphire 12x1L FTL @ Euro 100 per case, DAP Loen, 3 weeks
> Jagermeister 6x70cl Rf T2 FTL @ Euro 33 per case DAP Loen, 4-5 weeks
> Absolut Vodka 12x1L, 400 cases @ EUR 5.7/btl , on floor , T2 , DAP Loen
"""

AQT_WA = """01) Macallan Double Cask 12 YO | 700ml | 6 btls/case | 384 cases | GBP 318.00/case | FCA Antwerp | T2
04) Veuve Clicquot Yellow Label | 750ml | 6 btls/case | 24 cases total, same stock under either option | DAP Riga: EUR 144/case; EXW Breda: price TBC | MOQ 6 cases
12) Bombay Sapphire | 1L | 6 btls/case | 720 cases | HKD 765/case | FCA Hong Kong | MOQ 2 pallets
"""

EPIC = """PRODUCTS
BTL TYPE
REF / NRF
BTLS PER CASE
SIZE (CL)
VOL %
T1 / T2
QUANTITY
PRICE
DELIVERY
PER LOAD
STATUS

ABSOLUT BLUE
GLASS
REF
24
37,5
40
T2
1,404 cases
€ 73.00
Ready Stock
CHIVAS 12 YEARS
GLASS
REF
12
37,5
40
T1
2,310 cases
€ 85.00
Ready Stock
JOHN JAMESON
GLASS
REF
120
5
40
T1
1,430 cases
€ 96.00
Ready Stock
"""


def test_fbc_line_list_no_phantoms():
    prods = parse_line_list_offer(FBC)
    assert prods is not None and len(prods) == 4, prods
    by_name = {p["product_name"]: p for p in prods}
    # currency read from the cell, not defaulted to USD
    assert by_name["Bombay Sapphire"]["currency"] == "EUR"
    assert by_name["Jagermeister"]["currency"] == "EUR"


def test_aqt_wa_pipe_keeps_veuve_single():
    prods = parse_pipe_list_offer(AQT_WA)
    assert prods is not None and len(prods) == 3, prods
    veuve = [p for p in prods if "veuve" in p["product_name"].lower()]
    assert len(veuve) == 1, "Veuve two-delivery-option line must stay ONE row"
    assert veuve[0]["price_per_case"] == 144.0
    bombay = [p for p in prods if "bombay" in p["product_name"].lower()][0]
    assert bombay["currency"] == "HKD"


def test_epic_flattened_recovers_all_skus():
    prods = parse_flattened_price_table(EPIC)
    assert prods is not None and len(prods) == 3, prods
    prices = sorted(p["price_per_case"] for p in prods)
    assert prices == [73.0, 85.0, 96.0]


def test_parsers_noop_on_prose():
    prose = "Hi, we can offer Grey Goose 70cl at EUR 18.50 per bottle, 300 cases. Thanks."
    assert parse_line_list_offer(prose) is None
    assert parse_pipe_list_offer(prose) is None
    assert parse_flattened_price_table(prose) is None


def test_source_selfcheck_estimate():
    # The self-check estimate must equal the deterministic parser's exact count
    # for recognised formats, count price lines for a free-text offer, and give
    # NO usable estimate (low confidence) for prose so it never false-flags.
    from core.completeness import estimate_text_source_rows
    assert estimate_text_source_rows(FBC)["count"] == 4
    assert estimate_text_source_rows(AQT_WA)["count"] == 3
    assert estimate_text_source_rows(EPIC)["count"] == 3
    milanako = ("Aperol Intl Label\n1650 cases\n8.45 USD EXW\n"
                "Aperol Greece\n2250 cases\n8.45 GBP DAP\n"
                "Aperol Hungary\n2250 cases\n8.30 JPY DAP\nPayment 20% deposit")
    assert estimate_text_source_rows(milanako)["count"] == 3
    assert estimate_text_source_rows("Hi, call +65 6561 8718. Regards.")["confidence"] == "low"


def test_reconcile_names_the_missed_and_phantom_rows():
    # Tier-2: price-join the source to the stored rows and name exactly what is
    # missing / extra, not just a count.
    from core.reconcile import reconcile_source_vs_rows
    rows = parse_line_list_offer(FBC)
    assert reconcile_source_vs_rows(FBC, rows)["missed"] == []
    assert reconcile_source_vs_rows(FBC, rows)["phantom"] == []
    # drop Bombay → it is named as missed
    dropped = [r for r in rows if "bombay" not in r["product_name"].lower()]
    rec = reconcile_source_vs_rows(FBC, dropped)
    assert len(rec["missed"]) == 1 and "Bombay" in rec["missed"][0]
    assert rec["phantom"] == []
    # add a phantom priced row → it is named as phantom
    rec2 = reconcile_source_vs_rows(
        FBC, rows + [{"product_name": "Ghost", "brand": "Ghost",
                      "currency": "EUR", "price_per_case": 999}])
    assert rec2["missed"] == [] and any("Ghost" in p for p in rec2["phantom"])


def test_deterministic_rows_get_product_key():
    # The persistence loop drops any row without a product_key — every
    # deterministic row MUST get one from clean_product_data.
    for prods in (parse_line_list_offer(FBC), parse_pipe_list_offer(AQT_WA),
                  parse_flattened_price_table(EPIC)):
        assert prods
        for p in prods:
            cleaned = clean_product_data(p)
            assert cleaned.get("product_key") not in (None, "", "Not Found"), \
                f"missing product_key for {p.get('product_name')!r}"
