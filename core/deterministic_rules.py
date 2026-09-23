"""
Deterministic extraction rules that run alongside the LLM.

The philosophy: any fact that can be derived from the document structure
without reasoning should NOT depend on the LLM. Model quality shouldn't
influence values that a regex can extract reliably.

These rules run BEFORE the LLM (as prompt hints) AND AFTER the LLM (as
enforcement / defaulting), so an LLM that ignores or drops a header-level
fact still gets corrected before the row lands in the DB.
"""
import re
from typing import Optional


# Currency symbol → ISO code
_CURRENCY_SYMBOLS = {
    "€": "EUR", "$": "USD", "£": "GBP", "¥": "JPY", "₣": "CHF",
    "USD": "USD", "EUR": "EUR", "GBP": "GBP", "CHF": "CHF",
}

# Recognised currency ISO codes in text
_CURRENCY_ISO = ("EUR", "USD", "GBP", "CHF", "JPY", "AUD", "CAD")


def detect_document_currency(text: str) -> Optional[str]:
    """
    Scan the top of the document for a currency indicator that applies to
    every row. Returns ISO code (e.g. 'EUR') or None if ambiguous / not found.

    Patterns handled (in priority order):
      1. Explicit column header:  "Net price (EUR)" / "Price EUR" / "Price €"
      2. Meta line at the top:    "Prices - EXW Rotterdam (EUR)" / "All prices in USD"
      3. Very first currency token found in header block (first 10 lines).
    """
    if not text:
        return None

    header = "\n".join(text.split("\n")[:15])

    # 1. Column-header pattern like "Net price (EUR)" or "Price EUR"
    m = re.search(r"(?:net\s+)?price[^A-Z]{0,30}?\(?\s*(EUR|USD|GBP|CHF|€|\$|£|CHF)\s*\)?",
                  header, re.IGNORECASE)
    if m:
        raw = m.group(1).upper()
        return _CURRENCY_SYMBOLS.get(raw, raw if raw in _CURRENCY_ISO else None)

    # 2. Meta line: "Prices - EXW ... (EUR)" or "Prices in USD"
    m = re.search(r"prices?\s+(?:in|-)[^A-Z\n]{0,40}\(?\s*(EUR|USD|GBP|CHF|€|\$|£)\s*\)?",
                  header, re.IGNORECASE)
    if m:
        raw = m.group(1).upper()
        return _CURRENCY_SYMBOLS.get(raw, raw if raw in _CURRENCY_ISO else None)

    # 3. Fallback — first standalone ISO code in the header
    m = re.search(r"\b(EUR|USD|GBP|CHF)\b", header)
    if m:
        return m.group(1)

    # 4. Very last fallback — currency symbol in header
    for sym, iso in [("€", "EUR"), ("$", "USD"), ("£", "GBP")]:
        if sym in header:
            return iso

    return None


def detect_document_incoterm(text: str) -> Optional[str]:
    """Similar deterministic detection for document-wide incoterm.
    Only fires if the header explicitly names one (e.g. 'EXW Rotterdam')."""
    if not text:
        return None
    header = "\n".join(text.split("\n")[:15])
    m = re.search(r"\b(EXW|FOB|CIF|DAP|DDP|CFR|CPT|FCA)\b(?!\w)", header)
    if m:
        return m.group(1).upper()
    return None


def apply_deterministic_defaults(products: list, source_text: str) -> tuple:
    """
    Apply document-level defaults to LLM-extracted products.
    Returns (corrected_products, list_of_corrections).

    A correction fires when:
      - Document header explicitly states a currency, AND
      - A row has price_per_unit or price_per_case set but currency is null / empty / different
    Same shape for incoterm.
    """
    doc_currency = detect_document_currency(source_text)
    doc_incoterm = detect_document_incoterm(source_text)

    corrections = []
    for i, p in enumerate(products):
        # ── currency ────────────────────────────────────────────────────
        # Fill BLANKS from the document-level default only. Never override
        # a row that already has a currency set — the LLM had a reason for
        # it (mixed-currency files exist: FBC Premium Spirits mixes USD and
        # EUR by line). If the source really contradicts a row-level value
        # a manual correction wins via M4 edit; automatic override would
        # silently break real mixed-currency price lists.
        if doc_currency:
            has_price = p.get("price_per_unit") is not None or p.get("price_per_case") is not None
            row_cur = (p.get("currency") or "").upper().strip()
            if has_price and not row_cur:
                p["currency"] = doc_currency
                corrections.append(f"Row {i+1}: currency=None → {doc_currency} (inherited from document header)")

        # ── incoterm ────────────────────────────────────────────────────
        # Same principle: only fill blanks, never override.
        if doc_incoterm and not (p.get("incoterm") or "").strip():
            p["incoterm"] = doc_incoterm
            corrections.append(f"Row {i+1}: incoterm=None → {doc_incoterm} (inherited from document header)")

    return products, corrections


if __name__ == "__main__":
    tests = [
        ("Description | Alcohol % | Net contents | Carton | Net price\nBache-Gabrielsen VSOP 40% 0.5L | 40 | 0.5 | 24 | 6.2", None),
        ("PRICELIST Drinks per 01.04.2026\nDescription | Alcohol % | Net contents | Carton | Net price (EUR)\nSome row", "EUR"),
        ("Perfumes stock list\nPrices - EXW Rotterdam (EUR)\nMOV 50k", "EUR"),
        ("Prices in USD ex Singapore\nSomething", "USD"),
        ("Header without price info", None),
    ]
    for text, expected in tests:
        got = detect_document_currency(text)
        print(f"expected={expected}  got={got}  {'✓' if got == expected else '✗'}  |  {text[:60]!r}")
