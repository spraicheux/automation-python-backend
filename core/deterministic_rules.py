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


# City / country tokens for document-level location detection.
# Matches "EXW Rotterdam", "Ex Loendersloot", "DAP Riga", "EXW Spain",
# "Ex NewCorp", "Ex Singapore" etc. — the strong signal is `(EXW|Ex|DAP|...)` +
# a proper-noun place word within a few characters.
_LOCATION_PATTERN = re.compile(
    r"(?:EXW|Ex(?:work)?|DAP|DDP|FOB|CIF|CFR|CPT|FCA)"
    r"\s+([A-Z][A-Za-z]+)",
    re.IGNORECASE,
)

# Free-mail / generic domains that never identify a supplier from the email alone.
_GENERIC_MAIL_DOMAINS = {
    "gmail", "yahoo", "hotmail", "outlook", "icloud", "aol", "live",
    "protonmail", "gmx", "mail", "orange", "wanadoo", "yandex", "qq", "163",
}


def detect_document_location(text: str) -> Optional[str]:
    """
    Detect a document-level location by pairing an incoterm token with the
    proper-noun place that follows it in the header.
    """
    if not text:
        return None
    header = "\n".join(text.split("\n")[:15])
    m = _LOCATION_PATTERN.search(header)
    if not m:
        return None
    place = m.group(1).strip()
    # Reject clearly non-place words that might follow the incoterm token.
    if place.lower() in {"ready", "warehouse", "stock", "stocks", "now",
                         "available", "confirmed", "the"}:
        return None
    return place


def detect_supplier_from_metadata(text: str, source_filename: Optional[str] = None,
                                  sender_email: Optional[str] = None) -> Optional[str]:
    """
    Extract a supplier hint from document metadata when the extractor didn't
    pick one up. Priority:
      1. 'Répondre à:' / 'Reply to:' address's mailbox host (e.g. fbctrades)
      2. 'De:' / 'From:' explicit company name after the address
      3. Sender email domain
      4. Uppercase supplier code in the filename (e.g. 'HNS' in 'MIX SPIRITS ... HNS.xlsm')
    Never fabricates — only returns something present in the metadata.
    """
    if text:
        header = "\n".join(text.split("\n")[:20])
        m = re.search(
            r"(?:Répondre à|Reply[- ]?to)\s*:\s*(?:[A-Za-z ]+@)?([A-Za-z0-9.-]+@[A-Za-z0-9.-]+)",
            header, re.IGNORECASE,
        )
        if m:
            domain = m.group(1).split("@")[-1].split(".")[0]
            if domain and len(domain) >= 3:
                return domain.title().replace("-", " ")

        m = re.search(r"^De\s*:\s*([A-Z][A-Za-z0-9&' .-]{2,60})\s+[<]?[A-Za-z0-9._%+-]+@",
                      header, re.IGNORECASE | re.MULTILINE)
        if m:
            name = m.group(1).strip()
            if name and not name.lower().startswith(("subject", "objet", "date")):
                return name

    if sender_email and "@" in sender_email:
        domain = sender_email.split("@")[-1].split(".")[0].lower()
        # Free-mail addresses (gmail / yahoo / hotmail…) don't identify a
        # supplier company — leave supplier_name blank so the manual reviewer
        # sets it, rather than lying with "Gmail".
        if len(domain) >= 3 and domain not in _GENERIC_MAIL_DOMAINS:
            return domain.title().replace("-", " ")

    if source_filename:
        # e.g. "MIX SPIRITS DISCOUNT SALE HNS.xlsm" → uppercase 3-letter code at end
        stem = re.sub(r"\.(xlsx|xlsm|xls|pdf|txt|csv)$", "", source_filename, flags=re.IGNORECASE)
        m = re.search(r"\b([A-Z]{3,6})\s*$", stem)
        if m:
            return m.group(1)

    return None


def apply_deterministic_defaults(products: list, source_text: str,
                                 source_filename: Optional[str] = None,
                                 sender_email: Optional[str] = None) -> tuple:
    """
    Apply document-level defaults to LLM-extracted products.
    Returns (corrected_products, list_of_corrections).

    All document-level defaults (currency, incoterm, location, supplier) are
    filled only into rows that don't already have that field set. Row-level
    values coming from the LLM are always preserved so mixed-currency /
    mixed-incoterm files (like FBC Premium Spirits with per-line USD/EUR) work.
    """
    doc_currency = detect_document_currency(source_text)
    doc_incoterm = detect_document_incoterm(source_text)
    doc_location = detect_document_location(source_text)
    doc_supplier = detect_supplier_from_metadata(source_text, source_filename, sender_email)

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

        # ── location ──────────────────────────────────────────────────
        row_loc = (p.get("location") or "").strip()
        if doc_location and (not row_loc or row_loc.lower() == "not found"):
            p["location"] = doc_location
            corrections.append(f"Row {i+1}: location=None → {doc_location} (inherited from document header)")

        # ── supplier ──────────────────────────────────────────────────
        row_sup = (p.get("supplier_name") or "").strip()
        if doc_supplier and (not row_sup or row_sup.lower() == "not found"):
            p["supplier_name"] = doc_supplier
            corrections.append(f"Row {i+1}: supplier_name=None → {doc_supplier} (inherited from document metadata)")

        # ── €/case sanity: catch "Total Price" mis-mapping ──────────
        # LLM sometimes puts a source "Total Price" column (qty × unit price)
        # into price_per_case. That row has no real per-case price. Detect:
        #   units_per_case is unknown / 1 (no case pack in source)
        #   AND price_per_case ≈ quantity_case × price_per_unit
        # → null out price_per_case (it was a lot total, not a case price).
        upc = p.get("units_per_case")
        qc = p.get("quantity_case")
        ppc = p.get("price_per_case")
        ppu = p.get("price_per_unit")
        if (ppc and ppu and qc and
            (upc is None or upc == 1 or upc == 1.0) and
            qc > 1):
            expected_total = qc * ppu
            if abs(ppc - expected_total) / max(expected_total, 0.001) < 0.02:
                p["price_per_case"] = None
                p["price_per_case_eur"] = None
                # The numeric quantity was clearly in bottles (matched a
                # bottle-count × unit-price total), so the source used a
                # bottle unit. Record that so the dashboard shows "N btls"
                # instead of "N cs".
                if not (p.get("quantity_unit") or "").strip():
                    p["quantity_unit"] = "bottles"
                corrections.append(
                    f"Row {i+1}: price_per_case cleared + quantity_unit='bottles' — "
                    f"value equalled quantity × price_per_unit, so the source had "
                    f"no per-case pricing (looks like a mis-mapped 'Total Price' column)"
                )
        # Also: if units_per_case = 1 AND price_per_case == price_per_unit,
        # the "case" concept doesn't exist here — clear price_per_case to
        # keep the peer-key clean.
        elif (ppc and ppu and
              (upc == 1 or upc == 1.0) and
              abs(ppc - ppu) < 0.01):
            p["price_per_case"] = None
            p["price_per_case_eur"] = None
            corrections.append(
                f"Row {i+1}: price_per_case cleared — units_per_case=1 and "
                f"price_per_case equalled price_per_unit (no real case pack)"
            )

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
