"""
Deterministic source-row estimators for completeness control.

The pipeline must know how many product rows a source document really
contains, so it can refuse to mark a file as fully imported when the
LLM silently drops some. Per the client requirement:

  - For STRUCTURED XLSX files, the source-row count is DETERMINISTIC
    from the sheet/table structure. The LLM may be used as a secondary
    sanity cross-check but is never the authoritative source.

  - For LESS STRUCTURED PDFs, a MULTI-SIGNAL approach is used:
    whitespace-aware line counting after stripping headers / footers /
    totals, GS1-barcode count, and an optional LLM cross-check. No
    single signal is authoritative; the system picks the strongest and
    reports which it used.

Both estimators return a dict with:
    {"count": int, "method": str, "signals": {...}, "confidence": "high"|"medium"|"low"}

so callers can log the provenance of the number.
"""
from __future__ import annotations

import re
from typing import Any


# Header-cell tokens we recognise as product-name signals across the
# supplier files seen so far. Case-insensitive match; the row is treated
# as the header row when it contains at least one of these.
_PRODUCT_NAME_COLUMNS = (
    "description", "line", "product", "designation", "designacion",
    "libelle", "libellé", "produit", "name", "item", "article",
    "designation produit",
)
_PRICE_COLUMNS = (
    "price", "prix", "eur", "usd", "gbp", "net", "unit price",
    "pru", "ht", "€", "$",
)
_BARCODE_COLUMNS = (
    "ean", "barcode", "bar code", "gtin", "code ean", "upc", "ref ean",
)


def _row_as_strings(row: Any) -> list[str]:
    """Coerce a pandas row (Series) into a plain list of trimmed strings."""
    out = []
    for v in row:
        if v is None:
            out.append("")
            continue
        try:
            import pandas as pd
            if pd.isna(v):
                out.append("")
                continue
        except Exception:
            pass
        out.append(str(v).strip())
    return out


def _looks_like_header(cells: list[str]) -> bool:
    """A row is a header when at least one cell matches a product-name
    column token AND at least one matches a price or barcode column."""
    low = [c.lower() for c in cells if c]
    has_name = any(any(tok in c for tok in _PRODUCT_NAME_COLUMNS) for c in low)
    has_price = any(any(tok in c for tok in _PRICE_COLUMNS) for c in low)
    has_bar = any(any(tok in c for tok in _BARCODE_COLUMNS) for c in low)
    return has_name and (has_price or has_bar)


def _find_header_row(df) -> int:
    """Return the row index of the header. Scans the first 10 rows and
    returns the first that looks like a header; falls back to 0 (which
    is what pandas uses when the first row IS the header already, in
    which case cells comparing against our tokens will match the pandas
    column names)."""
    for i in range(min(10, len(df))):
        cells = _row_as_strings(df.iloc[i])
        if _looks_like_header(cells):
            return i
    return -1  # no explicit header row within the first 10 rows


def _column_roles(cells: list[str]) -> dict[str, int]:
    """For a header row, map {"name": col_idx, "price": col_idx, "bar": col_idx}
    where values are present. Columns absent from the header map to -1."""
    roles: dict[str, int] = {"name": -1, "price": -1, "bar": -1}
    for i, c in enumerate(cells):
        low = c.lower()
        if roles["name"] == -1 and any(tok in low for tok in _PRODUCT_NAME_COLUMNS):
            roles["name"] = i
        if roles["price"] == -1 and any(tok in low for tok in _PRICE_COLUMNS):
            roles["price"] = i
        if roles["bar"] == -1 and any(tok in low for tok in _BARCODE_COLUMNS):
            roles["bar"] = i
    return roles


_BARCODE_CELL_RE = re.compile(r"^\s*\d{8,14}\s*$")


def _cell_has_product_name(cell: str) -> bool:
    """A product-name cell has at least two alphabetic characters —
    stricter than non-empty so footer rows like 'TOTAL' or 'END' don't
    count as products, but loose enough to catch short-form brand +
    product codes seen on perfume stock lists ("CK BE ET 100 vp",
    "Q BY D&G EP 100 vp") where every word is 1-3 characters."""
    if not cell:
        return False
    letters = sum(1 for c in cell if c.isalpha())
    return letters >= 2


def _cell_has_price(cell: str) -> bool:
    """A price cell is parseable as a positive number (possibly with
    comma decimal / currency symbol / trailing €+ etc.)."""
    if not cell:
        return False
    s = str(cell).strip().replace("€", "").replace("$", "").replace("£", "")
    s = s.replace(",", ".").strip()
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s) > 0
    except (ValueError, TypeError):
        return False


def _cell_has_barcode(cell: str) -> bool:
    digits = re.sub(r"\D+", "", str(cell or ""))
    return 8 <= len(digits) <= 14


def estimate_xlsx_source_rows(df) -> dict:
    """Deterministic row count for a pandas DataFrame of a supplier
    XLSX. Counts non-header rows where (product_name AND (price OR
    barcode)). Header row is detected within the first 10 rows; if none
    is found, falls back to counting any row satisfying the signal.

    This is the AUTHORITATIVE expected_row_count for structured XLSX
    files. No LLM involvement."""
    if df is None or len(df) == 0:
        return {"count": 0, "method": "xlsx_deterministic",
                "signals": {"total_rows": 0, "header_row": -1},
                "confidence": "high"}

    header_idx = _find_header_row(df)
    if header_idx >= 0:
        header_cells = _row_as_strings(df.iloc[header_idx])
        roles = _column_roles(header_cells)
        data_start = header_idx + 1
    else:
        # Fallback: assume pandas treated row 0 as the header (so
        # df.columns carries the labels). Build roles from df.columns.
        header_cells = [str(c) for c in df.columns]
        roles = _column_roles(header_cells)
        data_start = 0

    name_col = roles["name"]
    price_col = roles["price"]
    bar_col = roles["bar"]

    count = 0
    for i in range(data_start, len(df)):
        cells = _row_as_strings(df.iloc[i])
        if not cells:
            continue
        # Name: if we know the column, enforce it; else scan all cells.
        if name_col >= 0 and name_col < len(cells):
            has_name = _cell_has_product_name(cells[name_col])
        else:
            has_name = any(_cell_has_product_name(c) for c in cells)
        if not has_name:
            continue
        if price_col >= 0 and price_col < len(cells):
            has_price = _cell_has_price(cells[price_col])
        else:
            has_price = any(_cell_has_price(c) for c in cells)
        if bar_col >= 0 and bar_col < len(cells):
            has_bar = _cell_has_barcode(cells[bar_col])
        else:
            has_bar = any(_cell_has_barcode(c) for c in cells)
        if has_price or has_bar:
            count += 1

    return {
        "count": count,
        "method": "xlsx_deterministic",
        "signals": {
            "total_rows": len(df),
            "header_row": header_idx,
            "name_col": name_col,
            "price_col": price_col,
            "bar_col": bar_col,
        },
        "confidence": "high",
    }


# ─── PDF multi-signal estimator ───────────────────────────────────────

_PDF_BARCODE_RE = re.compile(r"(?<!\d)(\d{8,14})(?!\d)")
_PDF_PRICE_HINT_RE = re.compile(r"\d{1,4}[.,]\d{1,2}\b")


_HEADER_FOOTER_HINTS = (
    "prix", "price", "total", "sous-total", "subtotal", "page",
    "www.", "@", "tel", "phone", "fax", "siret", "iban", "vat ",
    "copyright", "all rights reserved",
    "stock list", "price list",
)


def _is_header_footer_line(line: str) -> bool:
    low = line.lower().strip()
    if not low:
        return True
    if len(low) < 3:
        return True
    for hint in _HEADER_FOOTER_HINTS:
        if hint in low and len(low) < 80:
            return True
    return False


def estimate_pdf_source_rows(all_pages_text: list[str]) -> dict:
    """Multi-signal estimator for a PDF document. Returns the best
    deterministic count the signals support, together with every
    individual signal reading so the caller can log/trace.

    Signals used:
      - barcodes: count of GS1-length (8-14) digit tokens
      - product_lines: lines carrying both an alphabetic product name
        hint and a price hint, after stripping header/footer/total lines
      - longest_signal: max(barcodes, product_lines) — used as the
        authoritative count when the two are within 10% of each other,
        otherwise the one with higher confidence (barcode count is
        more reliable when ≥ 10 and barcode-dense)
    """
    combined = "\n".join(all_pages_text or [])
    if not combined.strip():
        return {"count": 0, "method": "pdf_multi_signal",
                "signals": {}, "confidence": "low"}

    # Barcode signal
    barcode_tokens = _PDF_BARCODE_RE.findall(combined)
    # Dedupe barcodes that appear twice (page-break repeat)
    barcode_tokens_unique = set()
    barcode_count_raw = 0
    for b in barcode_tokens:
        barcode_tokens_unique.add(b)
        barcode_count_raw += 1
    bar_count = len(barcode_tokens_unique) if barcode_tokens_unique else 0

    # Product-line signal
    lines = combined.split("\n")
    product_lines = 0
    for ln in lines:
        if _is_header_footer_line(ln):
            continue
        has_name = _cell_has_product_name(ln)
        has_price = bool(_PDF_PRICE_HINT_RE.search(ln))
        if has_name and has_price:
            product_lines += 1

    # Resolve to one count
    signals = {
        "barcode_count_unique": bar_count,
        "barcode_count_raw": barcode_count_raw,
        "product_lines": product_lines,
        "total_lines": len(lines),
    }

    if bar_count == 0 and product_lines == 0:
        return {"count": 0, "method": "pdf_multi_signal",
                "signals": signals, "confidence": "low"}

    if bar_count >= 10 and bar_count >= 0.9 * product_lines:
        # Dense barcode-carrying document (FBC-style). Barcodes are the
        # most reliable signal here.
        return {"count": bar_count, "method": "pdf_multi_signal",
                "signals": signals, "confidence": "high"}

    if product_lines >= bar_count * 1.2 and product_lines > 0:
        # Many more product-looking lines than barcodes. The source
        # likely has some non-barcode rows (cosmetics-style). Trust
        # the line signal.
        return {"count": product_lines, "method": "pdf_multi_signal",
                "signals": signals, "confidence": "medium"}

    # Default: pick the stronger signal.
    best = max(bar_count, product_lines)
    conf = "medium" if abs(bar_count - product_lines) < max(5, 0.1 * best) else "low"
    return {"count": best, "method": "pdf_multi_signal",
            "signals": signals, "confidence": conf}


if __name__ == "__main__":
    import pandas as pd
    import sys
    files = [
        "/Users/aliabdullah/Downloads/W23_niche_Offer_02.06.xlsx",
        "/Users/aliabdullah/Downloads/W39_Designer_Offer_23.09.xlsx",
        "/Users/aliabdullah/Downloads/Cosmetics_Offer_18.06.xlsx",
        "/Users/aliabdullah/Downloads/Cosmetics_Offer_09.06.2026.xlsx",
        "/Users/aliabdullah/Downloads/STOCK LIST PERFUMES 13.04.xlsx",
    ]
    for f in files:
        try:
            df = pd.read_excel(f, engine="openpyxl", header=None)
            r = estimate_xlsx_source_rows(df)
            print(f"{f.rsplit('/', 1)[-1]:50s} → count={r['count']:5d}  "
                  f"total={r['signals']['total_rows']:5d}  "
                  f"header_row={r['signals']['header_row']}  conf={r['confidence']}")
        except Exception as e:
            print(f"ERR {f}: {e}", file=sys.stderr)
