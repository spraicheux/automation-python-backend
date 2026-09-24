"""
One-shot backfill for two client-reported bugs on already-ingested rows:

  1. €/case bug — the LLM sometimes mapped a source "Total Price" column
     (qty × unit_price) into price_per_case. Detected by:
        price_per_case ≈ quantity_case × price_per_unit
        AND units_per_case is None / 1
     → clear price_per_case & price_per_case_eur.

  2. Supplier detection — rows where supplier_name is null / "Not Found"
     get re-derived from the filename (uppercase 3-6 char code at the end,
     e.g. "MIX SPIRITS DISCOUNT SALE HNS.xlsm" → "HNS") and sender_email.

Prints a summary and asks before writing. Run with --apply to commit.
"""
import argparse
import re
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

from core.deterministic_rules import detect_supplier_from_metadata
from models.offer_item import OfferItemDB
from models.source_file import SourceFileDB  # noqa: F401 — needed for mapper


def rows_needing_case_fix(db) -> list:
    """Rows where price_per_case looks like the lot total."""
    fixups = []
    q = db.query(OfferItemDB).filter(
        OfferItemDB.price_per_case.isnot(None),
        OfferItemDB.price_per_unit.isnot(None),
        OfferItemDB.quantity_case.isnot(None),
    ).all()
    for r in q:
        upc = r.units_per_case
        qc = r.quantity_case
        ppc = r.price_per_case
        ppu = r.price_per_unit
        if not (ppc and ppu and qc):
            continue
        if upc not in (None, 1, 1.0):
            continue
        if qc <= 1:
            continue
        expected_total = qc * ppu
        if expected_total <= 0:
            continue
        if abs(ppc - expected_total) / expected_total < 0.02:
            fixups.append(r)
    return fixups


def rows_needing_supplier_fix(db) -> list:
    """Rows where supplier is null/Not Found but we can derive it from metadata."""
    fixups = []
    q = db.query(OfferItemDB).filter(
        (OfferItemDB.supplier_name.is_(None)) |
        (OfferItemDB.supplier_name == "") |
        (OfferItemDB.supplier_name.ilike("not found"))
    ).all()
    for r in q:
        supplier = detect_supplier_from_metadata(
            text="",
            source_filename=r.source_filename,
            sender_email=r.sender_email,
        )
        if supplier:
            fixups.append((r, supplier))
    return fixups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually write changes")
    args = ap.parse_args()

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL not set — put it in .env or the shell env.")
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    db = Session()

    print("== €/case sanity backfill ==")
    case_fixups = rows_needing_case_fix(db)
    print(f"Rows to fix: {len(case_fixups)}")
    for r in case_fixups[:15]:
        print(f"  {r.uid[:12]} | {r.brand} {r.product_name} | "
              f"qty={r.quantity_case} × unit={r.price_per_unit} = "
              f"total={r.price_per_case}  (upc={r.units_per_case})")
    if len(case_fixups) > 15:
        print(f"  … and {len(case_fixups) - 15} more")

    print()
    print("== Supplier backfill ==")
    supplier_fixups = rows_needing_supplier_fix(db)
    print(f"Rows to fix: {len(supplier_fixups)}")
    for r, sup in supplier_fixups[:15]:
        print(f"  {r.uid[:12]} | {r.source_filename} | supplier → {sup}")
    if len(supplier_fixups) > 15:
        print(f"  … and {len(supplier_fixups) - 15} more")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to write.")
        return

    for r in case_fixups:
        r.price_per_case = None
        r.price_per_case_eur = None
    for r, sup in supplier_fixups:
        r.supplier_name = sup

    db.commit()
    print(f"\nApplied: {len(case_fixups)} case-price clears, "
          f"{len(supplier_fixups)} supplier assignments.")


if __name__ == "__main__":
    main()
