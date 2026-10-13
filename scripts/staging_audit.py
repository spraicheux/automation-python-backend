"""
R-16 Staging audit — before/after import diff for all source files.

Shows expected vs imported row counts for every source file and flags any
import_incomplete entries. Run this after re-ingesting a file to verify the
pipeline captured the expected number of rows.

Usage:
    python scripts/staging_audit.py
    python scripts/staging_audit.py --base-url http://localhost:8000
    python scripts/staging_audit.py --supplier "Maison Kurtz"
    python scripts/staging_audit.py --incomplete-only

Output: a table of source files with PASS / INCOMPLETE / UNKNOWN status.
"""
import argparse
import os
import sys
from datetime import datetime

try:
    import requests
except ImportError:
    print("requests not installed — run: pip install requests")
    sys.exit(1)

BASE_URL = os.getenv("BASE_URL", "https://loxo-offer-api.azurewebsites.net")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "valid-token")

# ANSI colours (disabled when output is piped)
_tty = sys.stdout.isatty()
GREEN  = "\033[32m" if _tty else ""
YELLOW = "\033[33m" if _tty else ""
RED    = "\033[31m" if _tty else ""
GREY   = "\033[90m" if _tty else ""
BOLD   = "\033[1m"  if _tty else ""
RESET  = "\033[0m"  if _tty else ""


def _fetch_sources(base_url: str, supplier_filter: str | None) -> list[dict]:
    session = requests.Session()
    session.headers["x-admin-token"] = ADMIN_TOKEN

    all_sources: list[dict] = []
    skip = 0
    limit = 100

    while True:
        r = session.get(f"{base_url}/api/sources",
                        params={"skip": skip, "limit": limit},
                        timeout=20)
        r.raise_for_status()
        body = r.json()
        batch = body.get("sources", [])
        if not batch:
            break
        all_sources.extend(batch)
        if len(all_sources) >= body.get("total", 0):
            break
        skip += limit

    if supplier_filter:
        lf = supplier_filter.lower()
        all_sources = [s for s in all_sources
                       if lf in (s.get("supplier_name") or "").lower()
                       or lf in (s.get("source_filename") or "").lower()]

    return all_sources


def _status(source: dict) -> tuple[str, str]:
    exp = source.get("expected_row_count")
    imp = source.get("imported_row_count")
    incomplete = source.get("import_incomplete", False)

    if exp is None and imp is None:
        return "UNKNOWN", GREY
    if incomplete or (exp is not None and imp is not None and imp < exp):
        return "INCOMPLETE", RED
    if exp is not None and imp is not None and imp == exp:
        return "PASS", GREEN
    if exp is None:
        return "NO ESTIMATE", GREY
    return "PASS", GREEN


def _render_table(sources: list[dict], incomplete_only: bool) -> None:
    rows = []
    counts = {"PASS": 0, "INCOMPLETE": 0, "UNKNOWN": 0, "NO ESTIMATE": 0}

    for s in sources:
        status, colour = _status(s)
        counts[status] = counts.get(status, 0) + 1
        if incomplete_only and status not in ("INCOMPLETE",):
            continue
        exp = s.get("expected_row_count")
        imp = s.get("imported_row_count")
        delta = ""
        if exp is not None and imp is not None and exp != imp:
            delta = f"  Δ {exp - imp:+d}"
        rows.append({
            "status": status,
            "colour": colour,
            "filename": (s.get("source_filename") or "")[:40],
            "supplier": (s.get("supplier_name") or "—")[:28],
            "expected": str(exp) if exp is not None else "—",
            "imported": str(imp) if imp is not None else "—",
            "delta": delta,
            "date": (s.get("created_at") or "")[:10],
        })

    # Header
    print(f"\n{BOLD}{'STATUS':<12}{'SUPPLIER':<30}{'EXPECTED':>10}{'IMPORTED':>10}  DELTA    DATE        FILENAME{RESET}")
    print("─" * 110)

    for r in rows:
        c = r["colour"]
        print(
            f"{c}{r['status']:<12}{RESET}"
            f"{r['supplier']:<30}"
            f"{r['expected']:>10}"
            f"{r['imported']:>10}"
            f"  {r['delta']:<8} "
            f"{r['date']:<12}"
            f"{GREY}{r['filename']}{RESET}"
        )

    print("─" * 110)
    total = len(sources)
    print(
        f"\n{BOLD}Summary:{RESET} {total} files — "
        f"{GREEN}{counts.get('PASS',0)} PASS{RESET}  "
        f"{RED}{counts.get('INCOMPLETE',0)} INCOMPLETE{RESET}  "
        f"{GREY}{counts.get('UNKNOWN',0) + counts.get('NO ESTIMATE',0)} no estimate{RESET}\n"
    )

    if counts.get("INCOMPLETE", 0):
        print(
            f"{YELLOW}Action:{RESET} {counts['INCOMPLETE']} file(s) have fewer imported rows "
            f"than expected.\n"
            f"  • Check the import log for parse errors on those files.\n"
            f"  • If the delta is structural duplicates (same EAN+price collapsed),\n"
            f"    reconcile with:\n"
            f"    POST {BASE_URL}/admin/reconcile-expected-against-imported"
            f"?source_filename=<name>&apply=true\n"
            f"    (add x-admin-token header)\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=BASE_URL,
                        help="API base URL (default: $BASE_URL or Azure)")
    parser.add_argument("--supplier", default=None,
                        help="Filter by supplier name or filename (substring, case-insensitive)")
    parser.add_argument("--incomplete-only", action="store_true",
                        help="Show only INCOMPLETE files")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    print(f"{GREY}Fetching sources from {base} …{RESET}", end="", flush=True)

    try:
        sources = _fetch_sources(base, args.supplier)
    except requests.RequestException as e:
        print(f"\n{RED}Error: {e}{RESET}")
        sys.exit(1)

    print(f"\r{GREY}Fetched {len(sources)} source file(s).{RESET}         ")
    print(f"Run at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    if not sources:
        print(f"{YELLOW}No source files found{' for that supplier' if args.supplier else ''}.{RESET}")
        return

    _render_table(sources, incomplete_only=args.incomplete_only)


if __name__ == "__main__":
    main()
