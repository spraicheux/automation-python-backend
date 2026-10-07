"""
Brand gazetteer + longest-prefix matcher.

Used as a deterministic fallback when a source file has no brand column
and the LLM returns brand=null. The matcher scans the product_name for
the longest known brand token that starts the normalized string.

Normalization is diacritic-insensitive, ampersand-tolerant and
whitespace-collapsed, so entries written as "Estée Lauder" also match
"Estee Lauder" and "Yves Saint Laurent" also matches "YSL" via the
alias table.

The gazetteer is curated, not scraped — one row per canonical brand
and optional aliases that resolve to it. Adding new brands is a plain
append to BRANDS (canonical form) or ALIASES (short form → canonical).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional


# Canonical brand names. Multi-word entries are supported. Entries that
# are a prefix of another entry MUST appear in both so longest-wins can
# find them (e.g. "Armani" and "Giorgio Armani" are both listed).
#
# Keep this list sorted by category for maintainability. The matcher
# rebuilds its length-descending index on first use, so order here is
# cosmetic only.
BRANDS: list[str] = [
    # ── Perfume houses — niche ──────────────────────────────────────
    "Acqua di Parma", "Bond No. 9", "Bond No 9",
    "Al Haramain", "Lattafa", "Afnan", "Rasasi", "Ard Al Zaafaran",
    "Jo Malone", "Jo Malone London",
    "Maison Francis Kurkdjian", "Maison Margiela", "Mancera",
    "Montale", "Nasomatto", "Nishane", "Parfums de Marly",
    "Penhaligon's", "Tiziana Terenzi", "Xerjoff",
    "By Kilian", "Clive Christian", "Creed", "Diptyque",
    "Frederic Malle", "Juliette Has a Gun", "Memo Paris",
    "Roja Dove", "Serge Lutens", "Amouage", "Byredo",
    # ── Perfume houses — designer ───────────────────────────────────
    "Giorgio Armani", "Armani",
    "Yves Saint Laurent", "Christian Dior", "Dior",
    "Dolce & Gabbana", "Dolce and Gabbana",
    "Jean Paul Gaultier", "Paco Rabanne", "Thierry Mugler",
    "Carolina Herrera", "Marc Jacobs", "Narciso Rodriguez",
    "Tom Ford", "Hugo Boss", "Boss",
    "Viktor & Rolf", "Viktor and Rolf",
    "Victor & Rolf",  # common misspelling seen in supplier files
    "Calvin Klein", "Ralph Lauren", "Elie Saab",
    "Issey Miyake", "Bvlgari", "Bulgari",
    "Guerlain", "Chanel", "Lancome", "Lancôme",
    "Givenchy", "Kenzo", "Mont Blanc", "Montblanc",
    "Burberry", "Versace", "Prada", "Gucci",
    "Salvatore Ferragamo", "Ferragamo",
    "Davidoff", "Azzaro", "Loewe", "Cartier",
    "Yves Rocher", "Nina Ricci", "Mugler",
    "Police", "Shakira", "Taylor Of London", "Tabac Original",
    "United Colors Of Benetton", "Benetton",
    # ── Cosmetics houses ────────────────────────────────────────────
    "Charlotte Tilbury", "Estée Lauder", "Estee Lauder",
    "Clinique", "La Mer", "La Prairie", "La Roche-Posay",
    "Shiseido", "Nars", "Benefit", "Benefit Cosmetics",
    "Urban Decay", "MAC", "MAC Cosmetics", "Bobbi Brown",
    "Too Faced", "Fenty Beauty", "Fenty Skin",
    "Rare Beauty", "Huda Beauty", "Pat McGrath Labs",
    "Beauty of Joseon",
    "Anastasia Beverly Hills", "Dermalogica", "Drunk Elephant",
    "Kiehl's", "L'Occitane", "L Occitane",
    "Nuxe", "Caudalie",
    "Biotherm", "Moroccanoil", "Elizabeth Arden", "Lancaster",
    "Rituals", "Hermes", "Hermès", "Dr. Barbara Sturm", "Dr. Hauschka",
    "Dr. Jart+", "Jo Malone London", "Jo Malone",
    "Sisley Paris", "Sisley", "Clarins",
    "Yves Rocher", "The Ordinary", "Paula's Choice",
    "Tatcha", "Glossier", "Milk Makeup", "Rare Beauty",
    "L'Oreal Paris", "L'Oréal Paris", "L'Oréal", "L'Oreal",
    "Loreal", "Loreal Paris",
    "Maybelline", "Maybelline New York", "Revlon",
    "Rimmel", "Rimmel London", "NYX", "NYX Professional Makeup",
    "Essie", "OPI", "Sally Hansen",
    "Pat McGrath", "Hourglass", "ILIA", "Ilia Beauty",
    "Natasha Denona", "Vieve", "Westman Atelier",
    # ── Numeric-leading brand (actually seen in supplier data) ──────
    "4711",
]


# Alias → canonical. Normalized match on both sides. Add new aliases as
# their short form (left) to the canonical brand (right). The canonical
# must already be in BRANDS.
ALIASES: dict[str, str] = {
    "YSL": "Yves Saint Laurent",
    "D&G": "Dolce & Gabbana",
    "D & G": "Dolce & Gabbana",
    "ADP": "Acqua di Parma",
    "JPG": "Jean Paul Gaultier",
    "CT": "Charlotte Tilbury",
    "EL": "Estée Lauder",
    "ABH": "Anastasia Beverly Hills",
    "V&R": "Viktor & Rolf",
    "VR": "Viktor & Rolf",
    "TF": "Tom Ford",
    "CK": "Calvin Klein",
    "RL": "Ralph Lauren",
    "BB": "Bobbi Brown",
    "MAC": "MAC Cosmetics",
}


_WS_RE = re.compile(r"\s+")
_PUNCT_STRIP_RE = re.compile(r"[.,/\\]")


def _normalize(s: str) -> str:
    """Normalize for comparison. Lowercase, strip diacritics, collapse
    whitespace, normalize ampersand to the word 'and', treat apostrophe
    as a space (so "L'Oreal" and "L Oreal" both normalize to "l oreal"
    — supplier files use both), strip stray punctuation (dots, commas,
    slashes)."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = s.replace("&", " and ")
    s = s.replace("'", " ").replace("'", " ").replace("'", " ")
    s = _PUNCT_STRIP_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


_INDEX: list[tuple[str, str]] | None = None  # (normalized_prefix, canonical)


def _ensure_index() -> list[tuple[str, str]]:
    """Build and cache the length-descending normalized index. Longer
    entries come first so a product name starting with "Yves Saint
    Laurent" matches that entry before falling back to "Yves"."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    entries: dict[str, str] = {}
    for brand in BRANDS:
        norm = _normalize(brand)
        if norm and norm not in entries:
            entries[norm] = brand
    for alias, canonical in ALIASES.items():
        norm = _normalize(alias)
        if norm and norm not in entries:
            entries[norm] = canonical
    _INDEX = sorted(entries.items(), key=lambda kv: (-len(kv[0]), kv[0]))
    return _INDEX


def extract_brand_prefix(product_name: Optional[str]) -> Optional[tuple[str, str]]:
    """Return (canonical_brand, stripped_product_name) if the product
    name starts with a known brand (longest wins), else None.

    The returned product_name has the brand prefix removed and leading
    separators stripped. The caller is responsible for deciding whether
    to overwrite the row's brand field (typically only when it is empty).

    Apostrophes in the original ("L'Occitane") expand to spaces in the
    normalized form ("l occitane"), so the raw-token count can be
    smaller than the normalized-token count. The stripper walks raw
    tokens, re-normalizes each one, and consumes until the normalized
    prefix has been fully covered — then returns the remaining raw
    tokens as the stripped product name.
    """
    if not product_name:
        return None
    norm = _normalize(product_name)
    if not norm:
        return None
    for prefix, canonical in _ensure_index():
        if norm == prefix or norm.startswith(prefix + " "):
            # How many normalized tokens must be covered.
            want_norm_tokens = prefix.split(" ")
            raw_tokens = product_name.split()
            covered: list[str] = []
            consumed_raw = 0
            for raw_tok in raw_tokens:
                raw_norm = _normalize(raw_tok)
                if not raw_norm:
                    consumed_raw += 1
                    continue
                covered.extend(raw_norm.split(" "))
                consumed_raw += 1
                if covered[:len(want_norm_tokens)] == want_norm_tokens:
                    break
            remainder = " ".join(raw_tokens[consumed_raw:]).strip()
            return canonical, remainder
    return None


if __name__ == "__main__":
    # Spot-check the matcher against examples from real supplier files.
    cases = [
        ("Yves Saint Laurent Libre EDP 50ml", "Yves Saint Laurent"),
        ("YSL Libre EDP 50ml", "Yves Saint Laurent"),
        ("Beauty of Joseon Calming Serum", "Beauty of Joseon"),
        ("4711 Echt Kolnisch Wasser 100ml", "4711"),
        ("Armani Eccentrico Mascara", "Armani"),
        ("Giorgio Armani Si Passione EDP 100ml", "Giorgio Armani"),
        ("Dolce & Gabbana Light Blue EDT 100ml", "Dolce & Gabbana"),
        ("Viktor Rolf Flowerbomb EDP 50ml", None),  # missing & — matcher honest
        ("Victor & Rolf Flowerbomb EDP 50ml", "Victor & Rolf"),
        ("Acqua di Parma Blu Mediterraneo 100ml", "Acqua di Parma"),
        ("Bond No. 9 Central Park 100ml", "Bond No. 9"),
        ("Random No-Brand Product 100ml", None),
        ("", None),
        (None, None),
    ]
    for name, expected in cases:
        got = extract_brand_prefix(name)
        ok = (got[0] == expected) if (got and expected) else (got is None and expected is None)
        mark = "✓" if ok else "✗"
        print(f"{mark} {name!r:60s} → {got}")
