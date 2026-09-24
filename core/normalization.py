"""
Product canonical identity (Milestone 5 — normalization).

Fuzzy resolver so "Bailey's", "Baileys", "Baileys Original Irish Cream" collapse
into one canonical key when compared for historical / best-price purposes.

Also handles Unicode / accents deterministically so "Peñasol" and "Rémy Martin"
produce a stable ASCII key ("penasol", "remy martin") that never depends on
which service processed the string. The backend is the single source of truth
for the peer identity — the API surfaces `peer_group_id` on every offer, and
the frontend uses that value directly instead of re-hashing.

MVP: normalize (Unicode fold → lowercase → drop non-ascii-alnum → drop noise
words) and use the result as the join key. Not an edit-distance engine.
"""
import re
import unicodedata

_NOISE_TOKENS = {
    "the", "and", "&", "original", "classic", "edition", "premium",
    "int", "intl", "international",
}


_APOSTROPHES = "'’‘ʼ`´"  # ASCII, ’ ‘ ʼ ` ´

# Latin-alphabet ligatures that NFKD leaves alone (they're "letters", not
# compatibility forms). Expand them by hand so "Æther" doesn't drop to
# "ther" and collide with anything starting with "ther".
_LIGATURES = {
    "æ": "ae", "Æ": "AE",
    "œ": "oe", "Œ": "OE",
    "ø": "o",  "Ø": "O",
    "ß": "ss",
    "ð": "d",  "Ð": "D",
    "þ": "th", "Þ": "TH",
}


def _fold(s: str) -> str:
    """
    Deterministic Unicode → ASCII-ish fold.
      - Apostrophes (ASCII + curly + prime + backtick + acute) are stripped
        first so "Daniel's" survives as "daniels" rather than splitting on
        the punctuation.
      - Hand-expanded Latin ligatures (Æ→AE, Œ→OE, ß→ss, Ø→O, Ð→D, Þ→TH)
        because NFKD leaves those atomic.
      - NFKD decomposes accented chars into base + combining marks
        (é → e + combining-acute; ñ → n + combining-tilde).
      - Combining marks (category "Mn") are dropped.
      - Compatibility decomposition also flattens things like ﬁ → fi, ² → 2.
      - casefold() lower-cases across scripts.
    Non-Latin scripts (Cyrillic, CJK) survive as-is at this stage; the
    subsequent [^a-z0-9] filter drops them. That collision risk is
    accepted for MVP — the alternative (transliteration) is a bigger
    dependency for the marginal benefit of matching cross-script spellings.
    """
    if not s:
        return ""
    raw = str(s)
    raw = raw.translate({ord(a): None for a in _APOSTROPHES})
    for lig, repl in _LIGATURES.items():
        if lig in raw:
            raw = raw.replace(lig, repl)
    decomposed = unicodedata.normalize("NFKD", raw)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.casefold()


def canonical_brand(s: str | None) -> str:
    if not s:
        return ""
    v = _fold(s)
    v = re.sub(r"[^a-z0-9]+", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v


def canonical_product(s: str | None) -> str:
    if not s:
        return ""
    v = _fold(s)
    v = re.sub(r"\b(gbx|gb|nrf|rf|coded|not\s+found)\b", " ", v)
    v = re.sub(r"[^a-z0-9]+", " ", v)
    tokens = [t for t in v.split() if t and t not in _NOISE_TOKENS]
    return " ".join(tokens)


def canonical_key(brand: str | None, product_name: str | None,
                  unit_volume_ml: float | int | None,
                  units_per_case: float | int | None,
                  incoterm: str | None = None) -> str:
    """
    The one peer-group identity used everywhere: benchmarks lookup, records
    serialization, alert dedup. Deriving it in more than one place is how
    silent mismatches (Peñasol → pe asol on the backend vs peñasol on the
    frontend) creep in — hence a single implementation here plus a
    `peer_group_id` field the API emits per row.
    """
    return "|".join([
        canonical_brand(brand),
        canonical_product(product_name),
        str(int(unit_volume_ml or 0)),
        str(int(units_per_case or 0)),
        (incoterm or "").upper(),
    ])


if __name__ == "__main__":
    # Smoke test — different spellings of the same product should collapse
    cases = [
        ("Baileys", "Original Irish Cream", 700, 6, "EXW"),
        ("Bailey's", "Original Irish Cream", 700, 6, "EXW"),
        ("Baileys", "The Original Irish Cream", 700, 6, "EXW"),
        ("BAILEYS", "Original Irish Cream Classic", 700, 6, "EXW"),
    ]
    for c in cases:
        print(canonical_key(*c))
