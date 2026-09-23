"""
Product canonical identity (Milestone 5 — normalization).

Fuzzy resolver so "Bailey's", "Baileys", "Baileys Original Irish Cream" collapse
into one canonical key when compared for historical / best-price purposes.

MVP approach: normalize the string (strip apostrophes, lowercase, collapse
whitespace, drop common noise words) and use it as the join key. Not a fully
fuzzy edit-distance engine — those come with cost we don't need yet.
"""
import re

_NOISE_TOKENS = {
    "the", "and", "&", "original", "classic", "edition", "premium",
    "int", "intl", "international",
}


def canonical_brand(s: str | None) -> str:
    if not s:
        return ""
    v = s.lower()
    v = v.replace("'", "").replace("'", "").replace("’", "")
    v = re.sub(r"[^a-z0-9]+", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v


def canonical_product(s: str | None) -> str:
    if not s:
        return ""
    v = s.lower()
    v = v.replace("'", "").replace("'", "").replace("’", "")
    v = re.sub(r"\b(gbx|gb|nrf|rf|coded|not\s+found)\b", " ", v)
    v = re.sub(r"[^a-z0-9]+", " ", v)
    tokens = [t for t in v.split() if t and t not in _NOISE_TOKENS]
    return " ".join(tokens)


def canonical_key(brand: str | None, product_name: str | None,
                  unit_volume_ml: float | int | None,
                  units_per_case: float | int | None,
                  incoterm: str | None = None) -> str:
    """The one key that must be used everywhere for peer comparison."""
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
