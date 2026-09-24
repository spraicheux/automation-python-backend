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


def _normalize_location(s: str | None) -> str:
    """
    Fold a location string the same way we fold brands, then take the first
    proper-noun token so "EXW Rotterdam" / "Rotterdam Port" / "rotterdam"
    collapse to "rotterdam" but "Rotterdam" and "Dubai" stay separate.
    """
    if not s:
        return ""
    v = _fold(s)
    if v.strip() in ("not found", "none", "unknown", "n a", "na", ""):
        return ""
    v = re.sub(r"[^a-z0-9]+", " ", v).strip()
    # Drop leading incoterm-like tokens someone shoved into the location field.
    tokens = v.split()
    while tokens and tokens[0] in {"exw", "fob", "cif", "dap", "ddp", "cfr", "cpt", "fca", "ex"}:
        tokens = tokens[1:]
    return tokens[0] if tokens else ""


def _norm_incoterm(s: str | None) -> str:
    """Normalise incoterm; unknown / 'Not Found' collapses to empty."""
    if not s:
        return ""
    v = str(s).upper().strip()
    if v in {"NOT FOUND", "NONE", "UNKNOWN", "N/A", "NA", ""}:
        return ""
    return v


def _norm_num(v, decimals: int = 0) -> str:
    """Numeric field to a canonical string. Empty for None so blanks collapse."""
    if v in (None, ""):
        return ""
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return ""


def _norm_text(s: str | None) -> str:
    """Short text field (vintage, age statement, edition) folded to bare letters."""
    if not s:
        return ""
    v = _fold(s)
    v = re.sub(r"[^a-z0-9]+", " ", v).strip()
    return v


def product_family_id(brand: str | None, product_name: str | None) -> str:
    """
    LEVEL A — FAMILY identity. Answers: "is this the same product family,
    regardless of size / ABV / edition?"
      Hennessy VS 350ml and Hennessy VS 700ml → same family
      Dior Sauvage EDT and Dior Sauvage EDP  → same family
    Used for: search, cross-size roll-up, storytelling ("all Hennessy VS
    offers this week"). NEVER for dedup, Best Price, or benchmarking —
    those need the SKU-level identity below.
    """
    return f"{canonical_brand(brand)}|{canonical_product(product_name)}"


# Historical alias — was used everywhere pre-split. Keep pointing at the
# family identity so old call sites don't silently switch meaning.
canonical_product_id = product_family_id


def sku_identity(
    brand: str | None,
    product_name: str | None,
    unit_volume_ml: float | int | None = None,
    units_per_case: float | int | None = None,
    alcohol_percent: float | int | None = None,
    vintage: str | None = None,
    age_statement: str | None = None,
    edition: str | None = None,
    ean_code: str | None = None,
    # Reserved for perfumes/cosmetics — same signature, per-category discriminators
    perfume_format: str | None = None,  # EDT | EDP | Parfum | Cologne
    retail_state: str | None = None,    # retail | tester | sample
    gender: str | None = None,
    shade: str | None = None,
) -> str:
    """
    LEVEL B — SKU / physical-product identity. Answers: "is this the same
    physical product, regardless of who's selling it or under what terms?"
    Dedup and 'same SKU across suppliers' comparisons must use THIS key,
    NOT the family key — otherwise two genuinely different SKUs (Hennessy
    VS 350ml vs 700ml) collapse together on the Best-Prices roll-up.

    W&S discriminators: volume, pack, ABV, vintage, age statement, edition,
    EAN. Perfume/cosmetic discriminators added by the same function so
    the split logic doesn't have to know which category it's serving.
    """
    return "|".join([
        product_family_id(brand, product_name),
        _norm_num(unit_volume_ml, 0),
        _norm_num(units_per_case, 0),
        _norm_num(alcohol_percent, 1),
        _norm_text(vintage),
        _norm_text(age_statement),
        _norm_text(edition),
        _norm_text(ean_code),
        _norm_text(perfume_format),
        _norm_text(retail_state),
        _norm_text(gender),
        _norm_text(shade),
    ])


def is_peer_group_qualified(incoterm: str | None, location: str | None) -> bool:
    """
    A peer comparison is only "qualified" (i.e. safe to raise a NEW XM LOW
    or trusted Best Price signal from) when both the commercial term
    (incoterm) and the origin (location) are known. Two rows both missing
    incoterm technically hash to the same peer key, but they are NOT a
    trustworthy apples-to-apples comparison — they might come from any
    incoterm at any location. Callers must downgrade the signal for
    unqualified peers.
    """
    return bool(_norm_incoterm(incoterm)) and bool(_normalize_location(location))


def peer_group_id(
    brand: str | None,
    product_name: str | None,
    unit_volume_ml: float | int | None = None,
    units_per_case: float | int | None = None,
    incoterm: str | None = None,
    location: str | None = None,
    alcohol_percent: float | int | None = None,
    vintage: str | None = None,
    age_statement: str | None = None,
    edition: str | None = None,
    ean_code: str | None = None,
    perfume_format: str | None = None,
    retail_state: str | None = None,
    gender: str | None = None,
    shade: str | None = None,
) -> str:
    """
    LEVEL C — COMMERCIAL peer-group identity. Answers: "are these two
    offers genuinely comparable for Best Price / historical benchmarking?"
    Builds on the SKU key (level B) by adding the terms of delivery.

    Includes every discriminator that could make two offers not apples-to-
    apples on a trading desk:
      - unit_volume_ml, units_per_case   → pack format
      - incoterm                         → term of delivery
      - location                         → EXW Rotterdam ≠ EXW Dubai
      - alcohol_percent (0.1° bin)       → 40% ≠ 43% is a different SKU
      - vintage                          → 2018 ≠ 2019 for wines
      - age_statement                    → 12YO ≠ 18YO for whisky
      - edition                          → Special / Limited / Reserve

    Multi-category will add: EAN, format (EDT/EDP/Parfum), tester/retail,
    size, gender, shade. Perfume-safe fields fold via `_norm_text` on the
    same principle — never store the raw string in the key, always the fold.

    Blank fields collapse (empty string), so a row missing ABV still peers
    with another row missing ABV. But a peer group that contains any row
    missing incoterm or location is NOT considered qualified for trusted
    signals — see is_peer_group_qualified(). The dashboard downgrades
    NEW XM LOW and Best Price alerts on unqualified peers.
    """
    return "|".join([
        sku_identity(
            brand, product_name,
            unit_volume_ml=unit_volume_ml,
            units_per_case=units_per_case,
            alcohol_percent=alcohol_percent,
            vintage=vintage,
            age_statement=age_statement,
            edition=edition,
            ean_code=ean_code,
            perfume_format=perfume_format,
            retail_state=retail_state,
            gender=gender,
            shade=shade,
        ),
        _norm_incoterm(incoterm),
        _normalize_location(location),
    ])


# ── Backward-compat shim ────────────────────────────────────────────────────
# Older call sites still pass the pre-split (brand, name, vol, upc, incoterm)
# signature. Keep the function name resolving to peer_group_id so nothing
# silently disagrees on identity mid-migration.
def canonical_key(brand: str | None, product_name: str | None,
                  unit_volume_ml: float | int | None,
                  units_per_case: float | int | None,
                  incoterm: str | None = None,
                  location: str | None = None,
                  alcohol_percent: float | int | None = None,
                  vintage: str | None = None) -> str:
    """Alias for peer_group_id. Prefer peer_group_id directly in new code."""
    return peer_group_id(
        brand, product_name,
        unit_volume_ml=unit_volume_ml,
        units_per_case=units_per_case,
        incoterm=incoterm,
        location=location,
        alcohol_percent=alcohol_percent,
        vintage=vintage,
    )


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
