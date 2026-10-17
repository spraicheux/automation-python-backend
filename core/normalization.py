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


def _norm_upc(v) -> str:
    """Units-per-case key component. A loose pack (None / 0) and a single-unit
    pack (1) are the same physical thing, so they must collapse to one key —
    otherwise the same-EAN offers (e.g. NARS 30ml, one supplier with upc=null
    and one with upc=1) split into two peer groups and a historical offer is
    dropped from Best Prices. Any pack of >1 keeps its real count.
    """
    try:
        n = float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        n = None
    if n is None or n <= 1:
        return "1"
    return f"{n:.0f}"


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


def ean_key(ean_code: str | None) -> str:
    """
    Normalise an EAN / GTIN to bare digits. Suppliers write them in many
    forms — "3348901234567", "3348901.234567", "EAN 3348901234567". Keep
    only the digits so all forms fold to the same slug; return "" when
    nothing usable is present, so downstream can treat "no EAN provided"
    as a fallback path (see the resolver logic in the Best Prices API).
    """
    if not ean_code:
        return ""
    digits = re.sub(r"\D+", "", str(ean_code))
    # A real EAN is 8, 12, 13 or 14 digits (EAN-8 / UPC / EAN-13 / GTIN-14).
    # Anything else is data we don't trust — treat as unknown.
    if len(digits) not in (8, 12, 13, 14):
        return ""
    # Validate the GS1 check digit (last digit) for EAN-8, EAN-13, GTIN-14.
    # EAN/GTIN check digit: sum alternating weights 3 and 1 from right-to-left
    # on all but the last digit; check digit = (10 - (sum % 10)) % 10.
    total = 0
    for i, d in enumerate(reversed(digits[:-1])):
        total += int(d) * (3 if i % 2 == 0 else 1)
    expected = (10 - (total % 10)) % 10
    if int(digits[-1]) != expected:
        return ""
    return digits


def sku_identity(
    brand: str | None,
    product_name: str | None,
    unit_volume_ml: float | int | None = None,
    units_per_case: float | int | None = None,
    alcohol_percent: float | int | None = None,
    vintage: str | None = None,
    age_statement: str | None = None,
    edition: str | None = None,
    # Multi-category (Phase 3 M1) — category comes first so a perfume
    # can never accidentally share an SKU with a cosmetic that has the
    # same brand + product name.
    category_slug: str | None = None,
    # Perfume / cosmetics discriminators
    perfume_format: str | None = None,  # EDT | EDP | Parfum | Cologne
    retail_state: str | None = None,    # retail | tester | sample
    gender: str | None = None,
    shade: str | None = None,
    product_type: str | None = None,    # cosmetics: lipstick | foundation | …
    size_weight_g: float | int | None = None,  # cosmetics (solids)
    range_name: str | None = None,      # e.g. Dior "Sauvage"
) -> str:
    """
    LEVEL B — SKU / physical-product identity from ATTRIBUTES ONLY.
    Answers: "is this the same physical product on paper, regardless of
    who's selling it or under what terms?" Dedup and "same SKU across
    suppliers" comparisons use this key ANDed with the EAN resolver
    (see ean_key + best-prices API): two rows are the same SKU when
    this key matches AND their EANs are compatible (both blank, one
    blank, or the same digits). Rows with contradicting EANs stay
    separate and get a needs-review flag.

    EAN is DELIBERATELY NOT in the hash. A supplier who left EAN blank
    would otherwise never match against a supplier who filled it in.
    Keeping EAN out of the key and reconciling it separately is what
    lets us honour the client's "one EAN known → fall back to full
    attributes" rule.
    """
    return "|".join([
        (category_slug or "").lower().strip(),  # first: split categories cleanly
        product_family_id(brand, product_name),
        _norm_text(range_name),
        _norm_num(unit_volume_ml, 0),
        _norm_upc(units_per_case),
        _norm_num(alcohol_percent, 1),
        _norm_text(vintage),
        _norm_text(age_statement),
        _norm_text(edition),
        _norm_text(perfume_format),
        _norm_text(retail_state),
        _norm_text(gender),
        _norm_text(shade),
        _norm_text(product_type),
        _norm_num(size_weight_g, 1),
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
    category_slug: str | None = None,
    perfume_format: str | None = None,
    retail_state: str | None = None,
    gender: str | None = None,
    shade: str | None = None,
    product_type: str | None = None,
    size_weight_g: float | int | None = None,
    range_name: str | None = None,
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
    # EAN sits in the peer key so two rows with genuinely different EANs
    # (different SKUs that share every text attribute — reformulations,
    # regional variants) don't get benchmarked against each other. Rows
    # without EAN slot into the ""-EAN bucket and rely on the attribute
    # match; the Best Prices resolver then reconciles blank-vs-known EAN
    # per the client's fallback rule (one EAN known → attributes decide).
    return "|".join([
        sku_identity(
            brand, product_name,
            unit_volume_ml=unit_volume_ml,
            units_per_case=units_per_case,
            alcohol_percent=alcohol_percent,
            vintage=vintage,
            age_statement=age_statement,
            edition=edition,
            category_slug=category_slug,
            perfume_format=perfume_format,
            retail_state=retail_state,
            gender=gender,
            shade=shade,
            product_type=product_type,
            size_weight_g=size_weight_g,
            range_name=range_name,
        ),
        ean_key(ean_code),
        _norm_incoterm(incoterm),
        _normalize_location(location),
    ])


def peer_group_id_from_ean(
    ean_code: str | None,
    incoterm: str | None = None,
    location: str | None = None,
) -> str | None:
    """
    EAN-primary peer group key. When a valid EAN is present, two records
    with the same EAN, incoterm, and delivery location are the same peer
    regardless of how each supplier formatted the product name. Returns
    None when no valid EAN is provided — callers fall back to the full
    attribute-based peer_group_id.
    """
    ean = ean_key(ean_code)
    if not ean:
        return None
    return f"ean:{ean}|{_norm_incoterm(incoterm)}|{_normalize_location(location)}"


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
