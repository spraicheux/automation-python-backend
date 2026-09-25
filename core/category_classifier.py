"""
Category classifier — Phase 3, Milestone 1.

Detects one of three top-level categories from a supplier document:

  wines_spirits | perfumes | cosmetics

Deterministic first: strong keyword signals from the filename and the
document header. Ambiguous cases return None and the LLM decides at
extraction time, so this classifier is precision-biased — it says "I
don't know" rather than mis-labelling. The category, once determined,
is normalised to a machine slug that identity keys, the dashboard
filter and the per-category prompt all consume, so a hand-typed
"Wine & Spirits" collapses with "wines_spirits" everywhere.

Architecture note. Adding a new category later is: (a) add its slug
here, (b) list its strong keywords, (c) extend the extraction prompt
for its category-specific fields, (d) extend sku_identity to include
those fields. Nothing else in the pipeline needs to know.
"""
from __future__ import annotations

import re
from typing import Optional

# Canonical machine slugs. Anything the LLM emits or a human hand-types
# folds through _normalize_category into one of these.
WINES_SPIRITS = "wines_spirits"
PERFUMES      = "perfumes"
COSMETICS     = "cosmetics"

_CANONICAL_SLUGS = {WINES_SPIRITS, PERFUMES, COSMETICS}


# ── Strong deterministic keywords ────────────────────────────────────
# Each list is ordered by specificity: rare / narrow terms first so they
# outrank generic ones ("cognac" wins over "spirits", "eau de toilette"
# wins over "toilette"). Terms are matched case-insensitively as whole
# tokens against a fold that keeps only [a-z0-9 ] — so "EDT" matches
# "EDT" but also "e.d.t." and "eau de toilette" as a phrase.

_PERFUME_TOKENS = [
    "eau de toilette", "eau de parfum", "eau de cologne", "eau fraiche",
    "extrait de parfum", "extrait", "parfum", "cologne", "fragrance",
    "fragrances", "perfume", "perfumery",
    "edt", "edp", "edc", "edf",
    "tester", "tester box", "vaporisateur", "vapo", "spray",
]

_COSMETICS_TOKENS = [
    "lipstick", "lip gloss", "lip balm", "lip liner",
    "foundation", "concealer", "primer", "powder",
    "mascara", "eyeliner", "eye shadow", "eyeshadow",
    "blush", "bronzer", "highlighter",
    "nail polish", "nail lacquer", "nail",
    "shampoo", "conditioner", "hair mask", "hair serum",
    "moisturiser", "moisturizer", "serum", "cream", "toner",
    "lotion", "cleanser", "scrub", "peel", "mask",
    "cosmetics", "beauty", "skincare", "skin care", "makeup", "make up",
    "haircare", "hair care", "body care", "bodycare",
]

_WINES_SPIRITS_TOKENS = [
    # Categories
    "whisky", "whiskey", "bourbon", "scotch",
    "vodka", "gin", "rum", "tequila", "mezcal",
    "cognac", "armagnac", "brandy",
    "liqueur", "aperitif", "digestif",
    "champagne", "prosecco", "cava", "crémant",
    "red wine", "white wine", "rose wine", "rosé wine", "sparkling wine",
    "wines", "spirits", "vins", "vins et spiritueux",
    # Format hints
    "abv", "alcohol", "alc/vol", "vintage", "aoc", "aop",
    "cs cognac", "cs whisky", "cs vodka",
    # Common brands strongly imply category
    "hennessy", "remy martin", "rémy martin", "martell", "camus",
    "jack daniels", "jack daniel", "johnnie walker",
    "chivas", "grand marnier", "grey goose", "absolut", "smirnoff",
    "bacardi", "baileys", "jägermeister", "jagermeister",
]


def _fold(text: str) -> str:
    """Lowercase, strip non-word chars, collapse spaces so token matching
    is punctuation- and accent-insensitive on the ASCII side."""
    if not text:
        return ""
    s = text.lower()
    # Preserve alphanumerics + spaces, drop everything else.
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _count_matches(fold: str, tokens: list[str]) -> int:
    """Number of listed tokens found as whole-word matches in the fold."""
    hits = 0
    for tok in tokens:
        if len(tok) <= 3:
            # Very short tokens (edt, edp, gin, rum) need whole-word boundaries
            # so "gin" doesn't match inside "engine" or "beginner".
            if re.search(rf"(?:^| ){re.escape(tok)}(?: |$)", fold):
                hits += 1
        else:
            if tok in fold:
                hits += 1
    return hits


def _normalize_category(raw: Optional[str]) -> Optional[str]:
    """Fold a human-written category name into one of the canonical slugs.

    A hand-typed "Wine & Spirits", "Wines and Spirits", "Beauty", "Fragrance"
    all resolve here so downstream code can index by slug and never worry
    about spelling variants.
    """
    if not raw:
        return None
    f = _fold(str(raw))
    if not f:
        return None
    if f in _CANONICAL_SLUGS:
        return f
    if any(t in f for t in ("wine", "wines", "spirit", "spirits", "alcohol",
                            "vin", "spiritueux")):
        return WINES_SPIRITS
    if any(t in f for t in ("perfume", "perfumes", "perfumery",
                            "parfum", "fragrance", "fragrances", "cologne")):
        return PERFUMES
    if any(t in f for t in ("cosmetic", "cosmetics", "beauty",
                            "skincare", "skin care", "haircare", "hair care",
                            "makeup", "make up")):
        return COSMETICS
    return None


def detect_category(text: Optional[str] = None,
                    source_filename: Optional[str] = None,
                    explicit_category: Optional[str] = None) -> Optional[str]:
    """
    Return one of wines_spirits / perfumes / cosmetics, or None when the
    signal is too weak to decide (in which case the LLM's own answer wins
    at extraction time).

    Precedence:
      1. explicit_category (e.g. LLM output or a manual override) is
         normalised to a canonical slug when possible.
      2. Filename tokens.
      3. Document header (first ~30 lines) with a margin rule: the top
         category has to beat the runner-up by ≥ 2 hits, otherwise we
         return None rather than commit to a guess.
    """
    if explicit_category:
        norm = _normalize_category(explicit_category)
        if norm:
            return norm

    filename_fold = _fold(source_filename or "")
    if filename_fold:
        if _count_matches(filename_fold, _PERFUME_TOKENS) >= 1 and \
           _count_matches(filename_fold, _COSMETICS_TOKENS) == 0:
            return PERFUMES
        if _count_matches(filename_fold, _COSMETICS_TOKENS) >= 1 and \
           _count_matches(filename_fold, _PERFUME_TOKENS) == 0:
            return COSMETICS
        if _count_matches(filename_fold, _WINES_SPIRITS_TOKENS) >= 1 and \
           _count_matches(filename_fold, _PERFUME_TOKENS) == 0 and \
           _count_matches(filename_fold, _COSMETICS_TOKENS) == 0:
            return WINES_SPIRITS

    header = "\n".join((text or "").split("\n")[:30])
    fold = _fold(header)
    if not fold:
        return None

    scores = {
        WINES_SPIRITS: _count_matches(fold, _WINES_SPIRITS_TOKENS),
        PERFUMES:      _count_matches(fold, _PERFUME_TOKENS),
        COSMETICS:     _count_matches(fold, _COSMETICS_TOKENS),
    }
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_slug, top_score = ordered[0]
    runner_score = ordered[1][1]

    # Precision bias: the winner has to be well clear of the runner-up.
    if top_score >= 3 and top_score - runner_score >= 2:
        return top_slug
    if top_score >= 5 and top_score - runner_score >= 1:
        return top_slug
    return None


if __name__ == "__main__":  # tiny smoke test
    samples = [
        ("Fwd Premium Spirits Whisky Cognac.pdf", "Whisky · Cognac · Vodka price list — EXW Rotterdam"),
        ("Dior Sauvage EDT & EDP stock.xlsx",     "Dior Sauvage EDT 100ml Tester · Dior Homme EDP Retail"),
        ("Chanel_lipsticks_offer.xlsx",           "Rouge Coco Bloom · Lip Gloss Levres Scintillantes shade 91"),
        ("mystery_stock.xlsx",                    "prices upon request"),
    ]
    for fn, txt in samples:
        print(f"{fn:45s} → {detect_category(txt, fn)!r}")
