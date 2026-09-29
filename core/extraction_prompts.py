"""
Category-specific extraction prompts.

The original SHARED_EXTRACTION_RULES in openai_client.py is ~11k tokens
and covers W&S deeply plus perfumes/cosmetics as add-ons. On dense
perfume PDFs the LLM was silently dropping most rows, likely because
the wine-heavy rule surface was crowding attention on category-specific
attribute fields. This module keeps the W&S rules where they are and
adds slim, focused prompts for perfumes and cosmetics — each carrying
only the rules relevant to that category. The router picks which
prompt to use based on filename + first-page text heuristics; if
detection is uncertain, we fall back to the omnibus rules so W&S
behaviour is unchanged.
"""


def detect_document_category(filename: str = "", sample_text: str = "") -> str:
    """
    Classify a source document as perfumes | cosmetics | wines_spirits
    from filename and a text sample (first page or first ~2k chars).

    Returns:
      "perfumes"      → dense perfume price lists (uses PERFUMES_RULES)
      "cosmetics"     → cosmetics offers (uses COSMETICS_RULES)
      "wines_spirits" → default; keeps the omnibus SHARED_EXTRACTION_RULES
    """
    hay = f"{filename or ''} {(sample_text or '')[:2000]}".lower()

    perfume_signals = (
        "perfume", "fragrance", "eau de parfum", "eau de toilette",
        " edp ", " edt ", " edc ", " parfum", "cologne",
    )
    cosmetics_signals = (
        "cosmetic", "makeup", "skincare", "skin care",
        "mascara", "lipstick", "foundation", "concealer",
        "eyeliner", "blush", "serum", "moisturi", "cleanser",
    )
    spirits_signals = (
        "whisky", "whiskey", "bourbon", "cognac", "vodka", "gin",
        "rum ", "tequila", "liqueur", "champagne", "wine ",
        " abv ", "% vol", "% abv",
    )

    p = sum(1 for w in perfume_signals if w in hay)
    c = sum(1 for w in cosmetics_signals if w in hay)
    s = sum(1 for w in spirits_signals if w in hay)

    if p >= 2 and p > c and p > s:
        return "perfumes"
    if c >= 2 and c > p and c > s:
        return "cosmetics"
    if s >= 2 and s > p and s > c:
        return "wines_spirits"
    # Ambiguous → default to omnibus (W&S) to preserve existing behaviour
    return "wines_spirits"


PERFUMES_RULES = r"""
CATEGORY: PERFUMES
──────────────────
You are extracting perfume offers. Every row that carries a product
name AND (a barcode OR a price) is a product. Emit exactly one row per
product. Never drop a product because the page also contains an email
header, a document title, or a footer.

PER-PRODUCT FIELDS:

  brand              — the perfume house (e.g. "Dior", "YSL", "Tom Ford",
                       "Viktor & Rolf"). Strip supplier tags like
                       "- PERFUMES ARABES -" from the brand column.
  product_name       — the perfume + line (e.g. "Sauvage", "Libre",
                       "Flowerbomb Extreme"). Do NOT include size,
                       concentration, or gender token in this field.
  range_name         — the umbrella range if any (e.g. Dior "Sauvage"
                       covers Sauvage EDT / Sauvage EDP / Sauvage Elixir).
                       Optional — leave null if none.
  ean_code           — the barcode column value, verbatim, digits only.
                       If the source row has a barcode, you MUST transcribe
                       it — dropping the EAN silently is a HARD FAILURE.
                       Strip separators ("3.348.901.234.567" → 12-digit
                       string). Never fabricate.

  perfume_format     — EDT | EDP | Parfum | Cologne | EDC | EDF.
                       Read from the source: "EDP", "EDT", "Eau de Parfum",
                       "EP" (= EDP), "ET" (= EDT). Leave null if absent.
  unit_volume_ml     — bottle size in ml as a number. "100ml" → 100.
                       "1.5ml", "7,5ml" → 1.5, 7.5. Leave null for sets.
  gender             — men | women | unisex. Signals: "(M)/(H)/Homme/Pour
                       Homme" = men; "(W)/(F)/Femme/Pour Femme/Donna" =
                       women; "(U)/Unisex" explicitly = unisex. If NONE
                       of those signals is present on the row, leave
                       gender NULL. Do NOT default to unisex.
  retail_state       — retail | tester | sample | miniature. ONLY set
                       when the row explicitly says so ("Tester", "TST",
                       "Sample", "SPL", "Mini", "Miniature"). Absence of
                       "tester" is NOT retail — leave retail_state NULL
                       so the peer group is not silently split.
  refillable_status  — "Refillable" if the source uses that word; else
                       "NRF" (non-refillable). Optional.

COMMERCIAL FIELDS (populate from source row / header):
  price_per_unit, currency, incoterm, location, offer_date,
  supplier_name, supplier_reference, quantity_case, quantity_unit,
  moq_cases (if present).

SET the category_slug field on every product to exactly "perfumes".

WHAT NOT TO DO:
  • Do not skip a row because it looks similar to one earlier on the
    page — different volume, different concentration, or different
    gender make it a distinct SKU.
  • Do not merge tester rows with retail rows.
  • Do not invent gender/retail_state when the source is silent.

RETURN FORMAT: {"products":[{...}, ...]}
"""


COSMETICS_RULES = r"""
CATEGORY: COSMETICS / BEAUTY
────────────────────────────
You are extracting cosmetic / beauty offers. Every row that carries a
product name AND (a barcode OR a price) is a product. Emit exactly one
row per product. Never drop a product because the page also contains
headers or footers.

PER-PRODUCT FIELDS:

  brand              — the cosmetics house (e.g. "Clinique", "Estee
                       Lauder", "Shiseido", "Nars"). If the source has
                       no Brand column, extract it from the product name.
  product_name       — the product without the brand + without the size
                       (e.g. "BADgal BANG Mascara", "Even Better Makeup").
  range_name         — the umbrella range if any (e.g. "Even Better").
                       Optional.
  ean_code           — same rule as perfumes — transcribe verbatim
                       when a barcode/EAN/GTIN column exists.

  product_type       — the physical product kind: lipstick, foundation,
                       mascara, cream, serum, shampoo, body wash, tint,
                       balm, nail polish, powder, brush, etc. Required
                       when identifiable; null when truly unclear.
  shade              — colour / variant string as written in the source
                       ("Rouge 91", "Fair 210", "Pitch Black",
                       "Neutral Light Brown"). Preserve original casing.
  unit_volume_ml     — for LIQUID products (creams, tints, serums, oils)
                       — bottle/tube size in ml as a number.
  size_weight_g      — for SOLID products (lipsticks, powders, bars) —
                       weight in grams as a number. Do NOT fold weight
                       into unit_volume_ml.
  gender             — men | women | unisex, ONLY if the source
                       explicitly says so (rare in cosmetics). Otherwise
                       leave null.

COMMERCIAL FIELDS: same set as perfumes (price / currency / incoterm /
location / offer_date / supplier / quantity_case / quantity_unit).

SET category_slug = "cosmetics" on every product.

WHAT NOT TO DO:
  • Do not merge different shades of the same product into one row —
    each shade is a distinct SKU.
  • Do not conflate ml (liquid) and g (solid) — use the correct field.
  • Do not skip a row because there is no explicit product_type; make
    a best-effort guess from the name, or leave it null.

RETURN FORMAT: {"products":[{...}, ...]}
"""


def rules_for_category(category: str, shared_omnibus: str) -> str:
    """
    Return the rule block to embed into an extraction prompt.

    W&S keeps the existing omnibus rules — that path is validated in
    production and shouldn't change. Perfumes and cosmetics get the
    slim, focused prompts above so the LLM's attention isn't consumed
    by wine-vintage / age-statement / bottle-can rules that don't apply.
    """
    if category == "perfumes":
        return PERFUMES_RULES
    if category == "cosmetics":
        return COSMETICS_RULES
    return shared_omnibus
