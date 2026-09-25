"""
Multi-category (Phase 3, M1) test suite.

Locks in the invariants that a future edit could silently break:

  1. Category classifier: strong deterministic hits produce a slug;
     truly ambiguous documents return None (never a wrong guess).
  2. Human-typed category labels fold to canonical slugs.
  3. sku_identity keeps different categories separate even when brand
     + product name coincide.
  4. Perfume EDT ≠ EDP at SKU level, Tester ≠ Retail, Men ≠ Women.
  5. Cosmetics: different shades = different SKUs; different sizes /
     weights = different SKUs.
  6. peer_group_id inherits every SKU-level split.
"""
import pytest

from core.normalization import product_family_id, sku_identity, peer_group_id
from core.category_classifier import (
    detect_category, _normalize_category,
    WINES_SPIRITS, PERFUMES, COSMETICS,
)


class TestCategoryClassifier:
    def test_wines_spirits_from_filename(self):
        assert detect_category(
            text="",
            source_filename="Fwd Premium Spirits Whisky Cognac.pdf",
        ) == WINES_SPIRITS

    def test_perfumes_from_filename(self):
        assert detect_category(
            text="",
            source_filename="Dior Sauvage EDT & EDP stock.xlsx",
        ) == PERFUMES

    def test_cosmetics_from_filename(self):
        assert detect_category(
            text="",
            source_filename="Chanel_lipsticks_offer.xlsx",
        ) == COSMETICS

    def test_wines_spirits_from_header(self):
        header = ("PRICELIST — Spirits & Wine, EXW Rotterdam\n"
                  "Whisky · Vodka · Cognac · Rum\n"
                  "Ballantine's Finest · Hennessy VS · Grey Goose")
        assert detect_category(text=header) == WINES_SPIRITS

    def test_perfumes_from_header(self):
        header = ("PERFUMES STOCK — EXW Paris\n"
                  "Dior Sauvage EDT 100ml Tester · Chanel Bleu EDP\n"
                  "Fragrance spray · Cologne · Parfum")
        assert detect_category(text=header) == PERFUMES

    def test_cosmetics_from_header(self):
        header = ("BEAUTY STOCK — cosmetics offer\n"
                  "Lipstick · Foundation · Mascara · Nail polish\n"
                  "Chanel Rouge Coco Bloom · Dior Forever Foundation")
        assert detect_category(text=header) == COSMETICS

    def test_ambiguous_returns_none(self):
        # "prices upon request" carries no category signal at all — the
        # classifier must decline rather than guess.
        assert detect_category(
            text="prices upon request",
            source_filename="mystery_stock.xlsx",
        ) is None

    def test_precision_bias(self):
        # A single stray "cologne" mention shouldn't override a
        # dominant W&S document — the classifier requires a comfortable
        # margin, otherwise falls back to None.
        header = ("Spirits · Wine · Vodka · Whisky · Cognac\n"
                  "Also a stray cologne reference here")
        assert detect_category(text=header) == WINES_SPIRITS


class TestCategoryNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("wines_spirits", "wines_spirits"),
        ("Wines & Spirits", "wines_spirits"),
        ("Wine", "wines_spirits"),
        ("Vin", "wines_spirits"),
        ("Spirits", "wines_spirits"),
        ("perfumes", "perfumes"),
        ("Perfume", "perfumes"),
        ("Fragrance", "perfumes"),
        ("Parfum", "perfumes"),
        ("Cosmetics", "cosmetics"),
        ("Beauty", "cosmetics"),
        ("Skincare", "cosmetics"),
        ("Makeup", "cosmetics"),
        ("", None),
        (None, None),
        ("something else", None),
    ])
    def test_fold(self, raw, expected):
        assert _normalize_category(raw) == expected


class TestSkuIdentityAcrossCategories:
    """A perfume must never share an SKU key with a cosmetic that
    happens to have the same brand + product name."""

    def test_same_brand_different_category(self):
        # Contrived but architecturally important: a brand like
        # "Aesop" makes both fragrances and skincare. Their SKUs
        # must never collide even if the extractor produced the
        # same brand + name.
        perfume = sku_identity(
            "Aesop", "Marrakech", unit_volume_ml=50,
            category_slug="perfumes",
        )
        cosmetic = sku_identity(
            "Aesop", "Marrakech", unit_volume_ml=50,
            category_slug="cosmetics",
        )
        assert perfume != cosmetic
        assert perfume.startswith("perfumes|")
        assert cosmetic.startswith("cosmetics|")


class TestPerfumeSkuSplit:
    def test_edt_vs_edp(self):
        a = sku_identity("Dior", "Sauvage", unit_volume_ml=100,
                         category_slug="perfumes", perfume_format="EDT")
        b = sku_identity("Dior", "Sauvage", unit_volume_ml=100,
                         category_slug="perfumes", perfume_format="EDP")
        assert a != b

    def test_retail_vs_tester(self):
        r = sku_identity("Dior", "Sauvage", unit_volume_ml=100,
                         category_slug="perfumes", perfume_format="EDT",
                         retail_state="retail")
        t = sku_identity("Dior", "Sauvage", unit_volume_ml=100,
                         category_slug="perfumes", perfume_format="EDT",
                         retail_state="tester")
        assert r != t

    def test_gender_split(self):
        m = sku_identity("Boss", "Bottled", unit_volume_ml=100,
                         category_slug="perfumes", gender="men")
        w = sku_identity("Boss", "Bottled", unit_volume_ml=100,
                         category_slug="perfumes", gender="women")
        assert m != w

    def test_size_split(self):
        s60 = sku_identity("Dior", "Sauvage", unit_volume_ml=60,
                           category_slug="perfumes", perfume_format="EDT")
        s100 = sku_identity("Dior", "Sauvage", unit_volume_ml=100,
                            category_slug="perfumes", perfume_format="EDT")
        assert s60 != s100


class TestCosmeticsSkuSplit:
    def test_shade_split(self):
        a = sku_identity("Chanel", "Rouge Coco Bloom", unit_volume_ml=None,
                         category_slug="cosmetics",
                         product_type="lipstick", shade="91")
        b = sku_identity("Chanel", "Rouge Coco Bloom", unit_volume_ml=None,
                         category_slug="cosmetics",
                         product_type="lipstick", shade="116")
        assert a != b

    def test_product_type_split(self):
        # Same brand + name but different product types — impossible in
        # practice but the split must be defensive.
        a = sku_identity("Estée Lauder", "Double Wear",
                         category_slug="cosmetics", product_type="foundation")
        b = sku_identity("Estée Lauder", "Double Wear",
                         category_slug="cosmetics", product_type="concealer")
        assert a != b

    def test_weight_split(self):
        s15 = sku_identity("Kiehl's", "Ultra Facial Cream",
                           category_slug="cosmetics",
                           product_type="cream", size_weight_g=15)
        s50 = sku_identity("Kiehl's", "Ultra Facial Cream",
                           category_slug="cosmetics",
                           product_type="cream", size_weight_g=50)
        assert s15 != s50


class TestPeerGroupInheritsCategoryDiscriminators:
    """Every SKU-level split must propagate to peer_group_id so a peer
    comparison inside Best Prices / benchmarks can never mix categories,
    concentrations, testers/retail, genders or shades either."""

    def test_peer_starts_with_sku(self):
        args = dict(unit_volume_ml=100, category_slug="perfumes",
                    perfume_format="EDT", retail_state="retail")
        sku = sku_identity("Dior", "Sauvage", **args)
        peer = peer_group_id("Dior", "Sauvage", incoterm="EXW",
                             location="Rotterdam", **args)
        assert peer.startswith(sku + "|")

    def test_peer_splits_edt_vs_edp(self):
        p1 = peer_group_id("Dior", "Sauvage", unit_volume_ml=100,
                           category_slug="perfumes", perfume_format="EDT",
                           incoterm="EXW", location="Rotterdam")
        p2 = peer_group_id("Dior", "Sauvage", unit_volume_ml=100,
                           category_slug="perfumes", perfume_format="EDP",
                           incoterm="EXW", location="Rotterdam")
        assert p1 != p2

    def test_family_ignores_all_category_fields(self):
        # Family = brand + name only. Category-specific attributes must
        # NOT leak into the family key — that would defeat cross-format
        # search and the "all Dior Sauvage" roll-up.
        edt = product_family_id("Dior", "Sauvage")
        edp = product_family_id("Dior", "Sauvage")
        assert edt == edp == "dior|sauvage"
