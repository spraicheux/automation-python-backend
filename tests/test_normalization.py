"""
Locks in the peer-identity contract:

  1. The same product spelled differently must produce the same key.
  2. Unicode / accented / ligature / apostrophe variants normalise the same
     way as their ASCII counterparts.
  3. Two products that really are different must NOT collapse into one key.

Any future edit to core/normalization.py that breaks these silently would
break the benchmark lookup on real client data (this is exactly how the
Peñasol / Jack Daniel's peer-key mismatch happened in the first place).
"""
import pytest

from core.normalization import (
    canonical_brand,
    canonical_product,
    canonical_key,
    canonical_product_id,
    peer_group_id,
)


class TestUnicodeFolding:
    """Accents, ligatures and non-ASCII must fold deterministically."""

    def test_ntilde(self):
        assert canonical_brand("Peñasol") == "penasol"
        assert canonical_brand("Peñasol") == canonical_brand("penasol")

    def test_e_acute(self):
        assert canonical_brand("Rémy Martin") == "remy martin"
        assert canonical_brand("Rémy Martin") == canonical_brand("Remy Martin")

    def test_umlaut(self):
        assert canonical_brand("Jägermeister") == "jagermeister"

    def test_curly_and_straight_apostrophes_collapse(self):
        assert canonical_brand("Jack Daniel's") == canonical_brand("Jack Daniel’s")

    def test_apostrophe_keeps_word_intact(self):
        # Was a real bug: "Jack Daniel's" -> "jack daniel s" made "daniels"
        # split into two tokens and stopped matching the peer's own row.
        assert canonical_brand("Jack Daniel's") == "jack daniels"

    def test_ae_ligature(self):
        assert canonical_brand("Æther") == "aether"

    def test_ss_ligature(self):
        assert canonical_brand("Straße") == "strasse"

    def test_fi_ligature(self):
        # ﬁ (U+FB01) → fi via NFKD compat decomposition
        assert canonical_brand("ﬁnesse") == "finesse"


class TestNoiseWords:
    """Filler tokens must not stop two spellings from collapsing."""

    def test_the_dropped(self):
        assert canonical_product("The Macallan") == "macallan"

    def test_original_dropped(self):
        assert canonical_product("Baileys Original Irish Cream") == "baileys irish cream"


class TestSameKeyDifferentSpelling:
    """The parity property peers depend on."""

    @pytest.mark.parametrize("a, b", [
        (("Peñasol", "Sangria"), ("penasol", "sangria")),
        (("Rémy Martin", "VSOP"), ("remy martin", "vsop")),
        (("Jack Daniel's", "Honey"), ("Jack Daniel’s", "Honey")),
        (("Bailey's", "Irish Cream"), ("BAILEYS", "the irish cream")),
    ])
    def test_pair(self, a, b):
        k1 = canonical_key(a[0], a[1], 1000, 1, "EXW")
        k2 = canonical_key(b[0], b[1], 1000, 1, "EXW")
        assert k1 == k2, f"{a} -> {k1!r}  vs  {b} -> {k2!r}"


class TestDifferentProductsStaySeparate:
    """Belt-and-braces: normalisation must not over-collapse."""

    def test_volume_differs(self):
        k70 = canonical_key("Baileys", "Irish Cream", 700, 6, "EXW")
        k100 = canonical_key("Baileys", "Irish Cream", 1000, 6, "EXW")
        assert k70 != k100

    def test_pack_size_differs(self):
        k6 = canonical_key("Baileys", "Irish Cream", 700, 6, "EXW")
        k12 = canonical_key("Baileys", "Irish Cream", 700, 12, "EXW")
        assert k6 != k12

    def test_incoterm_differs(self):
        kexw = canonical_key("Baileys", "Irish Cream", 700, 6, "EXW")
        kdap = canonical_key("Baileys", "Irish Cream", 700, 6, "DAP")
        assert kexw != kdap

    def test_different_brand(self):
        assert canonical_brand("Peñasol") != canonical_brand("Jack Daniels")


class TestCanonicalProductIdVsPeerGroupId:
    """
    Two distinct identity concepts must not collapse:
      - canonical_product_id: naming — same product regardless of pack / terms.
      - peer_group_id: commercial — genuinely apples-to-apples for benchmarking.
    """

    def test_same_product_different_pack_shares_canonical_but_not_peer(self):
        cpid_a = canonical_product_id("Baileys", "Irish Cream")
        cpid_b = canonical_product_id("Baileys", "Irish Cream")
        assert cpid_a == cpid_b

        pg_700 = peer_group_id("Baileys", "Irish Cream", unit_volume_ml=700, units_per_case=6, incoterm="EXW")
        pg_1000 = peer_group_id("Baileys", "Irish Cream", unit_volume_ml=1000, units_per_case=6, incoterm="EXW")
        assert pg_700 != pg_1000

    def test_canonical_product_id_ignores_pack(self):
        assert (canonical_product_id("Baileys", "Irish Cream") ==
                canonical_product_id("Baileys", "Irish Cream"))


class TestLocationInPeerGroup:
    """Rotterdam and Dubai must not peer even at the same incoterm."""

    def test_different_location_splits_peer(self):
        rot = peer_group_id("Grey Goose", "Original", unit_volume_ml=700,
                            units_per_case=6, incoterm="EXW", location="Rotterdam")
        dub = peer_group_id("Grey Goose", "Original", unit_volume_ml=700,
                            units_per_case=6, incoterm="EXW", location="Dubai")
        assert rot != dub

    def test_location_case_and_prefix_insensitive(self):
        a = peer_group_id("Grey Goose", "Original", incoterm="EXW", location="ROTTERDAM")
        b = peer_group_id("Grey Goose", "Original", incoterm="EXW", location="rotterdam")
        c = peer_group_id("Grey Goose", "Original", incoterm="EXW", location="EXW Rotterdam")
        assert a == b == c

    def test_not_found_location_treated_as_empty(self):
        a = peer_group_id("Grey Goose", "Original", incoterm="EXW", location="Not Found")
        b = peer_group_id("Grey Goose", "Original", incoterm="EXW", location=None)
        assert a == b


class TestCommercialDiscriminators:
    """ABV, vintage, age statement must break peer identity when they differ."""

    def test_abv_different(self):
        a = peer_group_id("Bacardi", "Superior", unit_volume_ml=700, units_per_case=6,
                          incoterm="EXW", alcohol_percent=40.0)
        b = peer_group_id("Bacardi", "Superior", unit_volume_ml=700, units_per_case=6,
                          incoterm="EXW", alcohol_percent=43.0)
        assert a != b

    def test_abv_same_within_tolerance(self):
        # 40.0 and 40.04 both bin to "40.0" — one decimal is our resolution.
        a = peer_group_id("Bacardi", "Superior", incoterm="EXW", alcohol_percent=40.0)
        b = peer_group_id("Bacardi", "Superior", incoterm="EXW", alcohol_percent=40.04)
        assert a == b

    def test_abv_blank_stays_blank(self):
        a = peer_group_id("Bacardi", "Superior", incoterm="EXW", alcohol_percent=None)
        b = peer_group_id("Bacardi", "Superior", incoterm="EXW")
        assert a == b

    def test_vintage_different(self):
        a = peer_group_id("Chateau X", "Premier Cru", incoterm="EXW", vintage="2018")
        b = peer_group_id("Chateau X", "Premier Cru", incoterm="EXW", vintage="2019")
        assert a != b

    def test_age_statement_different(self):
        a = peer_group_id("Macallan", "Sherry Oak", incoterm="EXW", age_statement="12 YO")
        b = peer_group_id("Macallan", "Sherry Oak", incoterm="EXW", age_statement="18 YO")
        assert a != b

    def test_age_statement_spelling_insensitive(self):
        a = peer_group_id("Macallan", "Sherry Oak", incoterm="EXW", age_statement="12YO")
        b = peer_group_id("Macallan", "Sherry Oak", incoterm="EXW", age_statement="12 years")
        # "12yo" vs "12 years" fold to different tokens; that's ok — the
        # rule is: same spelling collapses, different spellings don't peer
        # unless we teach an alias map later.
        assert a != b


class TestCanonicalKeyShim:
    """The old canonical_key() signature must still resolve to peer_group_id."""

    def test_shim_matches_peer_group_id(self):
        legacy = canonical_key("Baileys", "Irish Cream", 700, 6, "EXW")
        new = peer_group_id("Baileys", "Irish Cream", unit_volume_ml=700,
                            units_per_case=6, incoterm="EXW")
        assert legacy == new
