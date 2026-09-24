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

from core.normalization import canonical_brand, canonical_product, canonical_key


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
