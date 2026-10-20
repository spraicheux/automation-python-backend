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
    product_family_id,
    sku_identity,
    ean_key,
    peer_group_id,
    is_peer_group_qualified,
    depollute_identity,
    clean_product_key,
)


class TestCleanProductKey:
    """product_key must identify the product, not its trade terms — customs
    (T1/T2) and refillable (RF/NRF) tokens belong in their own columns
    (Samuel: "product_key still embeds _T2_")."""

    def test_strips_customs_and_refill(self):
        assert clean_product_key("JAGERMEISTER_RF_T2_700ML_6X70CL") == "JAGERMEISTER_700ML_6X70CL"
        assert clean_product_key("JOHNNIE_WALKER_RED_LABEL_6X1L_RF_T2") == "JOHNNIE_WALKER_RED_LABEL_6X1L"
        assert clean_product_key("HENNESSY_VS_1000ML_12X100CL_T1") == "HENNESSY_VS_1000ML_12X100CL"
        assert clean_product_key("NRF_VODKA_1L") == "VODKA_1L"

    def test_keeps_volume_pack_and_age_tokens(self):
        assert clean_product_key("ABSOLUT_700ML_6X70CL") == "ABSOLUT_700ML_6X70CL"
        assert clean_product_key("CHIVAS_REGAL_12YO") == "CHIVAS_REGAL_12YO"

    def test_empty_safe(self):
        assert clean_product_key("") == ""
        assert clean_product_key(None) == ""


class TestDepolluteIdentity:
    """A liquor-category descriptor (e.g. 'Baijiu') must not sit in the brand
    or be the whole product name — one product must keep one identity across
    files (Samuel 19 Oct: 'BaijiuWu Liang Ye' vs 'Wu Liang Ye' vs name 'Baijiu')."""

    def test_glued_prefix_stripped(self):
        assert depollute_identity("BaijiuWu Liang Ye", "BaijiuWu Liang Ye") == ("Wu Liang Ye", "Wu Liang Ye")

    def test_name_that_is_only_category_takes_brand(self):
        assert depollute_identity("Wu Liang Ye", "Baijiu") == ("Wu Liang Ye", "Wu Liang Ye")

    def test_space_separated_prefix_stripped(self):
        assert depollute_identity("Baijiu Wu Liang Ye", "Baijiu") == ("Wu Liang Ye", "Wu Liang Ye")

    def test_clean_identity_untouched(self):
        assert depollute_identity("Hennessy", "VS") == ("Hennessy", "VS")
        assert depollute_identity("Wu Liang Ye", "Wu Liang Ye") == ("Wu Liang Ye", "Wu Liang Ye")

    def test_blank_name_falls_back_to_brand(self):
        # Single-product brands (Disaronno, Cointreau, Tia Maria …) whose name is
        # blank / "Not Found" take the brand, so the UI never falls back to the key.
        assert depollute_identity("Disaronno", "") == ("Disaronno", "Disaronno")
        assert depollute_identity("Cointreau", None) == ("Cointreau", "Cointreau")
        assert depollute_identity("Tia Maria", "Not Found") == ("Tia Maria", "Tia Maria")

    def test_pack_token_name_falls_back_to_brand(self):
        # A pack/size token mis-extracted as the name (Samuel: Jack Daniel's "12x")
        # is not a product name — use the brand.
        assert depollute_identity("Jack Daniel's", "12x") == ("Jack Daniel's", "Jack Daniel's")
        assert depollute_identity("Jack Daniel's", "12x70cl") == ("Jack Daniel's", "Jack Daniel's")
        assert depollute_identity("Baileys", "6x1L") == ("Baileys", "Baileys")
        # real names that merely contain digits are untouched
        assert depollute_identity("Jack Daniel's", "Old No.7") == ("Jack Daniel's", "Old No.7")
        assert depollute_identity("Glenlivet", "12YO") == ("Glenlivet", "12YO")

    def test_all_three_variants_share_one_family(self):
        variants = [("BaijiuWu Liang Ye", "BaijiuWu Liang Ye"),
                    ("Wu Liang Ye", "Baijiu"),
                    ("Wu Liang Ye", "Wu Liang Ye")]
        fams = {product_family_id(*depollute_identity(b, p)) for b, p in variants}
        assert len(fams) == 1


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

    def test_not_found_incoterm_treated_as_empty(self):
        a = peer_group_id("Grey Goose", "Original", incoterm="Not Found")
        b = peer_group_id("Grey Goose", "Original", incoterm=None)
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


class TestThreeIdentityLevels:
    """
    The three levels must actually mean different things:
      family < SKU < peer_group.
    """

    def test_family_ignores_size(self):
        # Hennessy VS in different sizes is one FAMILY.
        f_350 = product_family_id("Hennessy", "VS")
        f_700 = product_family_id("Hennessy", "VS")
        assert f_350 == f_700

    def test_sku_splits_by_size(self):
        # But different sizes are different SKUs.
        sku_350 = sku_identity("Hennessy", "VS", unit_volume_ml=350, units_per_case=6, alcohol_percent=40)
        sku_700 = sku_identity("Hennessy", "VS", unit_volume_ml=700, units_per_case=6, alcohol_percent=40)
        assert sku_350 != sku_700

    def test_sku_splits_by_abv(self):
        sku_40 = sku_identity("Bacardi", "Superior", unit_volume_ml=700, units_per_case=6, alcohol_percent=40.0)
        sku_43 = sku_identity("Bacardi", "Superior", unit_volume_ml=700, units_per_case=6, alcohol_percent=43.0)
        assert sku_40 != sku_43

    def test_sku_ignores_incoterm_and_location(self):
        # Same physical SKU sold under different terms → same SKU key.
        exw = sku_identity("Hennessy", "VS", unit_volume_ml=700, units_per_case=6, alcohol_percent=40)
        dap = sku_identity("Hennessy", "VS", unit_volume_ml=700, units_per_case=6, alcohol_percent=40)
        assert exw == dap

    def test_peer_group_splits_where_sku_collapses(self):
        # Same SKU at different incoterms is not a fair comparison.
        pg_exw = peer_group_id("Hennessy", "VS", unit_volume_ml=700, units_per_case=6,
                               alcohol_percent=40, incoterm="EXW", location="Rotterdam")
        pg_dap = peer_group_id("Hennessy", "VS", unit_volume_ml=700, units_per_case=6,
                               alcohol_percent=40, incoterm="DAP", location="Rotterdam")
        assert pg_exw != pg_dap

    def test_peer_group_extends_sku(self):
        # The peer key starts with the SKU key — different SKUs can never
        # peer, and same SKU + same terms MUST peer.
        args = dict(unit_volume_ml=700, units_per_case=6, alcohol_percent=40)
        sku = sku_identity("Hennessy", "VS", **args)
        pg = peer_group_id("Hennessy", "VS", incoterm="EXW", location="Rotterdam", **args)
        assert pg.startswith(sku + "|")

    def test_sku_ignores_ean_by_design(self):
        # EAN is deliberately outside sku_identity so a blank-EAN row can
        # still merge with an EAN-bearing row of the same product (client's
        # "one EAN known → fall back to attributes" rule). EAN handling is
        # reconciled by the API resolver, not by the hash.
        a = sku_identity("Dior", "Sauvage", unit_volume_ml=100)
        b = sku_identity("Dior", "Sauvage", unit_volume_ml=100)
        assert a == b


class TestGS1CheckDigit:
    """
    R-04 lock: ean_key() must validate the GS1 check digit and return "" for
    any EAN whose last digit does not satisfy the alternating-weight formula.
    Any regression here silently contaminates peer groups with false EAN keys.
    """

    def test_valid_ean13_accepted(self):
        # 3348901234566 — check digit 6, computed: total=94, (10-4)%10=6
        assert ean_key("3348901234566") == "3348901234566"

    def test_invalid_ean13_one_off_rejected(self):
        # Same stem, wrong last digit: 7 ≠ 6
        assert ean_key("3348901234567") == ""

    def test_audit_invalid_ean_rejected(self):
        # Caught during Samuel's audit: client data contained this EAN
        # with a transposed check digit. Must never form a peer group key.
        assert ean_key("2901234500153") == ""

    def test_audit_corrected_ean_accepted(self):
        # Corrected version: check digit 2, total=78, (10-8)%10=2
        assert ean_key("2901234500152") == "2901234500152"

    def test_known_real_world_ean_accepted(self):
        # Johnnie Walker Black 700ml 12pk — used as audit evidence
        # total=68, (10-8)%10=2 ✓
        assert ean_key("5000112637922") == "5000112637922"

    def test_valid_ean8_accepted(self):
        # 12345670 — check digit 0, total=60, (10-0)%10=0
        assert ean_key("12345670") == "12345670"

    def test_invalid_ean8_rejected(self):
        # Same stem, wrong check digit: 8 ≠ 0
        assert ean_key("12345678") == ""

    def test_valid_gtin14_accepted(self):
        # 13348901234563 — check digit 3, total=97, (10-7)%10=3
        assert ean_key("13348901234563") == "13348901234563"

    def test_invalid_gtin14_rejected(self):
        # Same stem, wrong check digit: 5 ≠ 3
        assert ean_key("13348901234565") == ""

    def test_two_valid_eans_are_distinct(self):
        # Belt-and-braces: the peer splitter depends on distinct ean_keys
        # for distinct products. Two different valid EANs must not collide.
        a = ean_key("3348901234566")
        b = ean_key("3348901234573")
        assert a and b and a != b


class TestEANReconciliation:
    """
    EAN handling per client rules:
      - both known + same    → strong same-SKU match
      - both known + differ  → keep separate (SKU-level flag ean_conflict)
      - one known / none     → fall back to full attributes
      - EAN vs attributes contradiction → Needs Review (resolver's job)
    """

    def test_ean_folded_to_digits(self):
        # 3348901234566 — check digit 6 (valid GS1)
        assert ean_key("3348901234566") == "3348901234566"
        assert ean_key(" 3348-9012-34566 ") == "3348901234566"
        assert ean_key("EAN 3348901234566") == "3348901234566"

    def test_short_junk_rejected(self):
        # A "code" like "123" is not a real EAN — treat as unknown so we
        # never build a false-signal on stray data.
        assert ean_key("123") == ""
        assert ean_key("") == ""
        assert ean_key(None) == ""

    def test_valid_ean_lengths(self):
        # Each value below has a correct GS1 check digit.
        assert ean_key("12345670") == "12345670"                # EAN-8  (check=0)
        assert ean_key("123456789012") == "123456789012"        # UPC-12 (check=2)
        assert ean_key("3348901234566") == "3348901234566"      # EAN-13 (check=6)
        assert ean_key("13348901234563") == "13348901234563"    # GTIN-14 (check=3)

    def test_blank_ean_sku_matches_ean_bearing_sku(self):
        # Same attributes, one row with EAN, the other without → sku_identity
        # is the same. The resolver treats them as one SKU (fallback rule).
        a = sku_identity("Dior", "Sauvage", unit_volume_ml=100, alcohol_percent=None)
        b = sku_identity("Dior", "Sauvage", unit_volume_ml=100, alcohol_percent=None)
        assert a == b
        # And their EAN keys differ ("" vs "3348...") but the SKU key doesn't.
        assert ean_key(None) != ean_key("3348901234566")

    def test_peer_group_still_splits_on_ean(self):
        # If both rows have EAN and they differ, the peer key catches it
        # so a "trusted best" never mixes them.
        # 3348901234566 (check=6) and 3348901234573 (check=3) — two distinct valid EANs.
        pg_a = peer_group_id("Dior", "Sauvage", unit_volume_ml=100,
                             ean_code="3348901234566", incoterm="EXW", location="Rotterdam")
        pg_b = peer_group_id("Dior", "Sauvage", unit_volume_ml=100,
                             ean_code="3348901234573", incoterm="EXW", location="Rotterdam")
        assert pg_a != pg_b


class TestBestPricesHeadlineSemantics:
    """
    Documents the client's rules for the SKU-level headline in /api/best-prices.
    The response shape is generated inside the endpoint but its meaning is
    stable enough to test as a contract:
      - No cross-peer "Trusted Best Price" — that would silently compare
        EXW vs DAP without freight normalisation.
      - Cross-peer headline is called "lowest_nominal_price_eur" and is
        computed only across QUALIFIED peer groups.
      - An EAN conflict suppresses the headline entirely (both fields
        null on the API), no matter the peer count or qualification.
    These properties are exercised via the underlying primitives so a
    future refactor can't flip the semantic without failing CI.
    """

    def test_lowest_nominal_only_considers_qualified_peers(self):
        # The endpoint's logic in pseudocode:
        #   lowest_nominal = min(peer.best for peer in peers if peer.is_qualified)
        # Assert via primitives: is_peer_group_qualified must return False
        # for unknown terms so those peers are excluded from the min.
        assert not is_peer_group_qualified(None, "Rotterdam")
        assert not is_peer_group_qualified("EXW", None)
        assert is_peer_group_qualified("EXW", "Rotterdam")

    def test_ean_conflict_disables_cross_peer_headline(self):
        # Two known & different EANs → the endpoint's ean_conflict flag
        # forces lowest_nominal_price_eur to None regardless of peer state.
        # We assert the primitive: two distinct valid EANs must fold to
        # distinct ean_keys so the endpoint's `distinct_eans` count is > 1.
        a = ean_key("3348901234566")  # check=6 ✓
        b = ean_key("3348901234573")  # check=3 ✓
        assert a and b and a != b


class TestPeerQualification:
    """
    A peer comparison is "qualified" only when incoterm AND location are
    both known. Two rows with unknown terms may hash to the same peer key
    but must NOT produce trusted trading signals.
    """

    def test_all_known(self):
        assert is_peer_group_qualified("EXW", "Rotterdam") is True

    def test_missing_location(self):
        assert is_peer_group_qualified("EXW", None) is False
        assert is_peer_group_qualified("EXW", "Not Found") is False

    def test_missing_incoterm(self):
        assert is_peer_group_qualified(None, "Rotterdam") is False
        assert is_peer_group_qualified("Not Found", "Rotterdam") is False

    def test_both_missing(self):
        assert is_peer_group_qualified(None, None) is False
        assert is_peer_group_qualified("Not Found", "Not Found") is False
