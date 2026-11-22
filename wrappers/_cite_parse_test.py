"""Unit tests for the citation-string parser.

Run:  pytest wrappers/_cite_parse_test.py -q

The CELEX-aware tests check that a raw PDF filename such as
``CELEX_32014L0065_EN_TXT.pdf`` in the citation head resolves to the
canonical instrument id.
"""

from wrappers._cite_parse import parse_citation


# ---------------------------------------------------------------------------
# Short-name aliases (canonical happy path)
# ---------------------------------------------------------------------------

def test_mifid_short_name_resolves_to_canonical_id():
    out = parse_citation("MiFID II, Art. 25, para. 2 (CELEX 32014L0065)")
    assert out["instrument"] == "MIFID2_L1"
    assert out["article"] == "25"
    assert out["paragraph"] == "2"
    assert out["celex"] == "32014L0065"

def test_sfdr_short_name_resolves_to_canonical_id():
    out = parse_citation("SFDR, Art. 4, para. 1 (CELEX 32019R2088)")
    assert out["instrument"] == "SFDR_L1"
    assert out["article"] == "4"
    assert out["paragraph"] == "1"
    assert out["celex"] == "32019R2088"

def test_alias_match_is_case_insensitive():
    out = parse_citation("MIFID II, Art. 25")
    assert out["instrument"] == "MIFID2_L1"


def test_v1_filename_head_resolves_via_celex_alias():
    """A loader may emit the raw PDF filename as the citation head:
        ``CELEX_32014L0065_EN_TXT.pdf, p. 100``
    The CELEX number embedded in the head must be picked up so the parser
    returns the canonical instrument id rather than an empty string."""
    out = parse_citation("[CELEX_32014L0065_EN_TXT.pdf, p. 100]")
    assert out["instrument"] == "MIFID2_L1"

def test_v1_sfdr_filename_head_resolves_via_celex_alias():
    out = parse_citation("[CELEX_32019R2088_EN_TXT.pdf, p. 4]")
    assert out["instrument"] == "SFDR_L1"

def test_celex_alias_works_outside_the_head():
    """The fallback also catches CELEX numbers that show up only in the
    parenthetical suffix, not in the head (e.g. when the short name is missing
    and only the CELEX appears)."""
    out = parse_citation("Some unknown header, Art. 7 (CELEX 32014L0065)")
    assert out["instrument"] == "MIFID2_L1"


# ---------------------------------------------------------------------------
# Negative cases — must not silently invent an instrument
# ---------------------------------------------------------------------------

def test_unknown_document_returns_empty_instrument():
    out = parse_citation("Random Authority Bulletin, Art. 1")
    assert out["instrument"] == ""

def test_empty_string_is_safe():
    out = parse_citation("")
    assert out["instrument"] == ""
    assert out["article"] == ""
    assert out["paragraph"] is None
    assert out["point"] is None
    assert out["celex"] == ""


# ---------------------------------------------------------------------------
# Article / paragraph / point extraction
# ---------------------------------------------------------------------------

def test_article_paragraph_point_extracted():
    out = parse_citation("MiFID II, Art. 25, para. 2 (a)")
    assert out["article"] == "25"
    assert out["paragraph"] == "2"
    assert out["point"] == "a"

def test_article_with_letter_suffix():
    out = parse_citation("MiFID II, Art. 25a")
    assert out["article"] == "25a"
