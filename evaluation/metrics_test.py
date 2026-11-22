"""Unit tests for custom regulatory metrics.

Run:  pytest evaluation/metrics_test.py -q
"""

from evaluation.metrics import (
    AnswerRecord,
    article_accuracy,
    article_pinpoint_accuracy,
    citation_density,
    citation_validity_rate,
    compute_all,
    multi_instrument_coverage,
    refusal_correctness,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mk(system="V2", intent="FACTUAL_QA", text="ok", refused=False,
        citations=None, **gold):
    return AnswerRecord(
        question_id="q",
        system_id=system,
        intent=intent,
        answer_text=text,
        refused=refused,
        citations=citations or [],
        **gold,
    )


def _cit(i, instrument="MIFID2_L1", article="25", paragraph=None, point=None):
    return {
        "citation_id": i,
        "instrument": instrument,
        "article": article,
        "paragraph": paragraph,
        "point": point,
        "passage": "lorem ipsum",
    }


# ---------------------------------------------------------------------------
# CVR
# ---------------------------------------------------------------------------

def test_cvr_all_valid():
    r = _mk(text="See [C1] and [C2].", citations=[_cit(1), _cit(2)])
    assert citation_validity_rate(r) == 1.0

def test_cvr_some_invalid():
    r = _mk(text="See [C1] [C3] [C9].", citations=[_cit(1), _cit(3)])
    assert citation_validity_rate(r) == 2/3

def test_cvr_refused_is_none():
    r = _mk(text="I cannot answer.", refused=True)
    assert citation_validity_rate(r) is None

def test_cvr_no_citations_returns_none():
    r = _mk(text="An answer without citations.", citations=[_cit(1)])
    assert citation_validity_rate(r) is None

def test_cvr_comma_group():
    r = _mk(text="[C1, C2, C3]", citations=[_cit(1), _cit(2)])
    assert abs(citation_validity_rate(r) - 2/3) < 1e-9


# ---------------------------------------------------------------------------
# APA_pinpoint — strict: article + paragraph + point must all match
# ---------------------------------------------------------------------------

def test_apa_pinpoint_exact_match():
    r = _mk(
        text="Article 25(2)(a) applies.",
        citations=[_cit(1, article="25", paragraph="2", point="a")],
        gold_article_pinpoints={"25(2)(a)"},
    )
    assert article_pinpoint_accuracy(r) == 1.0

def test_apa_pinpoint_partial():
    r = _mk(
        text="Art. 25 and Art. 24.",
        citations=[_cit(1, article="25"), _cit(2, article="24")],
        gold_article_pinpoints={"25(1)", "26(2)"},
    )
    # predicted={25,24}, gold={25(1),26(2)}: no exact match
    assert article_pinpoint_accuracy(r) == 0.0

def test_apa_pinpoint_partial_f1():
    r = _mk(
        text="Article 25(1) and Article 24(3).",
        citations=[_cit(1, article="25", paragraph="1"), _cit(2, article="24", paragraph="3")],
        gold_article_pinpoints={"25(1)", "26(2)"},
    )
    # TP=1, P=0.5, R=0.5, F1=0.5
    assert article_pinpoint_accuracy(r) == 0.5

def test_apa_pinpoint_none_when_gold_names_no_paragraph():
    r = _mk(
        text="Article 25(1).",
        citations=[_cit(1, article="25", paragraph="1")],
        gold_article_pinpoints={"Article 25"},
    )
    assert article_pinpoint_accuracy(r) is None
    assert article_accuracy(r) == 1.0

def test_apa_pinpoint_none_when_no_gold():
    r = _mk(text="Art. 25", gold_article_pinpoints=None)
    assert article_pinpoint_accuracy(r) is None

def test_apa_pinpoint_none_when_refused():
    r = _mk(text="refused", refused=True, gold_article_pinpoints={"25"})
    assert article_pinpoint_accuracy(r) is None


# ---------------------------------------------------------------------------
# APA_article — article-only, ignores paragraph/point
# ---------------------------------------------------------------------------

def test_apa_article_exact_match():
    r = _mk(
        text="Article 25(2)(a) applies.",
        citations=[_cit(1, article="25", paragraph="2", point="a")],
        gold_article_pinpoints={"25(2)(a)"},
    )
    assert article_accuracy(r) == 1.0

def test_apa_article_credits_coarser_prose_against_paragraph_gold():
    """The core motivation for splitting APA. Gold is 'Article 88(1)' but the
    answer only says 'Article 88' — APA_pinpoint should be 0, APA_article 1.
    Without this distinction both systems look identical at APA=0 even
    though one of them is actually citing the right article."""
    r = _mk(
        text="Article 88 applies here.",
        citations=[_cit(1, article="", paragraph=None, point=None)],
        gold_article_pinpoints={"Article 88(1)"},
    )
    assert article_accuracy(r) == 1.0
    assert article_pinpoint_accuracy(r) == 0.0

def test_apa_article_partial():
    r = _mk(
        text="Art. 25 and Art. 24.",
        citations=[_cit(1, article="25"), _cit(2, article="24")],
        gold_article_pinpoints={"25", "26"},
    )
    # predicted articles={25,24}, gold articles={25,26}, F1=0.5
    assert article_accuracy(r) == 0.5

def test_apa_article_ignores_subarticle_differences():
    """Two predictions differing only in paragraph collapse to the same article,
    so they don't inflate precision."""
    r = _mk(
        text="Article 25(1) and Article 25(2).",
        citations=[_cit(1, article="25", paragraph="1"),
                   _cit(2, article="25", paragraph="2")],
        gold_article_pinpoints={"25(3)"},
    )
    # predicted = {"25"}  (both collapsed)
    # gold      = {"25"}
    # F1 = 1.0
    assert article_accuracy(r) == 1.0

def test_apa_article_none_when_no_gold():
    r = _mk(text="Art. 25", gold_article_pinpoints=None)
    assert article_accuracy(r) is None

def test_apa_article_none_when_refused():
    r = _mk(text="refused", refused=True, gold_article_pinpoints={"25"})
    assert article_accuracy(r) is None


# ---------------------------------------------------------------------------
# MIC
# ---------------------------------------------------------------------------

def test_mic_covered():
    r = _mk(
        intent="COMPARISON",
        citations=[_cit(1, instrument="A"), _cit(2, instrument="B")],
        gold_instruments={"A", "B"},
    )
    assert multi_instrument_coverage(r) == 1.0

def test_mic_missing_instrument():
    r = _mk(
        intent="COMPARISON",
        citations=[_cit(1, instrument="A"), _cit(2, instrument="A")],
        gold_instruments={"A", "B"},
    )
    assert multi_instrument_coverage(r) == 0.0

def test_mic_na_for_factual():
    r = _mk(intent="FACTUAL_QA", gold_instruments={"A", "B"})
    assert multi_instrument_coverage(r) is None

def test_mic_refused_is_zero():
    r = _mk(intent="COMPARISON", refused=True, gold_instruments={"A", "B"})
    assert multi_instrument_coverage(r) == 0.0

def test_mic_normalizes_v1_filename_instruments():
    """V1 stores raw PDF filenames in citation.instrument (its loader has no
    notion of our canonical ids). MIC must still credit it when the
    underlying CELEX numbers map to the gold instrument set — otherwise we
    score V1 at 0 on every COMPARISON row for a pure labeling reason."""
    r = _mk(
        intent="COMPARISON",
        citations=[
            _cit(1, instrument="CELEX_32014L0065_EN_TXT.pdf"),
            _cit(2, instrument="CELEX_32019R2088_EN_TXT.pdf"),
        ],
        gold_instruments={"MIFID2_L1", "SFDR_L1"},
    )
    assert multi_instrument_coverage(r) == 1.0


# ---------------------------------------------------------------------------
# RC
# ---------------------------------------------------------------------------

def test_rc_correct_refusal():
    r = _mk(intent="OFF_TOPIC", refused=True, text="I cannot answer.",
            gold_is_off_topic=True)
    assert refusal_correctness(r) == 1.0

def test_rc_over_refusal_is_zero():
    r = _mk(intent="FACTUAL_QA", refused=True, text="I cannot answer.",
            gold_is_off_topic=False)
    assert refusal_correctness(r) == 0.0

def test_rc_should_have_refused_but_didnt():
    r = _mk(intent="OFF_TOPIC", refused=False, text="It is 42.",
            gold_is_off_topic=True)
    assert refusal_correctness(r) == 0.0

def test_rc_text_marker_detection():
    # `refused` flag False but text contains marker — should still count
    # as a refusal for the purposes of RC.
    r = _mk(intent="OFF_TOPIC", refused=False,
            text="This is outside the scope of the provided material.",
            gold_is_off_topic=True)
    assert refusal_correctness(r) == 1.0


# ---------------------------------------------------------------------------
# CD
# ---------------------------------------------------------------------------

def test_cd_bounded():
    r = _mk(text="[C1][C1][C1] " * 50, citations=[_cit(1)])
    assert citation_density(r) <= 20.0

def test_cd_zero_when_no_citations():
    r = _mk(text="No citations here at all.", citations=[_cit(1)])
    assert citation_density(r) == 0.0

def test_cd_none_on_refusal():
    r = _mk(text="refused", refused=True)
    assert citation_density(r) is None


def test_cvr_accepts_bare_numeric_brackets():
    """V2's prompt template produces "[1]" / "[2]" — the metric must count
    them as citations, otherwise V2 shows spurious CVR=1.0 (no mentions
    parsed) or CD=0 (ditto) regardless of answer quality."""
    r = _mk(text="See [1] and [2].", citations=[_cit(1), _cit(2)])
    assert citation_validity_rate(r) == 1.0

def test_cvr_mixed_bracket_forms():
    r = _mk(text="See [C1] and [2] and [C3, 4].",
            citations=[_cit(i) for i in (1, 2, 3)])
    # mentioned = {1,2,3,4}; valid = {1,2,3}; hit=3; CVR=3/4
    assert abs(citation_validity_rate(r) - 3/4) < 1e-9

def test_cd_counts_bare_numeric_brackets():
    # 10-word answer with 2 bare-numeric citations → density 20.0 (or clipped)
    r = _mk(text="See [1] for this and [2] for that nine ten eleven.",
            citations=[_cit(1), _cit(2)])
    cd = citation_density(r)
    assert cd is not None and cd > 0.0, (
        "CD must credit bare [N] brackets, else v2 CD collapses to 0.")


# ---------------------------------------------------------------------------
# compute_all smoke
# ---------------------------------------------------------------------------

def test_compute_all_returns_all_keys():
    r = _mk(text="See [C1]. Article 25(2)(a).",
            citations=[_cit(1, article="25", paragraph="2", point="a")],
            gold_article_pinpoints={"25(2)(a)"})
    out = compute_all(r)
    assert set(out) == {"CVR", "APA_article", "APA_pinpoint", "MIC", "RC", "CD"}
