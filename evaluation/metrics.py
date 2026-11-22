"""
Custom regulatory metrics for evaluation.

All metrics are DETERMINISTIC — no LLM calls. They operate on the
structured AnswerRecord fields produced by pipelines. See harness.py for the
AnswerRecord schema.

Metrics implemented
-------------------
- CVR          : Citation Validity Rate        (per-answer, range [0,1])
- APA_article  : Article Accuracy (article number only)
- APA_pinpoint : Article Pinpoint Accuracy (article + paragraph + point, strict)
- MIC          : Multi-Instrument Coverage     (per-answer, 0/1; only for COMPARISON)
- RC           : Refusal Correctness           (per-answer, 0/1; only for OFF_TOPIC)
- CD           : Citation Density              (per-answer, count normalized by length)


Every function returns either a float in [0,1] or None when the metric is
not applicable (e.g. MIC on a non-comparison question). `None` must be
treated as MISSING by the aggregator — do NOT convert to 0 — otherwise
intent-conditional metrics will be biased downward.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence, Set

# ---------------------------------------------------------------------------
# Regex patterns — kept small and explicit so failures are easy to debug.
# ---------------------------------------------------------------------------

_CITATION_TOKEN_RE = re.compile(r"\[(C?\d+(?:\s*,\s*C?\d+)*)\]")

# A normalized article pinpoint looks like "Art. 25(2)(a)" or "Article 25"
# We match liberally to catch as many forms as the answer may use.
_ARTICLE_RE = re.compile(
    r"\b(?:Art\.|Article)\s*(\d+[A-Za-z]?)"   # Art. 25 or Article 25
    r"(?:\s*\(\s*([0-9]+[a-z]?)\s*\))?"          # optional (2)
    r"(?:\s*\(\s*([a-z])\s*\))?",                # optional (a)
    flags=re.IGNORECASE,
)

_REFUSAL_MARKERS = (
    "cannot answer",
    "unable to answer",
    "not covered by",
    "outside the scope",
    "no supporting passage",
    "i don't have",
    "i do not have",
    "no relevant information",
    # Observed in-the-wild LLM refusal phrasings (gemma4, deepseek-r1)
    "doesn't contain this information",
    "does not contain this information",
    "doesn't contain the",
    "does not contain the",
    "could not find this information",
    "could not find the information",
    "no information about",
    "no mention of",
    "the document doesn't contain",
    "the document does not contain",
    "the documents don't contain",
    "the documents do not contain",
    "i could not provide a grounded answer",
    "i am not designed to reply",
    "the provided context does not",
    "the provided text does not",
    "the provided documents do not",
    "there is no information",
    "there is no mention",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_citation_ids(answer_text: str) -> Set[int]:
    """Extract the set of citation IDs referenced by the answer.

    >>> sorted(_parse_citation_ids("See [C1] and [C3, C5] and [C1]"))
    [1, 3, 5]
    >>> _parse_citation_ids("nothing to cite")
    set()
    """
    ids: Set[int] = set()
    for m in _CITATION_TOKEN_RE.finditer(answer_text):
        for piece in m.group(1).split(","):
            piece = piece.strip().lstrip("C").strip()
            if piece.isdigit():
                ids.add(int(piece))
    return ids


def _normalize_article_pinpoint(
    article: str,
    paragraph: Optional[str] = None,
    point: Optional[str] = None,
) -> str:
    """Normalize a pinpoint to canonical form: '25', '25(2)', '25(2)(a)'.

    >>> _normalize_article_pinpoint("25")
    '25'
    >>> _normalize_article_pinpoint("25", "2")
    '25(2)'
    >>> _normalize_article_pinpoint("25", "2", "a")
    '25(2)(a)'
    """
    out = article.strip().lower()
    if paragraph:
        out += f"({paragraph.strip().lower()})"
    if point:
        out += f"({point.strip().lower()})"
    return out


def _extract_article_pinpoints(text: str) -> Set[str]:
    """Extract every article pinpoint mentioned in `text` as a canonical set.

    >>> _extract_article_pinpoints("see Art. 25(2)(a) and Article 24")
    {'24', '25(2)(a)'}
    """
    out: Set[str] = set()
    for m in _ARTICLE_RE.finditer(text):
        art, para, pt = m.group(1), m.group(2), m.group(3)
        if art is None:
            continue
        out.add(_normalize_article_pinpoint(art, para, pt))
    return out


# Matches a bare pinpoint like "25", "25(2)", "25(2)(a)" with optional
# surrounding whitespace.  Used ONLY to canonicalize gold entries that are
# already in article-free form (e.g. ESMA Q&A gold).
_BARE_PINPOINT_RE = re.compile(
    r"^\s*(\d+[A-Za-z]?)"                   # 25
    r"(?:\s*\(\s*([0-9]+[a-z]?)\s*\))?"     # (2)
    r"(?:\s*\(\s*([a-z])\s*\))?\s*$",       # (a)
    flags=re.IGNORECASE,
)


def _canonicalize_gold_pinpoint(g: str) -> Set[str]:
    """Canonicalize a single gold pinpoint string to the same format the
    predicted-side extractor produces.

    Accepts:
      - "Article 86(2)(a)"      → {"86(2)(a)"}
      - "Art. 25(2)"            → {"25(2)"}
      - "86(2)(a)"              → {"86(2)(a)"}      (bare ESMA form)
      - "MIFID2_L1(Article 67(3))" → {"67(3)"}      (inner Article rides along)
      - "MIFID2_L1(1)"          → set()             (names no article; ignored)
      - "" / None               → set()

    >>> _canonicalize_gold_pinpoint("Article 86(2)(a)")
    {'86(2)(a)'}
    >>> _canonicalize_gold_pinpoint("27(10)(a)")
    {'27(10)(a)'}
    >>> _canonicalize_gold_pinpoint("MIFID2_L1(1)")
    set()
    """
    if not g:
        return set()
    g = g.strip()
    # 1) Direct regex hit on "Article N(p)(pt)" anywhere in the string.
    found = _extract_article_pinpoints(g)
    if found:
        return found
    # 2) Bare form like "25(2)(a)" or "25" — no "Article" prefix.
    m = _BARE_PINPOINT_RE.match(g)
    if m:
        return {_normalize_article_pinpoint(
            m.group(1), m.group(2), m.group(3))}
    # 3) A string that names no article, like "MIFID2_L1(1)", is ignored:
    #    it cannot match any predicted pinpoint.
    return set()


def _canonicalize_gold_pinpoints(gold: Sequence[str]) -> Set[str]:
    """Canonicalize a whole gold set, unioning each entry's canonical form."""
    out: Set[str] = set()
    for g in gold or ():
        out |= _canonicalize_gold_pinpoint(g)
    return out


# Reduces a canonicalized pinpoint to its article-only form.
#   "25"        -> "25"
#   "25(2)"     -> "25"
#   "25(2)(a)"  -> "25"
#   "25a"       -> "25a"        (suffixed article numbers like "25a" stay intact)
#   ""          -> ""
_ARTICLE_ONLY_RE = re.compile(r"^(\d+[a-z]?)", re.IGNORECASE)


def _strip_to_article_only(pinpoint: str) -> str:
    """Return just the article portion of a canonicalized pinpoint.

    >>> _strip_to_article_only("25(2)(a)")
    '25'
    >>> _strip_to_article_only("25a")
    '25a'
    >>> _strip_to_article_only("")
    ''
    """
    if not pinpoint:
        return ""
    m = _ARTICLE_ONLY_RE.match(pinpoint)
    return m.group(1).lower() if m else ""


def _collect_predicted_pinpoints(rec) -> Set[str]:
    """Build the predicted pinpoint set for APA_article / APA_pinpoint.

    Shared between the two APA metrics so any future fix to what counts as
    a "predicted pinpoint" (e.g. prose-extraction changes, citation-key
    renames) lands in exactly one place.
    """
    predicted = _extract_article_pinpoints(rec.answer_text)
    for c in rec.citations:
        predicted.add(_normalize_article_pinpoint(
            c.get("article", ""),
            c.get("paragraph"),
            c.get("point"),
        ))
    predicted.discard("")
    return predicted


# ---------------------------------------------------------------------------
# AnswerRecord — thin dataclass shared with the harness
# ---------------------------------------------------------------------------

@dataclass
class AnswerRecord:
    """Everything a pipeline must produce per question.

    Fields marked (required) are used by at least one metric. Optional fields
    are for logging only.
    """

    # Identification
    question_id: str                      # required
    system_id: str
    intent: str                           # required, matches IntentLabel enum

    # Model output
    answer_text: str                      # required, full answer string
    refused: bool                         # required — did the system decline?

    # Retrieval / grounding evidence
    # Each element describes one cited passage. The order in this list MUST
    # match the [C1]..[Ck] citation IDs in `answer_text` (1-indexed).
    citations: list                       # required; list[CitationInfo-like dict]

    # Gold data (optional, filled in by the dataset builder per question)
    gold_article_pinpoints: Optional[Set[str]] = None
    gold_instruments: Optional[Set[str]] = None
    gold_is_off_topic: Optional[bool] = None
    gold_answer: Optional[str] = None

    # Book-keeping
    latency_s: Optional[float] = None
    retries: Optional[int] = None


# Each entry in `citations` is expected to be a dict with at least:
#   {
#       "citation_id": int,            # 1-based id matching [Ck]
#       "instrument": str,             # e.g. "MIFID2_L1"
#       "article": str,                # e.g. "25"
#       "paragraph": Optional[str],    # e.g. "2"
#       "point": Optional[str],        # e.g. "a"
#       "passage": str,                # retrieved text
#   }


# ---------------------------------------------------------------------------
# CVR — Citation Validity Rate
# ---------------------------------------------------------------------------

def citation_validity_rate(rec: AnswerRecord) -> Optional[float]:
    """Fraction of citation IDs in the answer that resolve to a valid passage.


    >>> r = AnswerRecord("q1","V2","FACTUAL_QA","see [C1] and [C9]", False,
    ...                  [{"citation_id":1,"instrument":"X","article":"1",
    ...                    "paragraph":None,"point":None,"passage":""}])
    >>> citation_validity_rate(r)
    0.5
    """
    if rec.refused:
        return None
    mentioned = _parse_citation_ids(rec.answer_text)
    if not mentioned:
        return None
    valid_ids = {c["citation_id"] for c in rec.citations}
    hit = sum(1 for i in mentioned if i in valid_ids)
    return hit / len(mentioned)


# ---------------------------------------------------------------------------
# APA — Article Pinpoint Accuracy
#
# Split into two metrics:
#   * APA_article  — F1 on just the article number ("88" vs "88")
#   * APA_pinpoint — F1 on the full pinpoint    ("88(1)" vs "88(1)")
#
# The split exists because gold is paragraph-level ("Article 88(1)") while
# answer prose typically says "Article 88". The strict single-metric version
# therefore reported 0 for "cited the right article but not the paragraph",
# hiding a real signal. Two metrics separate the two claims.
# ---------------------------------------------------------------------------

def _apa_f1(predicted: Set[str], gold: Set[str]) -> float:
    """Standard set F1 with the corner-case conventions APA uses:
    - both empty -> 1.0 (nothing to measure, nothing missed)
    - one side empty -> 0.0 (missed or hallucinated)
    """
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def article_accuracy(rec: AnswerRecord) -> Optional[float]:
    """APA_article: F1 on article numbers only.

    Credits a system for citing the right article even when the answer doesn't
    identify a specific paragraph. Predicted "88" matches gold "88(1)" because
    both reduce to article "88".

    >>> r = AnswerRecord("q1","V2","FACTUAL_QA","Article 88 applies.", False,
    ...                  [{"citation_id":1,"instrument":"X","article":"",
    ...                    "paragraph":None,"point":None,"passage":""}],
    ...                  gold_article_pinpoints={"Article 88(1)"})
    >>> article_accuracy(r)
    1.0
    """
    if rec.refused or not rec.gold_article_pinpoints:
        return None
    predicted = {_strip_to_article_only(p)
                 for p in _collect_predicted_pinpoints(rec)}
    predicted.discard("")
    gold = {_strip_to_article_only(g)
            for g in _canonicalize_gold_pinpoints(rec.gold_article_pinpoints)}
    gold.discard("")
    return _apa_f1(predicted, gold)


def article_pinpoint_accuracy(rec: AnswerRecord) -> Optional[float]:
    """APA_pinpoint: F1 on the full article + paragraph + point pinpoint.

    Strict. A system must cite the exact paragraph/point to earn credit.
    Use together with APA_article to separate "got the article" from
    "got the specific paragraph". Computed only where the gold reference
    names a paragraph; a gold reference to a whole article gives None.

    >>> r = AnswerRecord("q1","V2","FACTUAL_QA","Art. 25(2)(a)", False,
    ...                  [{"citation_id":1,"instrument":"X","article":"25",
    ...                    "paragraph":"2","point":"a","passage":""}],
    ...                  gold_article_pinpoints={"25(2)(a)"})
    >>> article_pinpoint_accuracy(r)
    1.0
    """
    if rec.refused or not rec.gold_article_pinpoints:
        return None
    predicted = _collect_predicted_pinpoints(rec)
    # Gold side: canonicalize each entry to the same bare form as predicted.
    # Accepted gold formats:
    #   ESMA Q&A:      "24(1)" / "27(10)(a)"       (bare)
    #   Synthetic:     "Article 86(2)(a)"          (Article-prefixed)
    #   Wrapped:       "MIFID2_L1(Article 67(3))"  (instrument-wrapped)
    gold = _canonicalize_gold_pinpoints(rec.gold_article_pinpoints)
    if not any("(" in g for g in gold):
        return None
    return _apa_f1(predicted, gold)


# ---------------------------------------------------------------------------
# MIC — Multi-Instrument Coverage
# ---------------------------------------------------------------------------

def _normalize_instrument_label(raw: str) -> str:
    """Map an arbitrary instrument label to its canonical id (e.g. MIFID2_L1).


    The same alias map (source of truth) lives in wrappers/_cite_parse.py so
    that newly-generated rows are already canonicalized at write time.
    Labels are normalized again at read time.

    Returns the original string unchanged if no alias matches (so already-
    canonical ids like "MIFID2_L1" pass through untouched).
    """
    from wrappers._cite_parse import _INSTRUMENT_ALIASES  # local import avoids cycle
    if not raw:
        return ""
    low = raw.lower()
    for alias, inst_id in _INSTRUMENT_ALIASES.items():
        if alias in low:
            return inst_id
    return raw


def multi_instrument_coverage(rec: AnswerRecord) -> Optional[float]:
    """For comparison questions, did the answer cite at least two instruments
    AND did those instruments equal (or superset) the gold instrument set?

    Returns 1.0 if covered, 0.0 otherwise. Returns None when the intent is
    not COMPARISON or when gold_instruments is missing.


    >>> r = AnswerRecord("q1","V2","COMPARISON","x", False,
    ...                  [{"citation_id":1,"instrument":"A","article":"1",
    ...                    "paragraph":None,"point":None,"passage":""},
    ...                   {"citation_id":2,"instrument":"B","article":"1",
    ...                    "paragraph":None,"point":None,"passage":""}],
    ...                  gold_instruments={"A","B"})
    >>> multi_instrument_coverage(r)
    1.0
    """
    if rec.intent != "COMPARISON":
        return None
    if rec.gold_instruments is None:
        return None
    if rec.refused:
        return 0.0
    cited = {_normalize_instrument_label(c.get("instrument", ""))
             for c in rec.citations}
    cited.discard("")
    if len(cited) < 2:
        return 0.0
    return 1.0 if rec.gold_instruments.issubset(cited) else 0.0


# ---------------------------------------------------------------------------
# RC — Refusal Correctness
# ---------------------------------------------------------------------------

def _looks_like_refusal(answer_text: str) -> bool:
    t = answer_text.lower()
    return any(m in t for m in _REFUSAL_MARKERS)


def refusal_correctness(rec: AnswerRecord) -> Optional[float]:
    """For off-topic questions, did the system refuse?

    - If gold_is_off_topic is True:  1.0 iff the system refused (via flag OR
      refusal-marker detection in the text).
    - If gold_is_off_topic is False: 1.0 iff the system did NOT refuse.
    - If gold_is_off_topic is None:  returns None (metric not applicable).

    Note: we accept both the structural `rec.refused` flag AND text markers
    because not all pipelines set the flag reliably.

    >>> r = AnswerRecord("q1","V2","OFF_TOPIC","I cannot answer.", True,
    ...                  [], gold_is_off_topic=True)
    >>> refusal_correctness(r)
    1.0
    """
    if rec.gold_is_off_topic is None:
        return None
    refused = rec.refused or _looks_like_refusal(rec.answer_text)
    return 1.0 if refused == bool(rec.gold_is_off_topic) else 0.0


# ---------------------------------------------------------------------------
# CD — Citation Density
# ---------------------------------------------------------------------------

def citation_density(rec: AnswerRecord) -> Optional[float]:
    """Number of distinct citation IDs per 100 words of answer text.

    - Returns None for refused answers (they have no body to score).
    - Unbounded above, but clipped to 20.0 (a hard cap prevents degenerate
      pipelines that spam [C1][C1][C1] from flattening distributions).

    >>> r = AnswerRecord("q1","V2","FACTUAL_QA",
    ...                  "See [C1] [C2] [C3]. "*20, False,
    ...                  [{"citation_id":i,"instrument":"X","article":"1",
    ...                    "paragraph":None,"point":None,"passage":""} for i in (1,2,3)])
    >>> cd = citation_density(r)
    >>> 0 < cd < 20
    True
    """
    if rec.refused:
        return None
    ids = _parse_citation_ids(rec.answer_text)
    words = max(1, len(rec.answer_text.split()))
    density = 100.0 * len(ids) / words
    return min(density, 20.0)


# ---------------------------------------------------------------------------
# Public API — compute everything for one answer
# ---------------------------------------------------------------------------

METRIC_FUNCS = {
    "CVR": citation_validity_rate,
    "APA_article": article_accuracy,
    "APA_pinpoint": article_pinpoint_accuracy,
    "MIC": multi_instrument_coverage,
    "RC": refusal_correctness,
    "CD": citation_density,
}


def compute_all(rec: AnswerRecord) -> dict:
    """Return {metric_name: float_or_None} for every metric."""
    return {name: fn(rec) for name, fn in METRIC_FUNCS.items()}


if __name__ == "__main__":
    # Quick smoke test so `python metrics.py` is informative.
    import doctest
    fails, attempted = doctest.testmod(verbose=False)
    print(f"doctests: {attempted - fails}/{attempted} passed")
