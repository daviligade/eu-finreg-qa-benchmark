"""
Extract question-answer pairs from ESMA PDFs.

ESMA's "Questions and Answers" documents have a conventional layout that we
can parse with a forgiving grammar:

    <N>. Question
       <one or more lines of question text>
    Answer
       <one or more lines of answer text, often verbose>
    [until the next numbered question or the end of the section]

We also parse the MiFID II Suitability Guidelines (ESMA35-43-3172), which
does NOT follow a Q&A structure but contains a detailed Q&A-style section
in its Annex. For those we fall back on a regex that targets
"Question N:" / "Answer:" patterns.

Output schema (one JSONL record per extracted pair)
---------------------------------------------------
{
  "id":           "ESMA_QA_INVPROT-q12",
  "source":       "ESMA_QA_INVPROT",
  "source_pdf":   "data/pdfs/esma35-43-349_...pdf",
  "page_from":    41,
  "page_to":      43,
  "number":       "12",
  "question":     "Can a firm ... ?",
  "gold_answer":  "<long ESMA reply verbatim>",
  "gold_article_pinpoints": ["25", "25(2)"],   # parsed from the answer text
  "gold_instruments":       ["MIFID2_L1"],
  "intent_hint":            "FACTUAL_QA",
  "off_topic":              false
}

This script is INTENTIONALLY conservative: we prefer to drop a messy pair
than to emit a corrupted one. All records include the page range so the
human expert validator can jump straight to the source to verify.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

log = logging.getLogger("esma_extractor")


# ---------------------------------------------------------------------------
# PDF text extraction (per-page, preserve order)
# ---------------------------------------------------------------------------

def _extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for i, p in enumerate(pdf.pages, start=1):
                txt = p.extract_text() or ""
                pages.append((i, txt))
            return pages
    except Exception as e:  # noqa: BLE001
        log.warning("pdfplumber failed (%s); falling back to pypdf", e)
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    return [(i, (p.extract_text() or "")) for i, p in enumerate(reader.pages, start=1)]


# ---------------------------------------------------------------------------
# QA record
# ---------------------------------------------------------------------------

@dataclass
class QARecord:
    id: str
    source: str
    source_pdf: str
    page_from: int
    page_to: int
    number: str
    question: str
    gold_answer: str
    gold_article_pinpoints: List[str]
    gold_instruments: List[str]
    intent_hint: str
    off_topic: bool


# ---------------------------------------------------------------------------
# Article pinpoint extraction (shared with metrics.py patterns)
# ---------------------------------------------------------------------------

_ARTICLE_RE = re.compile(
    r"\b(?:Art\.|Article)\s*(\d+[A-Za-z]?)"
    r"(?:\s*\(\s*([0-9]+[a-z]?)\s*\))?"
    r"(?:\s*\(\s*([a-z])\s*\))?",
    flags=re.IGNORECASE,
)

def _extract_pinpoints(text: str) -> List[str]:
    out: List[str] = []
    for m in _ARTICLE_RE.finditer(text):
        art, para, pt = m.group(1), m.group(2), m.group(3)
        if art is None:
            continue
        s = art.lower()
        if para:
            s += f"({para.lower()})"
        if pt:
            s += f"({pt.lower()})"
        if s not in out:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Parser (A) — numbered Q&A PDFs (e.g. esma35-43-349)
# ---------------------------------------------------------------------------

# Anchors we look for. Listed from most-specific to most-permissive; the
# parser picks the pattern that matches the MOST anchors in the document.
# Each pattern MUST have a single capture group for the question number.
_Q_START_PATTERNS = [
    # "1. Question ..." / "12a. Question ..."
    re.compile(r"^\s*(\d+[a-z]?)\s*\.\s*Question\b", re.IGNORECASE | re.MULTILINE),
    # "Question 1" / "Question 1:" / "Question 1 ..."
    re.compile(r"^\s*Question\s+(\d+[a-z]?)\b", re.IGNORECASE | re.MULTILINE),
    # "Q.1" / "Q1." / "Q 1"
    re.compile(r"^\s*Q\s*\.?\s*(\d+[a-z]?)\b", re.IGNORECASE | re.MULTILINE),
    # "Q&A 1" (ESMA sometimes uses this in newer compilations)
    re.compile(r"^\s*Q\s*&\s*A\s*(\d+[a-z]?)\b", re.IGNORECASE | re.MULTILINE),
    # Fallback: "1." on its own short line followed on any line by "?"
    re.compile(r"^\s*(\d+[a-z]?)\s*\.\s*$", re.MULTILINE),
]

# Answer header variants. We accept "Answer" as a standalone line, as a
# paragraph lead-in (followed by text on the same line), and "A:" / "A.".
_ANSWER_PATTERNS = [
    # "Answer" as a standalone (or nearly-standalone) header
    re.compile(r"(?:^|\n)\s*Answer\s*[:\.\s]", re.IGNORECASE),
    # "A:" / "A." paragraph lead-in (common in newer ESMA Q&A compilations)
    re.compile(r"(?:^|\n)\s*A\s*[:\.]\s", re.IGNORECASE),
    # Last-resort: the literal word "Answer" anywhere (with word boundaries)
    re.compile(r"\bAnswer\b", re.IGNORECASE),
]


def _concat_with_page_index(pages: List[Tuple[int, str]]) -> Tuple[str, List[int]]:
    """Return a single concatenated string and a char->page_no index."""
    buf = []
    index: List[int] = []
    for page_no, text in pages:
        buf.append(text)
        # Mark each character in this page with its page number.
        index.extend([page_no] * (len(text) + 1))  # +1 for the joining '\n'
        buf.append("\n")
    return "".join(buf), index


def _page_of(idx: List[int], pos: int) -> int:
    if not idx:
        return 0
    return idx[min(pos, len(idx) - 1)]


def _pick_best_q_pattern(text: str) -> Tuple[Optional[re.Pattern], List[re.Match]]:
    """Return the Q-anchor pattern that produces the most matches.

    ESMA's Q&A PDFs are inconsistent: different compilations use different
    numbering styles. Rather than trying them in a fixed order and stopping
    at the first hit (which may be a single false-positive), we pick the
    pattern with the highest match count.
    """
    best_pat: Optional[re.Pattern] = None
    best_matches: List[re.Match] = []
    for pat in _Q_START_PATTERNS:
        ms = list(pat.finditer(text))
        if len(ms) > len(best_matches):
            best_pat, best_matches = pat, ms
    return best_pat, best_matches


_LEADING_Q_HEADER = re.compile(
    r"^\s*(?:\d+[a-z]?\s*\.\s*Question|Question\s+\d+[a-z]?|Q\s*\.?\s*\d+[a-z]?"
    r"|Q\s*&\s*A\s*\d+[a-z]?)\s*[:\.\-–]?\s*",
    re.IGNORECASE,
)


def _parse_numbered_qa(
    text: str,
    page_index: List[int],
    source_id: str,
    pdf_path: str,
    instruments_hint: List[str],
    min_answer_words: int = 10,
    min_question_words: int = 3,
) -> List[QARecord]:
    """Iterate over "Question N ... Answer ... Question N+1" blocks."""
    recs: List[QARecord] = []

    best_pat, matches = _pick_best_q_pattern(text)
    if not matches:
        log.info("%s: no Q-anchor pattern matched.", source_id)
        return []
    log.info("%s: best Q-pattern %r produced %d anchors.",
             source_id, best_pat.pattern if best_pat else "?", len(matches))

    skipped_no_answer = 0
    skipped_short = 0

    for i, m in enumerate(matches):
        q_start = m.start()
        q_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[q_start:q_end]

        # Split at the first "Answer" header — try all patterns, take the
        # EARLIEST hit after the question text (so we don't grab a mention
        # of "Answer" inside the question).
        a_match: Optional[re.Match] = None
        for pat in _ANSWER_PATTERNS:
            for candidate in pat.finditer(block):
                # Reject anchors within the first 20 chars — that's usually
                # inside the question header itself.
                if candidate.start() < 20:
                    continue
                if a_match is None or candidate.start() < a_match.start():
                    a_match = candidate
                    break  # earliest hit for this pattern
            if a_match:
                break

        if not a_match:
            skipped_no_answer += 1
            continue

        q_text = block[: a_match.start()].strip()
        a_text = block[a_match.end():].strip()

        # Drop the leading "Question N" header from the question body.
        q_text = _LEADING_Q_HEADER.sub("", q_text, count=1).strip()

        # Minimum viable lengths — filters out spurious "Question" mentions
        # in running prose.
        if len(q_text.split()) < min_question_words or len(a_text.split()) < min_answer_words:
            skipped_short += 1
            continue

        number = m.group(1) if m.groups() else str(i + 1)
        page_from = _page_of(page_index, q_start)
        page_to = _page_of(page_index, q_end - 1)

        recs.append(QARecord(
            id=f"{source_id}-{len(recs) + 1:03d}-q{number}",
            source=source_id,
            source_pdf=pdf_path,
            page_from=page_from,
            page_to=page_to,
            number=str(number),
            question=re.sub(r"\s+", " ", q_text).strip(),
            gold_answer=re.sub(r"[ \t]+", " ", a_text).strip(),
            gold_article_pinpoints=_extract_pinpoints(a_text),
            gold_instruments=list(instruments_hint),
            intent_hint="FACTUAL_QA",
            off_topic=False,
        ))

    log.info("%s: accepted %d, skipped %d (no-answer) + %d (too short).",
             source_id, len(recs), skipped_no_answer, skipped_short)
    return recs


# ---------------------------------------------------------------------------
# Parser (B) — Guidelines PDFs (e.g. ESMA35-43-3172).
# These are NOT Q&A; they contain numbered "Guideline N" paragraphs with
# supporting guidance paragraphs. We extract each Guideline as a pseudo-QA
# pair: the Guideline heading becomes the question, and the supporting
# paragraphs become the gold answer. Intent is still FACTUAL_QA; a human
# curator can re-tag these.
# ---------------------------------------------------------------------------

_GUIDELINE_ANCHOR = re.compile(
    r"^\s*Guideline\s+(\d+)\s*[:\.\-–]?\s*(.*?)$",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_guidelines(
    text: str,
    page_index: List[int],
    source_id: str,
    pdf_path: str,
    instruments_hint: List[str],
    min_body_words: int = 30,
) -> List[QARecord]:
    """Extract "Guideline N: <title> / <body paragraphs>" as pseudo-QA."""
    recs: List[QARecord] = []
    matches = list(_GUIDELINE_ANCHOR.finditer(text))
    if not matches:
        log.info("%s: no 'Guideline N' anchors found.", source_id)
        return []
    log.info("%s: found %d 'Guideline N' anchors.", source_id, len(matches))

    for i, m in enumerate(matches):
        g_start = m.start()
        g_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[g_start:g_end]
        number = m.group(1)
        title = (m.group(2) or "").strip()
        # Body is whatever comes after the first newline of the heading.
        first_nl = block.find("\n")
        body = block[first_nl + 1:].strip() if first_nl >= 0 else ""
        if len(body.split()) < min_body_words:
            continue

        question = (
            f"According to ESMA Guideline {number} "
            f"({title or 'on MiFID II suitability'}), "
            f"what does the guideline require investment firms to do?"
        ).strip()

        page_from = _page_of(page_index, g_start)
        page_to = _page_of(page_index, g_end - 1)

        recs.append(QARecord(
            id=f"{source_id}-g{number}",
            source=source_id,
            source_pdf=pdf_path,
            page_from=page_from,
            page_to=page_to,
            number=str(number),
            question=question,
            gold_answer=re.sub(r"[ \t]+", " ", body).strip(),
            gold_article_pinpoints=_extract_pinpoints(body),
            gold_instruments=list(instruments_hint),
            intent_hint="FACTUAL_QA",
            off_topic=False,
        ))
    log.info("%s: extracted %d Guideline pseudo-QA records.",
             source_id, len(recs))
    return recs


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

DEFAULT_INSTRUMENTS_FOR = {
    # Map source_id -> instruments the Q&A targets. Used as a DEFAULT tag
    # that a human can revise later.
    "ESMA_QA_INVPROT": ["MIFID2_L1"],
    "ESMA_GL_SUIT":    ["MIFID2_L1"],
}


def extract_one(pdf_path: str, source_id: str) -> List[QARecord]:
    """Run the Q&A extractor on a single PDF.

    Strategy cascade:
      1. Numbered Q&A (most ESMA compilations)
      2. If <3 records came out AND the PDF looks like the Suitability
         Guidelines, fall back to the Guidelines parser.
    """
    pages = _extract_pages(pdf_path)
    if not pages:
        log.warning("No pages extracted from %s", pdf_path)
        return []
    text, index = _concat_with_page_index(pages)
    instruments = DEFAULT_INSTRUMENTS_FOR.get(source_id, [])

    # Primary strategy: Q&A anchors.
    recs = _parse_numbered_qa(text, index, source_id, pdf_path, instruments)

    # Fallback strategy: Guidelines (only if Q&A extraction was thin).
    is_guidelines_doc = (
        "guidelines" in source_id.lower()
        or "suit" in source_id.lower()
        or "guidelines" in Path(pdf_path).name.lower()
    )
    if len(recs) < 3 and is_guidelines_doc:
        log.info("%s: Q&A yielded %d records; trying Guidelines parser.",
                 source_id, len(recs))
        guideline_recs = _parse_guidelines(text, index, source_id, pdf_path, instruments)
        if guideline_recs:
            # Dedupe by id (Q&A records use q<N>, Guidelines use g<N>).
            recs.extend(guideline_recs)

    if not recs:
        log.warning(
            "%s: no records extracted. If this is the Suitability Guidelines "
            "(ESMA35-43-3172), that doc has no Q&A structure — use the "
            "synthetic generator to produce questions seeded from its "
            "content, or manually curate. For esma35-43-349 style PDFs, "
            "inspect the first page text to see what headers ESMA used.",
            source_id,
        )
    log.info("%s: total extracted %d records from %s",
             source_id, len(recs), pdf_path)
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="manifest.yaml",
                    help="Path to manifest.yaml with corpus.esma_qa entries.")
    ap.add_argument("--out", default="data/esma_qa.jsonl",
                    help="Output JSONL file.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap on records per source (useful for --pilot).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s %(message)s")

    import yaml
    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)

    all_recs: List[QARecord] = []
    for src in manifest["corpus"].get("esma_qa", []):
        path = src["path"]
        if not Path(path).exists():
            log.error("Missing PDF: %s — run `bash dataset/download_esma.sh` first.", path)
            continue
        recs = extract_one(path, src["id"])
        if args.limit:
            recs = recs[: args.limit]
        all_recs.extend(recs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in all_recs:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_recs)} records to {args.out}")


if __name__ == "__main__":
    main()
