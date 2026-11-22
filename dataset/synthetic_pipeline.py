"""
Multi-agent synthetic QA generation.

Four agents, orchestrated in sequence, each with a different LLM persona:

    G  — Generator     drafts a question + proposed gold answer given an
                       article (or pair of articles for COMPARISON).
    A  — Answerer      re-reads ONLY the cited passage(s) and writes a
                       defensive answer. This answer is what we keep as
                       the gold answer. G's draft is discarded.
    V  — Verifier      scores the pair on a rubric (clarity, groundedness,
                       pinpoint accuracy, contamination). Discards if any
                       score < 0.8.
    X  — Adversary     tries to break the question: asks a paraphrase, an
                       off-topic near-miss, or a comparison variant. These
                       become additional questions in the dataset.

Why four agents?
----------------
- Separation of roles reduces the chance that the same prompt both produces
  the question AND judges it (an obvious circularity).
- Discarding G's draft answer and rewriting with A forces the gold to be
  grounded in the passage rather than in G's (possibly hallucinated) draft.
- V is the quality gate: a single threshold (default 0.8/1.0 on every rubric
  axis) keeps only high-fidelity pairs.
- X increases linguistic diversity and supplies off-topic negatives needed
  for refusal-correctness evaluation.

Inputs
------
- AKN article dump (parsed from 32014L0065.akn, 32019R2088.akn, etc.)
- An Ollama model used for G/A/V/X (default: the "answerer" model in the
  manifest). For real use, pass different models per agent via CLI flags.

Output (one JSONL record per generated question)
------------------------------------------------
{
  "id": "SYN-MIFID2_L1-25-FACTUAL_QA-001",
  "source": "synthetic",
  "seed_article": "25",
  "seed_instrument": "MIFID2_L1",
  "intent_hint": "FACTUAL_QA",
  "question": "...",
  "gold_answer": "...",
  "gold_article_pinpoints": ["25(2)(a)"],
  "gold_instruments": ["MIFID2_L1"],
  "off_topic": false,
  "agent_v_score": 0.92,
  "agent_v_notes": "..."
}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("syn")


from lxml import etree


@dataclass
class Article:
    instrument: str
    number: str       # "25"
    heading: str
    text: str         # full article body (paragraphs joined)


def _load_akn_articles(path: str, instrument: str) -> List[Article]:
    """Parse an AKN file and yield articles with joined body text."""
    arts: List[Article] = []
    try:
        tree = etree.parse(path)
    except Exception as e:  # noqa: BLE001
        log.error("Failed to parse %s: %s", path, e)
        return arts

    # AKN namespaces vary; use local-name() to be tolerant.
    for art in tree.iter():
        local = etree.QName(art.tag).localname if art.tag else ""
        if local != "article":
            continue
        # Article number
        number = ""
        for n in art.iter():
            if etree.QName(n.tag).localname == "num" and n.text:
                number = n.text.strip().replace("Article", "").strip()
                break
        # Heading
        heading = ""
        for h in art.iter():
            if etree.QName(h.tag).localname == "heading" and h.text:
                heading = h.text.strip()
                break
        # Body text — concatenate all descendant <p> elements
        paras: List[str] = []
        for p in art.iter():
            if etree.QName(p.tag).localname == "p":
                if p.text:
                    paras.append(" ".join(p.itertext()).strip())
        body = "\n".join(t for t in paras if t)
        if number and body:
            arts.append(Article(instrument=instrument,
                                 number=number, heading=heading, text=body))
    return arts


# ---------------------------------------------------------------------------
# Ollama client (shared helper, avoids importing baselines._common to keep
# this script runnable standalone).
#
# Big-model reality: gemma4:31b on Apple Silicon can take 60–240s just to
# warm up its KV-cache on the first call, and 30–120s per subsequent call
# for long prompts. The default ollama-py timeout (300s) is NOT enough.
# We use:
#   - a long default timeout (900s) configurable via manifest.timeout_s
#   - retries with exponential backoff on httpx.ReadTimeout / 503
#   - a global client reuse — creating a new Client per call adds overhead
# ---------------------------------------------------------------------------

import time

_DEFAULT_TIMEOUT_S = 900
_OLLAMA_CLIENT_CACHE: Dict[Tuple[str, int], object] = {}


def _get_client(host: str, timeout_s: int):
    import ollama
    key = (host, timeout_s)
    if key not in _OLLAMA_CLIENT_CACHE:
        _OLLAMA_CLIENT_CACHE[key] = ollama.Client(host=host, timeout=timeout_s)
    return _OLLAMA_CLIENT_CACHE[key]


def _chat(model: str, prompt: str, host: str = "http://localhost:11434",
          system: Optional[str] = None, temperature: float = 0.2,
          timeout_s: int = _DEFAULT_TIMEOUT_S, max_retries: int = 3,
          sample_seed: int = 13) -> str:
    """Robust chat call. Retries on timeouts with exponential backoff."""
    import ollama  # noqa: F401  (keeps the dependency explicit)
    client = _get_client(host, timeout_s)
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            r = client.chat(
                model=model,
                messages=msgs,
                options={"temperature": temperature, "num_ctx": 8192,
                         "seed": sample_seed},
            )
            return r["message"]["content"]
        except Exception as e:  # noqa: BLE001
            # Import lazily so the script still runs when httpx is not directly installed
            last_err = e
            err_name = type(e).__name__
            # Only retry on transient errors; propagate on hard failures.
            transient = (
                "ReadTimeout" in err_name
                or "ConnectTimeout" in err_name
                or "ConnectError" in err_name
                or "RemoteProtocolError" in err_name
                or "ResponseError" in err_name
            )
            if not transient or attempt == max_retries:
                raise
            wait = 2 ** attempt  # 2, 4, 8s
            log.warning("_chat transient error (%s) — retry %d/%d in %ds",
                        err_name, attempt, max_retries, wait)
            time.sleep(wait)
    # Unreachable, but keeps type checker happy
    raise last_err if last_err else RuntimeError("_chat failed without exception")


def _warmup(model: str, host: str, timeout_s: int = _DEFAULT_TIMEOUT_S) -> None:
    """Send a tiny prompt to force Ollama to load the model into memory.

    Without this, the first real generation call often exceeds a sensible
    per-call timeout because it bundles the model-load time with the token
    generation time.
    """
    log.info("Warming up %s (this may take 30–180s on first run)…", model)
    t0 = time.time()
    try:
        _chat(model, "Reply with the single word OK.",
              host=host, temperature=0.0, timeout_s=timeout_s, max_retries=2)
        log.info("Model %s ready in %.1fs", model, time.time() - t0)
    except Exception as e:  # noqa: BLE001
        log.error("Warmup failed for %s: %s", model, e)
        raise


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPT_G_FACTUAL = """You are a regulatory legal drafter. I will give you ONE article of EU financial regulation.
Produce a single concise QUESTION that a compliance officer might ask whose answer is ENTIRELY contained in this article.
The question must be specific enough that a correct answer MUST cite a sub-paragraph, not just the article number.

Return STRICTLY JSON in this shape, with no other text:
{{"question": "...", "gold_draft": "...", "pinpoint": "<art>(<para>)(<point>)"}}

ARTICLE ({instrument}, Art. {number} — {heading}):
{text}
"""

PROMPT_G_COMPARISON = """You are a regulatory legal drafter. Below are TWO articles from DIFFERENT EU financial instruments. They were pre-selected because their wording suggests they may address a shared topic.

FIRST decide honestly: do these two articles genuinely address at least one COMMON legal point (for example: what must be disclosed, to whom, under which conditions, with which duties)?

- If NO common point exists, return STRICTLY: {{"skip": true, "reason": "<=15 words"}}
- If YES, write ONE question asking to compare the two instruments on that specific shared point. The question MUST name both instruments and MUST be answerable using ONLY these two articles.

Return STRICTLY JSON: {{"question": "...", "shared_point": "<=12 words"}}

ARTICLE A ({inst_a}, Art. {num_a} — {head_a}):
{text_a}

ARTICLE B ({inst_b}, Art. {num_b} — {head_b}):
{text_b}
"""

PROMPT_A_COMPARE = """You are answering a comparison question using ONLY the two articles provided. Your answer MUST address both instruments and cite the exact paragraph in each that supports the claim (e.g. "Article 24(4)" and "Article 6(1)"). If one side does not support a definitive answer, say so.

Return STRICTLY JSON:
{{"gold_answer": "...", "pinpoints": ["<art>(<para>)", ...]}}

ARTICLE A ({inst_a}, Art. {num_a} — {head_a}):
{text_a}

ARTICLE B ({inst_b}, Art. {num_b} — {head_b}):
{text_b}

QUESTION: {question}
"""

PROMPT_V_COMPARE = """Score the following comparison Q/A pair on a 0.0-1.0 scale for each rubric axis. Return STRICTLY JSON:

{{"clarity": <float>, "groundedness": <float>, "pinpoint_accuracy": <float>, "contamination_risk": <float>, "notes": "<=40 words>"}}

- clarity: is the question unambiguous and self-contained?
- groundedness: is the gold_answer derivable from BOTH articles below? Score LOW if the two articles do not truly share the point being compared, or if the answer leans on only one of them.
- pinpoint_accuracy: do the cited article/paragraph references actually contain the claims, on BOTH sides?
- contamination_risk: would this Q/A reward memorization of public regulatory documents? 1.0 means no such risk.

QUESTION: {question}
GOLD ANSWER: {gold_answer}

ARTICLE A ({inst_a}, Art. {num_a} — {head_a}):
{text_a}

ARTICLE B ({inst_b}, Art. {num_b} — {head_b}):
{text_b}
"""

PROMPT_G_SUMMARY = """Produce ONE question that asks to SUMMARIZE the scope or purpose of the article below.
Return STRICTLY JSON: {{"question": "..."}}

ARTICLE ({instrument}, Art. {number} — {heading}):
{text}
"""

PROMPT_G_OFFTOPIC = """Produce ONE SPECIFIC question about the FINANCIAL DOMAIN BELOW. The question must be PLAUSIBLY financial but OUTSIDE the scope of EU MiFID/SFDR regulation (which covers investment-firm conduct, investor protection, and sustainability disclosures in the EU).

DOMAIN: {topic_hint}

Your question must:
- Be concrete and answerable (not a vague "how does X work?")
- Clearly fall OUTSIDE EU MiFID/SFDR — e.g. about another jurisdiction, another instrument class, or another regulator
- NOT be a generic "mortgage rates" question

Return STRICTLY JSON: {{"question": "..."}}
"""

PROMPT_G_CONVERSATIONAL = """Produce ONE brief CONVERSATIONAL message a user might send to a chatbot, such as small-talk, greetings, or thanks. NOT a regulatory or factual question.

Style hint: {style_hint}

Examples:
- "Hi, can you help me with a MiFID question?"
- "Thanks, that was really useful!"
- "Good morning — are you ready to answer some questions?"

Return STRICTLY JSON: {{"question": "..."}}
"""

PROMPT_G_META = """Produce ONE META question a user might ask a financial-regulation chatbot about the CHATBOT ITSELF — its capabilities, scope, sources, or limitations. NOT a substantive regulatory question.

Angle hint: {angle_hint}

Examples:
- "What documents do you have access to?"
- "Can you cite both MiFID II and SFDR in a single answer?"
- "How current is your knowledge of ESMA guidelines?"
- "Can you explain how you retrieve passages?"

Return STRICTLY JSON: {{"question": "..."}}
"""

PROMPT_A_FACTUAL = """You are answering a compliance question using ONLY the article provided. You MUST cite the exact paragraph and point (e.g. "Article 25(2)(a)") that supports your claim. If the article does not support a definitive answer, say so.

Return STRICTLY JSON:
{{"gold_answer": "...", "pinpoints": ["<art>(<para>)(<point>)", ...]}}

ARTICLE ({instrument}, Art. {number} — {heading}):
{text}

QUESTION: {question}
"""

PROMPT_A_OFFTOPIC = """You are a regulatory assistant. The question below is OFF-TOPIC for the EU MiFID/SFDR corpus you have access to.
Return STRICTLY JSON: {{"gold_answer": "I cannot answer from the provided material."}}

QUESTION: {question}
"""

PROMPT_A_CONVERSATIONAL = """The message below is conversational small-talk, not a regulatory question. Produce a brief (1-2 sentence) friendly, helpful reply — professional but warm — that acknowledges the user and offers to help with MiFID II / SFDR questions.

Return STRICTLY JSON: {{"gold_answer": "..."}}

MESSAGE: {question}
"""

PROMPT_A_META = """The question below is a META question about a financial-regulation chatbot's own capabilities or scope, not a substantive regulatory question. Produce a brief (1-3 sentence) honest reply explaining what this chatbot (scope: EU MiFID II Level 1 + SFDR Level 1, using retrieval over the official texts) can or cannot answer.

Return STRICTLY JSON: {{"gold_answer": "..."}}

QUESTION: {question}
"""

PROMPT_V = """Score the following Q/A pair on a 0.0–1.0 scale for each rubric axis. Return STRICTLY JSON:

{{"clarity": <float>, "groundedness": <float>, "pinpoint_accuracy": <float>, "contamination_risk": <float>, "notes": "<≤40 words>"}}

- clarity: is the question unambiguous and self-contained?
- groundedness: is the gold_answer derivable from the cited passage?
- pinpoint_accuracy: does the cited article/paragraph/point actually contain the claim?
- contamination_risk: would this Q/A obviously reward a system that memorized the ESMA Q&A corpus? HIGH risk = LOW score here (i.e. 1.0 means no contamination risk).

QUESTION: {question}
GOLD ANSWER: {gold_answer}
CITED ARTICLE: {instrument}, Art. {number} — {heading}
ARTICLE TEXT: {text}
"""

PROMPT_X_PARA = """Rewrite the following regulatory question as a natural paraphrase that preserves its meaning and expected answer. Return STRICTLY JSON: {{"question": "..."}}
ORIGINAL: {question}
"""

# ---------------------------------------------------------------------------
# Variety pools for OFF_TOPIC / CONVERSATIONAL / META
# ---------------------------------------------------------------------------
# Without per-call topic injection the LLM collapses to a single archetype
# ("mortgage interest rates..." every time) and the dedup gate then rejects
# ~99% of attempts. Rotating through a broad domain pool makes each call a
# different prompt instance, which produces different answers. 30+ entries
# each is enough to fill a 60-row OFF_TOPIC target with headroom even if the
# LLM reuses some. We use Python's `random` module directly (not the seeded
# generator in the main loop) so topic choice isn't reproducible across runs
# — reproducibility matters for the STRUCTURED part of the dataset, but for
# adversarial off-topic noise, variety matters more than determinism.

_OFFTOPIC_DOMAINS = [
    "US SEC 10-K filings and quarterly reports",
    "cryptocurrency taxation rules in India under the 2022 VDA regime",
    "retail savings account interest rates in Japan",
    "Singapore REIT dividend yield performance",
    "Brazilian property and casualty insurance claim processing",
    "Canadian securities regulation under NI 31-103",
    "Chinese fintech IPO valuations on the Hong Kong exchange",
    "Swiss banking secrecy and FATCA compliance",
    "Australian superannuation fund mandatory-contribution rules",
    "UK FCA mortgage affordability rules post-Brexit",
    "Russia sanctions compliance for US financial institutions",
    "Middle Eastern sovereign wealth fund governance (ADIA, PIF)",
    "South African JSE bond listing requirements",
    "Indian RBI monetary policy transmission",
    "Korean won foreign-exchange intervention mechanics",
    "US SEC Regulation Best Interest and broker suitability",
    "FINRA broker-dealer net capital examinations",
    "IFRS 9 expected credit loss modelling",
    "LIBOR-to-SOFR transition for corporate loans",
    "central bank digital currency pilots in Nigeria and Ghana",
    "Islamic finance sukuk bond structures",
    "US private-equity carried-interest taxation",
    "venture capital term-sheet negotiation",
    "Hong Kong IPO underwriting fees",
    "ICE Europe energy-commodity futures margining",
    "California cap-and-trade carbon credit auctions",
    "US class-action lawsuits over ESG greenwashing",
    "robo-advisor fee structures in the US retail market",
    "US 401(k) fiduciary duty under ERISA",
    "FDIC deposit insurance coverage limits",
    "consumer credit score FICO calculation methodology",
    "auto loan origination in the US subprime market",
    "Canadian real estate mortgage stress testing (B-20 rules)",
    "Brazilian Selic rate and inflation targeting",
    "Australian banking royal commission outcomes",
    "Japanese yen-carry-trade unwind risk",
    "Chinese yuan internationalization via CIPS",
    "US student loan forgiveness programs",
    "UK NS&I savings products and Premium Bonds",
    "Mexican peso-denominated corporate bonds",
]

_CONVERSATIONAL_STYLES = [
    "casual greeting", "formal introduction", "thank-you message",
    "follow-up after a previous question", "request for help",
    "checking availability", "polite goodbye", "apology for a late question",
    "expressing frustration with something unrelated",
    "asking how the assistant is doing", "quick hello",
    "mentioning the time of day", "commenting on the weather briefly",
    "introducing themselves as a compliance officer",
    "introducing themselves as a law student",
    "asking if the assistant understands English",
]

_META_ANGLES = [
    "which documents are available", "coverage of MiFID Level 2 vs Level 1",
    "whether the assistant has access to ESMA guidelines",
    "how citations are produced", "update frequency of the knowledge base",
    "difference between MiFID II and MiFIR coverage",
    "whether the system uses LLMs or keyword search",
    "language support beyond English",
    "whether answers are legally binding",
    "how the assistant handles ambiguous questions",
    "what happens when a question is outside scope",
    "data sources and publication dates",
    "whether both MiFID II and SFDR can be cited in one answer",
    "reasoning transparency",
    "latency and response time",
    "privacy of user questions",
]


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

def _safe_json(text: str) -> Optional[dict]:
    """Extract the first JSON object from a noisy LLM response."""
    m = _JSON_BLOCK.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # Retry after trimming trailing commas — a common failure mode.
        cleaned = re.sub(r",(\s*[}\]])", r"\1", m.group(0))
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

@dataclass
class SynRecord:
    id: str
    source: str
    seed_article: str
    seed_instrument: str
    intent_hint: str
    question: str
    gold_answer: str
    gold_article_pinpoints: List[str] = field(default_factory=list)
    gold_instruments: List[str] = field(default_factory=list)
    off_topic: bool = False
    agent_v_score: float = 0.0
    agent_v_notes: str = ""


def _agent_g(art: Article, intent: str, model: str, host: str,
             timeout_s: int = _DEFAULT_TIMEOUT_S) -> Optional[dict]:
    # Intents without article grounding (OFF_TOPIC / CONVERSATIONAL / META)
    # get a per-call randomly-chosen "hint" injected into their prompt so
    # the LLM produces variety across attempts. Without this the LLM
    # collapses to a single archetype response every time (e.g. the
    # "mortgage interest rates" issue) and the dedup gate rejects 99% of
    # attempts. Temperature is lifted to 0.9 for these intents so even
    # same-hint calls explore more of the phrasing space.
    if intent == "FACTUAL_QA" or intent == "LEGAL_ADVICE":
        prompt = PROMPT_G_FACTUAL.format(
            instrument=art.instrument, number=art.number,
            heading=art.heading, text=art.text)
        temperature = 0.6
    elif intent == "SUMMARIZATION":
        prompt = PROMPT_G_SUMMARY.format(
            instrument=art.instrument, number=art.number,
            heading=art.heading, text=art.text)
        temperature = 0.6
    elif intent == "OFF_TOPIC":
        prompt = PROMPT_G_OFFTOPIC.format(
            topic_hint=random.choice(_OFFTOPIC_DOMAINS))
        temperature = 0.9
    elif intent == "CONVERSATIONAL":
        prompt = PROMPT_G_CONVERSATIONAL.format(
            style_hint=random.choice(_CONVERSATIONAL_STYLES))
        temperature = 0.9
    elif intent == "META":
        prompt = PROMPT_G_META.format(
            angle_hint=random.choice(_META_ANGLES))
        temperature = 0.9
    else:
        return None  # COMPARISON handled separately in generator()
    out = _chat(model, prompt, host=host, temperature=temperature, timeout_s=timeout_s)
    return _safe_json(out)


def _agent_g_compare(a: Article, b: Article, model: str, host: str,
                     timeout_s: int = _DEFAULT_TIMEOUT_S,
                     sample_seed: int = 13) -> Optional[dict]:
    prompt = PROMPT_G_COMPARISON.format(
        inst_a=a.instrument, num_a=a.number, head_a=a.heading, text_a=a.text,
        inst_b=b.instrument, num_b=b.number, head_b=b.heading, text_b=b.text,
    )
    return _safe_json(_chat(model, prompt, host=host, temperature=0.6,
                            timeout_s=timeout_s, sample_seed=sample_seed))


def _agent_a_compare(a: Article, b: Article, question: str, model: str,
                     host: str, timeout_s: int = _DEFAULT_TIMEOUT_S) -> Optional[dict]:
    """Gold answer for a comparison: the answerer sees BOTH articles.

    A comparison gold answer must be grounded in both instruments, so the
    single-article answer prompt is not used here.
    """
    prompt = PROMPT_A_COMPARE.format(
        inst_a=a.instrument, num_a=a.number, head_a=a.heading, text_a=a.text,
        inst_b=b.instrument, num_b=b.number, head_b=b.heading, text_b=b.text,
        question=question,
    )
    return _safe_json(_chat(model, prompt, host=host, temperature=0.0, timeout_s=timeout_s))


def _agent_v_compare(a: Article, b: Article, question: str, gold: str,
                     model: str, host: str,
                     timeout_s: int = _DEFAULT_TIMEOUT_S) -> Optional[dict]:
    """Checker for a comparison: reads BOTH articles, and its grounding
    axis explicitly punishes pairs that do not truly share the point."""
    prompt = PROMPT_V_COMPARE.format(
        inst_a=a.instrument, num_a=a.number, head_a=a.heading, text_a=a.text,
        inst_b=b.instrument, num_b=b.number, head_b=b.heading, text_b=b.text,
        question=question, gold_answer=gold,
    )
    return _safe_json(_chat(model, prompt, host=host, temperature=0.0, timeout_s=timeout_s))


_STOPWORDS = frozenset("""
the and for that with this are not any may such where when which their them
they have has been being other than within without under into from shall
also those these whose must each only means his her its our your out how
all can will would should could there here more most less least between
regard regards regarding respect accordance
article articles paragraph paragraphs subparagraph point points chapter
title annex section referred refer refers reference references
member state states competent authority authorities commission esma
european union parliament council regulation regulations directive
directives delegated implementing act acts provision provisions
requirement requirements ensure ensures ensuring order laid down set out
applicable apply applies application shall including include includes
inform information necessary appropriate relevant concerned person persons
entity entities firm firms undertaking undertakings financial investment
esas draft drafts technical standard standards regulatory submit submitted
develop develops developed developing power powers adopt adopted adopting
conferred committee joint procedure procedures granting refusing request
requests march june september december january period months month days
supervisory banking occupational pensions cooperation cooperate exchange
""".split())


def _salient_terms(text: str) -> set:
    """Words of 4+ letters that appear at least twice, minus boilerplate.

    Appearing twice filters passing mentions; the stopword list removes
    the legal boilerplate that every article shares. What survives is a
    rough topic fingerprint of the article.
    """
    counts: Dict[str, int] = {}
    for w in re.findall(r"[a-z]{4,}", (text or "").lower()):
        if w in _STOPWORDS:
            continue
        counts[w] = counts.get(w, 0) + 1
    return {w for w, c in counts.items() if c >= 2}


# Articles whose heading marks them as structural or procedural rather than
# substantive. A definitions article shares vocabulary with every topic
# without sharing a legal point, so it pairs with everything and produces
# empty comparisons. Same for amendment and entry-into-force plumbing.
_NON_SUBSTANTIVE_HEADING = re.compile(
    r"definition|amendment|entry into force|transposition|repeal|"
    r"exercise of the delegation|committee procedure|addressees|"
    r"transitional provision|review",
    re.IGNORECASE)


def _build_comparison_pairs(
    articles_by_instrument: Dict[str, List[Article]],
    top_k: int = 80,
    min_chars: int = 400,
) -> List[Tuple[Article, Article, float]]:
    """Rank every cross-instrument article pair by topic overlap.

    Returns the top_k pairs as (article_a, article_b, jaccard_score),
    best first. Deterministic: no model calls, no randomness. Structural
    and procedural articles (definitions, amendments, entry into force)
    are excluded on both sides; see _NON_SUBSTANTIVE_HEADING.
    """
    insts = list(articles_by_instrument.keys())
    if len(insts) < 2:
        return []

    def _substantive(arts: List[Article]) -> List[Article]:
        return [x for x in arts
                if len(x.text) >= min_chars
                and not _NON_SUBSTANTIVE_HEADING.search(x.heading or "")]

    scored: List[Tuple[Article, Article, float]] = []
    for i in range(len(insts)):
        for j in range(i + 1, len(insts)):
            arts_a = _substantive(articles_by_instrument[insts[i]])
            arts_b = _substantive(articles_by_instrument[insts[j]])
            terms_a = {id(x): _salient_terms(x.text) for x in arts_a}
            terms_b = {id(x): _salient_terms(x.text) for x in arts_b}
            # A tiny fingerprint (a short procedural article) produces
            # chance overlaps; demand a minimum topical vocabulary.
            MIN_TERMS = 8
            for xa in arts_a:
                ta = terms_a[id(xa)]
                if len(ta) < MIN_TERMS:
                    continue
                for xb in arts_b:
                    tb = terms_b[id(xb)]
                    if len(tb) < MIN_TERMS:
                        continue
                    inter = len(ta & tb)
                    if inter == 0:
                        continue
                    score = inter / len(ta | tb)
                    scored.append((xa, xb, score))
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:top_k]


def _agent_a(art: Article, question: str, intent: str, model: str, host: str,
             timeout_s: int = _DEFAULT_TIMEOUT_S) -> Optional[dict]:
    if intent == "OFF_TOPIC":
        prompt = PROMPT_A_OFFTOPIC.format(question=question)
    elif intent == "CONVERSATIONAL":
        prompt = PROMPT_A_CONVERSATIONAL.format(question=question)
    elif intent == "META":
        prompt = PROMPT_A_META.format(question=question)
    else:
        prompt = PROMPT_A_FACTUAL.format(
            instrument=art.instrument, number=art.number,
            heading=art.heading, text=art.text, question=question)
    return _safe_json(_chat(model, prompt, host=host, temperature=0.0, timeout_s=timeout_s))


def _agent_v(art: Article, question: str, gold: str, model: str, host: str,
             timeout_s: int = _DEFAULT_TIMEOUT_S) -> Optional[dict]:
    prompt = PROMPT_V.format(
        instrument=art.instrument, number=art.number, heading=art.heading,
        text=art.text, question=question, gold_answer=gold,
    )
    return _safe_json(_chat(model, prompt, host=host, temperature=0.0, timeout_s=timeout_s))


def _agent_x_paraphrase(question: str, model: str, host: str,
                        timeout_s: int = _DEFAULT_TIMEOUT_S) -> Optional[str]:
    out = _safe_json(_chat(model, PROMPT_X_PARA.format(question=question),
                           host=host, temperature=0.6, timeout_s=timeout_s))
    if not out:
        return None
    return out.get("question")


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def _is_acceptable(v: dict, threshold: float) -> Tuple[bool, float, str]:
    keys = ["clarity", "groundedness", "pinpoint_accuracy", "contamination_risk"]
    try:
        vals = [float(v.get(k, 0)) for k in keys]
    except (TypeError, ValueError):
        return False, 0.0, str(v.get("notes", ""))
    score = min(vals)  # worst-axis gate
    return score >= threshold, score, str(v.get("notes", ""))


def _count_by_intent(path: Path) -> Dict[str, int]:
    """Read an existing JSONL and count records per intent_hint (for resume)."""
    counts: Dict[str, int] = {}
    if not path.exists():
        return counts
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = obj.get("intent_hint", "")
                counts[k] = counts.get(k, 0) + 1
    except OSError:
        pass
    return counts


def _norm_question(q: str) -> str:
    """Case/space-normalized question text used for duplicate detection.

    Intentionally aggressive: lowercases, strips, collapses internal
    whitespace. Anything more lenient (e.g. stripping punctuation) would
    risk treating distinct questions as duplicates; anything stricter
    (e.g. exact-match only) misses whitespace-only variants that Agent X
    commonly produces during paraphrasing.
    """
    return " ".join((q or "").lower().split())


def _load_seen_questions(path: Path) -> set:
    """Read an existing JSONL and return the set of normalized question
    texts already on disk (for resume-safe dedup).
    """
    seen: set = set()
    if not path.exists():
        return seen
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = obj.get("question", "")
                if q:
                    seen.add(_norm_question(q))
    except OSError:
        pass
    return seen


def generate(
    articles_by_instrument: Dict[str, List[Article]],
    *,
    n_total: int,
    intent_distribution: Dict[str, float],
    model: str,
    host: str,
    out_path: Path,
    v_threshold: float = 0.8,
    seed: int = 13,
    paraphrase_ratio: float = 0.25,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    max_attempts_multiplier: int = 6,
    extra_targets: Optional[Dict[str, int]] = None,
) -> List[SynRecord]:
    """Generate synthetic QA records and stream-write them to `out_path`.

    Resume-safe: if `out_path` already contains records, this function reads
    the per-intent counts on disk and only generates the remainder. Each
    accepted record is appended to the JSONL file immediately so a crash
    doesn't lose prior work.
    """
    rng = random.Random(seed)
    records: List[SynRecord] = []

    # How many per intent.
    already = _count_by_intent(out_path)
    if extra_targets is not None:
        # Top-up mode: generate exactly the extras ON TOP of what is on
        # disk, ignoring the normal distribution (whose kinds are full).
        targets = {k: already.get(k, 0) + n for k, n in extra_targets.items()}
    else:
        targets = {k: max(1, int(round(n_total * v)))
                   for k, v in intent_distribution.items()}
    log.info("Intent targets: %s", targets)
    if already:
        log.info("Resuming: %s already contains %s", out_path, already)

    # Duplicate-question gate. Pre-populate from any records already on disk
    # so a resume never re-emits a collision and never re-collides against
    # existing questions.
    seen_questions: set = _load_seen_questions(out_path)
    if seen_questions:
        log.info("Loaded %d existing questions into dedup set", len(seen_questions))
    duplicates_rejected = 0

    # Flatten all articles with their instrument.
    all_articles: List[Article] = [a for arts in articles_by_instrument.values() for a in arts]
    if not all_articles:
        raise RuntimeError("No articles loaded — check AKN files.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Open in append-mode so prior records (if any) are kept.
    out_f = open(out_path, "a", encoding="utf-8")

    def _emit(rec: SynRecord) -> None:
        """Append a record to the JSONL and flush to disk immediately."""
        out_f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        out_f.flush()
        try:
            os.fsync(out_f.fileno())
        except OSError:
            pass
        records.append(rec)

    uid = sum(already.values())  # continue numbering past existing records
    # Lazily built the first time a COMPARISON question is attempted.
    comparison_pairs: Optional[List[Tuple[Article, Article, float]]] = None

    try:
        for intent, target in targets.items():
            produced = already.get(intent, 0)
            if produced >= target:
                log.info("Intent %s already at target (%d/%d) — skipping.",
                         intent, produced, target)
                continue
            attempts = 0
            max_attempts = target * max_attempts_multiplier
            while produced < target and attempts < max_attempts:
                attempts += 1
                try:
                    if intent == "COMPARISON":
                        if comparison_pairs is None:
                            comparison_pairs = _build_comparison_pairs(
                                articles_by_instrument)
                            log.info("Built %d topic-matched comparison "
                                     "pairs (best overlap %.3f)",
                                     len(comparison_pairs),
                                     comparison_pairs[0][2]
                                     if comparison_pairs else 0.0)
                        if not comparison_pairs:
                            log.warning("COMPARISON requires >=2 instruments "
                                        "with overlapping articles. Skipping.")
                            break
                        a, b, pair_score = comparison_pairs[
                            rng.randrange(len(comparison_pairs))]
                        g = _agent_g_compare(a, b, model, host,
                                             timeout_s=timeout_s,
                                             sample_seed=rng.randrange(1 << 30))
                        if not g:
                            continue
                        if g.get("skip"):
                            log.info("[COMPARISON] writer skipped pair "
                                     "%s-%s/%s-%s (overlap %.3f): %s",
                                     a.instrument, a.number,
                                     b.instrument, b.number, pair_score,
                                     str(g.get("reason", ""))[:60])
                            continue
                        if "question" not in g:
                            continue
                        question = g["question"].strip()
                        if _norm_question(question) in seen_questions:
                            duplicates_rejected += 1
                            log.info("[%s] duplicate rejected (total=%d)",
                                     intent, duplicates_rejected)
                            continue
                        a_json = _agent_a_compare(a, b, question, model, host,
                                                  timeout_s=timeout_s)
                        if not a_json:
                            continue
                        gold = a_json.get("gold_answer", "").strip()
                        pinpoints = a_json.get("pinpoints", []) or []
                        v = _agent_v_compare(a, b, question, gold, model, host,
                                             timeout_s=timeout_s) or {}
                        ok, score, notes = _is_acceptable(v, v_threshold)
                        if not ok:
                            log.info("[COMPARISON] checker rejected "
                                     "(worst=%.2f): %s", score, notes[:60])
                            continue
                        uid += 1
                        _emit(SynRecord(
                            id=f"SYN-{a.instrument}-{a.number}-{intent}-{uid:04d}",
                            source="synthetic",
                            seed_article=f"{a.number}|{b.number}",
                            seed_instrument=f"{a.instrument}|{b.instrument}",
                            intent_hint=intent,
                            question=question,
                            gold_answer=gold,
                            gold_article_pinpoints=pinpoints,
                            gold_instruments=[a.instrument, b.instrument],
                            off_topic=False,
                            agent_v_score=score,
                            agent_v_notes=notes,
                        ))
                        seen_questions.add(_norm_question(question))
                        produced += 1
                        log.info("[%s] %d/%d (attempts=%d)", intent, produced, target, attempts)
                        continue

                    art = rng.choice(all_articles)
                    g = _agent_g(art, intent, model, host, timeout_s=timeout_s)
                    if not g or "question" not in g:
                        continue
                    question = g["question"].strip()
                    if _norm_question(question) in seen_questions:
                        duplicates_rejected += 1
                        log.info("[%s] duplicate rejected (total=%d)",
                                 intent, duplicates_rejected)
                        continue
                    a_json = _agent_a(art, question, intent, model, host, timeout_s=timeout_s)
                    if not a_json:
                        continue
                    gold = a_json.get("gold_answer", "").strip()
                    pinpoints = a_json.get("pinpoints", []) or []
                    # Intents that don't ground in a specific article: we
                    # skip validator-V (it grades against article text) and
                    # emit with empty pinpoints/instruments. `off_topic` is
                    # the flag the RC metric keys off — only true for the
                    # actual OFF_TOPIC intent; CONVERSATIONAL / META should
                    # be answered, not refused.
                    ungrounded = intent in ("OFF_TOPIC", "CONVERSATIONAL", "META")
                    if ungrounded:
                        off_topic = intent == "OFF_TOPIC"
                        score, notes = 1.0, f"{intent.lower()} by design"
                        pinpoints = []
                    else:
                        off_topic = False
                        v = _agent_v(art, question, gold, model, host, timeout_s=timeout_s) or {}
                        ok, score, notes = _is_acceptable(v, v_threshold)
                        if not ok:
                            continue
                    uid += 1
                    _emit(SynRecord(
                        id=f"SYN-{art.instrument}-{art.number}-{intent}-{uid:04d}",
                        source="synthetic",
                        seed_article=art.number,
                        seed_instrument=art.instrument,
                        intent_hint=intent,
                        question=question,
                        gold_answer=gold,
                        gold_article_pinpoints=pinpoints,
                        gold_instruments=[art.instrument] if not ungrounded else [],
                        off_topic=off_topic,
                        agent_v_score=score,
                        agent_v_notes=notes,
                    ))
                    seen_questions.add(_norm_question(question))
                    produced += 1
                    log.info("[%s] %d/%d (attempts=%d)", intent, produced, target, attempts)

                    if not ungrounded and rng.random() < paraphrase_ratio:
                        para = _agent_x_paraphrase(question, model, host, timeout_s=timeout_s)
                        if para and para.lower() != question.lower():
                            para_norm = _norm_question(para)
                            if para_norm in seen_questions:
                                duplicates_rejected += 1
                                log.info("[%s-X] duplicate paraphrase rejected (total=%d)",
                                         intent, duplicates_rejected)
                            else:
                                uid += 1
                                _emit(SynRecord(
                                    id=f"SYN-{art.instrument}-{art.number}-{intent}-X-{uid:04d}",
                                    source="synthetic",
                                    seed_article=art.number,
                                    seed_instrument=art.instrument,
                                    intent_hint=intent,
                                    question=para.strip(),
                                    gold_answer=gold,
                                    gold_article_pinpoints=pinpoints,
                                    gold_instruments=[art.instrument],
                                    off_topic=False,
                                    agent_v_score=score,
                                    agent_v_notes=notes + " [paraphrase]",
                                ))
                                seen_questions.add(para_norm)
                except Exception as e:  # noqa: BLE001
                    # A transient error after all retries were exhausted, or
                    # an unexpected JSON shape. Log and keep going so one bad
                    # sample doesn't kill the batch.
                    log.error("[%s] attempt %d failed: %s", intent, attempts, e)
                    continue
    finally:
        out_f.close()

    log.info("Produced %d new records (%d total on disk); rejected %d duplicates",
             len(records), sum(_count_by_intent(out_path).values()),
             duplicates_rejected)
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="manifest.yaml")
    ap.add_argument("--out", default="data/synthetic.jsonl")
    ap.add_argument("--pilot", action="store_true",
                    help="Use manifest.dataset.pilot.n_synthetic instead of full.")
    ap.add_argument("--v-threshold", type=float, default=0.8)
    ap.add_argument("--paraphrase-ratio", type=float, default=0.25)
    ap.add_argument("--model", default=None,
                    help="Ollama model to use for ALL agents (G/A/V/X). "
                         "Defaults to manifest.models.answerer.name. "
                         "Tip: use a smaller model (e.g. qwen3:8b) here to "
                         "make generation tractable; gemma4:31b is very slow "
                         "for this generation-heavy step.")
    ap.add_argument("--host", default=None,
                    help="Ollama host URL. Defaults to the answerer host.")
    ap.add_argument("--timeout-s", type=int, default=None,
                    help="Per-call Ollama timeout in seconds. "
                         "Default: manifest.models.answerer.timeout_s or 900.")
    ap.add_argument("--no-warmup", action="store_true",
                    help="Skip the initial warmup call (not recommended for "
                         "large models — the first call usually exceeds the "
                         "per-call timeout without it).")
    ap.add_argument("--fresh", action="store_true",
                    help="Delete the output JSONL before starting, forcing a "
                         "fresh generation run instead of resuming.")
    ap.add_argument("--instrument-topup", default=None, metavar="INSTRUMENT",
                    help="Generate a small extra batch of grounded questions "
                         "seeded ONLY on the given instrument (e.g. SFDR_L1): "
                         "6 factual, 3 summaries, 3 client-style. Use after "
                         "fixing an instrument's source file, when normal "
                         "generation would consider its kinds already full.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s %(message)s")

    import yaml
    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)

    articles_by_inst: Dict[str, List[Article]] = {}
    for s in manifest["corpus"]["sources"]:
        akn_path = s["akn"]
        if not Path(akn_path).exists():
            log.error("Missing AKN: %s — copy your .akn files there first.", akn_path)
            continue
        articles_by_inst[s["id"]] = _load_akn_articles(akn_path, s["id"])
        log.info("Loaded %d articles for %s", len(articles_by_inst[s["id"]]), s["id"])

    if not articles_by_inst:
        print("No AKN articles loaded. Nothing to generate.", file=sys.stderr)
        sys.exit(2)

    n_total = (manifest["dataset"]["pilot"]["n_synthetic"] if args.pilot
               else manifest["dataset"]["full"]["n_synthetic"])

    extra_targets = None
    if args.instrument_topup:
        inst = args.instrument_topup
        if inst not in articles_by_inst:
            print(f"Unknown instrument {inst!r}; have "
                  f"{sorted(articles_by_inst)}", file=sys.stderr)
            sys.exit(2)
        articles_by_inst = {inst: articles_by_inst[inst]}
        extra_targets = {"FACTUAL_QA": 6, "SUMMARIZATION": 3,
                         "LEGAL_ADVICE": 3}
        log.info("Instrument top-up mode: %s only, extras %s",
                 inst, extra_targets)

    # Model/host resolution — CLI flags > manifest.answerer
    answerer_cfg = manifest["models"]["answerer"]
    model = args.model or answerer_cfg["name"]
    host = args.host or answerer_cfg["host"]
    timeout_s = args.timeout_s or int(answerer_cfg.get("timeout_s", _DEFAULT_TIMEOUT_S))
    if timeout_s < 300:
        log.warning("timeout_s=%d is very short for a local LLM; forcing 300s minimum", timeout_s)
        timeout_s = 300

    out_path = Path(args.out)
    if args.fresh and out_path.exists():
        log.info("--fresh: removing %s", out_path)
        out_path.unlink()

    # Warm up the model so the first real call doesn't bundle model-load time
    # into a single per-call timeout.
    if not args.no_warmup:
        try:
            _warmup(model, host, timeout_s=timeout_s)
        except Exception as e:  # noqa: BLE001
            print(f"FATAL: Ollama warmup failed for model {model!r} @ {host}: {e}",
                  file=sys.stderr)
            print("Check `ollama list` and that `ollama serve` is running.",
                  file=sys.stderr)
            sys.exit(3)

    records = generate(
        articles_by_inst,
        n_total=n_total,
        intent_distribution=manifest["dataset"]["intent_distribution"],
        model=model,
        host=host,
        out_path=out_path,
        v_threshold=args.v_threshold,
        seed=manifest.get("seed", 13),
        paraphrase_ratio=args.paraphrase_ratio,
        timeout_s=timeout_s,
        extra_targets=extra_targets,
    )

    total_on_disk = sum(_count_by_intent(out_path).values())
    print(f"Wrote {len(records)} new records ({total_on_disk} total) to {out_path}")


if __name__ == "__main__":
    main()
