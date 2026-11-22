"""Cross-family audit of the synthetic benchmark.


WHAT IT DOES
------------
For each row in data/synthetic.jsonl:
  - grounded rows (FACTUAL_QA, SUMMARIZATION, LEGAL_ADVICE, COMPARISON):
    re-score with the SAME rubric prompt used at generation time (PROMPT_V),
    against the same article text. COMPARISON rows get BOTH articles —
    better than generation, which only checked the first.
  - ungrounded rows (OFF_TOPIC, CONVERSATIONAL, META): there is no article
    to ground in, so the four axes are re-defined for the row's job and the
    output is marked audit_mode="ungrounded". For OFF_TOPIC the key check
    is: is the question TRULY outside MiFID/SFDR scope?

Results stream to data/synthetic_audit.jsonl (append; resume-safe: rows
already audited are skipped on restart). A summary prints at the end:
per-axis means, rows deepseek would have rejected (worst axis < 0.8), and
agreement with the original gemma scores.


Do not run this while an evaluation run is in progress: both need Ollama
and the two large models compete for memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from dataset.synthetic_pipeline import (
    Article,
    PROMPT_V,
    _load_akn_articles,
    _safe_json,
)

CONTAMINATION_CLARIFICATION = """

IMPORTANT clarification for contamination_risk: this axis asks ONLY
whether the question rewards memorization of PUBLIC documents (for
example the ESMA Q&A corpus or famous passages) INSTEAD of reading the
article provided here. A gold answer that follows directly from the
provided article text is the intended design and must NOT lower this
score. Score below 1.0 only if the question looks like a well-known
public Q&A item that a model could answer from memory alone.
"""

log = logging.getLogger("audit")

V_THRESHOLD = 0.8  # same acceptance floor used at generation time

# deepseek-r1 thinks out loud inside <think>...</think> before answering.
# The thinking often contains braces, which would confuse the JSON
# extractor — strip it first.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "")


# The rubric for rows that have no grounding article. Same four keys as
# PROMPT_V so downstream handling is identical, but each axis is defined
# for what these rows are FOR.
PROMPT_V_UNGROUNDED = """Score the following Q/A pair from a chatbot test set on a 0.0-1.0 scale for each axis. Return STRICTLY JSON:

{{"clarity": <float>, "groundedness": <float>, "pinpoint_accuracy": <float>, "contamination_risk": <float>, "notes": "<=40 words>"}}

This is a {kind} test item for a chatbot that only covers EU MiFID II and SFDR regulation. Axis meanings for THIS item type:
- clarity: is the message unambiguous and self-contained?
- groundedness: {groundedness_def}
- pinpoint_accuracy: 1.0 if the expected answer claims NO specific article references (it should not); lower if it invents any.
- contamination_risk: for this item type, being outside the MiFID II / SFDR corpus is REQUIRED and is NOT contamination. Score 1.0 unless the exact question is a well-known public Q&A item that a model could answer from memory alone.

QUESTION: {question}
EXPECTED ANSWER: {gold_answer}
"""

_GROUNDEDNESS_DEFS = {
    "OFF_TOPIC": ("is the question TRULY outside the scope of EU MiFID II / "
                  "SFDR (so refusing it is correct)? 1.0 = clearly outside "
                  "scope; 0.0 = actually answerable from MiFID/SFDR."),
    "CONVERSATIONAL": ("is this really small talk with no regulatory "
                       "content, and does the expected answer respond "
                       "naturally and offer help?"),
    "META": ("is this really a question about the chatbot itself (its "
             "sources, scope, or workings), and is the expected answer an "
             "honest description of a MiFID II + SFDR retrieval chatbot?"),
}

GROUNDED_INTENTS = {"FACTUAL_QA", "SUMMARIZATION", "LEGAL_ADVICE", "COMPARISON"}


# ---------------------------------------------------------------------------
# Ollama call — local, with the judge's longer context window
# ---------------------------------------------------------------------------

def _judge_chat(model: str, prompt: str, host: str, num_ctx: int,
                timeout_s: int, max_retries: int = 3) -> str:
    import ollama
    client = ollama.Client(host=host, timeout=timeout_s)
    last: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            r = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_ctx": num_ctx, "seed": 13},
            )
            return r["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last = e
            wait = 10 * attempt
            log.warning("judge call failed (%s), retry %d/%d in %ds",
                        type(e).__name__, attempt, max_retries, wait)
            time.sleep(wait)
    raise RuntimeError(f"All {max_retries} judge calls failed: {last}")


# ---------------------------------------------------------------------------
# Article lookup
# ---------------------------------------------------------------------------

def _load_articles(manifest: dict, base: Path) -> Dict[Tuple[str, str], Article]:
    """Map (instrument_id, article_number) -> Article for every AKN source."""
    out: Dict[Tuple[str, str], Article] = {}
    for s in manifest["corpus"]["sources"]:
        akn = s.get("akn")
        if not akn:
            continue
        path = Path(akn)
        if not path.is_absolute():
            path = base / path
        for art in _load_akn_articles(str(path), s["id"]):
            out[(s["id"], str(art.number))] = art
    return out


def _articles_for_row(row: dict,
                      lookup: Dict[Tuple[str, str], Article]) -> List[Article]:
    """Resolve the article(s) a grounded row was generated from.

    seed fields hold either one value ("34" / "MIFID2_L1") or a pair joined
    with "|" for COMPARISON rows ("11|87" / "SFDR_L1|MIFID2_L1").
    """
    nums = str(row.get("seed_article", "")).split("|")
    insts = str(row.get("seed_instrument", "")).split("|")
    if len(insts) == 1 and len(nums) > 1:
        insts = insts * len(nums)
    arts: List[Article] = []
    for num, inst in zip(nums, insts):
        art = lookup.get((inst.strip(), num.strip()))
        if art:
            arts.append(art)
    return arts


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_prompt(row: dict, arts: List[Article]) -> Tuple[str, str]:
    """Return (audit_mode, prompt)."""
    intent = row.get("intent_hint", "")
    if intent in GROUNDED_INTENTS:
        # Same rubric as generation-time agent V. For COMPARISON we join
        # both articles into the text block the rubric reads.
        a = arts[0]
        if len(arts) > 1:
            text = "\n\n---\n\n".join(
                f"({x.instrument}, Art. {x.number} — {x.heading})\n{x.text}"
                for x in arts)
            heading = " / ".join(f"Art. {x.number}" for x in arts)
            prompt = PROMPT_V.format(
                question=row.get("question", ""),
                gold_answer=row.get("gold_answer", ""),
                instrument=" + ".join(x.instrument for x in arts),
                number=heading, heading="comparison seed articles",
                text=text,
            )
        else:
            prompt = PROMPT_V.format(
                question=row.get("question", ""),
                gold_answer=row.get("gold_answer", ""),
                instrument=a.instrument, number=a.number,
                heading=a.heading, text=a.text,
            )
        return "grounded", prompt + CONTAMINATION_CLARIFICATION
    gdef = _GROUNDEDNESS_DEFS.get(intent)
    if gdef is None:
        return "unknown_intent", ""
    prompt = PROMPT_V_UNGROUNDED.format(
        kind=intent, groundedness_def=gdef,
        question=row.get("question", ""),
        gold_answer=row.get("gold_answer", ""),
    )
    return "ungrounded", prompt + CONTAMINATION_CLARIFICATION


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.yaml")
    ap.add_argument("--input", default="data/synthetic.jsonl")
    ap.add_argument("--output", default="data/synthetic_audit.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="audit only the first N unaudited rows (smoke test)")
    ap.add_argument("--redo-contamination", action="store_true",
                    help="re-audit only rows previously rejected with "
                         "contamination_risk as the worst axis (the "
                         "ambiguous-rubric artifact); their new records "
                         "override the old ones")
    ap.add_argument("--dry-run", action="store_true",
                    help="build prompts and report coverage; no model calls")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)
    base = Path(args.manifest).resolve().parent
    judge = manifest["models"]["judge"]
    model, host = judge["name"], judge["host"]
    num_ctx = int(judge.get("num_ctx", 16384))
    timeout_s = int(judge.get("request_timeout_s", 600))

    rows = [json.loads(l) for l in open(base / args.input) if l.strip()]
    lookup = _load_articles(manifest, base)
    log.info("Loaded %d rows, %d articles", len(rows), len(lookup))

    out_path = base / args.output
    current_ids = {r.get("id") for r in rows}
    done = set()
    prior = {}
    if out_path.exists():
        for l in open(out_path):
            try:
                rec = json.loads(l)
                if rec["id"] not in current_ids:
                    continue   # audit record of a deleted row; ignore
                prior[rec["id"]] = rec   # last record per id wins
                done.add(rec["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    if done:
        log.info("Resuming: %d rows already audited", len(done))

    if args.redo_contamination:
        axes_keys = ["ds_clarity", "ds_groundedness",
                     "ds_pinpoint_accuracy", "ds_contamination_risk"]
        redo_ids = set()
        for rec in prior.values():
            if rec.get("ds_accept"):
                continue
            worst = min(axes_keys, key=lambda k: rec.get(k, 1.0))
            if worst == "ds_contamination_risk":
                redo_ids.add(rec["id"])
        log.info("Redo-contamination: %d rows selected", len(redo_ids))
        todo = [r for r in rows if r.get("id") in redo_ids]
    else:
        todo = [r for r in rows if r.get("id") not in done]
    if args.limit:
        todo = todo[: args.limit]

    # Dry run: verify every row resolves to a prompt, then stop.
    if args.dry_run:
        counts = {"grounded": 0, "ungrounded": 0,
                  "missing_article": 0, "unknown_intent": 0}
        for r in todo:
            arts = _articles_for_row(r, lookup)
            intent = r.get("intent_hint", "")
            if intent in GROUNDED_INTENTS and not arts:
                counts["missing_article"] += 1
                print(f"  MISSING ARTICLE: {r['id']} "
                      f"(seed {r.get('seed_instrument')}/{r.get('seed_article')})")
                continue
            mode, prompt = _build_prompt(r, arts)
            counts[mode] = counts.get(mode, 0) + 1
        print("Dry run:", counts)
        return 0 if counts["missing_article"] == 0 else 1

    out_f = open(out_path, "a", encoding="utf-8")
    audited = failures = 0
    t0 = time.time()
    try:
        for i, r in enumerate(todo, 1):
            arts = _articles_for_row(r, lookup)
            intent = r.get("intent_hint", "")
            if intent in GROUNDED_INTENTS and not arts:
                log.warning("%s: seed article not found — skipping", r["id"])
                failures += 1
                continue
            mode, prompt = _build_prompt(r, arts)
            if not prompt:
                log.warning("%s: unknown intent %r — skipping", r["id"], intent)
                failures += 1
                continue
            try:
                raw = _judge_chat(model, prompt, host, num_ctx, timeout_s)
            except RuntimeError as e:
                log.error("%s: %s", r["id"], e)
                failures += 1
                continue
            v = _safe_json(_strip_think(raw))
            if not v:
                log.warning("%s: judge output was not valid JSON — skipping",
                            r["id"])
                failures += 1
                continue
            axes = ["clarity", "groundedness", "pinpoint_accuracy",
                    "contamination_risk"]
            try:
                scores = {k: float(v.get(k, 0)) for k in axes}
            except (TypeError, ValueError):
                log.warning("%s: judge scores were not numbers — skipping",
                            r["id"])
                failures += 1
                continue
            worst = min(scores.values())
            rec = {
                "id": r["id"],
                "intent": intent,
                "audit_mode": mode,
                **{f"ds_{k}": scores[k] for k in axes},
                "ds_worst": worst,
                "ds_accept": worst >= V_THRESHOLD,
                "ds_notes": str(v.get("notes", ""))[:300],
                "gemma_score": r.get("agent_v_score"),
                "judge_model": model,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            audited += 1
            el = time.time() - t0
            log.info("[%d/%d] %s worst=%.2f (%s) — %.0fs elapsed",
                     i, len(todo), r["id"], worst,
                     "ok" if rec["ds_accept"] else "REJECT", el)
    finally:
        out_f.close()

    # ----- summary over EVERYTHING audited so far (this run + previous) ----
    by_id = {}
    for l in open(out_path):
        if l.strip():
            rec = json.loads(l)
            if rec["id"] not in current_ids:
                continue   # deleted row; not part of the dataset anymore
            by_id[rec["id"]] = rec   # last record per id wins
    audit = list(by_id.values())
    n = len(audit)
    print(f"\nAudited rows on file: {n}  (this run: {audited}, failures: {failures})")
    if not n:
        return 1
    for k in ("ds_clarity", "ds_groundedness", "ds_pinpoint_accuracy",
              "ds_contamination_risk", "ds_worst"):
        vals = [a[k] for a in audit if k in a]
        print(f"  mean {k[3:]:22s} {sum(vals)/len(vals):.3f}")
    rejects = [a for a in audit if not a.get("ds_accept")]
    print(f"\nRows deepseek would REJECT (worst axis < {V_THRESHOLD}): {len(rejects)}")
    for a in rejects:
        print(f"  {a['id']:42s} worst={a['ds_worst']:.2f}  {a['ds_notes'][:70]}")
    both = [a for a in audit if a.get("gemma_score") is not None]
    if both:
        deltas = [a["ds_worst"] - float(a["gemma_score"]) for a in both]
        print(f"\nAgreement with generation-time gemma scores (n={len(both)}):")
        print(f"  mean(deepseek worst - gemma score) = {sum(deltas)/len(deltas):+.3f}")
        gemma_ok_ds_no = sum(1 for a in both
                             if float(a["gemma_score"]) >= V_THRESHOLD
                             and not a["ds_accept"])
        print(f"  accepted by gemma but rejected by deepseek: {gemma_ok_ds_no}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
