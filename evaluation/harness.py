"""
Evaluation harness — runs every system over the dataset, computes RAGAS +
custom metrics, runs paired statistical tests, and writes a report bundle.


Outputs (per run, into runs/<timestamp>/)
-----------------------------------------
    manifest.snapshot.yaml   verbatim copy of manifest.yaml at run time
    answers.jsonl            one record per (system, question)
    metrics.csv              one row per (system, question), all metrics
    summary.csv              one row per system, mean/SE per metric
    comparisons.csv          one row per (metric, system_a, system_b) with
                             paired-difference CI, p-value, effect size
    log.txt                  full run log

Pipeline interface
------------------
Every system module must expose:

    def build(manifest: dict, corpus: list[dict]) -> Pipeline
    class Pipeline:
        def answer(self, question: str, intent: str | None = None) -> dict

The dict returned by `answer` MUST contain at least:
    answer_text          str
    refused              bool
    citations            list[dict]   # see metrics.AnswerRecord.citations
    retrieved_contexts   list[str]    # for RAGAS context_precision/recall
    latency_s            float
    retries              int
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# macOS OpenMP workaround — MUST run BEFORE numpy / scipy / faiss / torch import.
# ---------------------------------------------------------------------------
# On macOS, faiss-cpu bundles its own libomp.dylib, and sentence-transformers /
# torch / scipy link against a DIFFERENT libomp.dylib (usually via llvm-openmp).
# When both get loaded into the same process, libomp's duplicate-init guard
# aborts the process with:
#   "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized."
# The env var KMP_DUPLICATE_LIB_OK=TRUE tells the OMP runtime to tolerate the
# duplicate initialization instead of aborting. This is a widely-used workaround
# and does not affect correctness for our workload (neither faiss nor our
# pipelines do heavy multi-threaded numerics inside the eval loop).
import os as _os  # noqa: E402
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Also limit OMP thread count so parallel FAISS/BLAS calls don't fight each
# other for cores — on an M-series Mac this gives more consistent latencies.
_os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import csv
import importlib
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

from evaluation.metrics import AnswerRecord, METRIC_FUNCS, compute_all
from evaluation.stats import bonferroni, compare_paired, fdr_bh

log = logging.getLogger("harness")


# ---------------------------------------------------------------------------
# Manifest + corpus
# ---------------------------------------------------------------------------

def _load_manifest(path: str) -> dict:
    with open(path) as f:
        m = yaml.safe_load(f)
    m["_manifest_path"] = os.path.abspath(path)
    return m


def _resolve_corpus(manifest: dict) -> List[dict]:
    """Resolve PDF/AKN paths relative to the manifest directory."""
    base = Path(manifest["_manifest_path"]).parent
    out = []
    for s in manifest["corpus"]["sources"]:
        out.append({
            "id":  s["id"],
            "pdf": str((base / s["pdf"]).resolve()),
            "akn": str((base / s["akn"]).resolve()),
        })
    return out


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[dict]:
    if not Path(path).exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("Skipping malformed JSONL line: %s", e)
    return out


def _stratified_sample(rows: List[dict], n: int, seed: int) -> List[dict]:
    """Draw n rows spread across intents in proportion to their share.


    Proportional allocation with largest remainder, so every intent with
    at least one row gets considered and the counts sum to exactly n.
    Seeded shuffle inside each intent group keeps the draw reproducible.
    """
    if n >= len(rows):
        return rows
    rng = random.Random(seed)
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(r.get("intent_hint", "FACTUAL_QA"), []).append(r)
    total = len(rows)
    # Ideal (fractional) share per intent, floored; spread the remainder
    # to the intents with the largest fractional parts.
    shares = {k: n * len(v) / total for k, v in groups.items()}
    counts = {k: int(s) for k, s in shares.items()}
    leftover = n - sum(counts.values())
    for k in sorted(shares, key=lambda k: shares[k] - counts[k], reverse=True):
        if leftover <= 0:
            break
        counts[k] += 1
        leftover -= 1
    out: List[dict] = []
    for k in sorted(groups):  # sorted: draw order independent of dict order
        g = list(groups[k])
        rng.shuffle(g)
        out.extend(g[:counts.get(k, 0)])
    return out


def _load_dataset(manifest: dict, *, pilot: bool,
                  conference: bool = False) -> List[dict]:
    """Concatenate ESMA + synthetic; sample down to the mode's sizes."""
    base = Path(manifest["_manifest_path"]).parent
    esma = _load_jsonl(str(base / "data/esma_qa.jsonl"))
    syn = _load_jsonl(str(base / "data/synthetic.jsonl"))
    seed = int(manifest.get("seed", 13))
    if conference:
        n_e = manifest["dataset"]["conference"]["n_esma"]
        n_s = manifest["dataset"]["conference"]["n_synthetic"]
        # ESMA: seeded random draw (a fixed head would be one topic
        # cluster of the source document). Synthetic: stratified across
        # the seven kinds. Both reproducible from the manifest seed.
        rng = random.Random(seed)
        esma = list(esma)
        rng.shuffle(esma)
        esma = esma[:n_e]
        syn = _stratified_sample(syn, n_s, seed=seed)
    elif pilot:
        n_e = manifest["dataset"]["pilot"]["n_esma"]
        n_s = manifest["dataset"]["pilot"]["n_synthetic"]
        esma = esma[:n_e]
        syn = _stratified_sample(syn, n_s, seed=seed)
    log.info("Dataset: %d ESMA + %d synthetic = %d questions",
             len(esma), len(syn), len(esma) + len(syn))
    out = []
    for q in esma + syn:
        # Normalize the schema across the two sources.
        # ESMA questions carry an editorial stamp from the source PDF,
        # "[Last update: 10 October 2016] How should firms...". That stamp
        # is not part of what any user would ask, and feeding it to the
        # systems adds noise (dates invite date-flavored answers). Strip it
        # at load time; the stored file keeps the stamp for traceability.
        # NOTE: this changes the question text, so judge-cache entries for
        # stamped questions from earlier runs will not be reused.
        q_text = q.get("question") or q.get("text") or ""
        q_text = re.sub(r"^\s*\[last update:[^\]]*\]\s*", "", q_text,
                        flags=re.IGNORECASE)
        out.append({
            "id": q.get("id"),
            "question": q_text,
            "intent": q.get("intent_hint", "FACTUAL_QA"),
            "gold_answer": q.get("gold_answer", ""),
            "gold_article_pinpoints": q.get("gold_article_pinpoints", []) or [],
            "gold_instruments": q.get("gold_instruments", []) or [],
            "off_topic": bool(q.get("off_topic", False)),
            "source": q.get("source", "unknown"),
        })
    # Drop questions with no text
    out = [q for q in out if q["question"].strip()]
    seen: Dict[str, int] = {}
    for q in out:
        seen[q["id"]] = seen.get(q["id"], 0) + 1
    dups = sorted(k for k, n in seen.items() if n > 1)
    if dups:
        raise ValueError(
            f"Dataset has {len(dups)} duplicate question id(s), e.g. "
            f"{dups[:5]} — ids must be unique because answers, judge "
            f"scores, and resume logic are all keyed on them.")
    return out


# ---------------------------------------------------------------------------
# Build pipelines
# ---------------------------------------------------------------------------

def _import_pipeline(module_path: str):
    """Import a system module and return its `build` callable."""
    mod = importlib.import_module(module_path)
    if not hasattr(mod, "build"):
        raise RuntimeError(f"Module {module_path} has no `build` function.")
    return mod.build


def _instantiate_systems(manifest: dict, corpus: List[dict],
                         enabled: Optional[List[str]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if enabled is not None:
        unknown = set(enabled) - set(manifest["systems"])
        if unknown:
            raise ValueError(
                f"--systems names unknown system id(s) {sorted(unknown)}; "
                f"defined: {sorted(manifest['systems'])}")
    for sys_id, entry in manifest["systems"].items():
        if enabled is not None:
            if sys_id not in enabled:
                continue
        elif not entry.get("enabled", True):
            continue
        log.info("Building system %s (%s)", sys_id, entry["module"])
        try:
            build = _import_pipeline(entry["module"])
            out[sys_id] = build(manifest, corpus)
        except Exception as e:
            log.error("Failed to build %s: %s\n%s", sys_id, e, traceback.format_exc())
            raise
    if not out:
        raise RuntimeError("No systems were built.")
    return out


# ---------------------------------------------------------------------------
# Run answers
# ---------------------------------------------------------------------------

def _run_answers(systems: Dict[str, Any], dataset: List[dict],
                 out_path: Path) -> List[dict]:
    """
    Run every system over every question and write one JSONL row per
    (system, question) to ``out_path``. Resume-safe:

    * If ``out_path`` already exists, previously-emitted rows are loaded and
      their (system_id, question_id) pairs are skipped.
    * Every new row is written + flushed + fsynced BEFORE the next call into
      the pipeline, so a crash cannot drop an answer we already paid for.
    * The file is opened in APPEND mode — never truncated — unless the
      caller explicitly deletes it first.

    Backward compatible with callers that deleted the file beforehand
    (fresh runs) because open-for-append on a non-existent path simply
    creates an empty file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Preload any rows already on disk from an earlier interrupted run.
    rows: List[dict] = []
    done: set = set()  # {(system_id, question_id)}
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            with out_path.open("r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        # Corrupt trailing line from a crash — ignore it.
                        continue
                    rows.append(r)
                    done.add((r.get("system_id"), r.get("question_id")))
            if rows:
                log.info("Resuming from existing answers file: %d rows already on disk.",
                         len(rows))
        except Exception as e:
            log.warning("Could not preload %s for resume (%s) — starting fresh.",
                        out_path, e)
            rows = []
            done = set()

    # Stream-write new rows. Open in append mode; every write is immediately
    # flushed and fsynced so a crash cannot lose an answer we've computed.
    with out_path.open("a", buffering=1) as fh:
        def _emit(row: dict) -> None:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            try:
                _os.fsync(fh.fileno())
            except OSError:
                # Some filesystems (e.g. tmpfs) don't support fsync — that's OK.
                pass

        for q in dataset:
            for sys_id, pipe in systems.items():
                if (sys_id, q["id"]) in done:
                    continue  # already answered in a prior run
                t0 = time.time()
                try:
                    a = pipe.answer(q["question"], intent=q.get("intent"))
                    err = None
                except Exception as e:
                    log.exception("System %s failed on %s", sys_id, q["id"])
                    a = {
                        "answer_text": "",
                        "refused": True,
                        "citations": [],
                        "retrieved_contexts": [],
                        "latency_s": time.time() - t0,
                        "retries": 0,
                    }
                    err = str(e)
                row = {
                    "question_id": q["id"],
                    "system_id": sys_id,
                    "intent": q["intent"],
                    "question": q["question"],
                    "gold_answer": q["gold_answer"],
                    "gold_article_pinpoints": q["gold_article_pinpoints"],
                    "gold_instruments": q["gold_instruments"],
                    "off_topic": q["off_topic"],
                    "answer_text": a.get("answer_text", ""),
                    "refused": bool(a.get("refused", False)),
                    "citations": a.get("citations", []),
                    "retrieved_contexts": a.get("retrieved_contexts", []),
                    "latency_s": float(a.get("latency_s", time.time() - t0)),
                    "retries": int(a.get("retries", 0)),
                    "error": err,
                }
                _emit(row)
                rows.append(row)
                done.add((sys_id, q["id"]))
    return rows


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------

def _to_record(row: dict) -> AnswerRecord:
    return AnswerRecord(
        question_id=row["question_id"],
        system_id=row["system_id"],
        intent=row["intent"],
        answer_text=row["answer_text"],
        refused=row["refused"],
        citations=row["citations"],
        gold_article_pinpoints=set(row["gold_article_pinpoints"] or []),
        gold_instruments=set(row["gold_instruments"] or []),
        gold_is_off_topic=bool(row["off_topic"]),
        gold_answer=row.get("gold_answer", ""),
    )


def _compute_custom_metrics(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows:
        rec = _to_record(r)
        m = compute_all(rec)
        out.append({
            "question_id": r["question_id"],
            "system_id": r["system_id"],
            "intent": r["intent"],
            **{k: v for k, v in m.items()},
            "latency_s": r["latency_s"],
            "refused": r["refused"],
        })
    return out


def _build_ragas_llm_and_embeddings(manifest: dict):
    """Wrap the manifest's Ollama judge + embedding model in RAGAS-compatible
    wrappers. Without this, RAGAS silently defaults to OpenAI and fails with
    "The api_key client option must be set...".

    Returns (llm, embeddings) or (None, None) if anything is missing.
    """
    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except Exception as e:
        log.error("Cannot build RAGAS Ollama wrappers (%s) — "
                  "install langchain-ollama + ragas.", e)
        return None, None

    judge_cfg = manifest.get("models", {}).get("judge", {}) or {}
    embed_cfg = manifest.get("models", {}).get("embed", {}) or {}

    judge_name = judge_cfg.get("name")
    judge_host = judge_cfg.get("host", "http://localhost:11434")
    judge_temp = float(judge_cfg.get("temperature", 0.0))
    judge_ctx  = int(judge_cfg.get("num_ctx", 8192))
    judge_request_timeout = float(judge_cfg.get("request_timeout_s", 600))

    embed_name = embed_cfg.get("name")
    embed_host = embed_cfg.get("host", "http://localhost:11434")

    if not judge_name or not embed_name:
        log.error("manifest.models.judge.name / models.embed.name missing — "
                  "cannot configure RAGAS. Skipping RAGAS metrics.")
        return None, None

    try:
        # Build ChatOllama defensively: `timeout` was renamed across
        # langchain-ollama releases (`request_timeout` → `timeout`, then
        # occasionally dropped as a public kwarg). Pass whichever the
        # installed version accepts; the num_ctx + base_url path is stable.
        chat_kwargs = dict(
            model=judge_name,
            base_url=judge_host,
            temperature=judge_temp,
            num_ctx=judge_ctx,
        )
        try:
            chat = ChatOllama(**chat_kwargs, timeout=judge_request_timeout)
        except TypeError:
            try:
                chat = ChatOllama(**chat_kwargs, request_timeout=judge_request_timeout)
            except TypeError:
                log.warning(
                    "ChatOllama rejected both `timeout` and `request_timeout` "
                    "kwargs — falling back to library default. Long-context "
                    "context_precision calls may time out as a result."
                )
                chat = ChatOllama(**chat_kwargs)
        emb = OllamaEmbeddings(model=embed_name, base_url=embed_host)
        llm = LangchainLLMWrapper(chat)
        embeddings = LangchainEmbeddingsWrapper(emb)
        log.info("RAGAS configured: judge=%s, embed=%s, host=%s",
                 judge_name, embed_name, judge_host)
        return llm, embeddings
    except Exception as e:
        log.error("Failed to instantiate Ollama RAGAS wrappers (%s).", e)
        return None, None


def _compute_ragas_metrics(
    rows: List[dict],
    manifest: dict,
    *,
    cache_dir: Optional[Path] = None,
) -> Dict[Tuple[str, str], dict]:
    """Run RAGAS on every (system, question). Returns
    {(system_id, question_id): {metric: score, ...}}.

    Optional — caller can skip via --skip-ragas.

    IMPORTANT: by default RAGAS tries to use OpenAI. We explicitly inject
    Ollama-backed LLM + embeddings (built from manifest.models.judge and
    .embed) into every metric via `evaluate(..., llm=..., embeddings=...)`.

    When `cache_dir` is provided, every (system, question, metric) tuple is
    looked up in a content-addressable cache keyed on the judge inputs
    (question, answer, contexts, gold, judge identity). Cache hits are
    reused; cache misses are sent to the judge and written back on success.
    NaN judge results are never cached so that failures get retried on the
    next pass rather than being permanently silenced.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        # RunConfig moved across ragas versions: try the stable paths in order.
        try:
            from ragas.run_config import RunConfig  # ragas >= 0.1.x
        except Exception:  # pragma: no cover
            from ragas import RunConfig  # type: ignore[attr-defined]
    except Exception as e:
        log.error("RAGAS import failed (%s) — skipping RAGAS metrics.", e)
        return {}

    # Build Ollama-backed LLM + embeddings ONCE and reuse across every system.
    ragas_llm, ragas_emb = _build_ragas_llm_and_embeddings(manifest)
    if ragas_llm is None or ragas_emb is None:
        log.error("RAGAS LLM/embeddings unavailable — skipping RAGAS metrics.")
        return {}

    judge_cfg = manifest["models"]["judge"]
    _timeout = int(judge_cfg.get("job_timeout_s", 1200))
    _workers = int(judge_cfg.get("max_workers", 1))
    ragas_run_config = RunConfig(
        timeout=_timeout,
        max_workers=_workers,
        max_retries=3,
        max_wait=60,
    )
    log.info("RAGAS RunConfig: timeout=%ds, max_workers=%d, max_retries=3",
             _timeout, _workers)

    # Cache handle (optional). Keyed on judge identity so swapping judges
    # (deepseek ↔ qwen) automatically invalidates everything — see
    # evaluation/ragas_cache.py for the full key schema.
    cache_handle: Optional["RagasCacheHandle"] = None
    if cache_dir is not None:
        from evaluation.ragas_cache import CacheHandle as RagasCacheHandle
        from evaluation import ragas_cache as _ragas_cache_mod
        judge_cfg = manifest.get("models", {}).get("judge", {}) or {}
        cache_handle = RagasCacheHandle(
            cache_dir=cache_dir,
            judge_name=str(judge_cfg.get("name", "")),
            judge_temperature=float(judge_cfg.get("temperature", 0.0)),
        )
        log.info("RAGAS cache enabled at %s (%d existing entries)",
                 cache_dir, _ragas_cache_mod.stats(cache_dir)["entries"])

    # Build the per-system, per-question scaffold we'll fill in. Non-refused
    # rows are the only ones we score — faithfulness on "I cannot answer" is
    # noise, and the judge will happily call it trivially faithful.
    metric_names = manifest["metrics"]["ragas"]
    metric_fns = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
    }
    sys_ids = sorted({r["system_id"] for r in rows})

    # Seed the output dict with Nones so even rows we fail to score show up
    # as explicit nulls rather than missing keys downstream.
    out: Dict[Tuple[str, str], dict] = {}
    eligible_by_sys: Dict[str, List[dict]] = {}
    for sys_id in sys_ids:
        sub = [r for r in rows
               if r["system_id"] == sys_id
               and r["answer_text"]
               and not r.get("refused", False)]
        eligible_by_sys[sys_id] = sub
        for r in sub:
            out[(sys_id, r["question_id"])] = {m: None for m in metric_names}

    # ---- per-metric loop: check cache, compute misses, write back ----------
    total_hits = 0
    total_misses = 0
    total_failures = 0
    for m_name in metric_names:
        if m_name not in metric_fns:
            log.warning("Unknown RAGAS metric '%s' in manifest — skipping.", m_name)
            continue
        metric_fn = metric_fns[m_name]

        for sys_id in sys_ids:
            sub = eligible_by_sys[sys_id]
            if not sub:
                continue

            # Partition rows into cache hits and misses for THIS metric.
            miss_rows: List[dict] = []
            miss_keys: List[str] = []
            miss_components: List[Dict[str, Any]] = []
            hits_here = 0
            for r in sub:
                contexts = r["retrieved_contexts"] or [""]
                if cache_handle is None:
                    miss_rows.append(r); miss_keys.append(""); miss_components.append({})
                    continue
                key, comps = cache_handle.key_for(
                    metric_name=m_name,
                    question=r["question"],
                    answer_text=r["answer_text"],
                    contexts=contexts,
                    ground_truth=r.get("gold_answer"),
                )
                cached = cache_handle.get(key)
                if cached is not None:
                    out[(sys_id, r["question_id"])][m_name] = cached
                    hits_here += 1
                    total_hits += 1
                else:
                    miss_rows.append(r); miss_keys.append(key); miss_components.append(comps)

            if not miss_rows:
                log.info("RAGAS %s/%s: %d/%d cache hits, 0 to compute",
                         sys_id, m_name, hits_here, len(sub))
                continue

            log.info("RAGAS %s/%s: %d cache hits, computing %d",
                     sys_id, m_name, hits_here, len(miss_rows))

            # Build a single-metric dataset of ONLY the misses.
            ds = Dataset.from_dict({
                "question":     [r["question"] for r in miss_rows],
                "answer":       [r["answer_text"] for r in miss_rows],
                "contexts":     [r["retrieved_contexts"] or [""] for r in miss_rows],
                "ground_truth": [r.get("gold_answer") for r in miss_rows],
            })
            try:
                res = evaluate(ds, metrics=[metric_fn],
                               llm=ragas_llm, embeddings=ragas_emb,
                               run_config=ragas_run_config)
                df = res.to_pandas()
            except Exception as e:
                log.error("RAGAS evaluate failed for %s/%s: %s",
                          sys_id, m_name, e)
                total_failures += len(miss_rows)
                continue

            for i, r in enumerate(miss_rows):
                if m_name not in df.columns:
                    total_failures += 1
                    continue
                try:
                    v = float(df.iloc[i][m_name])
                except (TypeError, ValueError):
                    total_failures += 1
                    continue
                if math.isnan(v):
                    total_failures += 1
                    continue
                out[(sys_id, r["question_id"])][m_name] = v
                total_misses += 1
                if cache_handle is not None:
                    cache_handle.put(miss_keys[i], v, miss_components[i])

    if cache_handle is not None:
        log.info("RAGAS summary: %d cache hits, %d fresh scores, %d judge failures",
                 total_hits, total_misses, total_failures)
    return out


# ---------------------------------------------------------------------------
# Aggregation + write CSVs
# ---------------------------------------------------------------------------

def _aggregate_per_system(metric_rows: List[dict],
                          ragas: Dict[Tuple[str, str], dict],
                          manifest: dict) -> Dict[str, Dict[str, float]]:
    """Mean ± SE per system, per metric (custom + RAGAS). Skips Nones."""
    sys_ids = sorted({r["system_id"] for r in metric_rows})
    metric_names = list(METRIC_FUNCS.keys()) + manifest["metrics"]["ragas"]

    summary: Dict[str, Dict[str, float]] = {}
    for sys_id in sys_ids:
        per_sys: Dict[str, float] = {}
        for m in metric_names:
            vals: List[float] = []
            for r in metric_rows:
                if r["system_id"] != sys_id:
                    continue
                if m in r:
                    v = r[m]
                else:
                    v = ragas.get((sys_id, r["question_id"]), {}).get(m)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                # Guard against NaN sneaking in from any upstream source
                # (RAGAS failures, CSV round-trip) — skip it like None.
                if math.isnan(fv):
                    continue
                vals.append(fv)
            if vals:
                arr = np.asarray(vals)
                per_sys[f"{m}_mean"] = float(arr.mean())
                per_sys[f"{m}_se"] = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
                per_sys[f"{m}_n"] = len(arr)
        summary[sys_id] = per_sys
    return summary


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Pairwise comparisons
# ---------------------------------------------------------------------------

def _pairwise_comparisons(metric_rows: List[dict],
                          ragas: Dict[Tuple[str, str], dict],
                          manifest: dict) -> List[dict]:
    sys_ids = sorted({r["system_id"] for r in metric_rows})
    if len(sys_ids) < 2:
        return []
    metric_names = list(METRIC_FUNCS.keys()) + manifest["metrics"]["ragas"]
    by_qid = {r["question_id"]: {} for r in metric_rows}
    for r in metric_rows:
        by_qid[r["question_id"]][r["system_id"]] = r

    results: List[dict] = []
    raw_p_values: List[float] = []
    for m in metric_names:
        # Decide non-inferiority margin (only for RC by default)
        ni = manifest["stats"]["noninferiority_margin_pp"] / 100.0 if m == "RC" else None
        for i, a in enumerate(sys_ids):
            for b in sys_ids[i+1:]:
                vals_a, vals_b = [], []
                for qid, rec in by_qid.items():
                    ra = rec.get(a); rb = rec.get(b)
                    if not ra or not rb:
                        vals_a.append(None); vals_b.append(None); continue
                    va = ra.get(m, ragas.get((a, qid), {}).get(m))
                    vb = rb.get(m, ragas.get((b, qid), {}).get(m))
                    vals_a.append(va); vals_b.append(vb)
                try:
                    cr = compare_paired(
                        vals_a, vals_b,
                        system_a=a, system_b=b, metric=m,
                        n_iter=manifest["stats"]["bootstrap"]["n_iter"],
                        confidence=manifest["stats"]["bootstrap"]["confidence"],
                        seed=manifest.get("seed", 13),
                        noninferiority_margin=ni,
                    )
                    results.append(cr.to_dict())
                    raw_p_values.append(cr.p_value)
                except Exception as e:
                    log.warning("compare_paired failed for %s vs %s on %s: %s",
                                a, b, m, e)

    # Multiple-comparison correction over the entire family.
    if raw_p_values:
        bonf = bonferroni(raw_p_values)
        bh = fdr_bh(raw_p_values)
        for i, r in enumerate(results):
            r["p_bonferroni"] = float(bonf[i])
            r["p_fdr_bh"] = float(bh[i])
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _find_latest_run_dir(runs_root: Path, prefix: str) -> Optional[Path]:
    """Return the most-recent directory under ``runs_root`` whose name starts
    with ``prefix`` (e.g. ``pilot-`` or ``full-``). Used by ``--resume``.
    """
    if not runs_root.exists():
        return None
    candidates = [p for p in runs_root.iterdir()
                  if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_cache_dir(manifest: dict) -> Path:
    """RAGAS cache lives under the project's .cache dir so it's shared across
    every run, not scoped to a single run dir. That's the whole point:
    iterating on metric fixes should reuse judge calls made in prior runs."""
    base = Path(manifest["_manifest_path"]).parent
    cache_root = base / manifest["paths"].get("cache_dir", ".cache")
    return cache_root / "ragas"


def _load_answers_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"answers.jsonl not found at {path}")
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _subsample_dataset(dataset: List[dict], per_system_limit: int, seed: int,
                       systems: Optional[List[str]] = None) -> List[dict]:
    """Deterministically keep the first `per_system_limit` questions (by a
    seeded shuffle) so EVERY system answers the SAME subset. Subsampling
    happens before answer generation, so all 4 systems share the same debug
    question set — that's required for paired comparisons to be meaningful."""
    import random as _random
    rng = _random.Random(seed)
    ordered = list(dataset)
    rng.shuffle(ordered)
    return ordered[:per_system_limit]


def _write_outputs(run_dir: Path, custom: List[dict],
                   ragas: Dict[Tuple[str, str], dict],
                   manifest: dict) -> None:
    """Merge custom + RAGAS scores, write the three CSVs we care about:
    metrics.csv (per-row), summary.csv (per-system means), comparisons.csv
    (pairwise tests). Shared by run() and recompute_metrics()."""
    merged_rows = []
    for r in custom:
        merged = dict(r)
        merged.update(ragas.get((r["system_id"], r["question_id"]), {}))
        merged_rows.append(merged)
    metric_fields = ["question_id", "system_id", "intent", "latency_s", "refused"] + \
        list(METRIC_FUNCS.keys()) + manifest["metrics"]["ragas"]
    _write_csv(run_dir / "metrics.csv", merged_rows, metric_fields)

    summary = _aggregate_per_system(custom, ragas, manifest)
    summary_rows = [{"system_id": s, **stats} for s, stats in summary.items()]
    _write_csv(run_dir / "summary.csv",
               summary_rows,
               sorted({k for r in summary_rows for k in r.keys()}))

    comps = _pairwise_comparisons(custom, ragas, manifest)
    if comps:
        _write_csv(run_dir / "comparisons.csv", comps, sorted(comps[0].keys()))


def _backup_existing_csvs(run_dir: Path) -> List[Path]:
    """Rename existing metrics/summary/comparisons to *.bak before overwrite.
    Returns the list of backup paths created — useful for logging / undo."""
    backups: List[Path] = []
    for name in ("metrics.csv", "summary.csv", "comparisons.csv"):
        src = run_dir / name
        if not src.exists():
            continue
        dst = src.with_suffix(src.suffix + ".bak")
        # If a .bak already exists (e.g. user ran recompute twice), keep the
        # older one as .bak.1, .bak.2, ... so we never silently clobber history.
        if dst.exists():
            i = 1
            while dst.with_suffix(dst.suffix + f".{i}").exists():
                i += 1
            dst = dst.with_suffix(dst.suffix + f".{i}")
        src.rename(dst)
        backups.append(dst)
    return backups


def run(manifest_path: str, *, pilot: bool, conference: bool = False,
        systems: Optional[List[str]] = None,
        skip_ragas: bool = False, resume: bool = False,
        ragas_subsample: Optional[int] = None,
        prefix_override: Optional[str] = None,
        judge_qids: Optional[set] = None) -> Path:
    """Normal evaluation run. When ``ragas_subsample`` is set, the run is
    treated as debug: output goes to runs/debug-<ts>/, NOT to the regular
    pilot-/full- directory, so debug runs can never pollute a real result."""
    manifest = _load_manifest(manifest_path)
    corpus = _resolve_corpus(manifest)
    base = Path(manifest["_manifest_path"]).parent

    runs_root = base / manifest["paths"]["runs_dir"]
    if prefix_override:
        prefix = prefix_override if prefix_override.endswith("-") \
            else prefix_override + "-"
    else:
        prefix = "conf-" if conference else ("pilot-" if pilot else "full-")
    run_dir: Optional[Path] = None
    debug_mode = ragas_subsample is not None and ragas_subsample > 0
    if debug_mode:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = runs_root / f"debug-{timestamp}"
        if resume:
            log.warning("--resume ignored in debug/subsample mode.")
    else:
        if resume:
            run_dir = _find_latest_run_dir(runs_root, prefix)
            if run_dir is None:
                log.warning("--resume requested but no prior %s run found; starting fresh.",
                            prefix.rstrip("-"))
            else:
                log.info("Resuming prior run dir: %s", run_dir)
        if run_dir is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            run_dir = runs_root / f"{prefix}{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot manifest verbatim
    shutil.copy2(manifest_path, run_dir / "manifest.snapshot.yaml")

    # Logging — also write to file
    fh = logging.FileHandler(run_dir / "log.txt")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("Pilot=%s  Skip-RAGAS=%s  Subsample=%s  Systems=%s",
             pilot, skip_ragas, ragas_subsample, systems or "ALL")

    # Build pipelines
    pipes = _instantiate_systems(manifest, corpus, enabled=systems)

    # Load dataset
    dataset = _load_dataset(manifest, pilot=pilot, conference=conference)
    if not dataset:
        raise RuntimeError("Empty dataset. Run dataset/esma_extractor.py and "
                           "dataset/synthetic_pipeline.py first.")

    if debug_mode:
        seed = int(manifest.get("seed", 13))
        dataset = _subsample_dataset(dataset, ragas_subsample, seed=seed)
        log.info("DEBUG subsample: reduced dataset to %d questions (seed=%d)",
                 len(dataset), seed)

    # Generate answers
    answers = _run_answers(pipes, dataset, run_dir / "answers.jsonl")

    # Compute metrics
    custom = _compute_custom_metrics(answers)
    cache_dir = None if skip_ragas else _resolve_cache_dir(manifest)
    # Optional judge filter (prompt study light mode): deterministic
    # metrics always cover every answer; judged metrics only the listed
    # questions. Extending later costs no rework: --recompute-metrics on
    # the same run dir WITHOUT the filter judges the remaining rows, and
    # the cache keeps every score already paid for.
    ragas_answers = answers
    if judge_qids is not None:
        ragas_answers = [a for a in answers if a["question_id"] in judge_qids]
        log.info("Judge filter: %d of %d answer rows eligible for judged "
                 "metrics.", len(ragas_answers), len(answers))
    ragas = {} if skip_ragas else _compute_ragas_metrics(ragas_answers, manifest,
                                                        cache_dir=cache_dir)

    _write_outputs(run_dir, custom, ragas, manifest)

    log.info("Run complete — outputs in %s", run_dir)
    return run_dir


def recompute_metrics(manifest_path: str, run_dir: Path, *,
                      skip_ragas: bool = False) -> Path:
    """Re-run metric computation over an existing run's answers.jsonl.

    Writes fresh metrics.csv / summary.csv / comparisons.csv into the SAME
    directory, backing up the prior versions to *.bak first. RAGAS is
    cache-backed, so every judge call we've already made is reused — only
    previously-failed (NaN) rows or rows whose underlying inputs changed
    will hit the judge.

    Use cases:
      - Changed a metric in evaluation/metrics.py: re-run, every custom
        metric is recomputed, every RAGAS call is a cache hit.
      - Changed a judge setting: cells without a score become cache misses,
        and only those are judged again.
      - Edited a system wrapper and want to re-score just that system:
        delete or regenerate that system's rows in answers.jsonl before
        running this; untouched systems hit the cache.
    """
    manifest = _load_manifest(manifest_path)
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    # Attach file logging to the run dir so this command's activity is
    # captured alongside the original run's log.
    fh = logging.FileHandler(run_dir / "log.txt")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("Recompute-metrics on %s  Skip-RAGAS=%s", run_dir, skip_ragas)

    # Load answers
    answers = _load_answers_jsonl(run_dir / "answers.jsonl")
    log.info("Loaded %d answer rows from %s/answers.jsonl", len(answers), run_dir)

    # Back up existing CSVs before we overwrite.
    backups = _backup_existing_csvs(run_dir)
    for b in backups:
        log.info("Backed up %s", b)

    # Custom metrics are cheap + deterministic — always recompute.
    custom = _compute_custom_metrics(answers)
    cache_dir = None if skip_ragas else _resolve_cache_dir(manifest)
    ragas = {} if skip_ragas else _compute_ragas_metrics(answers, manifest,
                                                        cache_dir=cache_dir)

    _write_outputs(run_dir, custom, ragas, manifest)
    log.info("Recompute complete — outputs in %s (prior files at *.bak)", run_dir)
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="manifest.yaml")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pilot", action="store_true", default=True)
    g.add_argument("--full", action="store_true")
    g.add_argument("--conference", action="store_true",
                   help="Headline run for the conference paper: "
                        "dataset.conference sizes (50 ESMA drawn by seed + "
                        "100 synthetic stratified). Output: runs/conf-<ts>/")
    ap.add_argument("--systems", default=None,
                    help="Comma-separated system IDs to enable (default: all enabled).")
    ap.add_argument("--skip-ragas", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse the most recent pilot-/full- run dir; skip "
                         "(system, question) pairs already written to "
                         "answers.jsonl. Use this to pick up after a crash.")
    ap.add_argument("--recompute-metrics", metavar="RUN_DIR", default=None,
                    help="Skip answer generation; re-run metric computation "
                         "over RUN_DIR/answers.jsonl. Backs up existing CSVs "
                         "to *.bak and overwrites them in place. Uses the "
                         "shared RAGAS cache so only changed / previously-"
                         "failed cells are sent to the judge.")
    ap.add_argument("--ragas-subsample", type=int, default=None, metavar="N",
                    help="Debug mode: only score N questions per system. "
                         "Output is forced to runs/debug-<ts>/ so real "
                         "pilot/full results are never overwritten.")
    ap.add_argument("--clear-cache", action="store_true",
                    help="Delete every cached RAGAS score before proceeding. "
                         "Use this if you upgrade ragas or change any prompt "
                         "template upstream of the judge.")
    ap.add_argument("--prefix", default=None, metavar="NAME",
                    help="Override the run-dir prefix (e.g. 'prompt-'). "
                         "Keeps side studies out of the conf-/pilot- "
                         "namespaces; --resume then matches this prefix.")
    ap.add_argument("--judge-qids", default=None, metavar="FILE",
                    help="Only send questions whose ids appear in FILE "
                         "(one per line, # comments allowed) to the judge. "
                         "Deterministic metrics still cover every answer.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    # Optional: wipe the RAGAS cache up front so the rest of the run is fresh.
    if args.clear_cache:
        from evaluation import ragas_cache as _rc
        manifest = _load_manifest(args.manifest)
        cache_dir = _resolve_cache_dir(manifest)
        n = _rc.clear(cache_dir)
        print(f"Cleared {n} RAGAS cache entries from {cache_dir}")

    enabled = [s.strip() for s in args.systems.split(",")] if args.systems else None

    # Mode 1: recompute over an existing run — no answer generation.
    if args.recompute_metrics:
        run_dir = recompute_metrics(args.manifest,
                                    Path(args.recompute_metrics),
                                    skip_ragas=args.skip_ragas)
        print(f"\nRecomputed: {run_dir}")
        return

    # Mode 2: normal run (optionally in debug subsample mode).
    judge_qids = None
    if args.judge_qids:
        with open(args.judge_qids) as f:
            judge_qids = {ln.strip() for ln in f
                          if ln.strip() and not ln.startswith("#")}
        if not judge_qids:
            raise SystemExit(f"--judge-qids file {args.judge_qids} is empty.")
    pilot = not (args.full or args.conference)
    run_dir = run(args.manifest, pilot=pilot, conference=args.conference,
                  systems=enabled,
                  skip_ragas=args.skip_ragas, resume=args.resume,
                  ragas_subsample=args.ragas_subsample,
                  prefix_override=args.prefix,
                  judge_qids=judge_qids)
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
