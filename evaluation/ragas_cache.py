"""
Content-addressable cache for RAGAS judge scores.


Design: each RAGAS score is a deterministic function of a small tuple of
inputs that the judge actually sees. We hash that tuple and use the hash
as the cache key. If any input changes, the hash changes and we recompute.
If nothing changes, we reuse. This is the same idea git blobs use.

Cache key components
--------------------
    judge_name          (e.g. "deepseek-r1:32b")
    judge_temperature   (float; affects sampling → score)
    metric_name         (e.g. "faithfulness")
    question            (the user's question string)
    answer_text         (the system's predicted answer)
    contexts            (list of retrieved-context strings, order-independent)
    ground_truth        (the gold answer we score against)

What's intentionally NOT in the key
-----------------------------------
    judge num_ctx       — doesn't affect the score when the prompt fits
    RAGAS version       — bumps here may silently change prompts; document
                          and use `clear()` on upgrade
    system_id, question_id — identifiers, not inputs to the judge

Layout on disk
--------------
    <cache_dir>/<hash[0:2]>/<hash>.json

Sharding by first 2 hex chars keeps any single directory under ~4k files
even at full-run scale. Each JSON file carries:
    {
      "score": float | null,
      "key_components": {... all inputs listed above ...},
      "created_at": "ISO-8601 UTC",
      "cache_schema": 1
    }

`key_components` is stored verbatim so we can audit a cache entry after
the fact — useful when chasing "why did this hit" mysteries.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger("ragas_cache")

CACHE_SCHEMA = 1


# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------

def _normalize_contexts(contexts: Optional[Iterable[str]]) -> List[str]:
    """Contexts are order-independent (judges see them as a set). Sort and
    deduplicate so the hash is stable under reordering."""
    if contexts is None:
        return []
    out = []
    seen = set()
    for c in contexts:
        s = "" if c is None else str(c)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return sorted(out)


def _normalize_str(s: Optional[str]) -> str:
    return "" if s is None else str(s)


def build_key_components(
    *,
    judge_name: str,
    judge_temperature: float,
    metric_name: str,
    question: Optional[str],
    answer_text: Optional[str],
    contexts: Optional[Iterable[str]],
    ground_truth: Optional[str],
) -> Dict[str, Any]:
    """Return the canonical dict that will be hashed and stored alongside the
    score. Always passes through this function so that every caller hashes
    the same shape — inline formatting changes would silently invalidate the
    whole cache."""
    return {
        "judge_name": _normalize_str(judge_name),
        "judge_temperature": float(judge_temperature),
        "metric_name": _normalize_str(metric_name),
        "question": _normalize_str(question),
        "answer_text": _normalize_str(answer_text),
        "contexts": _normalize_contexts(contexts),
        "ground_truth": _normalize_str(ground_truth),
    }


def hash_key(components: Dict[str, Any]) -> str:
    """SHA-256 of the canonical-JSON representation of the components."""
    blob = json.dumps(components, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _path_for(cache_dir: Path, key: str) -> Path:
    return cache_dir / key[:2] / f"{key}.json"


def get(cache_dir: Path, key: str) -> Optional[Optional[float]]:
    """Look up a cached score. Returns None on miss; returns the stored
    value (which itself may be a float OR None if we cached a judge failure)
    on hit. Callers that care about the hit/miss distinction should use
    `has()` first."""
    p = _path_for(cache_dir, key)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Corrupt cache entry at %s (%s) — ignoring.", p, e)
        return None
    v = payload.get("score")
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(fv):
        return None
    return fv


def has(cache_dir: Path, key: str) -> bool:
    """True iff a cache entry exists AND its stored score is a usable float.
    A file that exists but contains `null` / NaN is treated as a miss so
    that we'll retry failed judge calls rather than permanently burying
    them."""
    return get(cache_dir, key) is not None


def put(
    cache_dir: Path,
    key: str,
    score: Optional[float],
    components: Dict[str, Any],
) -> None:
    """Write a score atomically (write-temp-then-rename). If score is None
    or NaN we DON'T write an entry — callers treat missing entries as
    retries, and we don't want to cache judge failures."""
    if score is None:
        return
    try:
        fs = float(score)
    except (TypeError, ValueError):
        return
    if math.isnan(fs):
        return
    p = _path_for(cache_dir, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    payload = {
        "score": fs,
        "key_components": components,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache_schema": CACHE_SCHEMA,
    }
    tmp.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, p)


def clear(cache_dir: Path) -> int:
    """Delete every cached entry under cache_dir. Returns the count deleted.
    Use this on RAGAS upgrades, or whenever you suspect the prompts upstream
    of the judge have shifted under you."""
    if not cache_dir.exists():
        return 0
    n = 0
    for p in cache_dir.rglob("*.json"):
        try:
            p.unlink()
            n += 1
        except OSError as e:
            log.warning("Failed to delete %s: %s", p, e)
    return n


def stats(cache_dir: Path) -> Dict[str, int]:
    """Quick summary — count of entries on disk, for CLI / log output."""
    n = 0
    if cache_dir.exists():
        for _ in cache_dir.rglob("*.json"):
            n += 1
    return {"entries": n}


# ---------------------------------------------------------------------------
# Convenience wrapper used by the harness
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheHandle:
    """Bundles the cache directory with the judge identity. Pass this around
    so callers don't have to repeat judge_name/temperature on every lookup."""
    cache_dir: Path
    judge_name: str
    judge_temperature: float

    def key_for(
        self,
        *,
        metric_name: str,
        question: Optional[str],
        answer_text: Optional[str],
        contexts: Optional[Iterable[str]],
        ground_truth: Optional[str],
    ) -> tuple[str, Dict[str, Any]]:
        comps = build_key_components(
            judge_name=self.judge_name,
            judge_temperature=self.judge_temperature,
            metric_name=metric_name,
            question=question,
            answer_text=answer_text,
            contexts=contexts,
            ground_truth=ground_truth,
        )
        return hash_key(comps), comps

    def get(self, key: str) -> Optional[float]:
        return get(self.cache_dir, key)

    def has(self, key: str) -> bool:
        return has(self.cache_dir, key)

    def put(self, key: str, score: Optional[float], components: Dict[str, Any]) -> None:
        put(self.cache_dir, key, score, components)
