"""Unit tests for evaluation.ragas_cache."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from evaluation import ragas_cache as rc


@pytest.fixture
def handle(tmp_path: Path) -> rc.CacheHandle:
    return rc.CacheHandle(
        cache_dir=tmp_path / ".cache",
        judge_name="deepseek-r1:32b",
        judge_temperature=0.0,
    )


# ---------------------------------------------------------------------------
# Key stability / sensitivity
# ---------------------------------------------------------------------------

def test_same_inputs_same_key(handle: rc.CacheHandle) -> None:
    k1, _ = handle.key_for(
        metric_name="faithfulness",
        question="What is X?",
        answer_text="X is Y.",
        contexts=["chunk a", "chunk b"],
        ground_truth="Y.",
    )
    k2, _ = handle.key_for(
        metric_name="faithfulness",
        question="What is X?",
        answer_text="X is Y.",
        contexts=["chunk a", "chunk b"],
        ground_truth="Y.",
    )
    assert k1 == k2


def test_context_order_does_not_affect_key(handle: rc.CacheHandle) -> None:
    """Judges see contexts as a set. Order must not alter the cache key."""
    k1, _ = handle.key_for(
        metric_name="faithfulness",
        question="q",
        answer_text="a",
        contexts=["ctx1", "ctx2", "ctx3"],
        ground_truth="g",
    )
    k2, _ = handle.key_for(
        metric_name="faithfulness",
        question="q",
        answer_text="a",
        contexts=["ctx3", "ctx1", "ctx2"],
        ground_truth="g",
    )
    assert k1 == k2


def test_context_duplicates_are_deduplicated(handle: rc.CacheHandle) -> None:
    k1, _ = handle.key_for(
        metric_name="faithfulness", question="q", answer_text="a",
        contexts=["ctx1", "ctx2"], ground_truth="g",
    )
    k2, _ = handle.key_for(
        metric_name="faithfulness", question="q", answer_text="a",
        contexts=["ctx1", "ctx1", "ctx2", "ctx2"], ground_truth="g",
    )
    assert k1 == k2


@pytest.mark.parametrize("field,bad_value", [
    ("question",     "different question"),
    ("answer_text",  "different answer"),
    ("ground_truth", "different gold"),
    ("contexts",     ["different_ctx"]),
    ("metric_name",  "answer_relevancy"),
])
def test_input_change_invalidates_key(handle: rc.CacheHandle, field: str, bad_value) -> None:
    base = dict(
        metric_name="faithfulness",
        question="q",
        answer_text="a",
        contexts=["ctx"],
        ground_truth="g",
    )
    k_base, _ = handle.key_for(**base)
    mutated = dict(base)
    mutated[field] = bad_value
    k_mut, _ = handle.key_for(**mutated)
    assert k_base != k_mut, f"Changing {field} must invalidate the cache key"


def test_judge_identity_invalidates_key(tmp_path: Path) -> None:
    """Different judges must get different keys — we never want two judges
    to share scores."""
    h_ds = rc.CacheHandle(tmp_path, "deepseek-r1:32b", 0.0)
    h_qw = rc.CacheHandle(tmp_path, "qwen3:8b", 0.0)
    kw = dict(metric_name="faithfulness", question="q", answer_text="a",
              contexts=["ctx"], ground_truth="g")
    k_ds, _ = h_ds.key_for(**kw)
    k_qw, _ = h_qw.key_for(**kw)
    assert k_ds != k_qw


def test_judge_temperature_invalidates_key(tmp_path: Path) -> None:
    h0 = rc.CacheHandle(tmp_path, "deepseek-r1:32b", 0.0)
    h1 = rc.CacheHandle(tmp_path, "deepseek-r1:32b", 0.7)
    kw = dict(metric_name="faithfulness", question="q", answer_text="a",
              contexts=["ctx"], ground_truth="g")
    k0, _ = h0.key_for(**kw)
    k1, _ = h1.key_for(**kw)
    assert k0 != k1


# ---------------------------------------------------------------------------
# Read/write round-trip
# ---------------------------------------------------------------------------

def test_put_and_get_roundtrip(handle: rc.CacheHandle) -> None:
    key, comps = handle.key_for(
        metric_name="faithfulness", question="q", answer_text="a",
        contexts=["ctx"], ground_truth="g",
    )
    assert handle.get(key) is None
    assert not handle.has(key)
    handle.put(key, 0.875, comps)
    assert handle.has(key)
    assert handle.get(key) == pytest.approx(0.875)


def test_nan_not_cached(handle: rc.CacheHandle) -> None:
    """Failed judge calls produce NaN — we refuse to cache those so they
    get retried on the next pass instead of being permanently silenced."""
    key, comps = handle.key_for(
        metric_name="faithfulness", question="q", answer_text="a",
        contexts=["ctx"], ground_truth="g",
    )
    handle.put(key, float("nan"), comps)
    assert handle.get(key) is None
    assert not handle.has(key)


def test_none_not_cached(handle: rc.CacheHandle) -> None:
    key, comps = handle.key_for(
        metric_name="faithfulness", question="q", answer_text="a",
        contexts=["ctx"], ground_truth="g",
    )
    handle.put(key, None, comps)
    assert handle.get(key) is None


def test_corrupt_entry_treated_as_miss(handle: rc.CacheHandle) -> None:
    key, comps = handle.key_for(
        metric_name="faithfulness", question="q", answer_text="a",
        contexts=["ctx"], ground_truth="g",
    )
    # Manually corrupt the file
    from evaluation.ragas_cache import _path_for
    p = _path_for(handle.cache_dir, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("this is not valid JSON {")
    assert handle.get(key) is None
    assert not handle.has(key)


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

def test_clear(handle: rc.CacheHandle) -> None:
    for i in range(3):
        key, comps = handle.key_for(
            metric_name="faithfulness", question=f"q{i}",
            answer_text="a", contexts=["ctx"], ground_truth="g",
        )
        handle.put(key, 0.5, comps)
    assert rc.stats(handle.cache_dir)["entries"] == 3
    deleted = rc.clear(handle.cache_dir)
    assert deleted == 3
    assert rc.stats(handle.cache_dir)["entries"] == 0


def test_clear_nonexistent_dir(tmp_path: Path) -> None:
    assert rc.clear(tmp_path / "does-not-exist") == 0


def test_atomic_write_leaves_no_tmp_files(handle: rc.CacheHandle) -> None:
    """If os.replace works as advertised, no .tmp files should remain."""
    for i in range(5):
        key, comps = handle.key_for(
            metric_name="faithfulness", question=f"q{i}",
            answer_text="a", contexts=["ctx"], ground_truth="g",
        )
        handle.put(key, 0.5, comps)
    tmps = list(handle.cache_dir.rglob("*.tmp"))
    assert tmps == []


def test_key_components_stored_verbatim(handle: rc.CacheHandle) -> None:
    """The stored key_components should let us audit what was hashed."""
    import json
    from evaluation.ragas_cache import _path_for
    key, comps = handle.key_for(
        metric_name="faithfulness", question="q", answer_text="a",
        contexts=["ctx1", "ctx2"], ground_truth="g",
    )
    handle.put(key, 0.5, comps)
    payload = json.loads(_path_for(handle.cache_dir, key).read_text())
    assert payload["key_components"]["metric_name"] == "faithfulness"
    assert payload["key_components"]["contexts"] == ["ctx1", "ctx2"]
    assert payload["cache_schema"] == 1
