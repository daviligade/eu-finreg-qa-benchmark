"""Unit tests for the statistics module.

Run:  pytest evaluation/stats_test.py -q
"""

import numpy as np
import pytest

from evaluation.stats import (
    bonferroni,
    compare_paired,
    cohens_d_paired,
    fdr_bh,
    mcnemar,
    paired_bootstrap_bca,
    wilcoxon_signed_rank,
)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_contains_truth_on_clean_signal():
    rng = np.random.default_rng(7)
    # Construct a real effect of +0.1, bounded in [0,1]. Clip to stay legal.
    a = np.clip(rng.uniform(0.3, 0.9, 100), 0, 1)
    b = np.clip(a - 0.1, 0, 1)
    obs, lo, hi = paired_bootstrap_bca(a, b, n_iter=2000, seed=1)
    assert lo <= obs <= hi
    # Effect is positive; CI should not include 0.
    assert lo > 0

def test_bootstrap_zero_effect_includes_zero():
    rng = np.random.default_rng(7)
    a = rng.uniform(0, 1, 80)
    b = a.copy()  # zero difference
    obs, lo, hi = paired_bootstrap_bca(a, b, n_iter=2000, seed=1)
    assert obs == 0.0
    assert lo == 0.0 and hi == 0.0

def test_bootstrap_rejects_unequal_length():
    with pytest.raises(ValueError):
        paired_bootstrap_bca([0.5, 0.6], [0.5], n_iter=100)


# ---------------------------------------------------------------------------
# Wilcoxon / McNemar
# ---------------------------------------------------------------------------

def test_wilcoxon_detects_shift():
    rng = np.random.default_rng(7)
    a = rng.uniform(0.5, 0.9, 100)
    b = a - 0.2  # strong shift
    assert wilcoxon_signed_rank(a, b) < 0.01

def test_wilcoxon_identical_samples():
    a = np.array([0.5] * 20)
    b = np.array([0.5] * 20)
    assert wilcoxon_signed_rank(a, b) == 1.0

def test_mcnemar_rejects_bias():
    # 40 discordant pairs, 35/5 split -> should be highly significant
    a = np.array([1] * 35 + [0] * 5 + [1] * 10 + [0] * 10, dtype=float)
    b = np.array([0] * 35 + [1] * 5 + [1] * 10 + [0] * 10, dtype=float)
    assert mcnemar(a, b) < 0.001

def test_mcnemar_no_discordance():
    a = np.array([1, 0, 1, 0, 1], dtype=float)
    b = np.array([1, 0, 1, 0, 1], dtype=float)
    assert mcnemar(a, b) == 1.0

def test_mcnemar_rejects_non_binary():
    with pytest.raises(ValueError):
        mcnemar(np.array([0.0, 0.5]), np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------

def test_cohens_d_sign_and_magnitude():
    a = np.array([0.9] * 20)
    b = np.array([0.5] * 20)
    # Zero variance in differences -> function returns 0.0 by our convention.
    assert cohens_d_paired(a, b) == 0.0
    # Now with some variance
    rng = np.random.default_rng(1)
    a2 = rng.uniform(0.5, 0.9, 50)
    b2 = a2 - 0.1 + rng.normal(0, 0.05, 50)
    assert cohens_d_paired(a2, b2) > 0


# ---------------------------------------------------------------------------
# compare_paired orchestration
# ---------------------------------------------------------------------------

def test_compare_paired_selects_mcnemar_for_binary():
    a = [1] * 30 + [0] * 20
    b = [0] * 30 + [1] * 20
    r = compare_paired(a, b, system_a="V2", system_b="V1", metric="CVR")
    assert r.test_name == "mcnemar"

def test_compare_paired_selects_wilcoxon_for_continuous():
    rng = np.random.default_rng(1)
    a = rng.uniform(0.4, 0.9, 50).tolist()
    b = [x - 0.05 for x in a]
    r = compare_paired(a, b, system_a="V2", system_b="V1", metric="APA")
    assert r.test_name == "wilcoxon"

def test_compare_paired_drops_missing_pairs():
    a = [0.5, 0.6, None, 0.7, 0.8]
    b = [0.4, None, 0.6, 0.7, 0.9]
    r = compare_paired(a, b, system_a="V2", system_b="V1", metric="APA")
    assert r.n == 3
    assert r.n_pairs_discarded == 2

def test_noninferiority_pass_when_ci_within_margin():
    rng = np.random.default_rng(1)
    # a is on average slightly *worse* than b but within the 3pp margin.
    b = rng.uniform(0.7, 0.95, 80)
    a = b - rng.uniform(0.00, 0.02, 80)
    r = compare_paired(a, b, system_a="V2", system_b="V1",
                       metric="RC", noninferiority_margin=0.03, n_iter=2000)
    assert r.noninferior is True

def test_noninferiority_fail_when_ci_exceeds_margin():
    rng = np.random.default_rng(1)
    b = rng.uniform(0.7, 0.95, 80)
    a = b - rng.uniform(0.10, 0.15, 80)   # ~12pp worse
    r = compare_paired(a, b, system_a="V2", system_b="V1",
                       metric="RC", noninferiority_margin=0.03, n_iter=2000)
    assert r.noninferior is False


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------

def test_bonferroni_clips_at_one():
    out = bonferroni([0.04, 0.04, 0.04])
    assert np.allclose(out, [0.12, 0.12, 0.12])
    out = bonferroni([0.5, 0.5])
    assert np.allclose(out, [1.0, 1.0])

def test_fdr_bh_monotone_and_bounded():
    ps = [0.001, 0.01, 0.03, 0.06, 0.5]
    adj = fdr_bh(ps)
    # Monotone non-decreasing with raw p order
    assert np.all(np.diff(adj) >= -1e-9)
    assert np.all((adj >= 0) & (adj <= 1))
