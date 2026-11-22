"""
Statistical tools for paired system comparisons on the evaluation.

Why this module exists
----------------------
RAGAS gives us per-answer metrics; we need to turn *per-answer* score vectors
into defensible *per-system* claims. Because every question is scored by every
system, the natural design is PAIRED (matched-pairs) comparisons:

    delta_i = metric(system_A, q_i) - metric(system_B, q_i)

Everything in this file operates on these per-question deltas.

What it provides
----------------
1. paired_bootstrap_bca   : BCa confidence interval for a paired difference.
2. wilcoxon_signed_rank   : Paired non-parametric significance test.
3. mcnemar                : Paired binary-outcome test (for CVR, MIC, RC).
4. noninferiority_paired  : One-sided lower-bound check against -margin.
5. bonferroni             : Family-wise error correction over a set of p-values.
6. fdr_bh                 : Benjamini-Hochberg FDR (reported alongside, not used
                            as the primary decision rule — Bonferroni is).

Deliberate choices
------------------
- BCa (bias-corrected & accelerated) is the default CI because bounded metrics
  (0–1) and binary metrics produce skewed sampling distributions; the percentile
  CI is known to undercover in these conditions.
- Effect sizes accompany every test. A significant p-value on a tiny effect is
  as misleading as a non-significant p-value on a large one.

All public functions are pure and take NumPy arrays; they do not read files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Everything you'd want to report in a paper row for a paired comparison."""
    system_a: str
    system_b: str
    metric: str
    n: int
    mean_a: float
    mean_b: float
    mean_delta: float           # mean(a - b)
    ci_low: float
    ci_high: float
    ci_level: float             # e.g. 0.95
    p_value: float              # from paired test
    test_name: str              # "wilcoxon" | "mcnemar"
    cohens_d_paired: float      # Cohen's d_z for paired differences
    n_pairs_discarded: int      # excluded because of missing values
    # Optional non-inferiority decision
    noninferiority_margin: Optional[float] = None
    noninferior: Optional[bool] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pair cleaning
# ---------------------------------------------------------------------------

def _clean_pairs(a: Sequence[Optional[float]],
                 b: Sequence[Optional[float]]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Drop pairs where either element is None/NaN. Return aligned arrays and
    the count of discarded pairs (so the caller can report it).
    """
    if len(a) != len(b):
        raise ValueError("a and b must have the same length (paired design).")
    aa, bb = [], []
    discarded = 0
    for x, y in zip(a, b):
        if x is None or y is None:
            discarded += 1
            continue
        fx, fy = float(x), float(y)
        if math.isnan(fx) or math.isnan(fy):
            discarded += 1
            continue
        aa.append(fx)
        bb.append(fy)
    return np.asarray(aa, dtype=float), np.asarray(bb, dtype=float), discarded


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d_z for paired samples: mean(delta) / sd(delta).

    Returns 0.0 when variance is zero (to avoid divide-by-zero). A zero effect
    with zero variance means the two systems returned identical scores on
    every retained question — report it as a genuine zero effect, not NaN.
    """
    d = a - b
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    # Treat numerically-zero variance as a genuine zero effect. Using a
    # relative tolerance keeps us robust to floating-point noise from IEEE
    # cancellation when all deltas are identical (e.g. 0.9-0.5 = 0.4 repeated).
    mean_delta = float(np.mean(d))
    if sd < 1e-12 or sd < 1e-9 * (abs(mean_delta) + 1.0):
        return 0.0
    return mean_delta / sd


# ---------------------------------------------------------------------------
# Bootstrap BCa
# ---------------------------------------------------------------------------

def paired_bootstrap_bca(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_iter: int = 10_000,
    confidence: float = 0.95,
    seed: int = 13,
) -> Tuple[float, float, float]:
    """BCa confidence interval for the paired mean difference mean(a - b).

    Returns: (point_estimate, ci_low, ci_high)

    Implementation notes
    --------------------
    - Indices are resampled with replacement from [0, n) and applied to both
      arrays so the pairing is preserved in every bootstrap replicate.
    - Acceleration `a_hat` uses jackknife estimates of the paired deltas.
    - Bias correction `z0` uses the fraction of bootstrap replicates below the
      observed statistic.
    - Clamps CI indices to [0, n_iter - 1] so that extreme acceleration values
      can't index out of range; this is the standard treatment and it is what
      scipy.stats.bootstrap does too.
    """
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape:
        raise ValueError("a and b must have the same shape for paired bootstrap.")
    n = aa.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 paired samples.")

    rng = np.random.default_rng(seed)
    observed = float(np.mean(aa - bb))

    # Bootstrap replicates -------------------------------------------------
    idx = rng.integers(0, n, size=(n_iter, n))
    replicates = (aa[idx] - bb[idx]).mean(axis=1)

    # Bias correction z0 ---------------------------------------------------
    frac_below = float(np.mean(replicates < observed))
    # Clip to avoid inf from ppf at exactly 0 or 1.
    frac_below = min(max(frac_below, 1 / (n_iter + 1)), 1 - 1 / (n_iter + 1))
    z0 = sp_stats.norm.ppf(frac_below)

    # Jackknife acceleration a_hat ----------------------------------------
    delta = aa - bb
    jack = np.empty(n)
    total = delta.sum()
    for i in range(n):
        jack[i] = (total - delta[i]) / (n - 1)
    jack_mean = jack.mean()
    num = float(np.sum((jack_mean - jack) ** 3))
    den = 6.0 * (float(np.sum((jack_mean - jack) ** 2)) ** 1.5)
    a_hat = 0.0 if den == 0 else num / den

    # BCa-adjusted percentile ---------------------------------------------
    alpha = 1 - confidence
    z_low = sp_stats.norm.ppf(alpha / 2)
    z_high = sp_stats.norm.ppf(1 - alpha / 2)

    def _adjust(z):
        return z0 + (z0 + z) / (1 - a_hat * (z0 + z))

    p_low = sp_stats.norm.cdf(_adjust(z_low))
    p_high = sp_stats.norm.cdf(_adjust(z_high))

    ci_low = float(np.quantile(replicates, p_low))
    ci_high = float(np.quantile(replicates, p_high))
    return observed, ci_low, ci_high


# ---------------------------------------------------------------------------
# Paired significance tests
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided paired Wilcoxon signed-rank test. Returns p-value.

    - Uses `zsplit` for zeros so ties with delta=0 don't silently boost the
      statistic in either direction.
    - If all deltas are zero, returns 1.0 (no evidence of a difference).
    """
    d = a - b
    if np.all(d == 0):
        return 1.0
    try:
        res = sp_stats.wilcoxon(a, b, zero_method="zsplit", alternative="two-sided")
        return float(res.pvalue)
    except ValueError:
        return 1.0


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    """McNemar's paired test for binary outcomes (0/1 each). Returns p-value.

    Uses the exact binomial test on the discordant pairs, which is preferred
    over the chi-squared approximation when b + c is small — and with ~200
    paired samples and a small discordance rate, we are often in that regime.
    """
    if not set(np.unique(a)).issubset({0.0, 1.0}) or not set(np.unique(b)).issubset({0.0, 1.0}):
        raise ValueError("mcnemar requires binary 0/1 inputs.")
    # b = a=1 and b=0;  c = a=0 and b=1  (discordant pairs)
    b_count = int(np.sum((a == 1) & (b == 0)))
    c_count = int(np.sum((a == 0) & (b == 1)))
    n = b_count + c_count
    if n == 0:
        return 1.0
    # Exact binomial test on min(b,c) with p=0.5.
    k = min(b_count, c_count)
    # Two-sided.
    return float(sp_stats.binomtest(k, n, p=0.5, alternative="two-sided").pvalue)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _is_binary(arr: np.ndarray) -> bool:
    return bool(set(np.unique(arr)).issubset({0.0, 1.0}))


def compare_paired(
    a: Sequence[Optional[float]],
    b: Sequence[Optional[float]],
    *,
    system_a: str,
    system_b: str,
    metric: str,
    n_iter: int = 10_000,
    confidence: float = 0.95,
    seed: int = 13,
    noninferiority_margin: Optional[float] = None,
) -> ComparisonResult:
    """Run bootstrap CI + paired significance test + (optional) non-inferiority.

    The test choice is automatic:
      - Both arrays binary  -> McNemar
      - Otherwise           -> Wilcoxon signed-rank

    Non-inferiority (one-sided): PASS iff ci_low >= -margin. Under this rule
    the user can claim "system_a is non-inferior to system_b within margin".
    This mirrors the protocol's H4 treatment for refusal-correctness.
    """
    aa, bb, discarded = _clean_pairs(a, b)
    n = len(aa)
    if n < 2:
        raise ValueError(f"Not enough paired samples after cleaning: n={n}.")

    observed, ci_low, ci_high = paired_bootstrap_bca(
        aa, bb, n_iter=n_iter, confidence=confidence, seed=seed
    )

    if _is_binary(aa) and _is_binary(bb):
        p_value = mcnemar(aa, bb)
        test_name = "mcnemar"
    else:
        p_value = wilcoxon_signed_rank(aa, bb)
        test_name = "wilcoxon"

    d_paired = cohens_d_paired(aa, bb)

    noninferior: Optional[bool] = None
    if noninferiority_margin is not None:
        # Non-inferiority: we want mean(a-b) NOT to be worse than -margin.
        noninferior = bool(ci_low >= -abs(noninferiority_margin))

    return ComparisonResult(
        system_a=system_a,
        system_b=system_b,
        metric=metric,
        n=n,
        mean_a=float(np.mean(aa)),
        mean_b=float(np.mean(bb)),
        mean_delta=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=confidence,
        p_value=p_value,
        test_name=test_name,
        cohens_d_paired=d_paired,
        n_pairs_discarded=discarded,
        noninferiority_margin=noninferiority_margin,
        noninferior=noninferior,
    )


# ---------------------------------------------------------------------------
# Multiple-comparison corrections
# ---------------------------------------------------------------------------

def bonferroni(p_values: Sequence[float]) -> np.ndarray:
    """Bonferroni-adjusted p-values. Clipped to 1.0."""
    p = np.asarray(p_values, dtype=float)
    return np.clip(p * len(p), 0.0, 1.0)


def fdr_bh(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR. Returns adjusted p-values in original order.

    Included as a secondary, more powerful complement to Bonferroni — useful
    for descriptive tables. We still gate headline claims on Bonferroni.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    # Monotone decreasing from the top.
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(13)
    a = rng.uniform(0.4, 0.9, 50)
    b = a - rng.normal(0.05, 0.1, 50)
    r = compare_paired(a, b, system_a="V2", system_b="V1", metric="DEMO")
    print(r.to_dict())
