"""Small statistical helpers for reporting across seeds.

Used by the notebooks to attach uncertainty to the cost-to-tolerance numbers rather than
reporting bare medians: a percentile bootstrap confidence interval for the median, a paired
Wilcoxon signed-rank test between two methods on matched seeds, a paired TOST equivalence
test (for honest "the two methods are indistinguishable" claims, which must never be drawn
from a *failed* difference test), and a bootstrap CI for the paired median difference.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t
from scipy.stats import wilcoxon


def bootstrap_median_ci(
    samples: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the median of ``samples``.

    Returns ``(median, lo, hi)``. With too few samples the interval collapses to the
    point estimate rather than failing.
    """
    data = np.asarray(samples, dtype=float)
    median = float(np.median(data))
    if data.size < 2:
        return median, median, median
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, data.size, size=(n_resamples, data.size))
    boot = np.median(data[idx], axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(boot, alpha))
    hi = float(np.quantile(boot, 1.0 - alpha))
    return median, lo, hi


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value for ``a`` vs ``b``.

    Returns ``nan`` when the test is undefined (fewer than two pairs, or all-zero
    differences, e.g. both methods identical on every seed).
    """
    av: NDArray[np.float64] = np.asarray(a, dtype=float)
    bv: NDArray[np.float64] = np.asarray(b, dtype=float)
    if av.size != bv.size or av.size < 2 or np.allclose(av, bv):
        return float("nan")
    try:
        return float(wilcoxon(av, bv).pvalue)
    except ValueError:
        return float("nan")


def tost_equivalence(
    a: Sequence[float],
    b: Sequence[float],
    *,
    margin: float,
    alpha: float = 0.05,
) -> tuple[float, bool]:
    """Paired two-one-sided-t (TOST) equivalence test for ``a`` vs ``b``.

    Tests whether the mean paired difference ``mean(a - b)`` lies inside the equivalence
    band ``(-margin, +margin)``. Returns ``(p_value, equivalent)`` where ``p_value`` is the
    larger of the two one-sided p-values and ``equivalent`` is ``p_value < alpha``. This is
    the principled way to assert "no practically meaningful difference"; a non-significant
    *difference* test (e.g. Wilcoxon) does NOT license that claim.

    Parameters
    ----------
    a, b:
        Paired samples (matched seeds); must have equal length >= 2.
    margin:
        Half-width of the equivalence band, in the units of the samples (e.g. matrix
        inverses). Must be positive.
    alpha:
        Significance level for each one-sided test.

    Returns
    -------
    tuple
        ``(p_value, equivalent)``; ``(nan, False)`` when the test is undefined.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    if av.shape != bv.shape or av.size < 2:
        return float("nan"), False
    d = av - bv
    n = d.size
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    if sd == 0.0:
        # Degenerate: the difference is the constant ``mean`` with no spread.
        equivalent = abs(mean) < margin
        return (0.0 if equivalent else 1.0), equivalent
    se = sd / np.sqrt(n)
    df = n - 1
    p_lower = float(student_t.sf((mean + margin) / se, df))  # H1: mean > -margin
    p_upper = float(student_t.cdf((mean - margin) / se, df))  # H1: mean < +margin
    p_value = max(p_lower, p_upper)
    return p_value, bool(p_value < alpha)


def paired_diff_median_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for the median of the paired difference ``a - b``.

    Returns ``(median_diff, lo, hi)``. The half-width ``(hi - lo) / 2`` is the quantity to
    report when stating "no detectable difference at N seeds, CI half-width = ...".
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_median_ci(
        d, confidence=confidence, n_resamples=n_resamples, seed=seed
    )
