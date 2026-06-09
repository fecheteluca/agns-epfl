"""Unit tests for the reporting statistics on inputs with known answers.

The bootstrap median CI and the paired Wilcoxon test back the uncertainty reporting in the
notebooks, so their edge cases (too-few samples, all-equal inputs, size mismatch) and their
agreement with ``scipy.stats.wilcoxon`` are pinned here on inputs whose answers are known.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import wilcoxon

from agns.stats import (
    bootstrap_median_ci,
    paired_diff_median_ci,
    paired_wilcoxon,
    tost_equivalence,
)

# ----------------------------------------------------------------- bootstrap CI


def test_bootstrap_median_is_exact():
    """The point estimate equals the sample median."""
    median, lo, hi = bootstrap_median_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert median == 3.0
    assert lo <= median <= hi


def test_bootstrap_ci_within_data_range():
    """A percentile bootstrap CI cannot fall outside the observed data range."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    _, lo, hi = bootstrap_median_ci(data)
    assert min(data) <= lo <= hi <= max(data)


def test_bootstrap_single_sample_collapses():
    """With a single sample the interval collapses to the point estimate."""
    median, lo, hi = bootstrap_median_ci([7.5])
    assert median == lo == hi == 7.5


def test_bootstrap_constant_samples_collapse():
    """All-equal samples give a degenerate interval equal to the constant."""
    median, lo, hi = bootstrap_median_ci([4.0, 4.0, 4.0, 4.0])
    assert median == lo == hi == 4.0


def test_bootstrap_is_deterministic_in_seed():
    """The same seed yields the same interval; a different seed may differ."""
    data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    a = bootstrap_median_ci(data, seed=0)
    b = bootstrap_median_ci(data, seed=0)
    assert a == b


def test_bootstrap_ci_brackets_median_ordering():
    """Lower bound never exceeds the median and the median never exceeds the upper bound."""
    data = list(np.linspace(0.0, 10.0, 21))
    median, lo, hi = bootstrap_median_ci(data, n_resamples=5000, seed=1)
    assert lo <= median <= hi


# ----------------------------------------------------------------- Wilcoxon


def test_wilcoxon_nan_for_too_few_pairs():
    """Fewer than two pairs is undefined -> NaN."""
    assert math.isnan(paired_wilcoxon([1.0], [2.0]))


def test_wilcoxon_nan_for_identical_inputs():
    """All-zero differences (identical methods on every seed) -> NaN."""
    assert math.isnan(paired_wilcoxon([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))


def test_wilcoxon_nan_for_size_mismatch():
    """Unequal sample sizes are undefined -> NaN."""
    assert math.isnan(paired_wilcoxon([1.0, 2.0, 3.0], [1.0, 2.0]))


def test_wilcoxon_matches_scipy():
    """On a concrete paired sample, the helper equals scipy's two-sided p-value."""
    a = [12.0, 9.0, 11.0, 14.0, 10.0, 13.0, 8.0]
    b = [10.0, 8.0, 12.0, 11.0, 9.0, 10.0, 7.0]
    expected = float(wilcoxon(np.asarray(a), np.asarray(b)).pvalue)
    assert paired_wilcoxon(a, b) == expected


def test_wilcoxon_significant_for_consistent_shift():
    """A consistent shift across all pairs is detected at the 5% level."""
    a = [float(i) for i in range(1, 9)]
    b = [x + 5.0 for x in a]  # b strictly larger on every pair
    p = paired_wilcoxon(a, b)
    assert 0.0 <= p <= 1.0
    assert p < 0.05


def test_wilcoxon_is_symmetric():
    """The two-sided p-value does not depend on argument order."""
    a = [5.0, 3.0, 8.0, 2.0, 7.0, 6.0]
    b = [4.0, 4.0, 6.0, 3.0, 5.0, 5.0]
    assert paired_wilcoxon(a, b) == paired_wilcoxon(b, a)


# ----------------------------------------------------------------- TOST equivalence


def test_tost_declares_equivalence_for_tiny_paired_differences():
    """Pairs differing by a small amount are equivalent within a generous margin."""
    rng = np.random.RandomState(0)
    a = rng.normal(50.0, 1.0, size=30)
    b = a + rng.normal(0.0, 0.2, size=30)  # tiny, centered difference
    p, equivalent = tost_equivalence(a, b, margin=2.0)
    assert equivalent
    assert p < 0.05


def test_tost_rejects_equivalence_for_large_shift():
    """A shift well beyond the margin is not declared equivalent."""
    rng = np.random.RandomState(1)
    a = rng.normal(50.0, 1.0, size=30)
    b = a + 5.0  # far outside a margin of 1
    p, equivalent = tost_equivalence(a, b, margin=1.0)
    assert not equivalent
    assert p > 0.05


def test_tost_nan_for_size_mismatch():
    """Mismatched sizes are undefined -> (nan, False)."""
    p, equivalent = tost_equivalence([1.0, 2.0, 3.0], [1.0, 2.0], margin=1.0)
    assert math.isnan(p)
    assert equivalent is False


def test_tost_requires_positive_margin():
    """A non-positive margin is a usage error."""
    with pytest.raises(ValueError, match="margin must be positive"):
        tost_equivalence([1.0, 2.0], [1.0, 2.0], margin=0.0)


def test_tost_degenerate_constant_difference():
    """Zero-spread differences fall back to a direct margin comparison."""
    a = [10.0, 11.0, 12.0]
    b = [9.5, 10.5, 11.5]  # constant difference of +0.5, no spread
    p_in, eq_in = tost_equivalence(a, b, margin=1.0)
    assert eq_in and p_in == 0.0
    p_out, eq_out = tost_equivalence(a, b, margin=0.25)
    assert (not eq_out) and p_out == 1.0


# ----------------------------------------------------------- paired difference CI


def test_paired_diff_median_ci_matches_difference_bootstrap():
    """The paired-difference CI equals the bootstrap CI of the elementwise difference."""
    a = [12.0, 9.0, 11.0, 14.0, 10.0, 13.0]
    b = [10.0, 8.0, 12.0, 11.0, 9.0, 10.0]
    got = paired_diff_median_ci(a, b, seed=0)
    expected = bootstrap_median_ci(np.asarray(a) - np.asarray(b), seed=0)
    assert got == expected
    median, lo, hi = got
    assert lo <= median <= hi
