"""Tests for :mod:`agns.metrics`."""

from __future__ import annotations

import numpy as np
import pytest

from agns.metrics import (
    failure_rate,
    fit_loglog_slope,
    median_iqr_curves,
    wilcoxon_pair,
)

# ---------------------------------------------------------------------------
# fit_loglog_slope
# ---------------------------------------------------------------------------


def test_fit_loglog_slope_recovers_exponent() -> None:
    k = np.arange(1, 200, dtype=float)
    # Pure power law y = C / k^3 with C = 1.0.
    y = 1.0 / k**3
    fit = fit_loglog_slope(k, y, tail_fraction=0.5, min_y=1e-20)
    assert fit.slope == pytest.approx(-3.0, abs=1e-10)
    assert fit.r_squared > 0.999


def test_fit_loglog_slope_filters_floor() -> None:
    k = np.arange(1, 100, dtype=float)
    y = np.maximum(1.0 / k**2, 1e-12)  # would saturate at the floor near the tail
    fit = fit_loglog_slope(k, y, tail_fraction=1.0, min_y=1e-11)
    # The slope on the un-saturated portion is -2.
    assert fit.slope == pytest.approx(-2.0, abs=0.2)


def test_fit_loglog_slope_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="fewer than 3"):
        fit_loglog_slope(np.array([1.0, 2.0]), np.array([1.0, 1.0]), tail_fraction=1.0)


def test_fit_loglog_slope_validates_tail() -> None:
    with pytest.raises(ValueError):
        fit_loglog_slope(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]), tail_fraction=0.0)
    with pytest.raises(ValueError):
        fit_loglog_slope(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]), tail_fraction=1.5)


# ---------------------------------------------------------------------------
# median_iqr_curves
# ---------------------------------------------------------------------------


def test_median_iqr_curves_same_length() -> None:
    traces = [np.array([1.0, 0.5, 0.25]), np.array([2.0, 1.0, 0.5]), np.array([3.0, 1.5, 0.75])]
    agg = median_iqr_curves(traces)
    assert agg.n_seeds == 3
    np.testing.assert_allclose(agg.x, np.arange(3))
    np.testing.assert_allclose(agg.median, [2.0, 1.0, 0.5])


def test_median_iqr_curves_sticky_end_padding() -> None:
    traces = [np.array([1.0, 0.5]), np.array([2.0, 1.0, 0.5])]
    agg = median_iqr_curves(traces)
    # Shorter trace is extended at its last value 0.5; median at idx 2
    # is median([0.5, 0.5]) = 0.5.
    assert agg.median.shape == (3,)
    assert agg.median[-1] == pytest.approx(0.5)


def test_median_iqr_curves_empty_input() -> None:
    with pytest.raises(ValueError):
        median_iqr_curves([])
    with pytest.raises(ValueError):
        median_iqr_curves([np.array([])])


# ---------------------------------------------------------------------------
# wilcoxon_pair
# ---------------------------------------------------------------------------


def test_wilcoxon_pair_identical_inputs_returns_pvalue_one() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    stat, p = wilcoxon_pair(a, a)
    assert stat == 0.0
    assert p == 1.0


def test_wilcoxon_pair_obvious_difference() -> None:
    a = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    b = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    _stat, p = wilcoxon_pair(a, b)
    assert p < 0.1


def test_wilcoxon_pair_validates_input() -> None:
    with pytest.raises(ValueError):
        wilcoxon_pair(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        wilcoxon_pair(np.array([np.nan]), np.array([0.0]))


# ---------------------------------------------------------------------------
# failure_rate
# ---------------------------------------------------------------------------


def test_failure_rate_counts_nonfinite_as_failure() -> None:
    arr = np.array([1e-10, np.inf, 1.0, np.nan])
    # eps = 1e-8 -> the 1.0 and the two non-finites fail.
    assert failure_rate(arr, eps=1e-8) == pytest.approx(0.75)


def test_failure_rate_empty_input() -> None:
    with pytest.raises(ValueError):
        failure_rate(np.array([]), eps=1e-8)
