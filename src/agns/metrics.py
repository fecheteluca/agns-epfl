"""Statistical helpers for the experimental campaign.

Three families of operations:

1. **Rate fitting.**  :func:`fit_loglog_slope` returns the least-squares
   slope of ``log10(y) ~ slope * log10(x) + intercept`` on a configurable
   tail window, plus an R^2 goodness-of-fit so weak fits can be flagged.

2. **Multi-seed aggregation.**  :func:`median_iqr_curves` aligns per-seed
   traces and returns ``(median, lower, upper)`` curves on a common
   iteration grid for downstream plotting.

3. **Hypothesis testing.**  :func:`wilcoxon_pair` runs the paired
   Wilcoxon signed-rank test on two methods' end-of-run residuals, and
   :func:`failure_rate` reports the fraction of seeds for which a method
   failed to drop below a target ``eps``.

The implementation is conservative: any input that contains non-finite
entries, mis-matched seed counts, or an empty trace raises ``ValueError``
rather than silently producing NaNs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

__all__ = [
    "LogLogFit",
    "SeedAggregate",
    "failure_rate",
    "fit_loglog_slope",
    "median_iqr_curves",
    "wilcoxon_pair",
]


# ---------------------------------------------------------------------------
# Rate fitting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogLogFit:
    """Result of a least-squares fit on ``log10(y) ~ slope * log10(x) + intercept``.

    ``slope`` is the asymptotic decay exponent (negative for residuals
    decreasing along ``x``).  ``r_squared`` is the coefficient of
    determination on the log-log axes; values < 0.95 should be treated
    as "fit is suspect" rather than "rate confirmed".
    """

    slope: float
    intercept: float
    r_squared: float
    n_points: int


def fit_loglog_slope(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    tail_fraction: float = 0.5,
    min_y: float = 1e-10,
) -> LogLogFit:
    """Fit ``log10(y) = slope * log10(x) + intercept`` on the asymptotic tail.

    The asymptotic worst-case rate is visible only in a window that
    excludes the warm-up transient (slope is shallow) and the
    near-floor Q-superlinear endgame (slope is very steep, often
    dominated by floating-point noise near 1e-16).  Fitting on the
    entire trailing half conflates these regimes and reports a slope
    that represents neither.

    Window selection: take the trailing ``tail_fraction`` of the trace,
    then drop points whose residual has entered the Q-superlinear or
    floor regime (``y <= min_y``).

    Parameters
    ----------
    x, y : 1-D arrays of the same length
        ``x`` should be strictly positive; ``y`` should be non-negative.
        Entries ``y <= min_y`` are dropped.
    tail_fraction : float in (0, 1]
        Fraction of the trailing points considered before applying the
        ``min_y`` filter.  ``0.5`` -> second half; ``1.0`` -> entire trace.
    min_y : float
        Lower asymptotic-window floor; entries at or below this are
        skipped.  Default ``1e-10`` excludes the Q-superlinear endgame
        and the floating-point floor.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y shape mismatch: {x.shape} vs {y.shape}")
    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError(f"tail_fraction must lie in (0, 1], got {tail_fraction}")

    n = x.size
    start = max(0, int(n * (1.0 - tail_fraction)))
    x_tail = x[start:]
    y_tail = y[start:]
    keep = (x_tail > 0) & (y_tail > min_y) & np.isfinite(x_tail) & np.isfinite(y_tail)
    x_fit = x_tail[keep]
    y_fit = y_tail[keep]
    if x_fit.size < 3:
        raise ValueError(
            f"fewer than 3 usable points after filtering ({x_fit.size}); "
            f"cannot fit a slope -- consider extending the run or loosening min_y"
        )
    log_x = np.log10(x_fit)
    log_y = np.log10(y_fit)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    y_hat = slope * log_x + intercept
    ss_res = float(np.sum((log_y - y_hat) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return LogLogFit(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        n_points=int(x_fit.size),
    )


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedAggregate:
    """Median trace plus IQR bounds on a shared iteration grid."""

    x: NDArray[np.float64]
    median: NDArray[np.float64]
    lower: NDArray[np.float64]  # 25th percentile
    upper: NDArray[np.float64]  # 75th percentile
    n_seeds: int


def median_iqr_curves(
    traces: list[NDArray[np.float64]],
    *,
    common_grid: NDArray[np.float64] | None = None,
) -> SeedAggregate:
    """Align ``traces`` to a common iteration index and return median+IQR.

    Each trace is a 1-D array indexed by iteration ``k = 0, 1, ...``.
    Traces are not required to be the same length: shorter traces are
    extended at their last value (i.e. a method that stopped early
    continues to contribute its final residual).  The default grid is
    the integer range up to the longest trace.

    The 25th/75th percentile bounds are the standard IQR; consumers
    plotting on a log axis should clip the lower bound at a small
    positive number to avoid ``log(0)``.
    """
    if not traces:
        raise ValueError("traces must be non-empty")
    arrs = [np.asarray(t, dtype=np.float64).ravel() for t in traces]
    if any(a.size == 0 for a in arrs):
        raise ValueError("every trace must be non-empty")
    max_len = max(a.size for a in arrs)
    if common_grid is None:
        common_grid = np.arange(max_len, dtype=np.float64)
    padded = np.array(
        [np.pad(a, (0, max_len - a.size), mode="edge") for a in arrs],
        dtype=np.float64,
    )
    if padded.shape[1] != common_grid.size:
        raise ValueError(
            f"common_grid length {common_grid.size} != padded trace length {padded.shape[1]}"
        )
    median = np.median(padded, axis=0)
    lower = np.percentile(padded, 25.0, axis=0)
    upper = np.percentile(padded, 75.0, axis=0)
    return SeedAggregate(
        x=common_grid,
        median=median,
        lower=lower,
        upper=upper,
        n_seeds=len(arrs),
    )


# ---------------------------------------------------------------------------
# Hypothesis tests + failure rate
# ---------------------------------------------------------------------------


def wilcoxon_pair(a: NDArray[np.float64], b: NDArray[np.float64]) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test on per-seed final residuals.

    Returns ``(statistic, p_value)``.  Null hypothesis: the two methods
    have the same per-seed final residual distribution.  A small p-value
    favours one method over the other.

    Both arrays must contain finite entries and have the same length.
    Identical paired values short-circuit to ``(0.0, 1.0)``: scipy
    raises in that case but the empirically meaningful answer is "no
    evidence of difference".
    """
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"shape mismatch: {a_arr.shape} vs {b_arr.shape}")
    if a_arr.size < 1:
        raise ValueError("empty input")
    if not (np.all(np.isfinite(a_arr)) and np.all(np.isfinite(b_arr))):
        raise ValueError("non-finite entries in input")
    if np.array_equal(a_arr, b_arr):
        return (0.0, 1.0)
    stat = stats.wilcoxon(a_arr, b_arr)
    return (float(stat.statistic), float(stat.pvalue))


def failure_rate(final_residuals: NDArray[np.float64], *, eps: float) -> float:
    """Fraction of seeds where the final residual did not drop below ``eps``.

    A method that diverged (non-finite final residual) is counted as a
    failure: ``np.nan`` and ``np.inf`` always exceed any finite ``eps``.
    """
    arr = np.asarray(final_residuals, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("empty input")
    failed = ~np.isfinite(arr) | (arr > eps)
    return float(failed.mean())
