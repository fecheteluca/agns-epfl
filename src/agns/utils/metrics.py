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

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

__all__ = [
    "BOOTSTRAP_DEFAULT_RNG_SEED",
    "MAX_FIT_ABS_SLOPE",
    "MIN_FIT_LOG10_X_SPAN",
    "MIN_FIT_POINTS",
    "BootstrapCI",
    "LogLogFit",
    "SeedAggregate",
    "bootstrap_ratio_ci",
    "failure_rate",
    "fit_loglog_slope",
    "median_iqr_curves",
    "paired_wilcoxon",
    "wilcoxon_pair",
]

#: Default RNG seed for the bootstrap helpers.  Pinned so two callers
#: with the same input data and the same ``n_resamples`` get bit-stable
#: CI endpoints.  Documented in ``docs/statistics.md``.
BOOTSTRAP_DEFAULT_RNG_SEED: int = 0


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


#: Minimum number of surviving points required for a defensible log-log fit.
#: Three is the algebraic minimum (two would give an exact line through
#: any two points and zero residual); five buys enough redundancy that
#: a single outlier point cannot dominate the slope estimate.
MIN_FIT_POINTS = 5

#: Minimum log10(x_max / x_min) the surviving window must span before
#: the slope is considered meaningful.  0.15 decades catches the
#: pathological AGNS-near-floor cluster (e.g. 5 points at k = 11..15,
#: span ~0.135 decades) while still admitting fits on the trailing
#: half of short traces (length ~16 -> trailing half spans
#: log10(2) ~= 0.30 decades, shrinks to ~0.18-0.22 after ``min_y``
#: filters out the floor end).  Below 0.15 the slope estimate is
#: dominated by near-floor noise and the polyfit is unconstrained.
MIN_FIT_LOG10_X_SPAN = 0.15

#: Maximum physically-meaningful absolute slope for a polynomial-decay
#: rate ``O(1/k^(p+nu))``.  In the Holder smoothness theory ``p+nu``
#: rarely exceeds 4; values above ~6 signal that the trace is in a
#: superlinear (e.g. quadratic) convergence regime where log10(y) vs
#: log10(k) is no longer a line.  Above 8 the fit is structurally
#: wrong for the underlying process and we refuse to report a slope.
MAX_FIT_ABS_SLOPE = 8.0


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

    Raises
    ------
    ValueError
        If ``x`` and ``y`` differ in shape, ``tail_fraction`` is out of
        range, fewer than :data:`MIN_FIT_POINTS` points survive the
        window / ``min_y`` filter, the surviving x-axis spans fewer
        than :data:`MIN_FIT_LOG10_X_SPAN` decades, or the surviving
        points do not admit a finite least-squares line.  The fit
        never returns a non-finite slope.
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
    if x_fit.size < MIN_FIT_POINTS:
        raise ValueError(
            f"fewer than {MIN_FIT_POINTS} usable points after filtering "
            f"({x_fit.size}); cannot fit a defensible slope -- consider "
            f"extending the run or loosening min_y"
        )
    log_x = np.log10(x_fit)
    x_span = float(log_x.max() - log_x.min())
    if x_span < MIN_FIT_LOG10_X_SPAN:
        raise ValueError(
            f"surviving x-axis span {x_span:.2f} decades is below the "
            f"{MIN_FIT_LOG10_X_SPAN:.2f}-decade minimum; the window is too "
            f"narrow for a meaningful log-log slope (method likely skipped "
            f"the polynomial-decay regime)"
        )
    log_y = np.log10(y_fit)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        raise ValueError(
            "log-log least-squares fit is degenerate (non-finite slope); "
            "the filtered window does not admit a unique line"
        )
    if abs(slope) > MAX_FIT_ABS_SLOPE:
        raise ValueError(
            f"fitted |slope| = {abs(slope):.1f} exceeds the "
            f"{MAX_FIT_ABS_SLOPE:.1f} polynomial-decay ceiling; the method "
            f"is in a superlinear convergence regime and log(y) vs log(k) "
            f"is no longer a line"
        )
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


@dataclass(frozen=True)
class BootstrapCI:
    """Percentile bootstrap confidence interval for a ratio statistic.

    Attributes
    ----------
    point_estimate :
        ``median(a) / median(b)`` on the full input arrays.  ``nan``
        when ``median(b) == 0``; ``inf`` when the numerator is
        positive and the denominator is zero.
    lo, hi :
        2.5% / 97.5% percentile of the bootstrap distribution by
        default.  Both may be ``inf`` when every resample produced an
        infinite ratio.
    n_resamples :
        Number of resamples actually drawn.
    n_paired_seeds :
        Number of (a, b) pairs that contributed (those where both
        sides were finite; pairs with ``None`` on either side are
        propagated as infinity to the ratio rather than dropped, so
        this equals ``len(a)`` for typical inputs).
    rng_seed :
        Seed used to construct the bootstrap RNG.  Pinned so the CI
        is deterministic given the inputs.
    """

    point_estimate: float
    lo: float
    hi: float
    n_resamples: int
    n_paired_seeds: int
    rng_seed: int

    @property
    def covers_unity(self) -> bool:
        """``True`` iff the CI brackets ``1.0`` (the "no speedup" null)."""
        return bool(self.lo <= 1.0 <= self.hi)


def _to_float_with_inf(values: Sequence[float | int | None]) -> NDArray[np.float64]:
    """Cast a list of per-seed iter counts (``None`` -> ``inf``) to a float array."""
    out = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            out[i] = np.inf
        else:
            out[i] = float(v)
    return out


def bootstrap_ratio_ci(
    a: Sequence[float | int | None],
    b: Sequence[float | int | None],
    *,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    rng_seed: int = BOOTSTRAP_DEFAULT_RNG_SEED,
) -> BootstrapCI:
    """Pair-preserving percentile bootstrap CI for ``median(a) / median(b)``.

    The two input arrays are treated as **paired** (same seed indices
    on both sides).  On each of ``n_resamples`` iterations a vector
    of seed indices is drawn with replacement; both arrays are
    indexed with the *same* draw; the ratio of medians is recorded.
    The CI is the empirical ``[(1-ci)/2, (1+ci)/2]`` quantile of
    those recorded ratios.

    ``None`` entries (the convention this project uses for "never
    crossed eps") are propagated as ``+inf`` so that a resample
    dominated by non-crossers reports an infinite ratio, exactly
    mirroring the meaning the user expects.

    Parameters
    ----------
    a, b :
        Per-seed iter counts (or any paired numeric measurements).
        Must have the same length.
    n_resamples :
        Number of bootstrap iterations.  Default ``10_000`` keeps
        the Monte Carlo error at ~1% of the CI half-width.
    ci :
        Coverage probability.  Must lie in ``(0, 1)``; default 0.95.
    rng_seed :
        Seed for the bootstrap RNG.  Defaults to the project-wide
        :data:`BOOTSTRAP_DEFAULT_RNG_SEED`.

    Raises
    ------
    ValueError
        On mismatched shapes, empty inputs, or ``ci`` outside ``(0, 1)``.
    """
    if len(a) != len(b):
        raise ValueError(f"a and b length mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        raise ValueError("empty input")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must lie in (0, 1), got {ci}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    a_arr = _to_float_with_inf(a)
    b_arr = _to_float_with_inf(b)
    n = a_arr.size

    # Point estimate on the full data.
    med_a = float(np.median(a_arr))
    med_b = float(np.median(b_arr))
    with np.errstate(divide="ignore", invalid="ignore"):
        point = med_a / med_b if med_b != 0 else (np.inf if med_a > 0 else np.nan)

    # Pair-preserving resample.
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    ratios = np.empty(n_resamples, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(n_resamples):
            r = idx[i]
            ma = float(np.median(a_arr[r]))
            mb = float(np.median(b_arr[r]))
            if mb == 0.0:
                ratios[i] = np.inf if ma > 0 else np.nan
            else:
                ratios[i] = ma / mb

    # Drop NaNs (0/0 case) before quantiling so the CI endpoints stay
    # interpretable.  inf values are kept -- a CI dominated by inf is
    # the truthful answer when the denominator collapses.
    finite_mask = ~np.isnan(ratios)
    surviving = ratios[finite_mask]
    if surviving.size == 0:
        lo = hi = float("nan")
    elif np.all(np.isinf(surviving)):
        # Every resample produced +inf (numerator finite, denominator
        # collapsed to zero across every draw).  np.quantile would
        # raise a RuntimeWarning on inf-inf interpolation; short-circuit
        # to the truthful answer.
        lo = hi = float("inf")
    else:
        alpha = (1.0 - ci) / 2.0
        lo = float(np.quantile(surviving, alpha))
        hi = float(np.quantile(surviving, 1.0 - alpha))

    return BootstrapCI(
        point_estimate=float(point),
        lo=lo,
        hi=hi,
        n_resamples=int(n_resamples),
        n_paired_seeds=int(n),
        rng_seed=int(rng_seed),
    )


def paired_wilcoxon(
    a: Sequence[float | int | None],
    b: Sequence[float | int | None],
    *,
    min_paired: int = 5,
) -> tuple[float, float, int] | None:
    """Paired Wilcoxon signed-rank test, returning ``None`` when underpowered.

    Drops pairs where either side is ``None`` / non-finite, then
    calls :func:`scipy.stats.wilcoxon`.  When fewer than
    ``min_paired`` pairs survive, returns ``None`` (the test would
    be uninformative).  When the surviving pairs are bit-identical,
    returns ``(0.0, 1.0, n_paired)`` -- scipy's wilcoxon raises in
    that degenerate case but the empirically meaningful answer is
    "no evidence of difference".

    Returns
    -------
    ``(statistic, p_value, n_paired)`` or ``None``.
    """
    if len(a) != len(b):
        raise ValueError(f"a and b length mismatch: {len(a)} vs {len(b)}")
    a_arr = _to_float_with_inf(a)
    b_arr = _to_float_with_inf(b)
    paired_mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    a_paired = a_arr[paired_mask]
    b_paired = b_arr[paired_mask]
    n = int(a_paired.size)
    if n < int(min_paired):
        return None
    if np.array_equal(a_paired, b_paired):
        return (0.0, 1.0, n)
    stat = stats.wilcoxon(a_paired, b_paired)
    return (float(stat.statistic), float(stat.pvalue), n)


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
