"""Tests for :func:`agns.plots._helpers.common_x_range_across_methods`.

The helper is consumed by the convergence renderer when
``--equalise-x-axis`` is on: it returns the widest contiguous
abscissa range every plotted method covers, so a reader comparing
methods with different iter/cost budgets sees the "honest
comparable view".
"""

from __future__ import annotations

import pytest

from agns.plots._helpers import common_x_range_across_methods


def _record(
    *,
    iter_x: list[float] | None = None,
    time_median: list[float] | None = None,
    grad_median: list[float] | None = None,
) -> dict:
    """Build a minimal method record with whichever curves the test needs."""
    rec: dict = {}
    if iter_x is not None:
        rec["iter_curve"] = {"x": iter_x, "median": [1.0] * len(iter_x)}
    if time_median is not None:
        rec["time_curve"] = {"median": time_median, "x": list(range(len(time_median)))}
    if grad_median is not None:
        rec["grad_curve"] = {"median": grad_median, "x": list(range(len(grad_median)))}
    return rec


# ---------------------------------------------------------------------------
# Iter axis
# ---------------------------------------------------------------------------


class TestIterAxis:
    def test_returns_widest_common_range_on_iter_axis(self) -> None:
        """``x_max`` is the min of per-method maxima; ``x_min`` is the max of mins."""
        methods = {
            "newton": _record(iter_x=list(range(150))),         # 0..149
            "gradient": _record(iter_x=list(range(1500))),      # 0..1499
            "adam": _record(iter_x=list(range(2000))),          # 0..1999
        }
        rng = common_x_range_across_methods(methods, "iteration")
        assert rng is not None
        x_lo, x_hi = rng
        assert x_lo == 0.0
        # Smallest budget caps the upper bound: Newton's 149.
        assert x_hi == 149.0

    def test_returns_none_when_no_methods_have_iter_curve(self) -> None:
        methods = {"x": {}, "y": {}}
        assert common_x_range_across_methods(methods, "iteration") is None

    def test_skips_methods_with_missing_iter_curve(self) -> None:
        """Failure-only records (no iter_curve) are silently skipped.

        With the failure-only record skipped, the helper finds two
        usable methods (iter ranges 0..99 and 0..49) and returns the
        common upper bound 49.
        """
        methods = {
            "good_a": _record(iter_x=list(range(100))),
            "good_b": _record(iter_x=list(range(50))),
            # Failure-only record: no iter_curve.
            "failed": {"n_seeds_succeeded": 0, "failures": []},
        }
        rng = common_x_range_across_methods(methods, "iteration")
        assert rng is not None
        assert rng[1] == 49.0

    def test_skips_methods_with_empty_iter_curve(self) -> None:
        methods = {
            "good": _record(iter_x=list(range(10))),
            "empty": _record(iter_x=[]),
        }
        rng = common_x_range_across_methods(methods, "iteration")
        assert rng is not None
        assert rng == (0.0, 9.0)

    def test_returns_none_when_common_range_is_a_single_point(self) -> None:
        """If x_min == x_max, the common range is degenerate -> None.

        Clipping to a single point would produce an empty curve;
        the renderer would then plot nothing, which is worse than
        keeping the auto-axis.
        """
        methods = {
            "a": _record(iter_x=[5.0]),
            "b": _record(iter_x=[5.0]),
        }
        assert common_x_range_across_methods(methods, "iteration") is None


# ---------------------------------------------------------------------------
# Time and grad axes
# ---------------------------------------------------------------------------


class TestTimeAxis:
    def test_uses_time_curve_median_as_abscissa(self) -> None:
        methods = {
            "fast": _record(
                iter_x=list(range(10)),
                time_median=[0.001 * i for i in range(10)],   # 0..0.009 s
            ),
            "slow": _record(
                iter_x=list(range(10)),
                time_median=[0.1 * i for i in range(10)],     # 0..0.9 s
            ),
        }
        rng = common_x_range_across_methods(methods, "time_curve")
        assert rng is not None
        # Common upper bound = min(0.009, 0.9) ~= 0.009 (float-tolerant).
        assert rng[1] == pytest.approx(0.009)

    def test_skips_methods_without_time_curve(self) -> None:
        methods = {
            "no_time": _record(iter_x=[0, 1, 2]),                       # no time_curve
            "with_time": _record(iter_x=[0, 1, 2], time_median=[0.1, 0.2, 0.3]),
        }
        rng = common_x_range_across_methods(methods, "time_curve")
        # Only ``with_time`` contributes; range is its full extent.
        assert rng == (0.1, 0.3)


class TestGradAxis:
    def test_uses_grad_curve_median_as_abscissa(self) -> None:
        methods = {
            "agns": _record(iter_x=list(range(10)), grad_median=[5 * i for i in range(10)]),
            "gns": _record(iter_x=list(range(10)), grad_median=[1 * i for i in range(10)]),
        }
        rng = common_x_range_across_methods(methods, "grad_curve")
        assert rng is not None
        # min(45, 9) = 9.
        assert rng[1] == 9.0


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_methods_dict_returns_none() -> None:
    assert common_x_range_across_methods({}, "iteration") is None


def test_all_methods_unusable_returns_none() -> None:
    methods = {
        "no_curves": {},
        "failure_only": {"n_seeds_succeeded": 0},
    }
    assert common_x_range_across_methods(methods, "iteration") is None
