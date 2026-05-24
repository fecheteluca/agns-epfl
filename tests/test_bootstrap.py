"""Tests for the bootstrap CI + paired Wilcoxon helpers in metrics.py.

These helpers underpin the speedup table's CI/p_W columns; the
behavioural contracts here (pair-preservation, ``None`` handling,
RNG determinism) are load-bearing for the rendered table.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agns.utils.metrics import (
    BOOTSTRAP_DEFAULT_RNG_SEED,
    BootstrapCI,
    bootstrap_ratio_ci,
    paired_wilcoxon,
)

# ---------------------------------------------------------------------------
# bootstrap_ratio_ci
# ---------------------------------------------------------------------------


class TestBootstrapPointEstimate:
    def test_identical_inputs_have_point_estimate_one(self) -> None:
        ci = bootstrap_ratio_ci([5, 5, 5, 5, 5], [5, 5, 5, 5, 5], n_resamples=500)
        assert ci.point_estimate == pytest.approx(1.0)

    def test_double_inputs_have_point_estimate_two(self) -> None:
        ci = bootstrap_ratio_ci([10, 10, 10, 10, 10], [5, 5, 5, 5, 5], n_resamples=500)
        assert ci.point_estimate == pytest.approx(2.0)

    def test_halved_inputs_have_point_estimate_one_half(self) -> None:
        ci = bootstrap_ratio_ci([5, 5, 5, 5, 5], [10, 10, 10, 10, 10], n_resamples=500)
        assert ci.point_estimate == pytest.approx(0.5)


class TestBootstrapCICoverage:
    def test_identical_inputs_have_ci_tight_around_one(self) -> None:
        ci = bootstrap_ratio_ci([5, 5, 5, 5, 5], [5, 5, 5, 5, 5], n_resamples=2000)
        assert ci.lo == pytest.approx(1.0)
        assert ci.hi == pytest.approx(1.0)
        assert ci.covers_unity is True

    def test_clearly_different_medians_excludes_one(self) -> None:
        # With pair-preserving bootstrap and obviously different medians,
        # the CI should not bracket 1.0.
        a = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        b = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ci = bootstrap_ratio_ci(a, b, n_resamples=2000)
        assert ci.lo > 1.0
        assert ci.covers_unity is False

    def test_overlap_distribution_brackets_one(self) -> None:
        # Two arrays drawn from overlapping integer distributions:
        # CI should typically bracket 1.0.
        rng = np.random.default_rng(42)
        a = rng.integers(5, 11, size=20).tolist()
        b = rng.integers(5, 11, size=20).tolist()
        ci = bootstrap_ratio_ci(a, b, n_resamples=2000)
        assert ci.covers_unity is True


class TestBootstrapPairPreservation:
    def test_unpaired_shuffle_changes_ci(self) -> None:
        # Pair-preservation is load-bearing.  If the bootstrap were
        # unpaired (independent resampling of a and b), shuffling one
        # side would not change the CI.  Here we verify the opposite:
        # pairing matters, so swapping one side's order to break the
        # natural correlation visibly moves the CI.
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # b co-varies with a (perfect rank correlation).
        b = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ci_paired = bootstrap_ratio_ci(a, b, n_resamples=4000)
        # Break the pairing by reversing b.
        ci_broken = bootstrap_ratio_ci(a, list(reversed(b)), n_resamples=4000)
        # The point estimate is unchanged (median(a)/median(b) doesn't
        # care about pairing); the CI WIDTH does.
        assert ci_paired.point_estimate == pytest.approx(ci_broken.point_estimate)
        width_paired = ci_paired.hi - ci_paired.lo
        width_broken = ci_broken.hi - ci_broken.lo
        # The two widths should differ -- if they were equal, the
        # bootstrap is silently unpaired.  We allow them to differ in
        # either direction (sample-size dependent).
        assert width_paired != pytest.approx(width_broken, rel=1e-6)


class TestBootstrapDeterminism:
    def test_same_seed_gives_identical_endpoints(self) -> None:
        a, b = [3, 4, 5, 6, 7], [2, 3, 4, 5, 6]
        ci_a = bootstrap_ratio_ci(a, b, n_resamples=1000, rng_seed=42)
        ci_b = bootstrap_ratio_ci(a, b, n_resamples=1000, rng_seed=42)
        assert ci_a.lo == ci_b.lo
        assert ci_a.hi == ci_b.hi

    def test_different_seeds_give_different_endpoints(self) -> None:
        # On any non-degenerate input, two seeds should produce
        # measurably different bootstrap draws.  Use floats with a
        # wide spread so the resampled medians actually move across
        # different RNG draws (integer inputs with few distinct values
        # land on the same quantile bins regardless of seed).
        rng = np.random.default_rng(0)
        a = rng.uniform(1.0, 100.0, size=30).tolist()
        b = rng.uniform(1.0, 100.0, size=30).tolist()
        ci_a = bootstrap_ratio_ci(a, b, n_resamples=2000, rng_seed=1)
        ci_b = bootstrap_ratio_ci(a, b, n_resamples=2000, rng_seed=999)
        assert (ci_a.lo, ci_a.hi) != (ci_b.lo, ci_b.hi)

    def test_default_rng_seed_is_zero(self) -> None:
        # Documented invariant: the project-wide default seed is 0,
        # so a caller passing no seed gets the same numbers as a
        # caller passing rng_seed=0.
        assert BOOTSTRAP_DEFAULT_RNG_SEED == 0
        a, b = [1, 2, 3, 4, 5], [2, 2, 3, 4, 4]
        default = bootstrap_ratio_ci(a, b, n_resamples=500)
        explicit = bootstrap_ratio_ci(a, b, n_resamples=500, rng_seed=0)
        assert default.lo == explicit.lo
        assert default.hi == explicit.hi


class TestBootstrapNoneHandling:
    def test_none_in_denominator_collapses_to_zero_speedup(self) -> None:
        # When every b (AGNS) entry is None (never crossed eps), the
        # speedup = iters(GNS) / iters(AGNS) = finite / inf = 0.
        # Semantically: "AGNS infinitely slower than GNS on this campaign".
        ci = bootstrap_ratio_ci(
            [10, 10, 10, 10, 10],
            [None, None, None, None, None],
            n_resamples=500,
        )
        assert ci.point_estimate == 0.0
        # Every resample sees the same configuration, so the CI collapses.
        assert ci.lo == 0.0
        assert ci.hi == 0.0

    def test_none_in_numerator_collapses_to_infinity_speedup(self) -> None:
        # Symmetric: GNS never crosses eps but AGNS does -> speedup = inf.
        ci = bootstrap_ratio_ci(
            [None, None, None, None, None],
            [10, 10, 10, 10, 10],
            n_resamples=500,
        )
        assert math.isinf(ci.point_estimate)
        # Both endpoints inf (every resample has inf numerator).
        assert math.isinf(ci.lo)
        assert math.isinf(ci.hi)

    def test_both_sides_all_none_gives_nan(self) -> None:
        # Pathological inf/inf -> nan, which the renderer surfaces as ``--``.
        ci = bootstrap_ratio_ci(
            [None, None, None],
            [None, None, None],
            n_resamples=500,
        )
        assert math.isnan(ci.point_estimate)

    def test_mixed_none_partially_finite(self) -> None:
        # Half-None entries shouldn't crash; the helper propagates
        # them as +inf which affects medians.
        ci = bootstrap_ratio_ci(
            [5, 5, 5, 5, 5],
            [10, 10, 10, None, None],
            n_resamples=500,
        )
        # Point estimate uses full-array medians.  median([10,10,10,inf,inf]) == 10.
        assert ci.point_estimate == pytest.approx(0.5)


class TestBootstrapValidation:
    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            bootstrap_ratio_ci([1, 2], [1, 2, 3], n_resamples=100)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            bootstrap_ratio_ci([], [], n_resamples=100)

    def test_ci_outside_range_raises(self) -> None:
        with pytest.raises(ValueError, match="ci must lie in"):
            bootstrap_ratio_ci([1, 2], [1, 2], ci=1.5)
        with pytest.raises(ValueError, match="ci must lie in"):
            bootstrap_ratio_ci([1, 2], [1, 2], ci=0.0)

    def test_zero_resamples_raises(self) -> None:
        with pytest.raises(ValueError, match="n_resamples"):
            bootstrap_ratio_ci([1, 2], [1, 2], n_resamples=0)


class TestBootstrapCIDataclass:
    def test_covers_unity_lo_below_hi_above(self) -> None:
        ci = BootstrapCI(point_estimate=1.0, lo=0.5, hi=2.0, n_resamples=1, n_paired_seeds=1, rng_seed=0)
        assert ci.covers_unity is True

    def test_covers_unity_strictly_above(self) -> None:
        ci = BootstrapCI(point_estimate=2.0, lo=1.5, hi=2.5, n_resamples=1, n_paired_seeds=1, rng_seed=0)
        assert ci.covers_unity is False

    def test_covers_unity_strictly_below(self) -> None:
        ci = BootstrapCI(point_estimate=0.5, lo=0.3, hi=0.7, n_resamples=1, n_paired_seeds=1, rng_seed=0)
        assert ci.covers_unity is False

    def test_covers_unity_endpoint_exactly(self) -> None:
        # 1.0 sits exactly on the boundary -- count as covered.
        ci = BootstrapCI(point_estimate=1.0, lo=1.0, hi=1.5, n_resamples=1, n_paired_seeds=1, rng_seed=0)
        assert ci.covers_unity is True


# ---------------------------------------------------------------------------
# paired_wilcoxon
# ---------------------------------------------------------------------------


class TestPairedWilcoxon:
    def test_identical_inputs_returns_p_one(self) -> None:
        result = paired_wilcoxon([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert result is not None
        stat, p, n = result
        assert stat == 0.0
        assert p == 1.0
        assert n == 5

    def test_clear_difference_returns_small_p(self) -> None:
        # AGNS consistently faster than GNS across 10 seeds.
        gns = [20, 21, 19, 22, 18, 21, 20, 19, 22, 20]
        agns = [10, 11, 9, 12, 8, 11, 10, 9, 12, 10]
        result = paired_wilcoxon(gns, agns)
        assert result is not None
        _, p, n = result
        assert p < 0.01
        assert n == 10

    def test_underpowered_returns_none(self) -> None:
        # Only 4 paired seeds -- below the default min_paired=5.
        result = paired_wilcoxon([1, 2, 3, 4], [2, 3, 4, 5])
        assert result is None

    def test_explicit_min_paired_threshold(self) -> None:
        # Same 4-seed input, but caller lowers the threshold.
        result = paired_wilcoxon([1, 2, 3, 4], [2, 3, 4, 5], min_paired=4)
        assert result is not None

    def test_none_entries_dropped_before_test(self) -> None:
        # 8 entries total, 3 contain None -> 5 paired survive, just
        # enough to meet min_paired.
        result = paired_wilcoxon(
            [10, 11, 12, 13, 14, 15, None, None],
            [5, 6, 7, 8, 9, None, 20, 21],
        )
        assert result is not None
        _, _, n = result
        assert n == 5

    def test_too_many_nones_returns_none(self) -> None:
        result = paired_wilcoxon(
            [10, 11, None, None, None],
            [5, 6, 7, 8, 9],
        )
        assert result is None

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            paired_wilcoxon([1, 2], [1, 2, 3])
