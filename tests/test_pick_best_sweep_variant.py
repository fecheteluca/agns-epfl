"""Tests for the ``pick_best_sweep_variant`` / sweep-recognition helpers.

These helpers let downstream summarisers fold ``<base>__<param>=<value>``
sweep clones back under the base method's headline row.  The
behavioural contract pinned here: smallest median final residual
wins, ties break alphabetically, unusable records are skipped, and
the no-sweep case still works.
"""

from __future__ import annotations

from typing import Any

import pytest

from agns.tables._tex import (
    is_sweep_variant,
    pick_best_sweep_variant,
    sweep_variants_of,
)


def _rec(median: float | None, n_succeeded: int = 5) -> dict[str, Any]:
    fr: dict[str, Any] = {"median": median, "p25": median, "p75": median}
    return {
        "method": "x",
        "label": "x",
        "n_seeds_succeeded": n_succeeded,
        "final_residual_summary": fr,
    }


# ---------------------------------------------------------------------------
# is_sweep_variant
# ---------------------------------------------------------------------------


class TestIsSweepVariant:
    def test_base_key_itself_is_variant(self) -> None:
        assert is_sweep_variant("adam", "adam") is True

    def test_swept_key_is_variant(self) -> None:
        assert is_sweep_variant("adam__lr=1e-3", "adam") is True

    def test_unrelated_key_is_not_variant(self) -> None:
        assert is_sweep_variant("adahessian", "adam") is False

    def test_prefix_collision_is_not_variant(self) -> None:
        # ``adam_v2`` is not a sweep clone of ``adam`` -- the convention
        # requires the ``__param=value`` separator after the base.
        assert is_sweep_variant("adam_v2", "adam") is False


# ---------------------------------------------------------------------------
# sweep_variants_of
# ---------------------------------------------------------------------------


def test_sweep_variants_of_returns_all_matching_keys() -> None:
    records = {
        "adam": _rec(1.0),
        "adam__lr=1e-4": _rec(0.5),
        "adam__lr=1e-3": _rec(0.3),
        "adahessian": _rec(0.9),
        "lbfgs": _rec(0.1),
    }
    assert sweep_variants_of(records, "adam") == [
        "adam",
        "adam__lr=1e-3",
        "adam__lr=1e-4",
    ]


def test_sweep_variants_of_empty_when_no_match() -> None:
    records = {"lbfgs": _rec(0.1), "newton": _rec(0.2)}
    assert sweep_variants_of(records, "adam") == []


# ---------------------------------------------------------------------------
# pick_best_sweep_variant
# ---------------------------------------------------------------------------


def test_picks_variant_with_smallest_median() -> None:
    records = {
        "adam__lr=1e-4": _rec(0.5),
        "adam__lr=1e-3": _rec(0.3),
        "adam__lr=1e-2": _rec(0.4),
    }
    assert pick_best_sweep_variant(records, "adam") == "adam__lr=1e-3"


def test_returns_base_key_when_no_sweep_variants_exist() -> None:
    """Safe to call on non-swept records."""
    records = {"adam": _rec(0.5), "lbfgs": _rec(0.1)}
    assert pick_best_sweep_variant(records, "adam") == "adam"


def test_returns_none_when_base_absent_and_no_variants() -> None:
    records = {"lbfgs": _rec(0.1)}
    assert pick_best_sweep_variant(records, "adam") is None


def test_ties_break_alphabetically_for_determinism() -> None:
    records = {
        "adam__lr=1e-3": _rec(0.5),
        "adam__lr=1e-2": _rec(0.5),
        "adam__lr=1e-4": _rec(0.5),
    }
    # Three-way tie; alphabetically smallest key wins.
    assert pick_best_sweep_variant(records, "adam") == "adam__lr=1e-2"


def test_failure_only_variants_are_skipped() -> None:
    """A failure-only record (n_seeds_succeeded=0) is unrankable."""
    records = {
        "adam__lr=1e-4": {"n_seeds_succeeded": 0, "final_residual_summary": {"median": 0.0}},
        "adam__lr=1e-3": _rec(0.5),
    }
    assert pick_best_sweep_variant(records, "adam") == "adam__lr=1e-3"


def test_records_with_no_median_are_skipped() -> None:
    """Malformed records (missing final_residual_summary.median) are skipped."""
    records = {
        "adam__lr=1e-4": {"n_seeds_succeeded": 1, "final_residual_summary": {}},
        "adam__lr=1e-3": _rec(0.5),
    }
    assert pick_best_sweep_variant(records, "adam") == "adam__lr=1e-3"


def test_records_with_nan_median_are_skipped() -> None:
    """NaN median sorts inconsistently; skip rather than rank."""
    records = {
        "adam__lr=1e-4": _rec(float("nan")),
        "adam__lr=1e-3": _rec(0.5),
    }
    assert pick_best_sweep_variant(records, "adam") == "adam__lr=1e-3"


def test_all_variants_unusable_falls_back_to_base() -> None:
    records = {
        "adam": _rec(0.5),  # the un-swept base record
        "adam__lr=1e-4": {"n_seeds_succeeded": 0, "final_residual_summary": {"median": 0.0}},
        "adam__lr=1e-3": _rec(float("nan")),
    }
    # All sweep variants unusable; base is usable -> base wins
    # (it gets ranked among the candidates because it matches sweep_variants_of).
    assert pick_best_sweep_variant(records, "adam") == "adam"


@pytest.mark.parametrize("base", ["adam", "adahessian"])
def test_helper_distinguishes_overlapping_base_keys(base: str) -> None:
    records = {
        "adam__lr=1e-3": _rec(0.1),
        "adahessian__lr=1e-3": _rec(0.2),
    }
    chosen = pick_best_sweep_variant(records, base)
    assert chosen is not None and chosen.startswith(base + "__")
