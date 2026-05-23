"""Pure-function tests for :mod:`agns.plots._helpers`.

We do not exercise matplotlib rendering here (the end-to-end pipeline
test covers that via ``tests/cli/test_pipeline.py``); the unit tests
target the small functional helpers that have no side effects.
"""

from __future__ import annotations

import numpy as np
import pytest

from agns.plots._helpers import (
    floor_log,
    markevery,
    parse_sweep_key,
    resolve_style,
    short_label,
    sweep_xscale,
)
from agns.plots._style import RESID_FLOOR, STYLE_TABLE, UNKNOWN_STYLE


class TestFloorLog:
    def test_clips_below_floor(self) -> None:
        arr = np.array([1e-30, 1e-12, 1.0])
        out = floor_log(arr)
        assert out[0] == RESID_FLOOR
        assert out[1] == 1e-12
        assert out[2] == 1.0

    def test_preserves_positive_values_above_floor(self) -> None:
        arr = np.array([RESID_FLOOR * 2, 1.0])
        out = floor_log(arr)
        np.testing.assert_array_equal(out, arr)


class TestShortLabel:
    def test_strips_sweep_bracket_suffix(self) -> None:
        assert short_label("AGNS [gamma_0=0.1]") == "AGNS"

    def test_label_without_bracket_passes_through(self) -> None:
        assert short_label("AGNS") == "AGNS"

    def test_only_first_bracket_is_stripped(self) -> None:
        assert short_label("AGNS [a] [b]") == "AGNS"


class TestMarkevery:
    def test_returns_none_for_short_traces(self) -> None:
        assert markevery(5, target=8) is None
        assert markevery(8, target=8) is None

    def test_picks_step_for_long_traces(self) -> None:
        assert markevery(80, target=8) == 10
        assert markevery(81, target=8) == 10

    def test_never_returns_zero(self) -> None:
        assert markevery(9, target=8) == 1


class TestResolveStyle:
    def test_base_method_resolves_directly(self) -> None:
        style = resolve_style("agns_exact")
        assert style == STYLE_TABLE["agns_exact"]

    def test_sweep_clone_resolves_to_base(self) -> None:
        # ``agns_inexact__gamma_0=0p1`` should share style with ``agns_inexact``.
        style = resolve_style("agns_inexact__gamma_0=0p1")
        assert style == STYLE_TABLE["agns_inexact"]

    def test_unknown_key_returns_default(self) -> None:
        assert resolve_style("does_not_exist") == UNKNOWN_STYLE


class TestParseSweepKey:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("agns_inexact__gamma_0=0p1", ("agns_inexact", "gamma_0", "0p1")),
            ("agns__restart_mode=gradient", ("agns", "restart_mode", "gradient")),
        ],
    )
    def test_well_formed_sweep_keys(self, key: str, expected: tuple) -> None:
        assert parse_sweep_key(key) == expected

    def test_non_sweep_key_returns_none(self) -> None:
        assert parse_sweep_key("agns_exact") is None
        assert parse_sweep_key("gns_wsm") is None

    def test_malformed_suffix_returns_none(self) -> None:
        assert parse_sweep_key("agns__no_equals") is None
        assert parse_sweep_key("agns__=value") is None  # empty param
        assert parse_sweep_key("agns__param=") is None  # empty value


class TestSweepXscale:
    def test_log_when_values_span_wide_positive_range(self) -> None:
        assert sweep_xscale([0.01, 0.1, 1.0, 10.0, 100.0]) == "log"

    def test_linear_when_range_is_small(self) -> None:
        assert sweep_xscale([1.0, 2.0, 3.0, 4.0]) == "linear"

    def test_linear_when_zero_present(self) -> None:
        # log axis requires strictly positive values
        assert sweep_xscale([0.0, 1.0, 100.0]) == "linear"

    def test_linear_when_negative_present(self) -> None:
        assert sweep_xscale([-1.0, 1.0, 100.0]) == "linear"

    def test_linear_for_empty(self) -> None:
        assert sweep_xscale([]) == "linear"
