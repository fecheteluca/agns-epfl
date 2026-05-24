"""Pure-string tests for :mod:`agns.tables._tex`."""

from __future__ import annotations

import math

import pytest

from agns.tables._tex import (
    CELL_STATE_LEGEND,
    build_booktabs,
    escape_label,
    fmt_failed_cell,
    fmt_not_run_cell,
    fmt_resid,
    method_cell_state,
)


class TestFmtResid:
    def test_finite_value_renders_with_two_sig_figures(self) -> None:
        assert fmt_resid(1.234e-9) == r"\num{1.23e-09}"

    def test_zero_is_finite(self) -> None:
        assert fmt_resid(0.0) == r"\num{0.00e+00}"

    def test_nan_renders_textsc_nan(self) -> None:
        assert fmt_resid(math.nan) == r"\textsc{nan}"

    def test_positive_infinity_renders_inline_math(self) -> None:
        assert fmt_resid(math.inf) == r"$\infty$"

    def test_negative_infinity_renders_inline_math(self) -> None:
        assert fmt_resid(-math.inf) == r"$\infty$"


class TestEscapeLabel:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("foo_bar", r"foo\_bar"),
            ("50% off", r"50\% off"),
            ("a&b", r"a\&b"),
            ("#tag", r"\#tag"),
            ("plain", "plain"),
            ("agns_inexact 50% & #1", r"agns\_inexact 50\% \& \#1"),
        ],
    )
    def test_escapes_only_table_special_chars(self, raw: str, expected: str) -> None:
        assert escape_label(raw) == expected

    def test_dollar_and_backslash_pass_through(self) -> None:
        # Math snippets like ``$\gamma_0$`` are intentionally not escaped
        # so callers can embed inline math in cell labels.
        assert escape_label("$x$") == "$x$"
        assert escape_label(r"\alpha") == r"\alpha"


class TestBuildBooktabs:
    def test_minimal_table_has_all_rule_lines(self) -> None:
        tex = build_booktabs(
            "ll",
            ["A", "B"],
            [["1", "2"], ["3", "4"]],
        )
        for token in (
            r"\begin{table}",
            r"\toprule",
            r"\midrule",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ):
            assert token in tex
        assert r"A & B \\" in tex
        assert r"1 & 2 \\" in tex
        assert r"3 & 4 \\" in tex

    def test_caption_and_label_render_when_provided(self) -> None:
        tex = build_booktabs(
            "l",
            ["X"],
            [["v"]],
            caption="my caption",
            label="tab:my-label",
        )
        assert r"\caption{my caption}" in tex
        assert r"\label{tab:my-label}" in tex

    def test_caption_and_label_omitted_by_default(self) -> None:
        tex = build_booktabs("l", ["X"], [["v"]])
        assert r"\caption" not in tex
        assert r"\label" not in tex

    def test_empty_rows_produces_no_data_rows(self) -> None:
        tex = build_booktabs("l", ["X"], [])
        # No body row between \midrule and \bottomrule.
        assert tex.count(r"\\") == 1  # only the header row carries `\\`

    def test_colspec_is_preserved(self) -> None:
        tex = build_booktabs("lrcr", ["A", "B", "C", "D"], [])
        assert r"\begin{tabular}{lrcr}" in tex


# ---------------------------------------------------------------------------
# 3-state cell helpers
# ---------------------------------------------------------------------------


class TestMethodCellState:
    def test_none_record_is_not_run(self) -> None:
        assert method_cell_state(None) == "not_run"

    def test_zero_success_is_failed(self) -> None:
        assert method_cell_state({"n_seeds_succeeded": 0, "n_seeds_failed": 5}) == "failed"

    def test_at_least_one_success_is_converged(self) -> None:
        assert method_cell_state({"n_seeds_succeeded": 1, "n_seeds_failed": 4}) == "converged"

    def test_missing_n_seeds_succeeded_defaults_to_failed(self) -> None:
        # A record with no n_seeds_succeeded field is degenerate; treat
        # it as a failure (the alternative -- treating "missing" as
        # "converged" -- would silently render bogus numeric cells).
        assert method_cell_state({}) == "failed"


class TestFmtFailedCell:
    def test_renders_with_seed_counts(self) -> None:
        assert fmt_failed_cell(3, 5) == r"\textit{failed (3/5)}"

    def test_zero_failed_still_renders(self) -> None:
        # Degenerate but tolerated; the renderer should not crash on
        # weird counts from a corrupted aggregated JSON.
        assert fmt_failed_cell(0, 0) == r"\textit{failed (0/0)}"


class TestFmtNotRunCell:
    def test_is_textsc_n_slash_r(self) -> None:
        # The exact spelling is load-bearing -- the legend in
        # CELL_STATE_LEGEND references \textsc{n/r} verbatim.
        assert fmt_not_run_cell() == r"\textsc{n/r}"


class TestCellStateLegend:
    def test_mentions_all_three_states(self) -> None:
        # The legend must explain every cell shape the renderer can
        # produce; otherwise a reader looking at \textsc{n/r} or
        # \textit{failed (...)} has no decoder.
        assert "n/r" in CELL_STATE_LEGEND
        assert "failed" in CELL_STATE_LEGEND
        assert "median" in CELL_STATE_LEGEND
