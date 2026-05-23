"""Pure-string tests for :mod:`agns.tables._tex`."""

from __future__ import annotations

import math

import pytest

from agns.tables._tex import build_booktabs, escape_label, fmt_resid


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
