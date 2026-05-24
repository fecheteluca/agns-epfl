"""Renderer-integration tests for the ``equalise_x_axis`` mode.

The unit-level helper is covered in ``test_common_x_range.py``;
here we verify the end-to-end behaviour: when the convergence
renderer is invoked with ``equalise_x_axis=True``, the produced
matplotlib axis is clipped to the helper's reported range and the
plot title carries an explicit marker so the constraint is visible
to a reader.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agns.plots import convergence
from agns.plots._helpers import common_x_range_across_methods


def _method(iter_len: int) -> dict:
    """Minimal method record carrying an iter_curve of the given length.

    Residual decays so the log-y axis renders without clipping; the
    actual numerics don't matter for the test.
    """
    return {
        "iter_curve": {
            "x": list(range(iter_len)),
            "median": [10.0 / (k + 1) for k in range(iter_len)],
            "p25": [10.0 / (k + 1) for k in range(iter_len)],
            "p75": [10.0 / (k + 1) for k in range(iter_len)],
        }
    }


def test_equalise_x_axis_clips_to_common_range(tmp_path: Path) -> None:
    """With unequal budgets, the equalised render clips to the smallest max."""
    methods = {
        "newton": _method(iter_len=150),
        "gradient": _method(iter_len=1500),
        "adam": _method(iter_len=2000),
    }

    # Pre-compute what the helper says the common range should be.
    rng = common_x_range_across_methods(methods, "iteration")
    assert rng is not None
    expected_lo, expected_hi = rng

    # Render with equalisation on; inspect the live figure rather than
    # parsing the saved PDF/PNG.  We monkeypatch make_fig to expose the
    # axis to us... actually simpler: render, then grab the live ax via
    # plt.gca() before save_pair closes the figure.  Since save_pair
    # closes the figure, we replicate the renderer's call path here.
    fig, ax = plt.subplots()
    try:
        convergence._plot(
            ax,
            methods,
            base_dir=tmp_path,
            x_source="iteration",
            xlabel="iter",
            title="t",
            xscale="log",
            equalise_x_axis=True,
        )
        x_lo, x_hi = ax.get_xlim()
        assert x_hi == expected_hi
        # The helper's lower bound is 0 (iter index 0).  The renderer
        # clamps non-positive lower bounds to 1.0 on log axes so
        # matplotlib does not warn; verify the clamp fired.
        assert expected_lo == 0.0
        assert x_lo == 1.0
        # Title carries the "equal x-axis: [...]" marker.
        title_text = ax.get_title()
        assert "equal x-axis" in title_text
        assert "149" in title_text  # the actual common upper bound
    finally:
        plt.close(fig)


def test_equalise_x_axis_off_preserves_full_range(tmp_path: Path) -> None:
    """Default (off) leaves matplotlib auto-axis -- the union of method ranges."""
    methods = {
        "newton": _method(iter_len=150),
        "gradient": _method(iter_len=1500),
    }
    fig, ax = plt.subplots()
    try:
        convergence._plot(
            ax,
            methods,
            base_dir=tmp_path,
            x_source="iteration",
            xlabel="iter",
            title="t",
            xscale="log",
            equalise_x_axis=False,
        )
        _x_lo, x_hi = ax.get_xlim()
        # Auto-axis extends to the max of method ranges (with a small
        # matplotlib-internal padding factor).  We just assert the upper
        # bound is well past Newton's 149.
        assert x_hi > 149.0
        # Title is the unmodified caller-supplied string.
        assert ax.get_title() == "t"
    finally:
        plt.close(fig)


def test_equalise_x_axis_noop_when_common_range_unavailable(tmp_path: Path) -> None:
    """When the helper returns None, the renderer leaves the axis unclipped."""
    # Failure-only methods => no iter_curve anywhere.
    methods = {
        "failed_a": {"n_seeds_succeeded": 0},
        "failed_b": {"n_seeds_succeeded": 0},
    }
    fig, ax = plt.subplots()
    try:
        convergence._plot(
            ax,
            methods,
            base_dir=tmp_path,
            x_source="iteration",
            xlabel="iter",
            title="t",
            xscale="log",
            equalise_x_axis=True,
        )
        # No equal-axis marker -> the title stayed as-is.
        assert ax.get_title() == "t"
    finally:
        plt.close(fig)


def test_render_writes_pdf_and_png_with_equalise(tmp_path: Path) -> None:
    """End-to-end: ``convergence.render(...)`` with equalise produces both files."""
    methods = {
        "newton": _method(iter_len=20),
        "gradient": _method(iter_len=200),
    }
    out = tmp_path / "out"
    convergence.render(
        "iter_convergence",
        methods,
        out,
        base_methods_dir=tmp_path,
        title="demo",
        show_legend=False,
        equalise_x_axis=True,
    )
    assert (out / "iter_convergence.pdf").is_file()
    assert (out / "iter_convergence.png").is_file()


def test_render_rejects_unknown_kind(tmp_path: Path) -> None:
    """Defensive: an unknown kind raises rather than silently doing nothing."""
    import pytest

    with pytest.raises(ValueError, match="unknown convergence kind"):
        convergence.render(
            "unknown_kind",
            {"x": _method(iter_len=10)},
            tmp_path,
            base_methods_dir=tmp_path,
        )


def test_equalise_x_axis_caps_on_common_upper_bound_only(tmp_path: Path) -> None:
    """The clip is to the common range -- methods over-budget get their tails cropped.

    Newton ends at iter 9; gradient ends at iter 199.  With
    equalisation on, the abscissa upper bound is 9, so gradient's
    iter 10..199 should not appear (they're outside the visible
    range).  We assert this by checking that at least one line on
    the axis has its rendered x-data clipped at the limit.
    """
    methods = {
        "newton": _method(iter_len=10),
        "gradient": _method(iter_len=200),
    }
    fig, ax = plt.subplots()
    try:
        convergence._plot(
            ax,
            methods,
            base_dir=tmp_path,
            x_source="iteration",
            xlabel="iter",
            title=None,
            xscale="log",
            equalise_x_axis=True,
        )
        _, x_hi = ax.get_xlim()
        # Upper bound is the common max = 9 (Newton's last index).
        assert x_hi == 9.0

        # The data on the gradient line still contains the full 199-point
        # series (we don't subset the data, we just clip the axis).  But
        # the visible portion is bounded by ax.get_xlim().  Verify both
        # facts.
        lines = [ln for ln in ax.get_lines()]
        # At least one line still has its x-data extending beyond the limit
        # (proves we clipped the axis, not the data).
        assert any(np.asarray(ln.get_xdata()).max() > 9.0 for ln in lines)
    finally:
        plt.close(fig)
