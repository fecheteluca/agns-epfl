"""Median residual + IQR vs iteration / wall-clock / oracle calls.

One renderer covers all three convergence-panel types; the abscissa
is selected via ``x_source``:

* ``"iteration"`` --- iteration index from ``iter_curve["x"]``;
* ``"time_curve"`` --- cumulative wall-clock from ``time_curve["median"]``;
* ``"grad_curve"`` --- cumulative gradient-oracle count from ``grad_curve["median"]``.

The ordinate is always the suboptimality ``f(x_k) - f^*``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from agns.plots._helpers import (
    build_method_styles,
    floor_log,
    legend_below,
    make_fig,
    markevery,
    save_pair,
)

__all__ = ["render"]


def _plot(
    ax: Any,
    methods: dict[str, dict[str, Any]],
    base_dir: Path,
    x_source: str,
    xlabel: str,
    title: str | None,
    *,
    show_iqr: bool = True,
    xscale: str = "log",
) -> None:
    """Draw median + IQR curves for every method."""
    # Resolve styles up front so sweep clones get distinct colour shades,
    # linestyles, and labels (``AGNS-WGN [gamma_0=0.01]``) rather than
    # collapsing onto the base-method style and producing a legend of
    # ``[AGNS, AGNS, AGNS, AGNS, AGNS]`` repeats.
    styles = build_method_styles(methods, base_dir)
    for method_key in sorted(methods):
        rec = methods[method_key]
        resid = rec.get("iter_curve")
        if not resid:
            continue
        style = styles[method_key]

        med = floor_log(np.asarray(resid["median"], dtype=float))
        lo = floor_log(np.asarray(resid["p25"], dtype=float))
        hi = floor_log(np.asarray(resid["p75"], dtype=float))

        if x_source == "iteration":
            x = np.asarray(resid["x"], dtype=float)
        else:
            cost = rec.get(x_source)
            if not cost:
                continue
            # cumulative cost lives in ``median``; ``x`` is the iteration index.
            x = np.asarray(cost["median"], dtype=float)

        # curves are index-aligned; guard against a ragged record.
        n = min(x.size, med.size, lo.size, hi.size)
        if n == 0:
            continue
        x, med, lo, hi = x[:n], med[:n], lo[:n], hi[:n]

        # a log abscissa cannot show a non-positive coordinate.
        if xscale == "log":
            keep = x > 0.0
            x, med, lo, hi = x[keep], med[keep], lo[keep], hi[keep]
            if x.size == 0:
                continue

        if show_iqr:
            ax.fill_between(
                x,
                lo,
                hi,
                color=style["color"],
                alpha=0.13,
                linewidth=0,
                zorder=style["zorder"] - 1,
            )
        ax.plot(
            x,
            med,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markevery=markevery(x.size),
            markersize=5.5,
            markeredgewidth=0.0,
            zorder=style["zorder"],
        )
    ax.set_yscale("log")
    if xscale == "log":
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$f(x_k) - f^{\star}$")
    if title:
        ax.set_title(title)


# (x_source, x-axis label, x-scale) per kind.
_KINDS: dict[str, tuple[str, str, str]] = {
    "iter_convergence": ("iteration", r"Iteration $k$", "log"),
    "time_convergence": ("time_curve", r"Wall-clock time (s)", "log"),
    "grad_convergence": ("grad_curve", r"Gradient oracle calls", "log"),
}


def render(
    kind: str,
    methods: dict[str, dict[str, Any]],
    out_dir: Path,
    *,
    base_methods_dir: Path,
    title: str | None = None,
    show_legend: bool = True,
    rc_override: dict[str, Any] | None = None,
) -> None:
    """Render a convergence panel of the requested ``kind``."""
    if kind not in _KINDS:
        raise ValueError(f"unknown convergence kind {kind!r}; expected one of {sorted(_KINDS)}")
    x_source, xlabel, xscale = _KINDS[kind]
    fig, ax = make_fig(rc_override)
    _plot(ax, methods, base_methods_dir, x_source, xlabel, title, xscale=xscale)
    if show_legend:
        legend_below(ax)
    save_pair(fig, out_dir, kind)
