"""Log-log residual vs iteration with the fitted-slope annotation.

Each method's median trace is drawn on log-log axes; if the campaign
declares a Hölder smoothness exponent ``nu_ref`` (passed in from the
runner), the panel also overlays the theoretical ``C / K^{2+nu}``
reference line anchored at the first non-floor AGNS point.
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
    short_label,
)

__all__ = ["render"]


def _plot(
    ax: Any,
    methods: dict[str, dict[str, Any]],
    base_dir: Path,
    title: str | None,
    nu_ref: float | None,
) -> None:
    anchor: tuple[float, float] | None = None
    slope_lines: list[tuple[str, str, float]] = []
    # Resolve styles up front so sweep clones get distinct colour shades.
    styles = build_method_styles(methods, base_dir)
    for method_key in sorted(methods):
        rec = methods[method_key]
        curve = rec.get("iter_curve")
        fit = rec.get("rate_fit", {})
        if not curve:
            continue
        style = styles[method_key]
        x = np.asarray(curve["x"], dtype=float)
        med = floor_log(np.asarray(curve["median"], dtype=float))
        slope = fit.get("slope")
        r2 = fit.get("r_squared")
        slope_is_meaningful = (
            slope is not None
            and np.isfinite(slope)
            and r2 is not None
            and r2 > 0.9
            and abs(slope) > 0.1
            and abs(slope) < 10
        )
        if slope_is_meaningful:
            slope_lines.append((style["label"], style["color"], float(slope)))
        ax.loglog(
            x[1:],
            med[1:],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markevery=markevery(len(x) - 1),
            markersize=5.5,
            markeredgewidth=0.0,
            zorder=style["zorder"],
        )
        if anchor is None and method_key.startswith("agns") and len(x) > 2:
            for k_idx in range(1, len(med)):
                if med[k_idx] > 1e-12 and np.isfinite(med[k_idx]):
                    anchor = (float(x[k_idx]), float(med[k_idx]))
                    break

    if nu_ref is not None and anchor is not None:
        K_anchor, f_anchor = anchor
        exponent = 2.0 + float(nu_ref)
        C = f_anchor * (K_anchor**exponent)
        K_max = max(
            (
                np.asarray(methods[m]["iter_curve"]["x"], dtype=float).max()
                for m in methods
                if methods[m].get("iter_curve")
            ),
            default=K_anchor,
        )
        K_ref = np.geomspace(K_anchor, K_max, 64)
        f_ref = C / K_ref**exponent
        ax.loglog(
            K_ref,
            f_ref,
            color="0.4",
            linestyle=":",
            linewidth=1.2,
            label=rf"theory: $C/K^{{{exponent:.2f}}}$",
        )

    if slope_lines:
        slope_lines.sort(key=lambda t: t[2])
        for i, (lbl, col, sl) in enumerate(slope_lines[:6]):
            ax.text(
                0.02,
                0.04 + 0.05 * (len(slope_lines[:6]) - 1 - i),
                rf"{short_label(lbl)}: slope ${sl:+.2f}$",
                transform=ax.transAxes,
                fontsize=10,
                color=col,
                ha="left",
                va="bottom",
                zorder=20,
            )

    ax.set_xlabel(r"Iteration $k$")
    ax.set_ylabel(r"$f(x_k) - f^{\star}$")
    if title:
        ax.set_title(title)


def render(
    methods: dict[str, dict[str, Any]],
    out_dir: Path,
    *,
    base_methods_dir: Path,
    title: str | None = None,
    nu_ref: float | None = None,
    show_legend: bool = True,
    rc_override: dict[str, Any] | None = None,
) -> None:
    """Render the log-log rate panel."""
    fig, ax = make_fig(rc_override)
    _plot(ax, methods, base_methods_dir, title, nu_ref)
    if show_legend:
        legend_below(ax)
    save_pair(fig, out_dir, "rate_loglog")
