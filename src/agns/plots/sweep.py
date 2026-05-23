"""Sweep ablation panel: final residual vs a swept hyperparameter.

For each swept value we draw all per-seed final residuals as a
translucent point cloud and overlay the median as a connected line.
Reference methods that are not part of any sweep (e.g.\\ a
``gns_exact`` baseline included alongside the sweep) appear as dashed
horizontal lines for comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from agns.plots._helpers import (
    legend_below,
    make_fig,
    parse_sweep_key,
    save_pair,
    short_label,
    style_for,
    sweep_xscale,
)
from agns.plots._style import RESID_FLOOR, SWEEP_XLABEL

__all__ = ["has_sweep_methods", "render"]


def has_sweep_methods(methods: dict[str, dict[str, Any]]) -> bool:
    """Return True iff at least one method key has a sweep-style suffix."""
    return any(parse_sweep_key(k) is not None for k in methods)


def _plot(
    ax: Any,
    methods: dict[str, dict[str, Any]],
    base_dir: Path,
    title: str | None,
) -> None:
    sweep_groups: dict[tuple[str, str], list[tuple[Any, dict[str, Any]]]] = {}
    references: list[tuple[str, dict[str, Any]]] = []
    for method_key in sorted(methods):
        rec = methods[method_key]
        parsed = parse_sweep_key(method_key)
        if parsed is None:
            references.append((method_key, rec))
            continue
        base, param, raw_value = parsed
        try:
            value: float | str = float(raw_value.replace("p", "."))
        except ValueError:
            value = raw_value
        sweep_groups.setdefault((base, param), []).append((value, rec))

    if not sweep_groups:
        return

    xlabel = ""
    xscale = "linear"
    for (base, param), items in sweep_groups.items():
        numeric = all(isinstance(v, (int, float)) for v, _ in items)
        if numeric:
            items.sort(key=lambda kv: kv[0])
            values: list[Any] = [float(v) for v, _ in items]
            xs: list[float] = list(values)
            xscale = sweep_xscale([float(v) for v in values])
        else:
            values = [str(v) for v, _ in items]
            xs = [float(i) for i in range(len(values))]  # categorical positions

        style = style_for(base, items[0][1], base_dir)
        color = style["color"]

        seed_x: list[float] = []
        seed_y: list[float] = []
        med_y: list[float] = []
        for x, (_, rec) in zip(xs, items, strict=True):
            for r in rec.get("final_residuals") or []:
                if r is not None and np.isfinite(r):
                    seed_x.append(x)
                    seed_y.append(max(float(r), RESID_FLOOR))
            m = rec.get("final_residual_summary", {}).get("median")
            med_y.append(
                max(float(m), RESID_FLOOR)
                if isinstance(m, (int, float)) and np.isfinite(m)
                else RESID_FLOOR
            )

        if seed_x:
            ax.scatter(
                seed_x,
                seed_y,
                s=34,
                color=color,
                alpha=0.38,
                edgecolors="none",
                zorder=4,
            )
        ax.plot(
            xs,
            med_y,
            color=color,
            linestyle="-",
            linewidth=1.9,
            marker=style["marker"],
            markersize=8.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=6,
            label=short_label(style["label"]),
        )
        if not numeric:
            ax.set_xticks(xs)
            ax.set_xticklabels(values)
        xlabel = SWEEP_XLABEL.get(param, param.replace("_", " "))

    for ref_key, rec in references:
        style = style_for(ref_key, rec, base_dir)
        y = rec.get("final_residual_summary", {}).get("median")
        if isinstance(y, (int, float)) and np.isfinite(y):
            ax.axhline(
                max(float(y), RESID_FLOOR),
                color=style["color"],
                linestyle="--",
                linewidth=1.6,
                alpha=0.85,
                zorder=5,
                label=f"{style['label']} (ref.)",
            )

    ax.set_yscale("log")
    if xscale == "log":
        ax.set_xscale("log")
    ax.margins(x=0.08)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"final $f(x_K) - f^{\star}$")
    if title:
        ax.set_title(title)


def render(
    methods: dict[str, dict[str, Any]],
    out_dir: Path,
    *,
    base_methods_dir: Path,
    title: str | None = None,
    show_legend: bool = True,
    rc_override: dict[str, Any] | None = None,
) -> None:
    """Render the sweep ablation panel."""
    fig, ax = make_fig(rc_override)
    _plot(ax, methods, base_methods_dir, title)
    if show_legend:
        legend_below(ax)
    save_pair(fig, out_dir, "sweep")
