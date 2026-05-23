"""Cross-campaign scaling figure: time-to-eps vs problem size n.

Reads several aggregated JSONs (one per ``n``) and overlays the median
wall-clock time required for each method to reach
``f - f^* <= 1e-8`` against the problem size.  Methods with fewer than
two campaigns are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

from agns.plots._helpers import legend_below, make_fig, resolve_style, save_pair
from agns.plots._style import LEGEND_ABBREV

__all__ = ["render"]


def render(agg_paths: list[Path], out_dir: Path, *, target: float = 1e-8) -> None:
    """Render the scaling figure into ``out_dir``."""
    points: dict[str, list[tuple[int, float]]] = {}
    for path in agg_paths:
        agg = json.loads(path.read_text())
        n = int(agg["problem"]["params"]["n"])
        for key, rec in agg.get("methods", {}).items():
            resid = rec.get("iter_curve", {}).get("median", [])
            times = rec.get("time_curve", {}).get("median", [])
            for i, v in enumerate(resid):
                if v <= target and i < len(times):
                    t = float(times[i])
                    if t > 0.0:
                        points.setdefault(key, []).append((n, t))
                    break

    fig, ax = make_fig({"figure.figsize": (6.0, 3.7)})
    for key in sorted(points):
        series = sorted(points[key])
        if len(series) < 2:
            continue
        ns = [n for n, _ in series]
        ts = [t for _, t in series]
        color, linestyle, marker, lw, z = resolve_style(key)
        ax.loglog(
            ns,
            ts,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=lw,
            markersize=7,
            markeredgewidth=0.0,
            zorder=z,
            label=LEGEND_ABBREV.get(key, key),
        )
    ax.set_xlabel(r"Problem size $n$")
    ax.set_ylabel(rf"time to $f-f^{{\star}}\leq {target:.0e}$ (s)")
    ticks = sorted({n for s in points.values() for n, _ in s})
    if ticks:
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    legend_below(ax)
    save_pair(fig, out_dir, "scaling_time")
