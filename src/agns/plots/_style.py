"""Canonical per-method visual style.

Every plot module under :mod:`agns.plots` looks up its colour /
linestyle / marker / linewidth / zorder from :data:`STYLE_TABLE` and
its compact legend abbreviation from :data:`LEGEND_ABBREV`, so the same
method renders identically across figures.

Palette is colour-blind-safe (Wong / Tol).  Methods are grouped by
family so a reader sees AGNS variants in one colour, GNS in another,
Newton-type in a third, etc.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "LEGEND_ABBREV",
    "PAPER_RC",
    "RESID_FLOOR",
    "STYLE_TABLE",
    "SWEEP_XLABEL",
    "UNKNOWN_STYLE",
]


# Matplotlib rcParams shared by every figure.  Keep titles + legend
# tight so the figure fits in a single column without further tweaking
# downstream.
PAPER_RC: dict[str, Any] = {
    "figure.figsize": (6.0, 3.5),
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    # TeX is off by default: headless machines rarely have LaTeX installed.
    "text.usetex": False,
    "font.family": "serif",
    "axes.labelsize": 18,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
    "lines.linewidth": 2.4,
    "lines.markersize": 6,
    "axes.linewidth": 1.0,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.framealpha": 1.0,
    "legend.fancybox": False,
    "legend.borderpad": 0.20,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.35,
    "legend.labelspacing": 0.25,
    "legend.columnspacing": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


#: Floor a log-y curve at this value so an exact zero (machine precision)
#: does not blow up the rendering pipeline with ``log(0)`` warnings.
RESID_FLOOR: float = 1e-16


# (color, linestyle, marker, linewidth, zorder)
STYLE_TABLE: dict[str, tuple[str, str, str, float, int]] = {
    # AGNS (dark green).  Default restart=gradient; no-restart variants
    # share the colour but distinct linestyles + marker for the restart-
    # ablation panels.
    "agns_inexact": ("#117733", "-", "o", 2.6, 12),
    "agns_exact": ("#117733", "--", "s", 2.4, 11),
    "agns_wsm": ("#117733", ":", "D", 2.4, 11),
    "agns_exact_norestart": ("#117733", "-.", "v", 2.0, 9),
    "agns_inexact_norestart": ("#117733", "-.", "^", 2.0, 9),
    # GNS (wine).
    "gns_exact": ("#882255", "-", "X", 2.0, 10),
    "gns_inexact": ("#882255", "--", "p", 2.0, 10),
    "gns_wsm": ("#882255", ":", "h", 2.0, 10),
    # Newton-type (indigo).
    "newton": ("#332288", "-", "<", 1.8, 8),
    "super_newton": ("#332288", "--", ">", 1.8, 8),
    "cubic_newton": ("#332288", ":", "v", 1.8, 8),
    "trust_region": ("#332288", "-.", "*", 1.8, 8),
    # Accelerated-cubic family (olive).
    "accelerated_cubic_newton": ("#999933", "-", "d", 1.8, 6),
    "picard_acn_2008": ("#999933", "--", "H", 1.8, 6),
    # Quasi-Newton (sand).
    "lbfgs": ("#DDCC77", "-", "P", 1.8, 5),
    # First-order (light blue).
    "gradient": ("#88CCEE", "-", "+", 1.5, 3),
    "fast_gradient": ("#88CCEE", "--", "x", 1.5, 3),
    "heavy_ball": ("#88CCEE", ":", "*", 1.5, 3),
    # ML optimisers (mauve).
    "adam": ("#AA4499", "-", "*", 1.5, 2),
    "adahessian": ("#AA4499", "--", "X", 1.5, 2),
}


UNKNOWN_STYLE: tuple[str, str, str, float, int] = ("#444444", "-", "o", 1.5, 1)


# Compact legend labels.  Plots carry up to ~11 methods; the full names
# ("Super-Universal Newton", "Fast Gradient (Nesterov)", ...) make the
# legend wider than the axes.  These abbreviations match the
# tables/_tex.py method short-names so figures + tables agree.
LEGEND_ABBREV: dict[str, str] = {
    "agns_inexact": "AGNS-WGN",
    "agns_exact": "AGNS",
    "agns_wsm": "AGNS-WSM",
    "agns_exact_norestart": "AGNS-NOR",
    "agns_inexact_norestart": "AGNS-WGN-NOR",
    "gns_exact": "GNS",
    "gns_inexact": "GNS-WGN",
    "gns_wsm": "GNS-WSM",
    "newton": "Newton",
    "super_newton": "SUN",
    "cubic_newton": "Cubic-N",
    "accelerated_cubic_newton": "ACN",
    "picard_acn_2008": "Picard-ACN (N'08)",
    "trust_region": "Trust-reg",
    "lbfgs": "L-BFGS",
    "gradient": "GD",
    "fast_gradient": "FG",
    "heavy_ball": "Heavy-ball",
    "adam": "Adam",
    "adahessian": "AdaHessian",
}


#: x-axis labels for the sweep-axis plots, indexed by swept parameter name.
SWEEP_XLABEL: dict[str, str] = {
    "gamma_0": r"$\gamma_0$",
    "gamma0": r"$\gamma_0$",
    "momentum_offset": "Nesterov offset",
    "restart_mode": "restart rule",
    "inexact_rank": "Hessian rank",
}
