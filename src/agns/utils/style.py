"""Single source of truth for figure styling.

Every figure in the project draws its colours, line styles and fonts from here, so the same
method looks identical in every panel of every notebook. The palette is the
colourblind-safe Okabe-Ito set; AGNS is drawn prominently (saturated vermillion, thick, high
z-order) and the baselines are muted. Line *styles* are chosen so the methods remain
distinguishable in grayscale, and the approximate-Hessian AGNS variants share AGNS's colour as
a dashed line so "the approximation tracks the exact method" reads straight off the figure.

Fonts and sizes target a two-column ICLR/ICML LaTeX layout (serif body, Computer-Modern math),
and figures export as vector PDF with embedded TrueType text.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl

# Okabe-Ito colourblind-safe palette.
_BLACK = "#000000"
_ORANGE = "#E69F00"
_SKY = "#56B4E9"
_GREEN = "#009E73"
_BLUE = "#0072B2"
_VERMILLION = "#D55E00"
_PURPLE = "#CC79A7"
_GREY = "#999999"

# Column geometry for a two-column ICLR/ICML layout (inches).
COL_WIDTH = 3.25  # one text column -> compact "main paper" figures
TEXT_WIDTH = 6.75  # full text width -> multi-panel appendix figures

# Canonical method -> style. AGNS prominent (thick, high zorder); baselines muted.
METHOD_STYLE: dict[str, dict[str, Any]] = {
    "AGNS (exact)": {
        "color": _VERMILLION,
        "linestyle": "-",
        "linewidth": 2.4,
        "zorder": 6,
    },
    "AGNS (no restart)": {
        "color": _ORANGE,
        "linestyle": "--",
        "linewidth": 1.8,
        "zorder": 4,
    },
    "AGNS (function-value restart)": {
        "color": _PURPLE,
        "linestyle": "-.",
        "linewidth": 1.8,
        "zorder": 4,
    },
    "GNS": {"color": _BLUE, "linestyle": "-", "linewidth": 1.6, "zorder": 3},
    "Super-Universal Newton": {
        "color": _GREEN,
        "linestyle": "-.",
        "linewidth": 1.6,
        "zorder": 3,
    },
    "Cubic Newton": {"color": _SKY, "linestyle": "-.", "linewidth": 1.4, "zorder": 2},
    "Gradient Method": {
        "color": _GREY,
        "linestyle": ":",
        "linewidth": 1.4,
        "zorder": 2,
    },
    "Fast Gradient (Nesterov)": {
        "color": _BLACK,
        "linestyle": "--",
        "linewidth": 1.4,
        "zorder": 2,
    },
}

# Style shared by all approximate-Hessian AGNS variants: AGNS's colour, dashed (so the
# approximate curve visibly "tracks" the solid exact-AGNS curve).
_AGNS_APPROX_STYLE = {
    "color": _VERMILLION,
    "linestyle": "--",
    "linewidth": 2.0,
    "zorder": 5,
}
_DEFAULT_STYLE = {"color": _GREY, "linestyle": "-", "linewidth": 1.5, "zorder": 1}


def style_for(label: str) -> dict[str, Any]:
    """Return the fixed plot style for a method display label.

    Exact labels are looked up in :data:`METHOD_STYLE`; any other ``AGNS (...)`` label (the
    dynamically-named approximate-Hessian variants) gets the shared dashed-AGNS style; anything
    else falls back to a muted grey default.
    """
    if label in METHOD_STYLE:
        return dict(METHOD_STYLE[label])
    if label.startswith("AGNS"):
        return dict(_AGNS_APPROX_STYLE)
    return dict(_DEFAULT_STYLE)


def set_paper_style() -> None:
    """Apply the project-wide Matplotlib rcParams for paper-ready figures.

    Serif body font with Computer-Modern math, sizes legible at one-column width, thin axes,
    no global grid (panels add a light one), and vector-PDF export with embedded TrueType text.
    """
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def despine(ax: Any) -> None:
    """Remove the top and right spines and add a light grid (paper convention)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
