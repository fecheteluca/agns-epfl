from __future__ import annotations

from typing import Any

import matplotlib as mpl

_BLACK = "#000000"
_ORANGE = "#E69F00"
_SKY = "#56B4E9"
_GREEN = "#009E73"
_BLUE = "#0072B2"
_VERMILLION = "#D55E00"
_PURPLE = "#CC79A7"
_GREY = "#999999"

COL_WIDTH = 3.25  
TEXT_WIDTH = 6.75  

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
    """Remove the top and right spines (paper convention)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
