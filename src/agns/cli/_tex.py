"""Shared LaTeX-rendering helpers for the table-building CLIs.

Used by :mod:`agns.cli.make_tables` and :mod:`agns.cli.make_summary` so
the two renderers format residual cells identically.
"""

from __future__ import annotations

import math

__all__ = ["fmt_resid"]


def fmt_resid(v: float) -> str:
    """Format a residual as a ``siunitx`` ``\\num{}`` table cell.

    A NaN renders as ``\\textsc{nan}`` and an infinity as ``$\\infty$``
    so a diverged or missing entry stays visible in the typeset table.
    """
    if math.isnan(v):
        return r"\textsc{nan}"
    if math.isinf(v):
        return r"$\infty$"
    return f"\\num{{{v:.2e}}}"
