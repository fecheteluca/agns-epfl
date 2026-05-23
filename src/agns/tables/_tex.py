"""Shared LaTeX-rendering helpers for the table renderers.

Every table module under :mod:`agns.tables` consumes these helpers so
cells, escapes, and the booktabs scaffold are formatted identically
across the per-campaign, real-world-summary, restart-density, and
speedup tables.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["build_booktabs", "escape_label", "fmt_resid"]


_LATEX_ESCAPES = str.maketrans({"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"})


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


def escape_label(s: str) -> str:
    """LaTeX-escape the characters that would break a table cell.

    Covers ``_``, ``%``, ``&``, ``#``.  Does not touch ``$`` or ``\\``
    because user-supplied labels occasionally include math snippets
    (``$\\gamma_0$``) that we want to pass through verbatim.
    """
    return s.translate(_LATEX_ESCAPES)


def build_booktabs(
    colspec: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    caption: str = "",
    label: str = "",
) -> str:
    """Render a ``\\resizebox``-wrapped booktabs ``tabular`` block.

    Shared scaffold used by every table renderer in :mod:`agns.tables`.
    Callers are responsible for escaping cell contents with
    :func:`escape_label` if those cells may carry LaTeX-special
    characters.
    """
    parts: list[str] = []
    parts.append(r"\begin{table}[h]")
    parts.append(r"\centering")
    parts.append(r"\footnotesize")
    parts.append(r"\setlength{\tabcolsep}{4pt}")
    parts.append(r"\resizebox{\linewidth}{!}{%")
    parts.append("\\begin{tabular}{" + colspec + "}")
    parts.append(r"\toprule")
    parts.append(" & ".join(header) + r" \\")
    parts.append(r"\midrule")
    for row in rows:
        parts.append(" & ".join(row) + r" \\")
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}%")
    parts.append(r"}")
    if caption:
        parts.append(r"\caption{" + caption + r"}")
    if label:
        parts.append(r"\label{" + label + r"}")
    parts.append(r"\end{table}")
    return "\n".join(parts) + "\n"
