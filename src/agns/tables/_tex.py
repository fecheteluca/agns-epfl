"""Shared LaTeX-rendering helpers for the table renderers.

Every table module under :mod:`agns.tables` consumes these helpers so
cells, escapes, and the booktabs scaffold are formatted identically
across the per-campaign, real-world-summary, restart-density, and
speedup tables.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

__all__ = [
    "CELL_STATE_LEGEND",
    "build_booktabs",
    "escape_label",
    "fmt_failed_cell",
    "fmt_not_run_cell",
    "fmt_resid",
    "is_sweep_variant",
    "method_cell_state",
    "pick_best_sweep_variant",
    "sweep_variants_of",
]

#: Caption fragment describing the three cell states for cross-campaign
#: tables.  Renderers append this to their caption so a reader can decode
#: ``failed`` / ``n/r`` cells without consulting another source.
CELL_STATE_LEGEND = (
    r"\textit{Cell key:} numeric cell = converged across-seed residual "
    r"(median [p25, p75]); \textit{failed (n/N)} = method ran but n of N "
    r"seeds raised (modal exception type in footnote); "
    r"\textsc{n/r} = method not configured for this dataset."
)


def fmt_failed_cell(n_failed: int, n_total: int) -> str:
    r"""Render a "failed (n/N)" cell.

    Italic so it stands out from numeric residual cells and from
    ``\textsc{n/r}``.  Includes seed counts so a reader can tell
    "one bad seed" from "every seed exploded".
    """
    return rf"\textit{{failed ({int(n_failed)}/{int(n_total)})}}"


def fmt_not_run_cell() -> str:
    r"""Render the "not run / not configured" cell.

    Small-caps ``n/r`` -- visually distinct from numeric cells and from
    the italic failure cell.  Used when an aggregated record for this
    (dataset, method-column) combination is unavailable.
    """
    return r"\textsc{n/r}"


def is_sweep_variant(key: str, base_key: str) -> bool:
    """Return ``True`` if ``key`` is a sweep clone of ``base_key``.

    Sweep expansion produces keys of the form ``<base>__<param>=<value>``
    via :func:`agns.pipeline.sweeps.expand_sweeps`.  This helper does
    the same recognition without importing the sweep module.
    """
    return key == base_key or key.startswith(base_key + "__")


def sweep_variants_of(
    records: dict[str, dict[str, Any]],
    base_key: str,
) -> list[str]:
    """Return every key in ``records`` that is a clone of ``base_key``.

    Includes the bare ``base_key`` itself if present.  Useful when a
    summariser wants to fold ``adam__lr=*`` rows back under "Adam"
    for a high-level rollup.
    """
    return sorted(k for k in records if is_sweep_variant(k, base_key))


def pick_best_sweep_variant(
    records: dict[str, dict[str, Any]],
    base_key: str,
) -> str | None:
    """Return the swept-variant key whose median final residual is smallest.

    When ``base_key`` has no clones in ``records`` the helper returns
    ``base_key`` if it exists (so a non-swept call site sees the
    expected behaviour); otherwise ``None``.  Tie-breaks by key name
    (alphabetical) for deterministic selection.

    Records without a usable median (failure-only records, or missing
    ``final_residual_summary``) are skipped.  If every variant is
    unusable, returns ``None``.
    """
    candidates = [
        k for k in records if is_sweep_variant(k, base_key) and k in records
    ]
    if not candidates:
        return None
    scored: list[tuple[float, str]] = []
    for k in candidates:
        rec = records[k]
        # Failure-only records carry final_residual_summary={median: inf, ...};
        # skip rather than rank them.
        if int(rec.get("n_seeds_succeeded", 0)) == 0:
            continue
        fr = rec.get("final_residual_summary") or {}
        med = fr.get("median")
        if med is None:
            continue
        try:
            med_f = float(med)
        except (TypeError, ValueError):
            continue
        if math.isnan(med_f):
            continue
        scored.append((med_f, k))
    if not scored:
        # Every candidate was unusable.  If ``base_key`` itself was a
        # legitimate (non-swept) record, fall back to it; otherwise None.
        return base_key if base_key in records else None
    scored.sort()  # ascending median, ties broken alphabetically by key
    return scored[0][1]


def method_cell_state(rec: dict[str, Any] | None) -> str:
    """Classify a method record as one of ``"converged"``, ``"failed"``, ``"not_run"``.

    - ``not_run`` if ``rec is None``.
    - ``failed`` if the record carries ``n_seeds_succeeded == 0``.
    - ``converged`` otherwise (at least one seed produced a usable history).

    Used by cross-campaign renderers to pick the right cell formatter.
    """
    if rec is None:
        return "not_run"
    if int(rec.get("n_seeds_succeeded", 0)) == 0:
        return "failed"
    return "converged"


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
