"""Per-campaign residual + slope summary table.

Renders one row per method in a single campaign's aggregated JSON.
Median / IQR of the final residual, success count out of seeds, and
the log-log slope fit on the trailing-half median trace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agns.tables._tex import escape_label, fmt_resid

__all__ = ["render"]


_RATE_NOTE = (
    r"Slope is the least-squares fit of $\log_{10}(f(x_k) - f^\star)$ vs "
    r"$\log_{10}(k)$ on the trailing half of the median trace."
)


def _fmt_slope(v: Any) -> str:
    if not isinstance(v, (int, float)) or v != v:  # missing or NaN
        return "--"
    # Suppress slopes outside the meaningful asymptotic-rate range
    # (|slope| > 10 = Q-superlinear endgame / floor noise; |slope| < 0.1
    # = stalled).  In either case the number is uninformative.
    if abs(v) > 10.0 or abs(v) < 0.1:
        return "--"
    return f"{v:+.2f}"


def _fmt_r2(v: Any) -> str:
    if not isinstance(v, (int, float)) or v != v:
        return "--"
    if not (0.0 <= v <= 1.0):
        return "--"
    return f"{v:.3f}"


def _build_table(
    methods: dict[str, dict[str, Any]],
    *,
    n_seeds: int,
    order: list[str] | None,
    caption: str,
    label: str,
    rate_note: str | None = None,
) -> str:
    """Emit a booktabs-style per-campaign table."""
    keys = order if order is not None else sorted(methods.keys())
    cells: list[tuple[str, str, str, str, str]] = []
    for key in keys:
        if key not in methods:
            continue
        rec = methods[key]
        lab = escape_label(rec.get("label") or key)
        fr = rec.get("final_residual_summary", {})
        med = fmt_resid(float(fr.get("median", float("inf"))))
        p25 = fmt_resid(float(fr.get("p25", float("inf"))))
        p75 = fmt_resid(float(fr.get("p75", float("inf"))))
        n_ok = int(rec.get("n_seeds_succeeded", 0))
        slope_val = rec.get("rate_fit", {}).get("slope")
        slope = _fmt_slope(slope_val)
        r2_val = rec.get("rate_fit", {}).get("r_squared")
        # R^2 is only meaningful when the slope passed the sanity filter.
        r2 = _fmt_r2(r2_val) if slope != "--" else "--"
        residual = rf"{med} [{p25}, {p75}]"
        cells.append((lab, residual, f"{n_ok}/{n_seeds}", slope, r2))

    has_rate = any(c[3] != "--" for c in cells)
    if has_rate:
        colspec = "l c c c c"
        header = r"Method & final residual (med [IQR]) & succeeded & slope & $R^2$ \\"
        rows = [
            f"        {lab} & {res} & {succ} & {sl} & {r2} \\\\"
            for lab, res, succ, sl, r2 in cells
        ]
    else:
        colspec = "l c c"
        header = r"Method & final residual (med [IQR]) & succeeded \\"
        rows = [
            f"        {lab} & {res} & {succ} \\\\"
            for lab, res, succ, _sl, _r2 in cells
        ]
        if rate_note and caption.endswith(rate_note):
            caption = caption[: -len(rate_note)].rstrip()

    body = "\n".join(rows) if rows else "        % no methods to render"
    return rf"""\begin{{table}}[t]
    \centering
    \caption{{{caption}}}
    \label{{{label}}}
    \footnotesize
    \setlength{{\tabcolsep}}{{4pt}}
    \resizebox{{\linewidth}}{{!}}{{%
    \begin{{tabular}}{{{colspec}}}
        \toprule
        {header}
        \midrule
{body}
        \bottomrule
    \end{{tabular}}}}
\end{{table}}
"""


def render(
    aggregated: Path,
    output: Path,
    *,
    order: list[str] | None = None,
    caption: str | None = None,
    label: str | None = None,
) -> None:
    """Read one aggregated JSON and write the per-campaign table to ``output``."""
    if not aggregated.is_file():
        raise FileNotFoundError(f"aggregated JSON not found: {aggregated}")
    with open(aggregated) as fh:
        agg = json.load(fh)
    methods: dict[str, dict[str, Any]] = agg.get("methods", {})
    if not methods:
        raise ValueError(f"no methods in aggregated JSON: {aggregated}")

    campaign = agg.get("campaign", "campaign")
    n_seeds = int(agg.get("n_seeds", 0))
    campaign_caption = campaign.replace("_", r"\_")
    rate_note = None if caption else _RATE_NOTE
    final_caption = caption or (
        f"Campaign {campaign_caption}: summary across {n_seeds} seeds. "
        f"Final-residual statistics use the per-seed $f^\\star$. "
        f"{_RATE_NOTE}"
    )
    final_label = label or f"tab:{campaign}"

    tex = _build_table(
        methods,
        n_seeds=n_seeds,
        order=order,
        caption=final_caption,
        label=final_label,
        rate_note=rate_note,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex)
    print(f"  [per-campaign-table] wrote {output}")
