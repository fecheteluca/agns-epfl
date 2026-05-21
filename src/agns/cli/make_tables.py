"""Render a LaTeX summary table from an aggregated JSON.

Produces a single ``<campaign>.tex`` with the canonical summary table:

+-------------+-------------+-----------+--------+----------+
| Method      | residual (med [IQR])    | seeds  | slope    |
+=============+=========================+========+==========+
| AGNS-Pract. | 1.2e-09 [8.0e-10, ...]  |  9/10  | -3.04    |
| ...         |                         |        |          |
+-------------+-------------+-----------+--------+----------+

Median/IQR are printed with two significant digits, slope and ``R^2``
with two decimals, success count as ``<succeeded>/<n_seeds>``.  Methods
are listed alphabetically; override with ``--order``.

Usage::

    python -m agns.cli.make_tables \\
        --aggregated results/numerical/aggregated/rate_verification.json \\
        --output results/tables/rate_verification.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agns.cli._tex import fmt_resid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--aggregated", required=True, help="Path to <campaign>.json")
    p.add_argument("--output", required=True, help="Path to write the .tex file.")
    p.add_argument(
        "--order",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="Method order in the output table (default: alphabetical).",
    )
    p.add_argument(
        "--caption",
        default=None,
        help="Override the table caption.",
    )
    p.add_argument(
        "--label",
        default=None,
        help="Override the LaTeX label (default: tab:<campaign>).",
    )
    return p.parse_args()


def _fmt_slope(v: Any) -> str:
    if not isinstance(v, (int, float)) or v != v:  # missing or NaN
        return "--"
    # Suppress slopes outside the meaningful asymptotic-rate range.
    # |slope| > 10 typically indicates the Q-superlinear endgame or
    # near-floor noise (cf. _markevery filter in make_figures.py) and
    # would mislead a reader who interprets the number as the asymptotic
    # rate exponent.  |slope| < 0.1 is the "method stalled" regime where
    # the residual is essentially constant and a slope number is also
    # uninformative.
    if abs(v) > 10.0 or abs(v) < 0.1:
        return "--"
    return f"{v:+.2f}"


def _fmt_r2(v: Any) -> str:
    if not isinstance(v, (int, float)) or v != v:
        return "--"
    if not (0.0 <= v <= 1.0):
        return "--"
    return f"{v:.3f}"


def _escape_label(s: str) -> str:
    """LaTeX-escape only the characters that would break a table cell."""
    return s.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


_RATE_NOTE = (
    r"Slope is the least-squares fit of $\log_{10}(f(x_k) - f^\star)$ vs "
    r"$\log_{10}(k)$ on the trailing half of the median trace."
)


def _build_table(
    methods: dict[str, dict[str, Any]],
    *,
    n_seeds: int,
    order: list[str] | None,
    caption: str,
    label: str,
    rate_note: str | None = None,
) -> str:
    """Emit a booktabs-style summary table.

    Layout:

    * ``booktabs`` rules (``\\toprule`` / ``\\midrule`` / ``\\bottomrule``),
      no vertical rules.
    * ``\\footnotesize`` body.
    * ``\\resizebox{\\linewidth}{!}{...}`` wrapper to prevent overflow.
    * Combined ``residual (median [IQR])`` column.
    * Slope / ``R^2`` are suppressed (``--``) when the slope sits
      outside the meaningful asymptotic-rate range (see :func:`_fmt_slope`).
    * The slope / ``R^2`` *columns* are dropped entirely when no method in
      the campaign yields a meaningful fit -- a gradient-normalised campaign
      typically reaches machine precision in a near-vertical cliff, so every
      entry would read ``--`` and the columns would carry no information.
      When that happens ``rate_note`` (if given) is also omitted from the
      caption so the caption does not describe an absent column.
    """
    keys = order if order is not None else sorted(methods.keys())
    cells: list[tuple[str, str, str, str, str]] = []
    for key in keys:
        if key not in methods:
            continue
        rec = methods[key]
        lab = _escape_label(rec.get("label") or key)
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
        # Compact median [IQR] cell: ``\num{1.2e-12} [\num{5.5e-13},
        # \num{2.4e-12}]`` fits inside one column instead of three.
        residual = rf"{med} [{p25}, {p75}]"
        cells.append((lab, residual, f"{n_ok}/{n_seeds}", slope, r2))

    has_rate = any(c[3] != "--" for c in cells)

    if has_rate:
        colspec = "l c c c c"
        header = r"Method & final residual (med [IQR]) & succeeded & slope & $R^2$ \\"
        rows = [
            f"        {lab} & {res} & {succ} & {sl} & {r2} \\\\" for lab, res, succ, sl, r2 in cells
        ]
    else:
        colspec = "l c c"
        header = r"Method & final residual (med [IQR]) & succeeded \\"
        rows = [f"        {lab} & {res} & {succ} \\\\" for lab, res, succ, _sl, _r2 in cells]
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


def main() -> None:
    args = parse_args()
    agg_path = Path(args.aggregated).resolve()
    if not agg_path.is_file():
        sys.exit(f"aggregated JSON not found: {agg_path}")

    with open(agg_path) as fh:
        agg = json.load(fh)

    methods: dict[str, dict[str, Any]] = agg.get("methods", {})
    if not methods:
        sys.exit("no methods in aggregated JSON")

    campaign = agg.get("campaign", "campaign")
    n_seeds = int(agg.get("n_seeds", 0))
    # The campaign string flows into a LaTeX caption (where ``_`` is
    # subscript) and a label (where ``_`` is fine).  Escape only for
    # the caption.
    campaign_caption = campaign.replace("_", r"\_")
    # The caption ends with _RATE_NOTE; _build_table strips that sentence
    # again when the campaign has no meaningful slope column to describe.
    rate_note = None if args.caption else _RATE_NOTE
    caption = args.caption or (
        f"Campaign {campaign_caption}: summary across {n_seeds} seeds. "
        f"Final-residual statistics use the per-seed $f^\\star$. "
        f"{_RATE_NOTE}"
    )
    label = args.label or f"tab:{campaign}"

    tex = _build_table(
        methods,
        n_seeds=n_seeds,
        order=args.order,
        caption=caption,
        label=label,
        rate_note=rate_note,
    )

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"  [table] {out}")


if __name__ == "__main__":
    main()
