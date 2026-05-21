"""Cross-dataset rollup table over the real-world LIBSVM campaigns.

:mod:`agns.cli.make_tables` emits one ``<campaign>.tex`` per campaign:
a per-method table for a single dataset.  This script renders the
orthogonal view --- one row per *dataset*, comparing a small fixed set
of methods --- so a reader can compare the headline method against the
strongest baselines across every dataset at a glance.

Layout mirrors :mod:`agns.cli.make_tables` (booktabs rules,
``\\footnotesize`` body, ``\\resizebox`` wrapper, ``\\num{}`` cells)
for visual consistency.  Each method cell is the across-seed median
``[p25, p75]`` of the final residual ``f(x_k) - f^star``.

Usage::

    python -m agns.cli.make_summary \\
        --aggregated-dir results/numerical/aggregated \\
        --output results/tables/real_world_summary.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agns.cli._tex import fmt_resid

__all__ = ["main"]

# Campaign -> (display dataset name, problem-family abbreviation).  Row
# order in the rendered table follows this list.
REAL_WORLD_CAMPAIGNS: list[tuple[str, str, str]] = [
    ("real_lr_a9a", "a9a", "LR"),
    ("real_lr_w8a", "w8a", "LR"),
    ("real_lr_phishing", "phishing", "LR"),
    ("real_lr_covtype", "covtype", "LR"),
    ("real_svm_mushrooms", "mushrooms", "SVM"),
    ("real_ridge_cadata", "cadata", "Ridge"),
]

# Methods shown as columns: registry key + display name.
SUMMARY_METHODS: list[tuple[str, str]] = [
    ("agns_practice", "AGNS-Pract."),
    ("accelerated_cubic_newton", "ACN"),
    ("lbfgs", "L-BFGS"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--aggregated-dir",
        required=True,
        help="Directory holding the <campaign>.json aggregated files.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to write the summary .tex file.",
    )
    p.add_argument(
        "--label",
        default="tab:real-world-summary",
        help="LaTeX label.",
    )
    return p.parse_args()


def _resid_cell(rec: dict[str, Any]) -> str:
    fr = rec.get("final_residual_summary", {})
    med = fmt_resid(float(fr.get("median", float("inf"))))
    p25 = fmt_resid(float(fr.get("p25", float("inf"))))
    p75 = fmt_resid(float(fr.get("p75", float("inf"))))
    return rf"{med} [{p25}, {p75}]"


def _build_table(rows: list[dict[str, Any]], *, label: str) -> str:
    n_methods = len(SUMMARY_METHODS)
    colspec = "l l " + "c " * n_methods
    header_cells = ["Dataset", "Problem"] + [disp for _, disp in SUMMARY_METHODS]
    header = " & ".join(header_cells) + r" \\"

    body_lines: list[str] = []
    for row in rows:
        cells = [row["dataset"], row["problem"]] + row["method_cells"]
        body_lines.append("        " + " & ".join(cells) + r" \\")
    body = "\n".join(body_lines) if body_lines else "        % no datasets"

    caption = (
        "Real-world {\\sc libsvm} benchmarks: across-seed final residual "
        "$f(x_k) - f^\\star$ reported as median [p25, p75]."
    )

    return rf"""\begin{{table}}[t]
    \centering
    \caption{{{caption}}}
    \label{{{label}}}
    \footnotesize
    \setlength{{\tabcolsep}}{{4pt}}
    \resizebox{{\linewidth}}{{!}}{{%
    \begin{{tabular}}{{{colspec.strip()}}}
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
    agg_dir = Path(args.aggregated_dir).resolve()
    if not agg_dir.is_dir():
        sys.exit(f"aggregated directory not found: {agg_dir}")

    rows: list[dict[str, Any]] = []
    for campaign, dataset, problem in REAL_WORLD_CAMPAIGNS:
        agg_path = agg_dir / f"{campaign}.json"
        if not agg_path.is_file():
            print(f"  [summary] WARN: missing aggregated JSON for {campaign}: {agg_path}")
            continue
        with open(agg_path) as fh:
            agg = json.load(fh)
        methods: dict[str, dict[str, Any]] = agg.get("methods", {})
        method_cells: list[str] = []
        for key, _disp in SUMMARY_METHODS:
            if key not in methods:
                method_cells.append("--")
            else:
                method_cells.append(_resid_cell(methods[key]))
        rows.append(
            {
                "dataset": rf"{{\sc {dataset}}}",
                "problem": problem,
                "method_cells": method_cells,
            }
        )

    if not rows:
        sys.exit("No real-world aggregated JSONs found; nothing to render.")

    tex = _build_table(rows, label=args.label)
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"  [table] {out}")


if __name__ == "__main__":
    main()
