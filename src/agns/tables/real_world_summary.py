"""Cross-dataset rollup table over the real-world LIBSVM campaigns.

One row per dataset, comparing a small fixed set of methods (AGNS,
ACN, L-BFGS).  Complements :mod:`agns.tables.per_campaign`, which is
the per-method view for a single dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agns.tables._tex import build_booktabs, fmt_resid

__all__ = ["render"]

# Campaign -> (display dataset name, problem-family abbreviation).
# Row order in the rendered table follows this list.  Datasets added by
# the pipeline expansion (real-sim, gisette, madelon, splice, housing,
# bodyfat, mg, space_ga, mpg) are listed after the original nine so a
# diff against the historical real_world_summary stays minimal.
REAL_WORLD_CAMPAIGNS: list[tuple[str, str, str]] = [
    ("real_lr_a9a", "a9a", "LR"),
    ("real_lr_w8a", "w8a", "LR"),
    ("real_lr_phishing", "phishing", "LR"),
    ("real_lr_covtype", "covtype", "LR"),
    ("real_lr_ijcnn1", "ijcnn1", "LR"),
    ("real_lr_rcv1", "rcv1", "LR"),
    ("real_svm_mushrooms", "mushrooms", "SVM"),
    ("real_ridge_cadata", "cadata", "Ridge"),
    ("real_ridge_abalone", "abalone", "Ridge"),
    ("real_lr_real_sim", "real-sim", "LR"),
    ("real_lr_gisette", "gisette", "LR"),
    ("real_lr_madelon", "madelon", "LR"),
    ("real_lr_splice", "splice", "LR"),
    ("real_ridge_housing", "housing", "Ridge"),
    ("real_ridge_bodyfat", "bodyfat", "Ridge"),
    ("real_ridge_mg", "mg", "Ridge"),
    ("real_ridge_space_ga", "space\\_ga", "Ridge"),
    ("real_ridge_mpg", "mpg", "Ridge"),
]

# Methods shown as columns: registry key + display name.
SUMMARY_METHODS: list[tuple[str, str]] = [
    ("agns_inexact", "AGNS"),
    ("accelerated_cubic_newton", "ACN"),
    ("lbfgs", "L-BFGS"),
]


def _resid_cell(rec: dict[str, Any]) -> str:
    fr = rec.get("final_residual_summary", {})
    med = fmt_resid(float(fr.get("median", float("inf"))))
    p25 = fmt_resid(float(fr.get("p25", float("inf"))))
    p75 = fmt_resid(float(fr.get("p75", float("inf"))))
    return rf"{med} [{p25}, {p75}]"


def _build_table(rows: list[dict[str, Any]], *, label: str) -> str:
    n_methods = len(SUMMARY_METHODS)
    colspec = "l l " + ("c " * n_methods).rstrip()
    header = ["Dataset", "Problem"] + [disp for _, disp in SUMMARY_METHODS]
    body_rows = [
        [row["dataset"], row["problem"]] + row["method_cells"] for row in rows
    ] or [["% no datasets"] + [""] * (len(header) - 1)]
    caption = (
        "Real-world {\\sc libsvm} benchmarks: across-seed final residual "
        "$f(x_k) - f^\\star$ reported as median [p25, p75]."
    )
    return build_booktabs(colspec, header, body_rows, caption=caption, label=label)


def render(
    aggregated_dir: Path,
    output: Path,
    *,
    label: str = "tab:real-world-summary",
) -> None:
    """Read every real-world aggregated JSON and write the rollup table."""
    if not aggregated_dir.is_dir():
        raise FileNotFoundError(f"aggregated directory not found: {aggregated_dir}")

    rows: list[dict[str, Any]] = []
    for campaign, dataset, problem in REAL_WORLD_CAMPAIGNS:
        agg_path = aggregated_dir / f"{campaign}.json"
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
        raise RuntimeError("No real-world aggregated JSONs found; nothing to render.")

    tex = _build_table(rows, label=label)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex)
    print(f"  [real-world-summary] wrote {output}")
