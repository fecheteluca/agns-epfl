"""Cross-dataset rollup table over the real-world LIBSVM campaigns.

One row per dataset, comparing a small fixed set of methods (AGNS,
ACN, L-BFGS).  Complements :mod:`agns.tables.per_campaign`, which is
the per-method view for a single dataset.

Cell encoding:

  - ``converged`` -> numeric median [p25, p75]
  - ``failed``    -> ``\\textit{failed (n/N)}``
  - ``not_run``   -> ``\\textsc{n/r}``

A single method column may map to *several* registry keys via an
ordered fallback list.  E.g. AGNS picks ``agns_inexact`` when present
(LR / logsumexp / nonlinear-equations campaigns), and falls back to
``agns_exact`` (ridge campaigns, where the factory provides no
``approx_hess_fn``).  This is why the column label ("AGNS") and the
registry key may differ on a per-row basis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agns.tables._tex import (
    CELL_STATE_LEGEND,
    build_booktabs,
    fmt_failed_cell,
    fmt_not_run_cell,
    fmt_resid,
    method_cell_state,
)

__all__ = ["render"]

# Campaign -> (display dataset name, problem-family abbreviation).
# Row order in the rendered table follows this list.
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

# Each method column: (display name, [registry keys in fallback order]).
# The first key with a record in the aggregated JSON wins.  AGNS gets
# the inexact -> exact fallback because ridge campaigns only configure
# the exact variant (ridge has no approx_hess_fn).
SUMMARY_METHODS: list[tuple[str, list[str]]] = [
    ("AGNS", ["agns_inexact", "agns_exact"]),
    ("ACN", ["accelerated_cubic_newton"]),
    ("L-BFGS", ["lbfgs"]),
]


def _resid_cell(rec: dict[str, Any]) -> str:
    """Render the standard converged-residual cell (median [p25, p75])."""
    fr = rec.get("final_residual_summary", {})
    med = fmt_resid(float(fr.get("median", float("inf"))))
    p25 = fmt_resid(float(fr.get("p25", float("inf"))))
    p75 = fmt_resid(float(fr.get("p75", float("inf"))))
    return rf"{med} [{p25}, {p75}]"


def _resolve_method_record(
    methods: dict[str, dict[str, Any]],
    keys: list[str],
) -> dict[str, Any] | None:
    """Pick the first key in ``keys`` whose record exists in ``methods``.

    Returns ``None`` when no key resolves (the "not_run" / "not
    configured" case).
    """
    for key in keys:
        rec = methods.get(key)
        if rec is not None:
            return rec
    return None


def _cell_for(rec: dict[str, Any] | None, n_seeds: int) -> str:
    """Format a single cell from a method record, dispatching on state."""
    state = method_cell_state(rec)
    if state == "not_run":
        return fmt_not_run_cell()
    if state == "failed":
        assert rec is not None  # narrowed by state == "failed"
        return fmt_failed_cell(int(rec.get("n_seeds_failed", n_seeds)), n_seeds)
    assert rec is not None
    return _resid_cell(rec)


def _modal_exception_footnote(
    rows: list[dict[str, Any]],
) -> str:
    """Summarise the modal exception type per (dataset, method) failure cell.

    Empty string when no failure cells were rendered.  Appended to the
    caption so the reader can decode ``failed (n/N)`` cells without
    opening the aggregated JSONs.
    """
    fail_lines: list[str] = []
    for row in rows:
        for col_disp, exc_type in row.get("failure_types", {}).items():
            if not exc_type:
                continue
            fail_lines.append(
                f"{row['dataset']}/{col_disp}: \\texttt{{{exc_type}}}"
            )
    if not fail_lines:
        return ""
    return "  \\textit{Failure types:} " + "; ".join(fail_lines) + "."


def _build_table(rows: list[dict[str, Any]], *, label: str) -> str:
    n_methods = len(SUMMARY_METHODS)
    colspec = "l l " + ("c " * n_methods).rstrip()
    header = ["Dataset", "Problem"] + [disp for disp, _ in SUMMARY_METHODS]
    body_rows = [
        [row["dataset"], row["problem"], *row["method_cells"]] for row in rows
    ] or [["% no datasets", *[""] * (len(header) - 1)]]
    caption = (
        "Real-world {\\sc libsvm} benchmarks: across-seed final residual "
        "$f(x_k) - f^\\star$ reported as median [p25, p75].  "
        "\\textit{Metric note:} AGNS regularises with the problem-supplied "
        "$B$ (e.g. $X^\\top X / m + \\mathrm{reg}\\,I$ for LR / ridge); "
        "L-BFGS sees no $B$, and ACN uses $B$ only inside its cubic "
        "subproblem.  See README \\S``Metric ($B$) asymmetry''.  "
        + CELL_STATE_LEGEND
        + _modal_exception_footnote(rows)
    )
    return build_booktabs(colspec, header, body_rows, caption=caption, label=label)


def render(
    aggregated_dir: Path,
    output: Path,
    *,
    label: str = "tab:real-world-summary",
) -> None:
    """Read every real-world aggregated JSON and write the rollup table.

    A dataset whose aggregated JSON is missing entirely contributes a
    row of ``n/r`` cells (rather than being silently skipped) so the
    reader can distinguish "not yet run" from "ran but failed".  A
    method whose record is absent from the JSON (e.g. AGNS on a
    first-order-only campaign) also renders ``n/r``.
    """
    if not aggregated_dir.is_dir():
        raise FileNotFoundError(f"aggregated directory not found: {aggregated_dir}")

    # Detect the "user pointed at the wrong directory" case up front
    # (zero usable aggregated JSONs).  Every missing row becomes an
    # explicit n/r below, so we do the check here before populating
    # rows; a misconfigured rollup should fail loudly rather than
    # produce a wall of n/r cells.
    if not any((aggregated_dir / f"{c}.json").is_file() for c, _, _ in REAL_WORLD_CAMPAIGNS):
        raise RuntimeError(
            "No real-world aggregated JSONs found in "
            f"{aggregated_dir}; nothing to render."
        )

    rows: list[dict[str, Any]] = []
    for campaign, dataset, problem in REAL_WORLD_CAMPAIGNS:
        agg_path = aggregated_dir / f"{campaign}.json"
        if not agg_path.is_file():
            # Missing aggregated JSON entirely -- render a row of n/r so
            # the omission is visible (rather than silently dropping the
            # row, which would hide the gap from the reader).
            print(f"  [summary] WARN: missing aggregated JSON for {campaign}: {agg_path}")
            rows.append(
                {
                    "dataset": rf"{{\sc {dataset}}}",
                    "problem": problem,
                    "method_cells": [fmt_not_run_cell()] * len(SUMMARY_METHODS),
                    "failure_types": {},
                }
            )
            continue

        with open(agg_path) as fh:
            agg = json.load(fh)
        methods: dict[str, dict[str, Any]] = agg.get("methods", {})
        n_seeds = int(agg.get("n_seeds", 0))
        failures_summary: dict[str, dict[str, Any]] = agg.get("failures_summary", {}) or {}

        method_cells: list[str] = []
        failure_types: dict[str, str] = {}
        for disp, keys in SUMMARY_METHODS:
            rec = _resolve_method_record(methods, keys)
            method_cells.append(_cell_for(rec, n_seeds))
            # For failure cells, surface the modal exception type in the
            # caption footnote.  Look it up via the same fallback chain
            # so the column lines up with whichever variant was picked.
            if method_cell_state(rec) == "failed":
                for key in keys:
                    if key in failures_summary:
                        failure_types[disp] = str(
                            failures_summary[key].get("modal_exception_type", "")
                        )
                        break
        rows.append(
            {
                "dataset": rf"{{\sc {dataset}}}",
                "problem": problem,
                "method_cells": method_cells,
                "failure_types": failure_types,
            }
        )

    # ``rows`` is always non-empty at this point because the pre-loop
    # guard above raises when no JSONs are present; if any JSON is
    # present at all, every campaign gets a row (either real cells or
    # n/r).  Kept as a defensive invariant rather than a runtime guard.
    assert rows, "internal: row list unexpectedly empty"

    tex = _build_table(rows, label=label)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex)
    print(f"  [real-world-summary] wrote {output}")
