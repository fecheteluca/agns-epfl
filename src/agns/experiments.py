from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agns.oracles.problems import Problem
from agns.runner import run_variant
from agns.sampling import problem_for_seed
from agns.utils.config import load_config
from agns.utils.paths import DATA_DIR, FIGURES_DIR, TABLES_DIR
from agns.utils.plotting import (
    add_restart_markers,
    plot_methods,
    save_fig,
    write_booktabs_table,
)
from agns.utils.style import COL_WIDTH, style_for

History = dict[str, Any]


@dataclass
class ExperimentResult:
    """Everything a figure/table builder needs from one experiment run."""

    name: str
    problem: Problem
    method_runs: dict[str, list[History]]
    styles: dict[str, dict[str, Any]]
    cost_axes: list[str]
    config: dict[str, Any]


def _load_method_variants(names: list[str]) -> dict[str, dict[str, Any]]:
    """Load each ``config/methods/<name>.yaml`` keyed by its display label."""
    return {name: load_config(f"methods/{name}.yaml") for name in names}


def load_oracle_specs(names: list[str]) -> list[dict[str, Any]]:
    """Load a list of ``config/oracles/<name>.yaml`` specs (for cross-problem analyses)."""
    return [load_config(f"oracles/{name}.yaml") for name in names]


def _style(variant: dict[str, Any]) -> dict[str, Any]:
    """Canonical Matplotlib style for a method variant (from :mod:`agns.utils.style`)."""
    return style_for(variant.get("label", variant["method"]))


def run_spec_over_seeds(
    oracle_spec: dict[str, Any],
    methods: list[str | dict[str, Any]],
    *,
    n_iters: int,
    seeds: list[int],
    data_dir: Any = DATA_DIR,
) -> tuple[dict[str, list[History]], dict[str, dict[str, Any]], Problem]:
    """Run several method variants over seeds, with a fresh problem per seed.

    This is the single seed-aware primitive shared by every experiment path. For each seed
    it builds the per-seed problem (:func:`agns.sampling.problem_for_seed`, applying the
    oracle's ``vary`` protocol), runs every method on that problem, and tags each history
    with its own ``f_star`` (which varies across instance-sampled seeds). Seeds loop on the
    outside so at most one problem is held in memory at a time.

    ``methods`` items are either config names (loaded from ``config/methods/``) or inline
    variant dicts (e.g. a base method plus an ``approx`` override).

    Returns
    -------
    tuple
        ``(method_runs, styles, representative_problem)`` -- ``method_runs`` keyed by display
        label (one history per seed, each carrying ``f_star``); ``representative_problem`` is
        the first seed's problem, used only for labels/metadata.
    """
    variants: list[tuple[str, dict[str, Any]]] = []
    styles: dict[str, dict[str, Any]] = {}
    for item in methods:
        variant = load_config(f"methods/{item}.yaml") if isinstance(item, str) else item
        label = variant.get("label", variant["method"])
        variants.append((label, variant))
        styles[label] = _style(variant)

    method_runs: dict[str, list[History]] = {label: [] for label, _ in variants}
    representative: Problem | None = None
    for seed in seeds:
        problem = problem_for_seed(oracle_spec, seed, data_dir)
        if representative is None:
            representative = problem
        for label, variant in variants:
            _, _, history = run_variant(variant, problem, n_iters=n_iters, seed=seed)
            if history is not None:
                history["f_star"] = problem.f_star
            method_runs[label].append(history)
    assert representative is not None, "seeds must be non-empty"
    return method_runs, styles, representative


def experiment_result(
    oracle_spec: dict[str, Any],
    methods: list[str | dict[str, Any]],
    *,
    n_iters: int,
    seeds: list[int],
    cost_axes: tuple[str, ...] = ("iterations", "matrix_inverses"),
    name: str = "",
) -> ExperimentResult:
    """Bundle a seed-varied run on an *ad-hoc* oracle spec into an :class:`ExperimentResult`.

    Lets the figure/summary helpers be reused on oracle specs built outside a named
    experiment config (non-convex families, conditioned instances, approx variants). The
    spec's ``vary`` protocol controls how seeds are sampled.
    """
    method_runs, styles, problem = run_spec_over_seeds(
        oracle_spec, methods, n_iters=n_iters, seeds=seeds
    )
    return ExperimentResult(
        name=name or problem.name,
        problem=problem,
        method_runs=method_runs,
        styles=styles,
        cost_axes=list(cost_axes),
        config={},
    )


def run_experiment(name: str, *, quick: bool = False) -> ExperimentResult:
    """Run all methods of an experiment across seeds (fresh problem per seed).

    Parameters
    ----------
    name:
        Experiment config stem under ``config/experiments/``.
    quick:
        If ``True``, use ``quick_seeds``/``quick_n_iters`` for a fast smoke run.

    Returns
    -------
    ExperimentResult
    """
    cfg = load_config(f"experiments/{name}.yaml")
    oracle_name = cfg.get("quick_oracle", cfg["oracle"]) if quick else cfg["oracle"]
    oracle_cfg = load_config(f"oracles/{oracle_name}.yaml")

    seeds = cfg["quick_seeds"] if quick else cfg["seeds"]
    n_iters = cfg["quick_n_iters"] if quick else cfg["n_iters"]

    method_runs, styles, problem = run_spec_over_seeds(
        oracle_cfg, cfg["methods"], n_iters=n_iters, seeds=seeds
    )
    return ExperimentResult(
        name=name,
        problem=problem,
        method_runs=method_runs,
        styles=styles,
        cost_axes=cfg.get("cost_axes", ["iterations"]),
        config=cfg,
    )


def convergence_figure(
    result: ExperimentResult,
    *,
    quantities: tuple[str, ...] = ("func", "grad_norm"),
    restart_label: str | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    """Build a (quantities x cost-axes) grid of median+IQR convergence curves.

    Parameters
    ----------
    result:
        The experiment result to plot.
    quantities:
        Which traced quantities to show as rows (``"func"`` -> ``f - f*``,
        ``"grad_norm"`` -> gradient norm).
    title:
        Optional figure suptitle.
    restart_label:
        If given, overlay restart-event markers from that method's seed-0 run.
    save_as:
        Filename (under ``results/figures/``) to save a vector PDF.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cost_axes = result.cost_axes
    nrows, ncols = len(quantities), len(cost_axes)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False
    )
    f_star = result.problem.f_star

    for r, quantity in enumerate(quantities):
        for c, x_key in enumerate(cost_axes):
            ax = axes[r][c]
            plot_methods(
                ax,
                result.method_runs,
                x_key=x_key,
                f_star=f_star,
                y_key=quantity,
                styles=result.styles,
            )
            if (
                restart_label
                and restart_label in result.method_runs
                and quantity == "func"
            ):
                runs = result.method_runs[restart_label]
                if runs and x_key == "iterations":
                    add_restart_markers(ax, runs[0], y_key="func")
            if r == 0 and c == 0:
                ax.legend(loc="upper right")

    fig.tight_layout()
    if save_as:
        save_fig(fig, FIGURES_DIR / save_as)
    return fig


def main_convergence_figure(
    result: ExperimentResult,
    *,
    x_key: str = "matrix_inverses",
    quantity: str = "func",
    methods: list[str] | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    """A compact, one-column single-panel convergence figure for the main paper.

    Shows the median residual (or gradient norm) against a single cost axis for the chosen
    methods, sized for one ICLR/ICML column. Use :func:`convergence_figure` for the fuller
    multi-panel appendix version.
    """
    runs = (
        result.method_runs
        if methods is None
        else {m: result.method_runs[m] for m in methods}
    )
    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.80))
    plot_methods(ax, runs, x_key=x_key, f_star=result.problem.f_star, y_key=quantity)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save_as:
        save_fig(fig, FIGURES_DIR / save_as)
    return fig


def summarize_result(
    result: ExperimentResult, *, tolerance: float = 1e-6
) -> pd.DataFrame:
    """Per-method median cost to reach ``tolerance``, computed from an existing run.

    Returns a tidy DataFrame with the median iterations, matrix inverses and gradient
    calls to first reach ``f - f* < tolerance`` (or the run length if never reached),
    the median final residual, and the fraction of seeds that reached the tolerance.
    """
    rows = []
    for label, runs in result.method_runs.items():
        iters, mis, gcs, finals, reached = [], [], [], [], []
        for h in runs:
            f_star = h.get("f_star", result.problem.f_star)
            resid = np.asarray(h["func"], dtype=float) - f_star
            hit = np.where(resid < tolerance)[0]
            idx = int(hit[0]) if hit.size else len(resid) - 1
            reached.append(bool(hit.size))
            iters.append(idx)
            gcs.append(float(np.asarray(h["grad_calls"])[idx]))
            if "matrix_inverses" in h:  # first-order methods have none
                mis.append(float(np.asarray(h["matrix_inverses"])[idx]))
            finals.append(float(resid[-1]))
        rows.append(
            {
                "Method": label,
                "Iters": float(np.median(iters)),
                "Matrix inv.": float(np.median(mis)) if mis else float("nan"),
                "Grad calls": float(np.median(gcs)),
                "Final f-f*": float(np.median(finals)),
                "Reached": float(np.mean(reached)),
            }
        )
    return pd.DataFrame(rows)


def cost_samples(
    result: ExperimentResult,
    method_label: str,
    *,
    tolerance: float = 1e-6,
    cost_key: str = "matrix_inverses",
) -> list[float]:
    """Per-seed cost (along ``cost_key``) for one method to reach ``tolerance``.

    Feeds the statistical helpers in :mod:`agns.stats` (bootstrap CI, paired Wilcoxon).
    """
    out = []
    for h in result.method_runs[method_label]:
        f_star = h.get("f_star", result.problem.f_star)
        resid = np.asarray(h["func"], dtype=float) - f_star
        hit = np.where(resid < tolerance)[0]
        idx = int(hit[0]) if hit.size else len(resid) - 1
        if cost_key == "iterations":
            out.append(float(idx))
        else:
            out.append(float(np.asarray(h[cost_key])[idx]))
    return out


def family_gate_table(
    family_specs: list[dict[str, Any]],
    methods: list[str | dict[str, Any]],
    *,
    n_iters: int,
    seeds: list[int],
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Validity-gate table: per (family, method) reached-fraction, uphill steps, residual.

    For the sanity notebook. Runs every method on every family (fresh problem per seed) and
    reports, per pair, the fraction of seeds that reached ``f - f* < tolerance``, the worst-
    case number of uphill steps across seeds (a monotonicity diagnostic), and the median
    final residual. Computation stays here so the notebook only asserts on the table.
    """
    rows = []
    for spec in family_specs:
        runs, _, problem = run_spec_over_seeds(
            spec, methods, n_iters=n_iters, seeds=seeds
        )
        for label, histories in runs.items():
            reached, uphill, finals = [], [], []
            for h in histories:
                f_star = h.get("f_star", problem.f_star)
                resid = np.asarray(h["func"], dtype=float) - f_star
                reached.append(bool((resid < tolerance).any()))
                diffs = np.diff(np.asarray(h["func"], dtype=float))
                uphill.append(int((diffs > 1e-12).sum()))
                finals.append(float(resid.min()))
            rows.append(
                {
                    "Family": problem.label,
                    "Method": label,
                    "Reached": float(np.mean(reached)),
                    "Max uphill steps": int(np.max(uphill)),
                    "Median final f-f*": float(np.median(finals)),
                }
            )
    return pd.DataFrame(rows)


def restart_summary(result: ExperimentResult) -> pd.DataFrame:
    """Median number of adaptive restarts per (AGNS) method, from an existing run."""
    rows = []
    for label, runs in result.method_runs.items():
        counts = [
            int(h["n_restarts_total"][0]) for h in runs if "n_restarts_total" in h
        ]
        if counts:
            rows.append({"Method": label, "Median restarts": float(np.median(counts))})
    return pd.DataFrame(rows)


def monotonicity_summary(
    result: ExperimentResult, *, atol: float = 1e-12
) -> pd.DataFrame:
    """Median number of uphill steps per method (a proxy for momentum oscillation)."""
    rows = []
    for label, runs in result.method_runs.items():
        ups = [
            int((np.diff(np.asarray(h["func"], dtype=float)) > atol).sum())
            for h in runs
        ]
        rows.append({"Method": label, "Median uphill steps": float(np.median(ups))})
    return pd.DataFrame(rows)


def cost_to_tolerance_table(
    datasets: list[str],
    method_names: list[str],
    *,
    tolerance: float,
    n_iters: int,
    seeds: list[int]
) -> pd.DataFrame:
    """Median matrix-inverses and wall-clock to reach ``tolerance`` per method x dataset.

    Returns a tidy DataFrame and optionally writes a booktabs LaTeX table.
    """
    rows = []
    for dataset in datasets:
        oracle_cfg = load_config(f"oracles/{dataset}.yaml")
        method_runs, _, _ = run_spec_over_seeds(
            oracle_cfg, method_names, n_iters=n_iters, seeds=seeds
        )
        for label_m, runs in method_runs.items():
            mis, times = [], []
            for h in runs:
                f_star = h.get("f_star", 0.0)
                resid = np.asarray(h["func"], dtype=float) - f_star
                hit = np.where(resid < tolerance)[0]
                idx = int(hit[0]) if hit.size else len(resid) - 1
                if "matrix_inverses" in h:
                    mis.append(float(np.asarray(h["matrix_inverses"])[idx]))
                times.append(float(np.asarray(h["time"])[idx]))
            rows.append(
                {
                    "Dataset": oracle_cfg.get("label", dataset),
                    "Method": label_m,
                    "Matrix inv.": np.median(mis) if mis else float("nan"),
                    "Time (s)": np.median(times),
                }
            )
    df = pd.DataFrame(rows)
    return df
