"""Cross-problem analyses that strengthen each research question.

Builds on :func:`agns.experiments.run_spec_over_seeds`:

* :func:`cross_problem_figure` -- the same methods on several problem families side by side;
* :func:`performance_profile` -- Dolan-More profiles aggregated over a set of instances;
* :func:`sweep` -- vary one problem attribute (e.g. condition number) or method parameter
  (e.g. momentum offset) and trace the cost to reach tolerance.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from agns.experiments import History, _load_method_variants, run_spec_over_seeds
from agns.utils.paths import DATA_DIR, FIGURES_DIR
from agns.utils.plotting import (
    plot_methods,
    plot_performance_profile,
    plot_sweep,
    save_fig,
)
from agns.utils.style import COL_WIDTH

Spec = dict[str, Any]


def _cost_to_tolerance(
    history: History, f_star: float, tolerance: float, cost_key: str
) -> float:
    """Cost (along ``cost_key``) to first reach ``f - f* < tolerance``; ``inf`` if never."""
    resid = np.asarray(history["func"], dtype=float) - f_star
    hit = np.where(resid < tolerance)[0]
    if not hit.size:
        return float("inf")
    idx = int(hit[0])
    if cost_key == "iterations":
        return float(idx)
    if cost_key not in history:
        return float("inf")
    return float(np.asarray(history[cost_key])[idx])


def _labels(method_names: list[str]) -> dict[str, str]:
    """Map method-config name -> display label."""
    return {
        name: variant.get("label", variant["method"])
        for name, variant in _load_method_variants(method_names).items()
    }


def cross_problem_figure(
    problem_specs: list[Spec],
    method_names: list[str],
    *,
    n_iters: int,
    seeds: list[int],
    x_key: str = "matrix_inverses",
    data_dir: Any = DATA_DIR,
    save_as: str | None = None,
) -> plt.Figure:
    """Functional-residual convergence of the same methods across several problems."""
    fig, axes = plt.subplots(
        1, len(problem_specs), figsize=(4.8 * len(problem_specs), 3.8), squeeze=False
    )
    for ax, spec in zip(axes[0], problem_specs, strict=True):
        runs, styles, problem = run_spec_over_seeds(
            spec, method_names, n_iters=n_iters, seeds=seeds, data_dir=data_dir
        )
        plot_methods(
            ax, runs, x_key=x_key, f_star=problem.f_star, y_key="func", styles=styles
        )
        ax.set_title(problem.label, fontsize=11)
    axes[0][0].legend(loc="upper right")
    fig.tight_layout()
    if save_as:
        save_fig(fig, FIGURES_DIR / save_as)
    return fig


def performance_profile(
    instance_specs: list[Spec],
    method_names: list[str],
    *,
    tolerance: float,
    n_iters: int,
    seeds: list[int],
    cost_key: str = "matrix_inverses",
    data_dir: Any = DATA_DIR,
) -> dict[str, NDArray[np.float64]]:
    """Per-method performance ratios over all ``(instance, seed)`` pairs.

    For each instance and seed the per-method cost to reach ``tolerance`` is divided by the
    best method's cost on that pair; failures map to ``inf``. The returned ratios feed
    :func:`agns.utils.plotting.plot_performance_profile`.
    """
    labels = _labels(method_names)
    ratios: dict[str, list[float]] = {lab: [] for lab in labels.values()}
    for spec in instance_specs:
        runs, _, _ = run_spec_over_seeds(
            spec, method_names, n_iters=n_iters, seeds=seeds, data_dir=data_dir
        )
        for si in range(len(seeds)):
            costs = {
                lab: _cost_to_tolerance(
                    runs[lab][si],
                    runs[lab][si].get("f_star", 0.0),
                    tolerance,
                    cost_key,
                )
                for lab in labels.values()
            }
            best = min(costs.values())
            for lab, c in costs.items():
                if not np.isfinite(best) or best <= 0:
                    ratios[lab].append(1.0 if c == best else float("inf"))
                else:
                    ratios[lab].append(c / best)
    return {lab: np.asarray(v, dtype=float) for lab, v in ratios.items()}


def sweep(
    base_oracle: Spec,
    method_names: list[str],
    *,
    vary: str,
    target: str,
    values: list[float],
    tolerance: float,
    n_iters: int,
    seeds: list[int],
    cost_key: str = "matrix_inverses",
    data_dir: Any = DATA_DIR,
) -> pd.DataFrame:
    """Median cost to reach tolerance as one attribute is swept.

    ``vary="oracle"`` sets ``base_oracle[target] = value`` (e.g. ``target="cond"``);
    ``vary="method"`` sets ``params[target] = value`` on each method (e.g.
    ``target="momentum_offset"``). Returns a long-form DataFrame (``value`` is stored under
    the ``target`` column) with the median over the finite-cost seeds.
    """
    labels = _labels(method_names)
    rows = []
    for value in values:
        if vary == "oracle":
            spec = {**base_oracle, target: value}
            runs, _, _ = run_spec_over_seeds(
                spec, method_names, n_iters=n_iters, seeds=seeds, data_dir=data_dir
            )
            for lab in labels.values():
                rows.append(
                    _sweep_row(target, value, lab, runs[lab], tolerance, cost_key)
                )
        elif vary == "method":
            variants = _load_method_variants(method_names)
            for variant in variants.values():
                lab = variant.get("label", variant["method"])
                cfg = {
                    **variant,
                    "params": {**variant.get("params", {}), target: value},
                }
                runs, _, _ = run_spec_over_seeds(
                    base_oracle, [cfg], n_iters=n_iters, seeds=seeds, data_dir=data_dir
                )
                rows.append(
                    _sweep_row(target, value, lab, runs[lab], tolerance, cost_key)
                )
        else:
            raise ValueError("vary must be 'oracle' or 'method'")
    return pd.DataFrame(rows)


def _sweep_row(
    target: str,
    value: float,
    label: str,
    histories: list[History],
    tolerance: float,
    cost_key: str,
) -> dict[str, Any]:
    """Median finite cost (and solved fraction) for one sweep point (per-seed ``f_star``)."""
    costs = [
        _cost_to_tolerance(h, h.get("f_star", 0.0), tolerance, cost_key)
        for h in histories
    ]
    finite = [c for c in costs if np.isfinite(c)]
    return {
        target: value,
        "Method": label,
        cost_key: float(np.median(finite)) if finite else float("nan"),
        "solved": float(np.mean(np.isfinite(costs))),
    }


def plot_sweep_figure(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    logx: bool = False,
    logy: bool = True,
    save_as: str | None = None,
    compact: bool = False,
) -> plt.Figure:
    """Single-axis sweep figure from a :func:`sweep` DataFrame.

    ``compact=True`` sizes it for one paper column (the "main" version); otherwise it uses the
    larger appendix size.
    """
    figsize = (COL_WIDTH, COL_WIDTH * 0.80) if compact else (5.5, 4.0)
    fig, ax = plt.subplots(figsize=figsize)
    plot_sweep(ax, df, x=x, y=y, hue="Method", logx=logx, logy=logy)
    fig.tight_layout()
    if save_as:
        save_fig(fig, FIGURES_DIR / save_as)
    return fig


def performance_profile_figure(
    ratios: dict[str, NDArray[np.float64]],
    *,
    save_as: str | None = None,
    compact: bool = False,
) -> plt.Figure:
    """Single-axis performance-profile figure.

    ``compact=True`` sizes it for one paper column (the "main" version).
    """
    figsize = (COL_WIDTH, COL_WIDTH * 0.80) if compact else (5.5, 4.0)
    fig, ax = plt.subplots(figsize=figsize)
    plot_performance_profile(ax, ratios)
    fig.tight_layout()
    if save_as:
        save_fig(fig, FIGURES_DIR / save_as)
    return fig
