from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from agns.utils.style import despine, style_for

History = dict[str, Any]

COST_LABELS: dict[str, str] = {
    "iterations": "Iterations",
    "matrix_inverses": "Matrix inverses",
    "grad_calls": "Gradient calls",
    "func_calls": "Function calls",
    "time": "Time (s)",
}

_SWEEP_LABELS: dict[str, str] = {
    **COST_LABELS,
    "kappa": r"$\nu$ (curvature variation)",
    "cond": "Condition number",
    "momentum_offset": "Momentum offset",
    "solved": "Fraction solved",
}

_FLOOR = 1e-16  


def residual(
    history: History, f_star: float, y_key: str = "func"
) -> NDArray[np.float64]:
    """Return the positive residual ``history[y_key] - f_star`` (floored for log scale).

    When the history carries its own ``f_star`` (injected per seed by
    :func:`agns.experiments.run_spec_over_seeds`, since instance-sampled seeds each have a
    different optimum), that value takes precedence over the passed ``f_star``.
    """
    f_star = history.get("f_star", f_star)
    arr = np.asarray(history[y_key], dtype=float) - f_star
    return np.maximum(arr, _FLOOR)


def cost_axis(history: History, x_key: str) -> NDArray[np.float64]:
    """Return the x-axis values for ``x_key`` ("iterations" -> index, else a history key)."""
    if x_key == "iterations":
        return np.arange(len(history["func"]), dtype=float)
    return np.asarray(history.get(x_key, []), dtype=float)


def has_cost_axis(runs: Sequence[History], x_key: str) -> bool:
    """Whether ``runs`` expose a usable (non-empty) cost axis for ``x_key``.

    First-order methods do not record ``matrix_inverses``; such methods are simply
    omitted from a matrix-inverses panel rather than crashing or plotting at ``x = 0``.
    """
    if x_key == "iterations":
        return True
    return all(len(cost_axis(run, x_key)) == len(run["func"]) for run in runs)


def quantity_series(history: History, y_key: str, f_star: float) -> NDArray[np.float64]:
    """Extract the plotted y-series for ``y_key`` (floored for log scale).

    ``"func"`` -> residual ``f - f*``; ``"grad_norm"`` -> the dual gradient norm, falling
    back to ``sqrt(grad_sqr_norm)`` for the first-order methods that store the squared norm.
    """
    if y_key == "grad_norm":
        if "grad_norm" in history:
            arr = np.asarray(history["grad_norm"], dtype=float)
        elif "grad_sqr_norm" in history:
            arr = np.sqrt(np.asarray(history["grad_sqr_norm"], dtype=float))
        else:
            return np.empty(0)
        return np.maximum(arr, _FLOOR)
    return residual(history, f_star, y_key)


def has_quantity(runs: Sequence[History], y_key: str) -> bool:
    """Whether every run exposes a non-empty series for ``y_key``."""
    return all(
        len(quantity_series(run, y_key, 0.0)) == len(run["func"]) for run in runs
    )


def aggregate_curves(
    runs: Sequence[History],
    *,
    x_key: str,
    f_star: float,
    y_key: str = "func",
    n_grid: int = 200,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Aggregate per-seed runs into median and inter-quartile band on a shared grid.

    Each run is reduced to ``(cost, log10(residual))`` and linearly interpolated onto a
    shared log-spaced (for cost) / linear grid; the median and 25/75 percentiles are taken
    in log-residual space and exponentiated back.

    Parameters
    ----------
    runs:
        One history per seed.
    x_key:
        Cost axis key (see :data:`COST_LABELS`).
    f_star:
        Optimum used to form the residual.
    y_key:
        History key for the plotted quantity (``"func"`` or ``"grad_norm"``).
    n_grid:
        Number of shared grid points.

    Returns
    -------
    tuple
        ``(grid, median, lo, hi)`` arrays.
    """
    curves: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    x_max = 0.0
    for run in runs:
        x = cost_axis(run, x_key)
        y = quantity_series(run, y_key, f_star)
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        # Enforce strictly increasing x for interpolation.
        keep = np.concatenate(([True], np.diff(x) > 0))
        curves.append((x[keep], np.log10(y[keep])))
        x_max = max(x_max, float(x[keep][-1]) if len(x[keep]) else 0.0)

    grid = np.linspace(0.0, x_max, n_grid)
    stacked = np.vstack(
        [np.interp(grid, x, logy, left=logy[0], right=logy[-1]) for x, logy in curves]
    )
    median = 10 ** np.median(stacked, axis=0)
    lo = 10 ** np.percentile(stacked, 25, axis=0)
    hi = 10 ** np.percentile(stacked, 75, axis=0)
    return grid, median, lo, hi


def plot_methods(
    ax: plt.Axes,
    method_runs: dict[str, Sequence[History]],
    *,
    x_key: str,
    f_star: float,
    y_key: str = "func",
    styles: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Draw median curves with IQR bands for several methods on one axis.

    Colours and line styles come from :func:`agns.utils.style.style_for` (the single source of
    truth), so a method looks identical in every figure. The ``styles`` argument is accepted for
    backward compatibility but the canonical style takes precedence.
    """
    for label, runs in method_runs.items():
        if not has_cost_axis(runs, x_key) or not has_quantity(runs, y_key):
            continue  # e.g. first-order methods on a matrix-inverses axis
        grid, median, lo, hi = aggregate_curves(
            runs, x_key=x_key, f_star=f_star, y_key=y_key
        )
        style = style_for(label)
        (line,) = ax.semilogy(grid, median, label=label, **style)
        ax.fill_between(
            grid, lo, hi, alpha=0.15, color=line.get_color(), zorder=style["zorder"] - 1
        )
    ax.set_xlabel(COST_LABELS.get(x_key, x_key))
    ylabel = r"$\|\nabla f(x_k)\|$" if y_key == "grad_norm" else r"$f(x_k) - f^\star$"
    ax.set_ylabel(ylabel)
    despine(ax)


def add_restart_markers(
    ax: plt.Axes, history: History, *, color: str = "k", y_key: str = "func"
) -> None:
    """Mark adaptive-restart events along a single run's curve."""
    events = history.get("restart_events", [])
    if not events:
        return
    f_star = 0.0
    y = (
        residual(history, f_star, y_key)
        if y_key == "func"
        else np.asarray(history[y_key], dtype=float)
    )
    xs = cost_axis(history, "iterations")
    for e in events:
        if e < len(y):
            ax.scatter(xs[e], y[e], marker="x", color=color, s=40, zorder=5)


def plot_performance_profile(
    ax: plt.Axes,
    ratios: dict[str, NDArray[np.float64]],
    *,
    tau_max: float = 32.0,
    styles: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Plot Dolan-More performance profiles from per-method performance ratios.

    ``ratios[label]`` is the array of per-instance ratios (cost / best-cost-on-instance,
    ``inf`` for failures). The profile ``rho(tau)`` is the fraction of instances solved
    within a factor ``tau`` of the best method; it is drawn against ``log2(tau)``.
    """
    taus = np.logspace(0, np.log2(tau_max), 100, base=2.0)
    for label, r in ratios.items():
        r = np.asarray(r, dtype=float)
        rho = [(r <= t).mean() for t in taus]
        style = style_for(label)
        ax.step(np.log2(taus), rho, where="post", label=label, **style)
    ax.set_xlabel(r"$\log_2 \tau$")
    ax.set_ylabel(r"fraction within $\tau\times$ best")
    ax.set_ylim(-0.02, 1.02)
    despine(ax)
    ax.legend(loc="lower right")


def plot_sweep(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str,
    logx: bool = False,
    logy: bool = False,
    styles: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Line plot of ``y`` vs ``x`` for each ``hue`` group of a long-form DataFrame."""
    plot = ax.loglog if (logx and logy) else (ax.semilogx if logx else ax.plot)
    for key, sub in df.groupby(hue):
        sub = sub.sort_values(x)
        plot(
            sub[x].to_numpy(),
            sub[y].to_numpy(),
            marker="o",
            markersize=4,
            label=str(key),
            **style_for(str(key)),
        )
    if logy and not logx:
        ax.set_yscale("log")
    ax.set_xlabel(_SWEEP_LABELS.get(x, x))
    ax.set_ylabel(_SWEEP_LABELS.get(y, y))
    despine(ax)
    ax.legend()


def save_fig(fig: plt.Figure, path: str | Path) -> None:
    """Save a figure as vector PDF (tight bbox)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def write_booktabs_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    caption: str,
    label: str,
    float_format: str = "%.2f",
) -> None:
    """Write a DataFrame to a booktabs LaTeX table for ``\\input`` into a LaTeX document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = df.to_latex(
        index=False,
        escape=False,
        float_format=lambda v: float_format % v,
        column_format="l" + "r" * (df.shape[1] - 1),
        caption=caption,
        label=label,
        position="t",
    )
    path.write_text(body, encoding="utf-8")
    print(f"output: {path}")
