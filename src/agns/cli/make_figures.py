"""Render every figure for the AGNS report.

Two modes, selected on the command line:

* *per-campaign* (``--aggregated <campaign>.json --output-dir DIR``):
  one PDF + PNG per figure type, written under ``results/figures/<campaign>/``.
* *report figures* (``--report-figures --results-dir DIR``): the two
  cross-campaign figures that are not tied to a single campaign, written
  under ``results/paper_figures/``.

Per-campaign figure types (each subject to ``--types`` filtering):

* ``iter_convergence`` --- median residual + IQR vs iteration.  Log-y.
* ``time_convergence`` --- median residual + IQR vs wall-clock time.
* ``grad_convergence`` --- median residual + IQR vs gradient calls.
* ``rate_loglog``      --- log-log residual vs iteration with the
                           fitted slope annotated.
* ``sweep``            --- per-seed final-residual cloud + median vs the
                           swept hyperparameter (ablation campaigns only).

Report figures:

* ``picard_diagnostic`` --- AGNS-Theory outer-iteration diagnostic
  (residual and estimating-sequence weight vs iteration, several rho).
* ``scaling_time``      --- wall-clock to target vs problem size n.

Visual conventions: figure size 6 x 3.5 in, a compact legend below the
axes, fixed colour-blind-safe palette grouped by method family.  Each
method gets a unique ``(colour, linestyle, marker)`` triple so plots
remain unambiguous in colour, in greyscale, and for colour-blind
readers.  PDF metadata is sanitised so re-running on the same input
emits byte-comparable PDFs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agns.config import ConfigError, load_config
from agns.utils.io import load_pickle

__all__ = ["main"]

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

_PAPER_RC: dict[str, Any] = {
    "figure.figsize": (6.0, 3.5),
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    # TeX is off by default: headless machines rarely have a LaTeX install.
    "text.usetex": False,
    "font.family": "serif",
    "axes.labelsize": 18,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
    "lines.linewidth": 2.4,
    "lines.markersize": 6,
    "axes.linewidth": 1.0,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.framealpha": 1.0,
    "legend.fancybox": False,
    "legend.borderpad": 0.20,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.35,
    "legend.labelspacing": 0.25,
    "legend.columnspacing": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

_RESID_FLOOR = 1e-16

# ---------------------------------------------------------------------------
# Canonical per-method visual style.
# ---------------------------------------------------------------------------

# (color, linestyle, marker, linewidth, zorder)
_STYLE_TABLE: dict[str, tuple[str, str, str, float, int]] = {
    # AGNS-Practice (dark green).
    "agns_practice": ("#117733", "-", "o", 2.6, 12),
    "agns_practice_exact": ("#117733", "--", "s", 2.4, 11),
    "agns_practice_wsm": ("#117733", ":", "D", 2.4, 11),
    # AGNS-Theory family (orange) -- kept visually distinct from the
    # AGNS-Practice green: the two are the paper's central comparison and
    # must be separable in colour, in greyscale (via linestyle/marker), and
    # for colour-blind readers.
    "agns_theory": ("#EE7733", "-", "^", 2.6, 12),
    "agns_holder": ("#EE7733", "--", "v", 2.4, 11),
    "agns_inexact": ("#EE7733", ":", "P", 2.4, 11),
    "agns_sc": ("#EE7733", "-.", "p", 2.4, 11),
    # GNS (wine).
    "gns_exact": ("#882255", "-", "X", 2.0, 10),
    "gns_inexact": ("#882255", "--", "p", 2.0, 10),
    "gns_wsm": ("#882255", ":", "h", 2.0, 10),
    # Newton-type (indigo).
    "newton": ("#332288", "-", "<", 1.8, 8),
    "super_newton": ("#332288", "--", ">", 1.8, 8),
    "cubic_newton": ("#332288", ":", "v", 1.8, 8),
    "trust_region": ("#332288", "-.", "*", 1.8, 8),
    # Accelerated-cubic family (olive).
    "accelerated_cubic_newton": ("#999933", "-", "d", 1.8, 6),
    "monteiro_svaiter_acn": ("#999933", "--", "H", 1.8, 6),
    # Quasi-Newton (sand).
    "lbfgs": ("#DDCC77", "-", "P", 1.8, 5),
    # First-order (light blue).
    "gradient": ("#88CCEE", "-", "+", 1.5, 3),
    "fast_gradient": ("#88CCEE", "--", "x", 1.5, 3),
    "heavy_ball": ("#88CCEE", ":", "*", 1.5, 3),
    # ML optimisers (mauve).
    "adam": ("#AA4499", "-", "*", 1.5, 2),
    "adahessian": ("#AA4499", "--", "X", 1.5, 2),
}

_UNKNOWN_STYLE: tuple[str, str, str, float, int] = ("#444444", "-", "o", 1.5, 1)

# Compact legend labels.  Plots carry up to ~11 methods; the full names
# ("Super-Universal Newton", "Fast Gradient (Nesterov)", ...) make the
# legend wider than the axes and force the figure into a squashed
# aspect ratio.  These abbreviations keep the legend within the plot
# width and are consistent with the abbreviations used in the report's
# tables (GD, FG, AGNS-Pr, AGNS-Th, SUN).
_LEGEND_ABBREV: dict[str, str] = {
    "agns_practice": "AGNS-Pr",
    "agns_practice_exact": "AGNS-Pr (exact)",
    "agns_practice_wsm": "AGNS-Pr (WSM)",
    "agns_theory": "AGNS-Th",
    "agns_holder": "AGNS-Hol",
    "agns_inexact": "AGNS-Inx",
    "agns_sc": "AGNS-SC",
    "gns_exact": "GNS",
    "gns_inexact": "GNS-WGN",
    "gns_wsm": "GNS-WSM",
    "newton": "Newton",
    "super_newton": "SUN",
    "cubic_newton": "Cubic-N",
    "accelerated_cubic_newton": "ACN",
    "monteiro_svaiter_acn": "MS-ACN",
    "trust_region": "Trust-reg",
    "lbfgs": "L-BFGS",
    "gradient": "GD",
    "fast_gradient": "FG",
    "heavy_ball": "Heavy-ball",
    "adam": "Adam",
    "adahessian": "AdaHessian",
}

_SWEEP_XLABEL: dict[str, str] = {
    "rho": r"$\rho$",
    "theta": r"$\theta$",
    "gamma_0": r"$\gamma_0$",
    "gamma0": r"$\gamma_0$",
    "fp_max": r"$\mathrm{FP}_{\max}$",
    "delta": r"$\delta$",
    "delta_budget": r"$\delta$",
    "momentum_offset": "Nesterov offset",
    "restart_mode": "restart rule",
    "warm_start": "warm-start",
    "inexact_rank": "Hessian rank",
}

# Inexact-Hessian delta-sweep: Viridis ramp on log10(delta).
_INEXACT_DELTAS: dict[str, float] = {
    "agns_inexact_d0": 0.0,
    "agns_inexact_d1em3": 1e-3,
    "agns_inexact_d1em2": 1e-2,
    "agns_inexact_d1em1": 1e-1,
    "agns_inexact_d1": 1.0,
    "agns_inexact_d10": 10.0,
}


def _inexact_sweep_style(method_key: str) -> tuple[str, str, str, float, int] | None:
    delta = _INEXACT_DELTAS.get(method_key)
    if delta is None:
        return None
    if delta == 0.0:
        return ("#EE7733", "-", "P", 2.4, 12)
    t = (np.log10(delta) + 3.0) / 4.0  # delta=1e-3 -> 0, delta=10 -> 1
    t = float(np.clip(t, 0.0, 1.0))
    cmap = plt.get_cmap("viridis")
    color = matplotlib.colors.to_hex(cmap(t))
    markers = ["o", "s", "D", "^", "P", "X"]
    idx = round(t * (len(markers) - 1))
    return (color, "-", markers[idx], 2.0, 8 + idx)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--aggregated",
        default=None,
        help="Path to <campaign>.json (per-campaign mode; required unless "
        "--report-figures is given).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the per-campaign figures into.",
    )
    p.add_argument(
        "--report-figures",
        action="store_true",
        help="Render the cross-campaign report figures (picard_diagnostic, "
        "scaling_time) into <results-dir>/paper_figures/ instead.",
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Results root for --report-figures mode (default: results).",
    )
    p.add_argument(
        "--types",
        nargs="+",
        default=[
            "iter_convergence",
            "time_convergence",
            "grad_convergence",
            "rate_loglog",
            "sweep",
        ],
        help="Figure types to produce.",
    )
    p.add_argument("--title", default=None, help="Override figure title.")
    p.add_argument(
        "--no-title",
        action="store_true",
        help="Suppress titles (captions carry the take-away).",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Limit to this subset of method keys (default: every method in the JSON).",
    )
    p.add_argument(
        "--base-methods-dir",
        default="configs/_base/methods",
        help="Path to _base/methods YAMLs (for cosmetic-metadata fallback).",
    )
    p.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Override the default figure size in inches.",
    )
    p.add_argument(
        "--nu-ref",
        type=float,
        default=None,
        help=(
            "Overlay a reference slope ``C / K^{2+nu}`` on the "
            "rate_loglog figure.  ``C`` is fitted to the first non-floor "
            "point of the AGNS-Theory trajectory."
        ),
    )
    p.add_argument(
        "--no-legend",
        action="store_true",
        help="Suppress the per-figure legend.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Style resolution
# ---------------------------------------------------------------------------


def _resolve_style(method_key: str) -> tuple[str, str, str, float, int]:
    base = method_key.split("__", 1)[0]
    inexact = _inexact_sweep_style(base)
    if inexact is not None:
        return inexact
    return _STYLE_TABLE.get(base, _UNKNOWN_STYLE)


def _style_for(method_key: str, rec: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    color, linestyle, marker, linewidth, zorder = _resolve_style(method_key)
    base = method_key.split("__", 1)[0]
    abbr = _LEGEND_ABBREV.get(base)
    if abbr is not None:
        label: str = abbr
    else:
        raw: str | None = rec.get("label")
        if raw is None:
            base_yaml = base_dir / f"{method_key}.yaml"
            if base_yaml.is_file():
                try:
                    cfg = load_config(base_yaml)
                    yaml_label = cfg.get("label")
                    if yaml_label is not None:
                        raw = str(yaml_label)
                except ConfigError:
                    pass
        label = _short_label(raw) if raw else method_key
    return {
        "color": color,
        "linestyle": linestyle,
        "marker": marker,
        "linewidth": linewidth,
        "zorder": zorder,
        "label": label,
    }


# ---------------------------------------------------------------------------
# Plot primitives
# ---------------------------------------------------------------------------


def _floor_log(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, _RESID_FLOOR)


def _legend_below(ax: Any, *, ncol: int | None = None) -> None:
    """Place a compact legend below the axes.

    With the abbreviated method labels the legend fits within the plot
    width: up to five methods on one row, more on two rows.  This keeps
    the legend from being wider or taller than the plot it annotates.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    n = len(handles)
    if ncol is None:
        # At most five columns keeps the legend within the plot width;
        # balance the entries across the resulting rows so the last row
        # is not left with a lone entry.
        rows = math.ceil(n / 5)
        ncol = math.ceil(n / rows)
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=ncol,
        borderaxespad=0.0,
        fontsize=11,
        columnspacing=1.0,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.3,
    )


def _short_label(label: str) -> str:
    if " [" in label:
        return label.split(" [", 1)[0]
    return label


def _markevery(n_points: int, target: int = 8) -> int | None:
    if n_points <= target:
        return None
    return max(1, n_points // target)


def _plot_curves(
    ax: Any,
    methods: dict[str, dict[str, Any]],
    base_dir: Path,
    x_source: str,
    xlabel: str,
    title: str | None,
    *,
    show_iqr: bool = True,
    xscale: str = "log",
) -> None:
    """Plot the median residual (with IQR band) against the chosen abscissa.

    The ordinate is *always* the suboptimality ``f(x_k) - f^*`` and is read
    from ``iter_curve`` (``median`` for the line, ``p25``/``p75`` for the
    band).  The abscissa depends on ``x_source``:

    * ``"iteration"`` --- the iteration index ``iter_curve["x"]``;
    * ``"time_curve"`` / ``"grad_curve"`` --- the *cumulative* wall-clock
      time / gradient-oracle count, which the aggregator stores in the
      ``median`` field of those curves (their ``x`` field is the iteration
      index and is **not** an abscissa).

    All three per-method curves are produced by the same seed aggregation
    and are therefore index-aligned: entry ``i`` of every curve describes
    the same outer iteration ``i``.  This lets a cost-convergence figure
    pair the residual at iteration ``i`` with the cost spent to reach it.
    """
    for method_key in sorted(methods):
        rec = methods[method_key]
        resid = rec.get("iter_curve")
        if not resid:
            continue
        style = _style_for(method_key, rec, base_dir)

        med = _floor_log(np.asarray(resid["median"], dtype=float))
        lo = _floor_log(np.asarray(resid["p25"], dtype=float))
        hi = _floor_log(np.asarray(resid["p75"], dtype=float))

        if x_source == "iteration":
            x = np.asarray(resid["x"], dtype=float)
        else:
            cost = rec.get(x_source)
            if not cost:
                continue
            # cumulative cost lives in `median`; `x` is the iteration index.
            x = np.asarray(cost["median"], dtype=float)

        # curves are index-aligned; guard against a ragged record.
        n = min(x.size, med.size, lo.size, hi.size)
        if n == 0:
            continue
        x, med, lo, hi = x[:n], med[:n], lo[:n], hi[:n]

        # a log abscissa cannot show a non-positive coordinate (iteration 0,
        # or a zero-cost initial point); drop those points.
        if xscale == "log":
            keep = x > 0.0
            x, med, lo, hi = x[keep], med[keep], lo[keep], hi[keep]
            if x.size == 0:
                continue

        if show_iqr:
            ax.fill_between(
                x,
                lo,
                hi,
                color=style["color"],
                alpha=0.13,
                linewidth=0,
                zorder=style["zorder"] - 1,
            )
        ax.plot(
            x,
            med,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markevery=_markevery(x.size),
            markersize=5.5,
            markeredgewidth=0.0,
            zorder=style["zorder"],
        )
    ax.set_yscale("log")
    if xscale == "log":
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$f(x_k) - f^{\star}$")
    if title:
        ax.set_title(title)


def _parse_sweep_key(method_key: str) -> tuple[str, str, str] | None:
    if "__" not in method_key:
        return None
    base, suffix = method_key.split("__", 1)
    if "=" not in suffix:
        return None
    param, value = suffix.split("=", 1)
    if not param or not value:
        return None
    return base, param, value


def _sweep_xscale(values: list[float]) -> str:
    """Pick the x-axis scale for a numeric hyperparameter sweep.

    A log axis when the swept values are all positive and span a wide
    multiplicative range (e.g.\\ ``gamma_0`` over ``{0.01, ..., 100}``), so
    geometrically-spaced points are not crammed against the origin; a
    linear axis otherwise (``rho``, ``theta``, ``fp_max``, ...).
    """
    if values and all(v > 0.0 for v in values) and max(values) / min(values) >= 50.0:
        return "log"
    return "linear"


def _plot_sweep(
    ax: Any,
    methods: dict[str, dict[str, Any]],
    base_dir: Path,
    title: str | None,
) -> None:
    """Final residual vs.\\ a swept hyperparameter.

    For every swept value the campaign records ten per-seed final
    residuals; we draw all ten as a translucent point cloud (the residual
    ordinate separates the seeds, so no horizontal jitter is needed and the
    bimodal converged/diverged split is visible directly) and overlay the
    median as a connected line.  This replaces an earlier IQR-whisker
    rendering whose error bars spanned a dozen decades and read as noise.
    """
    sweep_groups: dict[tuple[str, str], list[tuple[Any, dict[str, Any]]]] = {}
    references: list[tuple[str, dict[str, Any]]] = []
    for method_key in sorted(methods):
        rec = methods[method_key]
        parsed = _parse_sweep_key(method_key)
        if parsed is None:
            references.append((method_key, rec))
            continue
        base, param, raw_value = parsed
        try:
            value: float | str = float(raw_value.replace("p", "."))
        except ValueError:
            value = raw_value
        sweep_groups.setdefault((base, param), []).append((value, rec))

    if not sweep_groups:
        return

    xlabel = ""
    xscale = "linear"
    for (base, param), items in sweep_groups.items():
        numeric = all(isinstance(v, (int, float)) for v, _ in items)
        if numeric:
            items.sort(key=lambda kv: kv[0])
            values: list[Any] = [float(v) for v, _ in items]
            xs: list[float] = list(values)
            xscale = _sweep_xscale([float(v) for v in values])
        else:
            values = [str(v) for v, _ in items]
            xs = [float(i) for i in range(len(values))]  # categorical positions

        style = _style_for(base, items[0][1], base_dir)
        color = style["color"]

        seed_x: list[float] = []
        seed_y: list[float] = []
        med_y: list[float] = []
        for x, (_, rec) in zip(xs, items, strict=True):
            for r in rec.get("final_residuals") or []:
                if r is not None and np.isfinite(r):
                    seed_x.append(x)
                    seed_y.append(max(float(r), _RESID_FLOOR))
            m = rec.get("final_residual_summary", {}).get("median")
            med_y.append(
                max(float(m), _RESID_FLOOR)
                if isinstance(m, (int, float)) and np.isfinite(m)
                else _RESID_FLOOR
            )

        if seed_x:
            ax.scatter(
                seed_x,
                seed_y,
                s=34,
                color=color,
                alpha=0.38,
                edgecolors="none",
                zorder=4,
            )
        ax.plot(
            xs,
            med_y,
            color=color,
            linestyle="-",
            linewidth=1.9,
            marker=style["marker"],
            markersize=8.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=6,
            label=_short_label(style["label"]),
        )
        if not numeric:
            ax.set_xticks(xs)
            ax.set_xticklabels(values)
        xlabel = _SWEEP_XLABEL.get(param, param.replace("_", " "))

    for ref_key, rec in references:
        style = _style_for(ref_key, rec, base_dir)
        y = rec.get("final_residual_summary", {}).get("median")
        if isinstance(y, (int, float)) and np.isfinite(y):
            ax.axhline(
                max(float(y), _RESID_FLOOR),
                color=style["color"],
                linestyle="--",
                linewidth=1.6,
                alpha=0.85,
                zorder=5,
                label=f"{style['label']} (ref.)",
            )

    ax.set_yscale("log")
    if xscale == "log":
        ax.set_xscale("log")
    ax.margins(x=0.08)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"final $f(x_K) - f^{\star}$")
    if title:
        ax.set_title(title)


def _plot_rate_loglog(
    ax: Any,
    methods: dict[str, dict[str, Any]],
    base_dir: Path,
    title: str | None,
    nu_ref: float | None = None,
) -> None:
    """Log-log residual vs iteration with the fitted slope annotated."""
    anchor: tuple[float, float] | None = None
    slope_lines: list[tuple[str, str, float]] = []
    for method_key in sorted(methods):
        rec = methods[method_key]
        curve = rec.get("iter_curve")
        fit = rec.get("rate_fit", {})
        if not curve:
            continue
        style = _style_for(method_key, rec, base_dir)
        x = np.asarray(curve["x"], dtype=float)
        med = _floor_log(np.asarray(curve["median"], dtype=float))
        slope = fit.get("slope")
        r2 = fit.get("r_squared")
        slope_is_meaningful = (
            slope is not None
            and np.isfinite(slope)
            and r2 is not None
            and r2 > 0.9
            and abs(slope) > 0.1
            and abs(slope) < 10
        )
        if slope_is_meaningful:
            slope_lines.append((style["label"], style["color"], float(slope)))
        ax.loglog(
            x[1:],
            med[1:],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markevery=_markevery(len(x) - 1),
            markersize=5.5,
            markeredgewidth=0.0,
            zorder=style["zorder"],
        )
        if anchor is None and method_key.startswith("agns_theory") and len(x) > 2:
            for k_idx in range(1, len(med)):
                if med[k_idx] > 1e-12 and np.isfinite(med[k_idx]):
                    anchor = (float(x[k_idx]), float(med[k_idx]))
                    break

    if nu_ref is not None and anchor is not None:
        K_anchor, f_anchor = anchor
        exponent = 2.0 + float(nu_ref)
        C = f_anchor * (K_anchor**exponent)
        K_max = max(
            (
                np.asarray(methods[m]["iter_curve"]["x"], dtype=float).max()
                for m in methods
                if methods[m].get("iter_curve")
            ),
            default=K_anchor,
        )
        K_ref = np.geomspace(K_anchor, K_max, 64)
        f_ref = C / K_ref**exponent
        ax.loglog(
            K_ref,
            f_ref,
            color="0.4",
            linestyle=":",
            linewidth=1.2,
            label=rf"theory: $C/K^{{{exponent:.2f}}}$",
        )

    if slope_lines:
        slope_lines.sort(key=lambda t: t[2])
        for i, (lbl, col, sl) in enumerate(slope_lines[:6]):
            short = _short_label(lbl)
            ax.text(
                0.02,
                0.04 + 0.05 * (len(slope_lines[:6]) - 1 - i),
                rf"{short}: slope ${sl:+.2f}$",
                transform=ax.transAxes,
                fontsize=10,
                color=col,
                ha="left",
                va="bottom",
                zorder=20,
            )

    ax.set_xlabel(r"Iteration $k$")
    ax.set_ylabel(r"$f(x_k) - f^{\star}$")
    if title:
        ax.set_title(title)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _save_pair(fig: Any, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    metadata = {"CreationDate": None, "ModDate": None, "Producer": None}
    fig.savefig(pdf_path, metadata=metadata)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  [figure] {pdf_path}")


def _make_fig(rc_override: dict[str, Any] | None = None) -> tuple[Any, Any]:
    plt.rcParams.update(_PAPER_RC)
    if rc_override:
        plt.rcParams.update(rc_override)
    fig, ax = plt.subplots()
    return fig, ax


# ---------------------------------------------------------------------------
# Report figures -- cross-campaign, written to results/paper_figures/.
# ---------------------------------------------------------------------------

# rho value -> (history-key suffix, colour, legend label).  Two stable
# settings (0.30, 0.85) bracket a divergent one (0.50); 1/sqrt(2) is the
# prefactor-optimal default.
_RHO_CASES: list[tuple[str, str, str]] = [
    ("0p3", "#0077BB", r"$\rho=0.30$"),
    ("0p5", "#CC3311", r"$\rho=0.50$"),
    ("0p707107", "#EE7733", r"$\rho=1/\sqrt{2}$"),
    ("0p85", "#009988", r"$\rho=0.85$"),
]


def _forward_fill_median(seeds: list[np.ndarray]) -> np.ndarray:
    """Median over seeds at each index over the full horizon.

    Seeds stop at different iterations -- a converged seed halts early, a
    divergent one runs the whole budget -- so each trace is forward-filled
    with its last value to the longest length before the median is taken.
    """
    horizon = max(s.size for s in seeds)
    rows: list[np.ndarray] = []
    for s in seeds:
        if s.size < horizon:
            s = np.concatenate([s, np.full(horizon - s.size, s[-1])])
        rows.append(s)
    return np.median(np.vstack(rows), axis=0)


def _render_picard_diagnostic(raw_dir: Path, agg_path: Path, out_dir: Path) -> None:
    """AGNS-Theory outer-iteration diagnostic.

    The median suboptimality (upper panel) and estimating-sequence weight
    ``A_k`` (lower panel) vs the outer iteration, for a divergent and two
    stable values of ``rho`` -- evidence for the late-stage overshoot of
    the non-contractive Picard coupling.
    """
    agg = json.loads(agg_path.read_text())
    per_seed_f_star: dict[str, float] = agg.get("per_seed_f_star", {})
    seed_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())

    plt.rcParams.update(_PAPER_RC)
    fig, (ax_res, ax_a) = plt.subplots(2, 1, sharex=True, figsize=(6.0, 5.4))
    fig.subplots_adjust(hspace=0.12, left=0.16, right=0.97, top=0.86, bottom=0.10)

    for suffix, colour, label in _RHO_CASES:
        res_seeds: list[np.ndarray] = []
        a_seeds: list[np.ndarray] = []
        for sd in seed_dirs:
            pkl = sd / f"agns_theory__rho={suffix}_history.pkl"
            if not pkl.is_file() or sd.name not in per_seed_f_star:
                continue
            hist = load_pickle(pkl)
            func = np.asarray(hist["func"], dtype=float)
            res_seeds.append(np.maximum(func - per_seed_f_star[sd.name], _RESID_FLOOR))
            a_seeds.append(np.maximum(np.asarray(hist["A_k"], dtype=float), 1e-12))
        if not res_seeds:
            continue
        res = _forward_fill_median(res_seeds)
        a_k = _forward_fill_median(a_seeds)
        ax_res.semilogy(np.arange(res.size), res, color=colour, linewidth=2.2, label=label)
        ax_a.semilogy(np.arange(a_k.size), a_k, color=colour, linewidth=2.2)

    ax_res.set_ylabel(r"$f(x_k) - f^{\star}$")
    ax_a.set_ylabel(r"weight $A_k$")
    ax_a.set_xlabel(r"Outer iteration $k$")
    for ax in (ax_res, ax_a):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_res.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        borderaxespad=0.0,
        fontsize=11,
        columnspacing=1.2,
        handlelength=1.6,
        handletextpad=0.4,
    )
    _save_pair(fig, out_dir, "picard_diagnostic")


def _render_scaling(agg_paths: list[Path], out_dir: Path) -> None:
    """Median wall-clock to reach ``f - f* <= 1e-8`` vs the problem size n."""
    target = 1e-8
    points: dict[str, list[tuple[int, float]]] = {}
    for path in agg_paths:
        agg = json.loads(path.read_text())
        n = int(agg["problem"]["params"]["n"])
        for key, rec in agg.get("methods", {}).items():
            resid = rec.get("iter_curve", {}).get("median", [])
            times = rec.get("time_curve", {}).get("median", [])
            for i, v in enumerate(resid):
                if v <= target and i < len(times):
                    t = float(times[i])
                    if t > 0.0:
                        points.setdefault(key, []).append((n, t))
                    break

    fig, ax = _make_fig({"figure.figsize": (6.0, 3.7)})
    for key in sorted(points):
        series = sorted(points[key])
        if len(series) < 2:
            continue
        ns = [n for n, _ in series]
        ts = [t for _, t in series]
        color, linestyle, marker, lw, z = _resolve_style(key)
        ax.loglog(
            ns,
            ts,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=lw,
            markersize=7,
            markeredgewidth=0.0,
            zorder=z,
            label=_LEGEND_ABBREV.get(key, key),
        )
    ax.set_xlabel(r"Problem size $n$")
    ax.set_ylabel(r"time to $f-f^{\star}\leq 10^{-8}$ (s)")
    ticks = sorted({n for s in points.values() for n, _ in s})
    if ticks:
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    _legend_below(ax)
    _save_pair(fig, out_dir, "scaling_time")


def _run_report_figures(results_dir: Path) -> None:
    """Render the cross-campaign report figures into ``paper_figures/``."""
    agg = results_dir / "numerical" / "aggregated"
    raw = results_dir / "numerical" / "raw"
    out = results_dir / "paper_figures"

    rd_agg = agg / "ablation_rho_diagnostic.json"
    rd_raw = raw / "ablation_rho_diagnostic"
    if rd_agg.is_file() and rd_raw.is_dir():
        _render_picard_diagnostic(rd_raw, rd_agg, out)
    else:
        print("  [figure] WARN: ablation_rho_diagnostic data missing; skipping picard_diagnostic")

    scaling = [agg / f"{c}.json" for c in ("scaling_lr", "scaling_lr_n200", "scaling_lr_n800")]
    present = [p for p in scaling if p.is_file()]
    if len(present) >= 2:
        _render_scaling(present, out)
    else:
        print("  [figure] WARN: scaling campaigns missing; skipping scaling_time")


def main() -> None:
    args = parse_args()

    if args.report_figures:
        _run_report_figures(Path(args.results_dir).resolve())
        return

    if not args.aggregated or not args.output_dir:
        sys.exit(
            "per-campaign mode requires --aggregated and --output-dir "
            "(or pass --report-figures for the cross-campaign figures)"
        )
    agg_path = Path(args.aggregated).resolve()
    if not agg_path.is_file():
        sys.exit(f"aggregated JSON not found: {agg_path}")
    out_dir = Path(args.output_dir).resolve()
    base_dir = Path(args.base_methods_dir).resolve()

    with open(agg_path) as fh:
        agg = json.load(fh)

    methods: dict[str, dict[str, Any]] = dict(agg.get("methods", {}))
    if args.methods is not None:
        methods = {k: v for k, v in methods.items() if k in args.methods}
    if not methods:
        sys.exit("no methods to plot (filter excluded everything)")

    title = None if args.no_title else args.title
    rc_override: dict[str, Any] = {}
    if args.figsize:
        rc_override["figure.figsize"] = (float(args.figsize[0]), float(args.figsize[1]))

    # (x_source, x-axis label, x-scale).  ``x_source`` selects the
    # abscissa; the ordinate is always the residual (see ``_plot_curves``).
    type_to_handler = {
        "iter_convergence": ("iteration", r"Iteration $k$", "log"),
        "time_convergence": ("time_curve", r"Wall-clock time (s)", "log"),
        "grad_convergence": ("grad_curve", r"Gradient oracle calls", "log"),
    }

    show_legend = not args.no_legend
    for ftype in args.types:
        if ftype in type_to_handler:
            x_source, xlabel, xscale = type_to_handler[ftype]
            fig, ax = _make_fig(rc_override)
            _plot_curves(ax, methods, base_dir, x_source, xlabel, title, xscale=xscale)
            if show_legend:
                _legend_below(ax)
            _save_pair(fig, out_dir, ftype)
        elif ftype == "rate_loglog":
            fig, ax = _make_fig(rc_override)
            _plot_rate_loglog(ax, methods, base_dir, title, nu_ref=args.nu_ref)
            if show_legend:
                _legend_below(ax)
            _save_pair(fig, out_dir, "rate_loglog")
        elif ftype == "sweep":
            has_sweep = any(_parse_sweep_key(k) is not None for k in methods)
            if has_sweep:
                fig, ax = _make_fig(rc_override)
                _plot_sweep(ax, methods, base_dir, title)
                if show_legend:
                    _legend_below(ax)
                _save_pair(fig, out_dir, "sweep")
        else:
            print(f"  [figure] WARN: unknown figure type {ftype!r}, skipping")


if __name__ == "__main__":
    main()
