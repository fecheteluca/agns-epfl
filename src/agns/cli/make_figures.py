"""Render every campaign figure from an aggregated JSON.

Input: a single aggregated-results JSON written by
:mod:`agns.cli.aggregate`.  Output: one PDF (vector) and one PNG
(preview) per figure type, written under
``results/figures/<campaign>/``.

Figure types (each subject to ``--types`` filtering):

* ``iter_convergence`` --- median residual + IQR vs iteration.  Log-y.
* ``time_convergence`` --- median residual + IQR vs wall-clock time.
* ``grad_convergence`` --- median residual + IQR vs gradient calls.
* ``rate_loglog``      --- log-log residual vs iteration with the
                           fitted slope annotated.
* ``sweep``            --- final-residual median+IQR vs the swept
                           hyperparameter (ablation campaigns only).

Visual conventions: figure size 6 x 3.5 in, legend below the axes,
fixed colour-blind-safe palette grouped by method family.  Each method
gets a unique ``(colour, linestyle, marker)`` triple so plots remain
unambiguous in colour, in greyscale, and for colour-blind readers.  PDF
metadata is sanitised so re-running on the same input emits byte-
comparable PDFs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agns.config import ConfigError, load_config

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
    # TeX is off by default: CI machines rarely have a LaTeX install.
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
    # AGNS-Theory (medium green).
    "agns_theory": ("#44AA77", "-", "^", 2.6, 12),
    "agns_holder": ("#44AA77", "--", "v", 2.4, 11),
    "agns_inexact": ("#44AA77", ":", "P", 2.4, 11),
    "agns_sc": ("#44AA77", "-.", "p", 2.4, 11),
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
        return ("#44AA77", "-", "P", 2.4, 12)
    t = (np.log10(delta) + 3.0) / 4.0  # delta=1e-3 -> 0, delta=10 -> 1
    t = float(np.clip(t, 0.0, 1.0))
    cmap = plt.get_cmap("viridis")
    color = matplotlib.colors.to_hex(cmap(t))
    markers = ["o", "s", "D", "^", "P", "X"]
    idx = round(t * (len(markers) - 1))
    return (color, "-", markers[idx], 2.0, 8 + idx)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--aggregated", required=True, help="Path to <campaign>.json")
    p.add_argument("--output-dir", required=True, help="Directory to write figures into.")
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
    label: str | None = rec.get("label")
    if label is None:
        base_yaml = base_dir / f"{method_key}.yaml"
        if base_yaml.is_file():
            try:
                base = load_config(base_yaml)
                yaml_label = base.get("label")
                if yaml_label is not None:
                    label = str(yaml_label)
            except ConfigError:
                pass
    return {
        "color": color,
        "linestyle": linestyle,
        "marker": marker,
        "linewidth": linewidth,
        "zorder": zorder,
        "label": label or method_key,
    }


# ---------------------------------------------------------------------------
# Plot primitives
# ---------------------------------------------------------------------------


def _floor_log(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, _RESID_FLOOR)  # type: ignore[no-any-return]


def _legend_below(ax: Any, *, ncol: int | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    n = len(handles)
    if ncol is None:
        if n <= 3:
            ncol = n
        elif n <= 6:
            ncol = 3
        else:
            ncol = 4
    ax.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=ncol,
        borderaxespad=0.0,
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
    curve_key: str,
    xlabel: str,
    title: str | None,
    *,
    show_iqr: bool = True,
    xscale: str = "linear",
) -> None:
    for method_key in sorted(methods):
        rec = methods[method_key]
        curve = rec.get(curve_key)
        if not curve:
            continue
        style = _style_for(method_key, rec, base_dir)
        x = np.asarray(curve["x"], dtype=float)
        med = _floor_log(np.asarray(curve["median"], dtype=float))
        if show_iqr:
            lo = _floor_log(np.asarray(curve["p25"], dtype=float))
            hi = _floor_log(np.asarray(curve["p75"], dtype=float))
            ax.fill_between(
                x, lo, hi,
                color=style["color"],
                alpha=0.13,
                linewidth=0,
                zorder=style["zorder"] - 1,
            )
        x_plot = x[1:] if xscale == "log" else x
        y_plot = med[1:] if xscale == "log" else med
        ax.plot(
            x_plot, y_plot,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markevery=_markevery(len(x_plot)),
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


def _plot_sweep(
    ax: Any,
    methods: dict[str, dict[str, Any]],
    base_dir: Path,
    title: str | None,
) -> None:
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
    for (base, param), items in sweep_groups.items():
        if all(isinstance(v, (int, float)) for v, _ in items):
            items.sort(key=lambda kv: kv[0])  # type: ignore[arg-type, return-value]
            xs_num = [float(v) for v, _ in items]
            xs_cat: list[str] = []
            categorical = False
        else:
            xs_num = []
            xs_cat = [str(v) for v, _ in items]
            categorical = True

        ys = [r["final_residual_summary"]["median"] for _, r in items]
        lo = [r["final_residual_summary"]["p25"] for _, r in items]
        hi = [r["final_residual_summary"]["p75"] for _, r in items]

        style = _style_for(base, items[0][1], base_dir)
        lab = _short_label(style["label"])
        marker_kwargs = {
            "color": style["color"],
            "linestyle": style["linestyle"],
            "linewidth": style["linewidth"],
            "marker": style["marker"],
            "markersize": 7,
            "markeredgewidth": 0.0,
            "zorder": style["zorder"],
            "label": lab,
        }
        yerr = [
            [max(m - lo_i, 0) for m, lo_i in zip(ys, lo, strict=True)],
            [max(hi_i - m, 0) for m, hi_i in zip(ys, hi, strict=True)],
        ]
        if categorical:
            ax.errorbar(range(len(xs_cat)), ys, yerr=yerr, **marker_kwargs)
            ax.set_xticks(range(len(xs_cat)))
            ax.set_xticklabels(xs_cat)
        else:
            ax.errorbar(xs_num, ys, yerr=yerr, **marker_kwargs)
        xlabel = _SWEEP_XLABEL.get(param, param.replace("_", " "))

    for ref_key, rec in references:
        style = _style_for(ref_key, rec, base_dir)
        y = rec.get("final_residual_summary", {}).get("median")
        if y is not None and np.isfinite(y):
            ax.axhline(
                y,
                color=style["color"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=f"{style['label']} (ref.)",
            )

    ax.set_yscale("log")
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
            K_ref, f_ref,
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


def main() -> None:
    args = parse_args()
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

    type_to_handler = {
        "iter_convergence": ("iter_curve", r"Iteration $k$", "log"),
        "time_convergence": ("time_curve", r"Wall-clock time (s)", "log"),
        "grad_convergence": ("grad_curve", r"Gradient oracle calls", "log"),
    }

    show_legend = not args.no_legend
    for ftype in args.types:
        if ftype in type_to_handler:
            curve_key, xlabel, xscale = type_to_handler[ftype]
            fig, ax = _make_fig(rc_override)
            _plot_curves(ax, methods, base_dir, curve_key, xlabel, title, xscale=xscale)
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
