#!/usr/bin/env python3
"""
main.py — Optimization benchmark runner.

Usage:
    python main.py --config config/logsumexp_synthetic.yaml
    python main.py --config config/chebyshev.yaml --output-dir results/my_run
    python main.py --config config/nonlinear_equations.yaml --methods gns_exact agns_lookahead
    python main.py --config config/rosenbrock_nle.yaml --no-plot --seed 99

Each YAML config describes one experimental setup: the problem, the set of
methods to compare, and the output settings.  main.py loads the problem,
runs every listed method, saves per-method history pickles + a JSON summary,
and generates convergence plots.
"""

import argparse
import json
import os
import pickle
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")       # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Ensure src/ is on the path regardless of where main.py is called from
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from datasets import DATASET_REGISTRY
from models import METHOD_REGISTRY, run_method
from utils import plot_only_iterations, plot_only_time, plot_only_operations


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run optimization benchmarks from a YAML config."
    )
    p.add_argument("--config", required=True, help="Path to experiment YAML config.")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override the save_dir specified in the config.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed specified in the problem params.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="Run only this subset of methods (by key, e.g. gns_exact agns_lookahead).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation (useful for automated sweeps).",
    )
    p.add_argument(
        "--warnings",
        action="store_true",
        help="Enable verbose runtime warnings from the methods.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_PLOT_STYLE = {
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "lines.linewidth": 2,
}


def _resolve_f_star(histories, f_star_known):
    """Return the best known lower bound on f*."""
    if f_star_known is not None:
        return f_star_known
    best = min(min(h["func"]) for h in histories if h)
    return best


def _make_plot(
    ax,
    histories,
    x_data_key,
    labels,
    colors,
    linestyles,
    linewidths,
    f_star,
    threshold,
    max_pts,
    xlabel,
):
    for hist, lbl, col, ls, lw in zip(
        histories, labels, colors, linestyles, linewidths
    ):
        func = np.array(hist["func"])
        resid = func - f_star
        resid = np.clip(resid, 1e-20, None)   # avoid log(0)

        n_total = len(resid)
        cutoff = np.searchsorted(-resid, -threshold) + 1
        n_pts = min(n_total, cutoff, max_pts)

        if x_data_key == "iterations":
            x = np.arange(n_pts)
        else:
            raw = np.array(hist[x_data_key])
            x = raw[:n_pts]

        ax.semilogy(x, resid[:n_pts], label=lbl, color=col, linestyle=ls, linewidth=lw)

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(r"$f(x_k) - f^\star$", fontsize=18)
    ax.legend(fontsize=14)
    ax.grid(True)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_experiment(cfg, args):
    problem_cfg = cfg["problem"]
    methods_list = cfg["methods"]
    common_cfg = cfg.get("common", {})
    output_cfg = cfg.get("output", {})

    # --- resolve output directory -------------------------------------------
    save_dir = args.output_dir or output_cfg.get("save_dir", "results/default")
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- build problem instance ---------------------------------------------
    problem_type = problem_cfg["type"]
    problem_params = dict(problem_cfg.get("params", {}))

    if args.seed is not None:
        problem_params["seed"] = args.seed

    if problem_type not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown problem type '{problem_type}'. "
            f"Available: {sorted(DATASET_REGISTRY)}"
        )

    print(f"\n{'='*60}")
    print(f"  Problem : {problem_type}")
    print(f"  Params  : {problem_params}")
    print(f"  Output  : {save_dir}")
    print(f"{'='*60}\n")

    problem = DATASET_REGISTRY[problem_type](**problem_params)

    oracle = problem["oracle"]
    x_0 = problem["x_0"]
    f_star_known = problem.get("f_star")
    B = problem.get("B")
    Binv = problem.get("Binv")
    approx_hess_fn_standard = problem.get("approx_hess_fn")
    approx_hess_fn_wsm = problem.get("approx_hess_fn_wsm")

    # --- filter methods list ------------------------------------------------
    if args.methods is not None:
        methods_list = [m for m in methods_list if m["name"] in args.methods]
        if not methods_list:
            raise ValueError(
                f"None of the requested methods {args.methods} found in config."
            )

    # --- run each method ----------------------------------------------------
    histories, labels, colors, linestyles, linewidths = [], [], [], [], []
    summary = {}

    for mcfg in methods_list:
        mkey = mcfg["name"]
        mlabel = mcfg.get("label", mkey)

        if mkey not in METHOD_REGISTRY:
            print(f"  [!] Unknown method '{mkey}' — skipping.")
            continue

        spec = METHOD_REGISTRY[mkey]

        # pick approx_hess_fn based on whether this method uses WSM
        if spec.uses_wsm:
            ahfn = approx_hess_fn_wsm
        else:
            ahfn = approx_hess_fn_standard

        # per-method overrides merge on top of common config
        method_common = {**common_cfg}
        method_common.update({
            k: v
            for k, v in mcfg.items()
            if k not in ("name", "label", "color", "linestyle", "linewidth")
        })
        method_common["B"] = B
        method_common["Binv"] = Binv
        method_common["f_star"] = f_star_known

        if args.warnings:
            method_common["warnings"] = True

        print(f"  [{mkey}] {mlabel} ...", end="", flush=True)
        t0 = time.perf_counter()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_final, status, history = run_method(
                    spec, oracle, x_0, method_common, oracle, ahfn
                )
        except Exception as exc:
            print(f" FAILED ({exc})")
            continue

        elapsed = time.perf_counter() - t0
        n_recorded = len(history["func"]) - 1
        f_final = float(history["func"][-1])
        grad_calls = int(history.get("grad_calls", [0])[-1])
        matrix_inv = int(history.get("matrix_inverses", [0])[-1])

        print(
            f" done | {status} | {elapsed:.3f}s | "
            f"f={f_final:.4e} | grads={grad_calls} | inv={matrix_inv}"
        )

        # record
        histories.append(history)
        labels.append(mlabel)
        colors.append(mcfg.get("color", "black"))
        linestyles.append(mcfg.get("linestyle", "-"))
        linewidths.append(float(mcfg.get("linewidth", 2.0)))

        summary[mkey] = {
            "label": mlabel,
            "status": status,
            "wall_time_s": elapsed,
            "n_iters": n_recorded,
            "f_final": f_final,
            "grad_calls": grad_calls,
            "matrix_inverses": matrix_inv,
        }

        # save per-method history
        hist_path = os.path.join(save_dir, f"{mkey}_history.pkl")
        with open(hist_path, "wb") as fh:
            pickle.dump(history, fh, protocol=4)

    if not histories:
        print("No methods ran successfully — nothing to save.")
        return

    # --- resolve f_star for plotting ----------------------------------------
    f_star = _resolve_f_star(histories, f_star_known)

    # --- save JSON summary --------------------------------------------------
    summary_data = {
        "problem": problem_cfg,
        "f_star": float(f_star),
        "methods": summary,
    }
    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary_data, fh, indent=2)
    print(f"\n  [summary] {summary_path}")

    # --- plots --------------------------------------------------------------
    if args.no_plot:
        return

    threshold = common_cfg.get("eps", 1e-8)
    max_pts = max(
        m.get("n_iters", common_cfg.get("n_iters", 1000)) for m in methods_list
    )
    title = output_cfg.get("title", problem_type.replace("_", " ").title())

    plt.rcParams.update(_PLOT_STYLE)

    # iterations plot
    if output_cfg.get("plot_iterations", True):
        fig, ax = plt.subplots()
        _make_plot(
            ax, histories, "iterations", labels, colors, linestyles, linewidths,
            f_star, threshold, max_pts, "Iterations"
        )
        ax.set_title(title)
        save_figure(fig, os.path.join(plots_dir, "iterations.png"))

    # wall-clock time plot
    if output_cfg.get("plot_time", True):
        fig, ax = plt.subplots()
        _make_plot(
            ax, histories, "time", labels, colors, linestyles, linewidths,
            f_star, threshold, max_pts, "Wall-clock time (s)"
        )
        ax.set_title(title)
        save_figure(fig, os.path.join(plots_dir, "time.png"))

    # gradient oracle calls plot
    if output_cfg.get("plot_grad_calls", True):
        grad_capable = [
            (h, l, c, ls, lw)
            for h, l, c, ls, lw in zip(histories, labels, colors, linestyles, linewidths)
            if "grad_calls" in h
        ]
        if grad_capable:
            hs, ls_, cs, lss, lws = zip(*grad_capable)
            fig, ax = plt.subplots()
            _make_plot(
                ax, hs, "grad_calls", ls_, cs, lss, lws,
                f_star, threshold, max_pts * 3, "Gradient oracle calls"
            )
            ax.set_title(title)
            save_figure(fig, os.path.join(plots_dir, "grad_calls.png"))

    # matrix inversion count plot
    if output_cfg.get("plot_matrix_inverses", False):
        inv_capable = [
            (h, l, c, ls, lw)
            for h, l, c, ls, lw in zip(histories, labels, colors, linestyles, linewidths)
            if "matrix_inverses" in h and max(h["matrix_inverses"]) > 0
        ]
        if inv_capable:
            hs, ls_, cs, lss, lws = zip(*inv_capable)
            fig, ax = plt.subplots()
            _make_plot(
                ax, hs, "matrix_inverses", ls_, cs, lss, lws,
                f_star, threshold, max_pts * 3, "Matrix inversions"
            )
            ax.set_title(title)
            save_figure(fig, os.path.join(plots_dir, "matrix_inverses.png"))

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"Config file not found: {args.config}")

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    run_experiment(cfg, args)


if __name__ == "__main__":
    main()
