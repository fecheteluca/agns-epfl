"""Run one benchmark campaign for one or more seeds.

Loads a YAML config (via :func:`agns.pipeline.config.load_config`), instantiates
the problem and every method through :mod:`agns.pipeline.registry`, runs each
method, and writes raw per-method history pickles plus a
``summary.json`` to ``<save_dir>/seed_<i>/``.

The default output root is ``results/numerical/raw/<campaign>``, where
``<campaign>`` is derived from the config file name.  Override with
``--output-dir``.

Plotting is not performed here; use :mod:`agns.cli.make_plots` on the
aggregated JSON.

Usage examples::

    python -m agns.cli.run_benchmark --config configs/rate_verification/logsumexp.yaml
    python -m agns.cli.run_benchmark --config <cfg> --seeds 0 1 2
    python -m agns.cli.run_benchmark --config <cfg> --methods gns_exact agns_inexact
"""

from __future__ import annotations

import os

# Pin BLAS / OpenMP threads BEFORE importing numpy / scipy.  Threading
# libraries cache the thread count at import time, so setting these env
# vars later has no effect.  Pinning to 1 gives byte-stable reductions
# across re-runs.
for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

from agns.pipeline.config import load_config
from agns.pipeline.registry import (
    METHOD_REGISTRY,
    PROBLEM_REGISTRY,
    resolve_method_name,
    run_method,
)
from agns.pipeline.sweeps import expand_sweeps
from agns.utils.io import save_json, save_pickle
from agns.utils.seed import set_global_seed

#: Schema version embedded in every ``<mkey>_failure.json`` file.  Bump
#: when the on-disk layout changes; downstream consumers can branch on it.
_FAILURE_JSON_SCHEMA_VERSION = 1

__all__ = ["main", "run_experiment"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run an optimisation-benchmark campaign from a YAML config.",
    )
    p.add_argument("--config", required=True, help="Path to a campaign YAML.")
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Override the save_dir from the config.  When omitted, the "
            "output goes to results/numerical/raw/<campaign>."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single-seed run.  Mutually exclusive with --seeds.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help=("Multi-seed campaign: one sub-directory 'seed_<i>/' per seed."),
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="Run only this subset of methods (by registry key).",
    )
    p.add_argument(
        "--warnings",
        action="store_true",
        help="Enable verbose runtime warnings from the methods.",
    )
    return p.parse_args()


def _resolve_save_dir(
    cli_output_dir: str | None,
    cfg: dict[str, Any],
    config_path: Path,
) -> Path:
    """Pick the campaign output directory.

    Precedence: ``--output-dir`` > ``cfg.output.save_dir`` >
    ``results/numerical/raw/<config-stem>``.
    """
    if cli_output_dir is not None:
        return Path(cli_output_dir)
    save_dir = cfg.get("output", {}).get("save_dir")
    if save_dir:
        return Path(save_dir)
    return Path("results/numerical/raw") / config_path.stem


def run_experiment(cfg: dict[str, Any], args: argparse.Namespace, save_dir: Path) -> None:
    """Run every method for a single (seed, save_dir) combination."""
    problem_cfg = cfg["problem"]
    methods_list = cfg["methods"]
    common_cfg = cfg.get("common", {})

    save_dir.mkdir(parents=True, exist_ok=True)

    problem_type = problem_cfg["type"]
    problem_params = dict(problem_cfg.get("params", {}))
    if args.seed is not None:
        problem_params["seed"] = args.seed
        # Pin the global NumPy / Python / torch RNG state so any method
        # that draws random numbers (e.g. adahessian Hutchinson probes,
        # torch oracle init) does so deterministically per-seed.
        set_global_seed(int(args.seed))

    if problem_type not in PROBLEM_REGISTRY:
        raise ValueError(
            f"Unknown problem type {problem_type!r}.  Available: {sorted(PROBLEM_REGISTRY)}"
        )

    print(f"\n{'=' * 60}")
    print(f"  Problem : {problem_type}")
    print(f"  Params  : {problem_params}")
    print(f"  Output  : {save_dir}")
    print(f"{'=' * 60}\n")

    problem = PROBLEM_REGISTRY[problem_type](**problem_params)

    oracle = problem["oracle"]
    x_0 = problem["x_0"]
    f_star = problem.get("f_star")
    B = problem.get("B")
    Binv = problem.get("Binv")
    approx_hess_fn_standard = problem.get("approx_hess_fn")
    approx_hess_fn_wsm = problem.get("approx_hess_fn_wsm")

    selected_methods = methods_list
    if args.methods is not None:
        wanted = set(args.methods)
        selected_methods = [m for m in methods_list if m["name"] in wanted]
        if not selected_methods:
            raise ValueError(
                f"None of the requested methods {sorted(wanted)} appear in the config."
            )

    # Up-front configuration check: a WSM (Woodbury rank-1) method needs the
    # problem to supply ``approx_hess_fn_wsm``.  Catch the mismatch here --
    # loudly, before any method runs -- rather than letting it surface as a
    # mid-run failure swallowed by the per-method guard below.
    for mcfg in selected_methods:
        mname = mcfg["name"]
        try:
            canonical = resolve_method_name(mname)
        except KeyError:
            continue  # the per-method loop below prints "Unknown method"
        spec = METHOD_REGISTRY[canonical]
        if spec.uses_wsm and approx_hess_fn_wsm is None:
            raise ValueError(
                f"method {mname!r} uses the WSM (Woodbury rank-1) backend, but "
                f"problem {problem_type!r} provides no 'approx_hess_fn_wsm'. "
                f"The WSM backend is only valid for problems with a rank-1 "
                f"Fisher Hessian (e.g. nonlinear_equations)."
            )

    summary: dict[str, Any] = {}
    failures: dict[str, Any] = {}

    for mcfg in selected_methods:
        mname = mcfg["name"]
        # ``key`` allows the same method to appear with multiple
        # hyperparameter settings without pickle collisions; defaults
        # to ``name``.
        mkey = mcfg.get("key", mname)
        mlabel = mcfg.get("label", mkey)

        try:
            canonical_mname = resolve_method_name(mname)
        except KeyError:
            print(f"  [!] Unknown method {mname!r} -- skipping.")
            continue

        spec = METHOD_REGISTRY[canonical_mname]
        ahfn = approx_hess_fn_wsm if spec.uses_wsm else approx_hess_fn_standard

        method_common = dict(common_cfg)
        method_common.update(
            {
                k: v
                for k, v in mcfg.items()
                if k not in {"name", "key", "label", "color", "linestyle", "linewidth"}
            }
        )
        method_common["B"] = B
        method_common["Binv"] = Binv
        method_common["f_star"] = f_star
        if args.warnings:
            method_common["warnings"] = True
        # Thread the run seed into methods whose registry default kwargs
        # declare a ``seed`` knob (currently only AdaHessian's
        # Hutchinson probes).  Without this the method's RNG is pinned
        # at its registry default and across-seed variance collapses
        # to data variance only.  Gated on ``spec.default_kw`` so we
        # do not forward ``seed`` to methods whose run_fn would raise
        # on the unexpected kwarg.  The mcfg/common precedence still
        # wins: a YAML ``seed: 42`` override beats this plumbing.
        if (
            args.seed is not None
            and "seed" in spec.default_kw
            and "seed" not in method_common
        ):
            method_common["seed"] = int(args.seed)

        print(f"  [{mkey}] {mlabel} ...", end="", flush=True)
        t0 = time.perf_counter()

        try:
            with warnings.catch_warnings():
                # Default: silence Python warnings.  ``--warnings`` opts in
                # to the full Python warning stream in addition to the
                # method-internal verbose mode (already passed via
                # ``method_common["warnings"]``).
                if not args.warnings:
                    warnings.simplefilter("ignore")
                _x_final, status, history = run_method(
                    spec,
                    oracle,
                    x_0,
                    method_common,
                    oracle,
                    ahfn,
                )
        except Exception as exc:
            # A bare ``except`` that prints and continues would collapse
            # every failure into a missing pickle, which downstream
            # becomes a ``--`` cell indistinguishable from "method not
            # configured".  Persist a structured failure record alongside
            # where the history pickle would have lived; the pickle is
            # deliberately NOT written so the aggregator's
            # "pickle exists <-> method succeeded" invariant holds.
            elapsed = time.perf_counter() - t0
            exc_type = type(exc).__name__
            tb_text = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            failure_record = {
                "method_key": mkey,
                "method_label": mlabel,
                "exception_type": exc_type,
                "message": str(exc),
                "traceback": tb_text,
                "wall_time_s_before_failure": float(elapsed),
                "schema_version": _FAILURE_JSON_SCHEMA_VERSION,
            }
            save_json(failure_record, save_dir / f"{mkey}_failure.json")
            # Slim per-method copy in summary.json so a single look at the
            # seed directory's summary tells the operator what failed without
            # opening every *_failure.json.
            failures[mkey] = {
                "label": mlabel,
                "exception_type": exc_type,
                "message": str(exc),
                "wall_time_s_before_failure": float(elapsed),
            }
            print(f" FAILED ({exc_type}: {exc})")
            continue

        elapsed = time.perf_counter() - t0
        if history is None:
            print(f" done | {status} | {elapsed:.3f}s | (no trace)")
            continue
        n_recorded = len(history["func"]) - 1
        f_final = float(history["func"][-1])
        grad_calls = int(history.get("grad_calls", [0])[-1] or 0)
        matrix_inv = int(history.get("matrix_inverses", [0])[-1] or 0)

        print(
            f" done | {status} | {elapsed:.3f}s | "
            f"f={f_final:.4e} | grads={grad_calls} | inv={matrix_inv}"
        )

        summary[mkey] = {
            "label": mlabel,
            "status": status,
            "wall_time_s": elapsed,
            "n_iters": n_recorded,
            "f_final": f_final,
            "grad_calls": grad_calls,
            "matrix_inverses": matrix_inv,
        }

        save_pickle(history, save_dir / f"{mkey}_history.pkl")

    if not summary and not failures:
        print("No methods ran -- nothing to save.")
        return
    if not summary:
        # Every method failed: still write summary.json so the operator
        # has one canonical place to look, and so the aggregator can
        # discover the per-seed failures.  Without this branch we used
        # to silently abandon the seed.
        print("No methods succeeded; writing summary.json with failures only.")

    # Write summary.json with the problem-declared f_star verbatim
    # (None if the factory did not declare one).  Filling None with
    # the per-seed minimum across methods would conflate "declared by
    # the problem" with "best a method happened to achieve here", and
    # the aggregator's declared-f_star precedence would then pick the
    # latter up as if it were authoritative.  The residual anchor must
    # come from either the problem (declared) or the cached reference
    # solution -- never from the methods being benchmarked.
    declared_f_star = float(f_star) if f_star is not None else None

    summary_data = {
        "problem": problem_cfg,
        "f_star": declared_f_star,
        "methods": summary,
        "failures": failures,
    }
    save_json(summary_data, save_dir / "summary.json")
    print(f"\n  [summary] {save_dir / 'summary.json'}")


def main() -> None:
    args = parse_args()

    if args.seed is not None and args.seeds is not None:
        sys.exit("--seed and --seeds are mutually exclusive")

    config_path = Path(args.config)
    if not config_path.is_file():
        sys.exit(f"Config file not found: {config_path}")

    cfg = load_config(config_path)
    cfg = expand_sweeps(cfg)
    save_dir = _resolve_save_dir(args.output_dir, cfg, config_path)

    if args.seeds is None:
        run_experiment(cfg, args, save_dir)
        return

    base_output = save_dir.resolve()
    print(f"\n=== Multi-seed campaign: {len(args.seeds)} seeds -> {base_output} ===")
    for s in args.seeds:
        seed_dir = base_output / f"seed_{s}"
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = s
        seed_args.seeds = None
        seed_args.output_dir = str(seed_dir)
        run_experiment(cfg, seed_args, seed_dir)


if __name__ == "__main__":
    main()
