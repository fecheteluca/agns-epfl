"""Compute and cache reference ``f_star`` values for every problem instance.

The aggregator (:mod:`agns.cli.aggregate`) prefers a cached reference
``f_star`` over the fallback "per-seed minimum across methods" rule.
This CLI populates that cache.

For every campaign YAML passed on the command line:
  1. Skip any problem whose factory already declares ``f_star`` (the
     declared value is authoritative and goes into ``summary.json`` at
     run time -- no cache entry needed).
  2. For every seed in ``--seeds``: build the problem instance, run
     ``newton`` with a tight tolerance and a large iteration budget.
     If Newton fails (``linalg_error`` / ``line_search_diverged``),
     fall back to ``fast_gradient``.  Take the minimum ``func``
     observed across the successful solvers as ``f_ref``.
  3. Persist a :class:`~agns.pipeline.reference_cache.ReferenceEntry`
     per (problem_type, params_hash, seed) under
     ``results/reference_solutions/``.
  4. Rewrite ``manifest.json`` at the end.

Usage::

    python -m agns.cli.compute_references \\
        --configs configs/real_world/lr_a9a.yaml configs/real_world/lr_w8a.yaml \\
        --seeds 0 1 2 3 4

    # Or via the console script registered in pyproject.toml:
    agns-references --configs <path> --seeds 0 1 2 3 4

    # All configs that point to a factory without a declared f_star:
    agns-references --all-configs --seeds 0 1 2 3 4

Non-convex problems (``matrix_completion_synthetic``,
``mlp_classifier_synthetic``) get the same treatment but are tagged
``local_minimum_warning=True`` in the entry so downstream consumers
know not to claim global optimality.
"""

from __future__ import annotations

import os

# BLAS pinning before any numpy import (mirrors agns.cli.run_benchmark).
# Without this the reference solver may produce ULP-different f_ref
# values across hosts / re-runs.
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
import datetime as _dt
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from agns.methods import fast_gradient, newton
from agns.oracles import PROBLEM_REGISTRY
from agns.pipeline.config import load_config
from agns.pipeline.reference_cache import (
    DEFAULT_REFERENCE_DIR,
    ReferenceEntry,
    canonical_params_hash,
    current_git_sha,
    rebuild_manifest,
    save_entry,
)
from agns.utils.seed import set_global_seed

__all__ = ["compute_one_reference", "main"]


# Non-convex problem types get ``local_minimum_warning=True`` on every entry.
_NONCONVEX_PROBLEM_TYPES: frozenset[str] = frozenset(
    {
        "matrix_completion_synthetic",
        "mlp_classifier_synthetic",
        "phase_retrieval_synthetic",
        "rosenbrock_2d",
        "rosenbrock_nle",
    }
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--configs",
        nargs="+",
        metavar="CFG",
        help="Campaign YAMLs to process (path relative to repo root or absolute).",
    )
    src.add_argument(
        "--all-configs",
        action="store_true",
        help="Process every YAML under configs/ that points to a no-f_star factory.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        metavar="SEED",
        help="Seeds to compute references for (default: 0..4).",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_REFERENCE_DIR),
        help="Cache root (default: results/reference_solutions).",
    )
    p.add_argument(
        "--newton-iters",
        type=int,
        default=10_000,
        help="Max iterations for the primary Newton solve.",
    )
    p.add_argument(
        "--fast-gradient-iters",
        type=int,
        default=50_000,
        help="Max iterations for the fast-gradient fallback solve.",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=1e-16,
        help="Function-gap tolerance passed to the reference solver.",
    )
    p.add_argument(
        "--grad-tol",
        type=float,
        default=1e-14,
        help="Gradient-norm tolerance passed to the reference solver.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if a cache entry already exists.",
    )
    p.add_argument(
        "--configs-root",
        default=None,
        help=(
            "Root directory containing the ``configs/`` tree to scan with "
            "``--all-configs``.  Defaults to the repository root inferred "
            "from this module's installation path."
        ),
    )
    return p.parse_args()


def _now_utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _final_func_and_grad_norm(history: dict[str, Any]) -> tuple[float, float | None]:
    """Pull the last ``f``/``||g||`` from a method history dict."""
    f_arr = np.asarray(history.get("func", []), dtype=np.float64)
    f_final = float(f_arr.min()) if f_arr.size else float("inf")
    g_arr = np.asarray(history.get("grad_norm", []), dtype=np.float64)
    g_final = float(g_arr[-1]) if g_arr.size else None
    return f_final, g_final


def compute_one_reference(
    problem_type: str,
    params: dict[str, Any],
    seed: int,
    *,
    newton_iters: int = 10_000,
    fast_gradient_iters: int = 50_000,
    eps: float = 1e-16,
    grad_tol: float = 1e-14,
) -> ReferenceEntry:
    """Run the reference solver(s) for one problem instance, return an entry.

    Tries Newton first; on failure (linalg / line-search) or if its
    final ``f`` is non-finite, falls back to fast_gradient.  Takes the
    minimum ``f`` observed across successful solvers as ``f_ref``.

    Exposed as a public function (not just a CLI helper) so tests and
    notebooks can compute a single reference without going through
    argparse.
    """
    full_params = dict(params)
    full_params["seed"] = int(seed)
    set_global_seed(int(seed))

    if problem_type not in PROBLEM_REGISTRY:
        raise ValueError(f"Unknown problem_type {problem_type!r}")

    factory = PROBLEM_REGISTRY[problem_type]
    problem = factory(**full_params)
    oracle = problem["oracle"]
    x_0 = problem["x_0"]
    B = problem.get("B")
    Binv = problem.get("Binv")

    common_kw: dict[str, Any] = {
        "n_iters": newton_iters,
        "eps": eps,
        "grad_tol": grad_tol,
        "trace": True,
        "B": B,
        "Binv": Binv,
        "f_star": None,  # don't let the stopping check short-circuit on residual
        "warnings": False,
    }

    candidates: list[tuple[str, float, float | None, int, str, float]] = []

    # --- Primary: Newton (PSD-regularised, Armijo backtracking). ---
    # Newton needs the dense ``n x n`` Hessian; on high-dim sparse problems
    # (rcv1: n=47k, real-sim: n=20k) the densification overflows host RAM
    # before any optimisation iteration runs.  Catch ``MemoryError``
    # (and the analogous numpy allocation failure) and treat it as a
    # silent "Newton not applicable here" --- the fallback solver below
    # will pick up the slack.  ``status_newton`` is set to a sentinel so
    # the fallback predicate fires unconditionally.
    t0 = time.perf_counter()
    status_newton: str = ""
    hist_newton: dict[str, Any] | None = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, status_newton, hist_newton = newton(oracle, x_0, **common_kw)
    except MemoryError as exc:
        status_newton = f"memory_error: {exc}"
        hist_newton = None
    dt_newton = time.perf_counter() - t0
    if hist_newton is not None:
        f_n, g_n = _final_func_and_grad_norm(hist_newton)
        if np.isfinite(f_n):
            n_iters_newton = max(len(hist_newton.get("func", [])) - 1, 0)
            candidates.append(("newton", f_n, g_n, n_iters_newton, status_newton, dt_newton))

    # --- Fallback: fast_gradient if Newton failed or didn't produce a finite f. ---
    needs_fallback = not candidates or status_newton in {
        "line_search_diverged",
        "linalg_error",
        "diverged_nonfinite",
    } or status_newton.startswith("memory_error")
    if needs_fallback:
        fg_kw = dict(common_kw)
        fg_kw["n_iters"] = fast_gradient_iters
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, status_fg, hist_fg = fast_gradient(oracle, x_0, **fg_kw)
        dt_fg = time.perf_counter() - t0
        if hist_fg is not None:
            f_fg, g_fg = _final_func_and_grad_norm(hist_fg)
            if np.isfinite(f_fg):
                n_iters_fg = max(len(hist_fg.get("func", [])) - 1, 0)
                candidates.append(("fast_gradient", f_fg, g_fg, n_iters_fg, status_fg, dt_fg))

    if not candidates:
        raise RuntimeError(
            f"every reference solver failed for {problem_type}(seed={seed}); "
            f"add a problem-specific reference solver or declare f_star in the factory"
        )

    # Pick the smallest f_ref across successful solvers.
    candidates.sort(key=lambda row: row[1])
    chosen = candidates[0]
    solver, f_ref, grad_norm_ref, n_iters_spent, status, wall_time_s = chosen

    return ReferenceEntry(
        problem_type=problem_type,
        params={k: v for k, v in params.items() if k != "seed"},
        seed=int(seed),
        f_ref=float(f_ref),
        grad_norm_ref=grad_norm_ref,
        solver=solver,
        n_iters_spent=int(n_iters_spent),
        eps_target=float(eps),
        grad_tol_target=float(grad_tol),
        status=status,
        wall_time_s=float(wall_time_s),
        local_minimum_warning=(problem_type in _NONCONVEX_PROBLEM_TYPES),
        code_commit=current_git_sha(),
        computed_at=_now_utc_iso(),
    )


def _problem_already_declares_f_star(problem_type: str, params: dict[str, Any]) -> bool:
    """Return ``True`` iff the factory declares a finite ``f_star``.

    We instantiate the factory to find out.  This is cheap for synthetic
    problems and unavoidable for the LIBSVM ones (we have to load the
    data anyway when we eventually compute references).
    """
    if problem_type not in PROBLEM_REGISTRY:
        return False
    full_params = dict(params)
    full_params.setdefault("seed", 0)
    try:
        problem = PROBLEM_REGISTRY[problem_type](**full_params)
    except Exception:  # pragma: no cover  - defensive
        return False
    fs = problem.get("f_star")
    return fs is not None and np.isfinite(float(fs))


def _iter_all_campaign_yamls(configs_root: Path) -> list[Path]:
    """Return every YAML under ``configs_root/configs/`` that is a leaf campaign.

    Skips both ``_base/`` (reusable includes, not campaigns) and
    ``extras/`` (opt-in heavy campaigns that exceed default-pipeline
    resource budgets, e.g. dense ``n x n`` Hessians on 47 k-feature
    LIBSVM datasets).  Callers that need an extras config can still pass
    it explicitly via ``--configs``.
    """
    out: list[Path] = []
    for yaml_path in sorted((configs_root / "configs").rglob("*.yaml")):
        if "_base" in yaml_path.parts or "extras" in yaml_path.parts:
            continue
        out.append(yaml_path)
    return out


def _load_problem_meta(config_path: Path) -> tuple[str, dict[str, Any]]:
    """Return ``(problem_type, params)`` for a leaf campaign YAML."""
    cfg = load_config(config_path)
    problem = cfg.get("problem", {})
    problem_type = problem.get("type")
    params = dict(problem.get("params", {}) or {})
    return str(problem_type), params


def _infer_repo_root() -> Path:
    """Climb from this module up until we find a directory containing ``configs/``.

    Works for both editable installs (``pip install -e .`` from the
    repo) and a checkout where the module sits under ``src/agns/cli``.
    """
    here = Path(__file__).resolve()
    for ancestor in (here.parent, *here.parents):
        if (ancestor / "configs").is_dir() and (ancestor / "pyproject.toml").is_file():
            return ancestor
    # Fallback: assume the repo root is two levels above src/agns/cli/.
    return here.parent.parent.parent.parent


def main() -> None:
    args = _parse_args()

    if args.configs_root is not None:
        configs_root = Path(args.configs_root).resolve()
    else:
        configs_root = _infer_repo_root()

    if args.all_configs:
        configs = _iter_all_campaign_yamls(configs_root)
    else:
        configs = [Path(c) for c in args.configs]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate (problem_type, params_hash) across campaigns so the same
    # instance is not solved twice.
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for cfg_path in configs:
        try:
            problem_type, params = _load_problem_meta(cfg_path)
        except Exception as exc:  # pragma: no cover  - defensive
            print(f"  [skip] {cfg_path}: cannot load problem meta ({exc})", file=sys.stderr)
            continue
        if not problem_type:
            print(f"  [skip] {cfg_path}: no problem.type", file=sys.stderr)
            continue
        if _problem_already_declares_f_star(problem_type, params):
            print(f"  [skip] {problem_type}: factory already declares f_star")
            continue
        key = (problem_type, canonical_params_hash(params))
        seen[key] = params  # last config wins; params identical by hash

    if not seen:
        print("Nothing to compute (all problems already declare f_star).")
        rebuild_manifest(output_dir)
        return

    print(
        f"Computing reference solutions for {len(seen)} unique problem instances "
        f"x {len(args.seeds)} seeds = {len(seen) * len(args.seeds)} entries."
    )

    for (problem_type, h), params in seen.items():
        for seed in args.seeds:
            # Honour --force; otherwise skip when an entry exists.
            target = output_dir / problem_type / f"{h}_seed_{seed}.json"
            if target.exists() and not args.force:
                print(f"  [skip] {problem_type}/{h}_seed_{seed}: cache hit")
                continue
            print(f"  [solve] {problem_type}/{h}_seed_{seed} ...", end="", flush=True)
            t0 = time.perf_counter()
            try:
                entry = compute_one_reference(
                    problem_type,
                    params,
                    seed,
                    newton_iters=args.newton_iters,
                    fast_gradient_iters=args.fast_gradient_iters,
                    eps=args.eps,
                    grad_tol=args.grad_tol,
                )
            except Exception as exc:
                dt = time.perf_counter() - t0
                print(f" FAILED in {dt:.2f}s: {exc}")
                continue
            save_entry(output_dir, entry)
            dt = time.perf_counter() - t0
            mark = " (local min)" if entry.local_minimum_warning else ""
            print(
                f" {entry.solver} | f_ref={entry.f_ref:.6e} | "
                f"iters={entry.n_iters_spent} | {dt:.2f}s{mark}"
            )

    manifest_path = rebuild_manifest(output_dir)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
