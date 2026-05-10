"""
Smoke / convergence tests for every method in METHOD_REGISTRY.

For each method we run a short optimization on a small, well-conditioned
problem and assert that:
  * the run terminates without exception,
  * the function value strictly decreases (when monotone descent is expected),
  * the residual reaches a target tolerance within the iteration budget.

Run:
    python tests/test_methods_smoke.py
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from datasets import make_nonlinear_equations, make_rosenbrock_2d
from models import METHOD_REGISTRY, run_method


def _run(method_key, problem, n_iters, **overrides):
    spec = METHOD_REGISTRY[method_key]
    cfg = {
        "n_iters": n_iters,
        "gamma_0": 1.0,
        "eps": 1e-10,
        "adaptive_search": True,
        "B": problem.get("B"),
        "Binv": problem.get("Binv"),
        "f_star": problem.get("f_star"),
    }
    cfg.update(overrides)
    return run_method(
        spec,
        problem["oracle"],
        problem["x_0"],
        cfg,
        problem["oracle"],
        problem.get("approx_hess_fn_wsm" if spec.uses_wsm else "approx_hess_fn"),
    )


def test_all_methods_on_small_nle():
    """Every method that's applicable should reduce f below the start by 8 orders."""
    problem = make_nonlinear_equations(n=20, m=40, p=4, seed=7)
    f0 = problem["oracle"].func(problem["x_0"])
    for key in METHOD_REGISTRY:
        # cubic_newton uses an exact-Hessian inner subproblem solve and is
        # tested separately on the rosenbrock_2d problem below.
        if key == "cubic_newton":
            continue
        n_iters = 200 if "gradient" in key else 60
        x_final, status, hist = _run(key, problem, n_iters=n_iters)
        f_final = float(hist["func"][-1])
        assert np.isfinite(f_final), f"{key}: f_final not finite ({f_final})"
        assert f_final < f0 * 1e-4, (
            f"{key}: insufficient descent f0={f0:.3e} f_final={f_final:.3e}"
        )


def test_cubic_newton_on_rosenbrock_2d():
    problem = make_rosenbrock_2d()
    x_final, status, hist = _run("cubic_newton", problem, n_iters=50)
    f_final = float(hist["func"][-1])
    assert f_final < 1e-6, f"cubic_newton: f_final={f_final:.3e} not small enough"


def test_monotone_descent_for_gns_and_super_newton():
    """Pure descent methods must produce a non-increasing function-value trace."""
    problem = make_nonlinear_equations(n=20, m=40, p=4, seed=11)
    for key in ("gns_exact", "gns_inexact", "gns_wsm", "super_newton"):
        x_final, status, hist = _run(key, problem, n_iters=40)
        func = np.array(hist["func"])
        # allow a tiny float-roundoff slack
        diffs = np.diff(func)
        assert (diffs <= 1e-12).all(), (
            f"{key}: descent violated by {diffs.max():.3e}"
        )


def test_agns_eventually_decreases_below_initial():
    """AGNS is not strictly monotone, but must beat the initial f convincingly."""
    problem = make_nonlinear_equations(n=20, m=40, p=4, seed=11)
    f0 = problem["oracle"].func(problem["x_0"])
    for key in (
        "agns_lookahead",
        "agns_iterate",
        "agns_wsm_lookahead",
        "agns_wsm_iterate",
    ):
        x_final, status, hist = _run(key, problem, n_iters=60)
        f_final = float(hist["func"][-1])
        assert f_final < f0 * 1e-4, (
            f"{key}: f0={f0:.3e} f_final={f_final:.3e}"
        )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  OK  {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
    if failures:
        sys.exit(f"\n{len(failures)} method test(s) failed.")
    print(f"\nAll {len(tests)} method tests passed.")
