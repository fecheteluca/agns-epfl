"""
Finite-difference correctness tests for every oracle in src/oracles.py.

For each oracle, we check that the analytic gradient and Hessian match
central-difference approximations to within a tight tolerance.

Run with:
    python -m pytest tests/                     # if pytest is available
    python tests/test_oracles.py                # falls back to plain asserts
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from oracles import (
    ChebyshevOracle,
    LogSumExpOracle,
    NonlinearEquationsOracle,
    NonlinearEquationsRosenbrockOracle,
    PDifferenceOracle,
    PolytopeFeasibility,
    RosenbrockOracle,
    create_log_sum_exp_oracle,
)

GRAD_TOL = 1e-6
HESS_TOL = 1e-6


def _grad_fd(f, x, h=1e-6):
    n = x.shape[0]
    g = np.zeros(n)
    for i in range(n):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


def _hess_fd(grad_fn, x, h=1e-5):
    n = x.shape[0]
    H = np.zeros((n, n))
    for i in range(n):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        H[:, i] = (grad_fn(xp) - grad_fn(xm)) / (2 * h)
    return 0.5 * (H + H.T)


def _relerr(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)


def _check(oracle, x, name):
    g_err = _relerr(oracle.grad(x), _grad_fd(oracle.func, x))
    H_err = _relerr(oracle.hess(x), _hess_fd(oracle.grad, x))
    assert g_err < GRAD_TOL, f"{name}: grad relerr {g_err:.2e} > {GRAD_TOL}"
    assert H_err < HESS_TOL, f"{name}: hess relerr {H_err:.2e} > {HESS_TOL}"
    return g_err, H_err


# ---------------------------------------------------------------------------
# Individual oracle tests
# ---------------------------------------------------------------------------

def test_polytope_feasibility_p2():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 4)); b = rng.standard_normal(5)
    o = PolytopeFeasibility(A, b, 2)
    _check(o, rng.standard_normal(4) * 0.5, "PolytopeFeasibility(p=2)")


def test_polytope_feasibility_p4():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 4)); b = rng.standard_normal(5)
    o = PolytopeFeasibility(A, b, 4)
    _check(o, rng.standard_normal(4) * 0.5, "PolytopeFeasibility(p=4)")


def test_logsumexp():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((6, 4)); b = rng.standard_normal(6); mu = 0.7
    o = create_log_sum_exp_oracle(A, b, mu)
    _check(o, rng.standard_normal(4), "LogSumExp")


def test_pdifference_p3():
    rng = np.random.default_rng(2)
    o = PDifferenceOracle(p=3, n=5, a=0.5)
    _check(o, rng.standard_normal(5), "PDifference(p=3)")


def test_nonlinear_equations_linear_p2():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((3, 5)); b = rng.standard_normal(3)
    o = NonlinearEquationsOracle(p=2, A=A, b=b)
    _check(o, rng.standard_normal(5), "NonlinearEquations(p=2,linear)")


def test_nonlinear_equations_linear_p4():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((3, 5)); b = rng.standard_normal(3)
    o = NonlinearEquationsOracle(p=4, A=A, b=b)
    _check(o, rng.standard_normal(5), "NonlinearEquations(p=4,linear)")


def test_nonlinear_equations_general_with_hess_u_list():
    """
    Regression test: the generic NonlinearEquationsOracle must include the
    sum_i ‖u‖^(p-2) u_i ∇²u_i(x) term with the correct coefficient.
    """
    def func_u(x):
        return np.array([x[0] ** 2 + x[1], x[0] * x[1]])

    def jac_u(x):
        return np.array([[2 * x[0], 1.0], [x[1], x[0]]])

    H_u1 = np.array([[2.0, 0.0], [0.0, 0.0]])
    H_u2 = np.array([[0.0, 1.0], [1.0, 0.0]])

    for p in (2, 3, 4):
        o = NonlinearEquationsOracle(
            p=p, func_u=func_u, jac_u=jac_u, hess_u_list=[H_u1, H_u2]
        )
        _check(o, np.array([0.7, -0.3]), f"NonlinearEquations(p={p},nonlinear u)")


def test_chebyshev_p3():
    rng = np.random.default_rng(4)
    o = ChebyshevOracle(n=4, p=3)
    _check(o, rng.uniform(-0.5, 0.5, 4), "Chebyshev(p=3)")


def test_chebyshev_p4():
    rng = np.random.default_rng(4)
    o = ChebyshevOracle(n=4, p=4)
    _check(o, rng.uniform(-0.5, 0.5, 4), "Chebyshev(p=4)")


def test_rosenbrock_2d():
    o = RosenbrockOracle(a=1.0, b=100.0)
    _check(o, np.array([0.5, -0.3]), "Rosenbrock2D")


def test_nonlinear_equations_rosenbrock_p2():
    o = NonlinearEquationsRosenbrockOracle(p=2)
    _check(o, np.array([0.5, -0.3]), "NonlinearEquationsRosenbrock(p=2)")


def test_nonlinear_equations_rosenbrock_p5():
    o = NonlinearEquationsRosenbrockOracle(p=5)
    _check(o, np.array([0.5, -0.3]), "NonlinearEquationsRosenbrock(p=5)")


# ---------------------------------------------------------------------------
# Plain-Python harness (so the file can run without pytest installed)
# ---------------------------------------------------------------------------

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
        sys.exit(f"\n{len(failures)} oracle test(s) failed.")
    print(f"\nAll {len(tests)} oracle tests passed.")
