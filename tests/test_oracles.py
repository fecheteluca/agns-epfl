"""Oracle correctness: analytic derivatives, known optima, approximate-Hessian builders.

Every oracle's analytic ``grad``/``hess`` is checked against a central finite-difference
of its own ``func``/``grad`` (so the check is independent of the closed-form algebra), each
synthetic family is verified to attain ``f(x*) == f_star`` at its analytic optimum, and the
approximate-Hessian builders are checked for shape, symmetry, and positive semi-definiteness
(the property the regularized Newton solve relies on).
"""

from __future__ import annotations

import numpy as np
import pytest

from agns.oracles.approximations import (
    approx_hess_fn_chebyshev,
    approx_hess_fn_fisher_term,
    approx_hess_fn_logsumexp,
    approx_hess_nonlinear_equations,
)
from agns.oracles.chebyshev import ChebyshevOracle
from agns.oracles.nonlinear_equations import NonlinearEquationsOracle
from agns.oracles.polytope import PolytopeFeasibility
from agns.oracles.rosenbrock import (
    NonlinearEquationsRosenbrockOracle,
    RosenbrockOracle,
)
from agns.oracles.separable import SeparableCurvatureOracle
from tests.conftest import (
    finite_diff_grad,
    finite_diff_hess,
    make_logsumexp_problem,
)

# --------------------------------------------------------------- oracle instances


def _well_conditioned_nleq(n: int = 6, p: int = 4, seed: int = 7):
    """A generic nonlinear-equations oracle with a known optimum ``x* = A^{-1} b``."""
    rng = np.random.RandomState(seed)
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    b = rng.standard_normal(n)
    oracle = NonlinearEquationsOracle(p=p, A=A, b=b)
    return oracle, np.linalg.solve(A, b)


def _feasible_polytope(n: int = 5, m: int = 12, p: int = 4, seed: int = 0):
    """A polytope-feasibility oracle with a strictly interior point ``x_feas`` (f = 0)."""
    rng = np.random.RandomState(seed)
    A = rng.standard_normal((m, n))
    x_feas = rng.standard_normal(n)
    b = A.dot(x_feas) + np.abs(rng.standard_normal(m)) + 0.1  # strictly feasible
    return PolytopeFeasibility(A, b, p), x_feas


def _oracle_cases():
    """(id, oracle, eval_point) cases for the finite-difference derivative checks."""
    rng = np.random.RandomState(1)
    lse = make_logsumexp_problem(n=8, m=14).oracle
    nleq, _ = _well_conditioned_nleq()
    poly, _ = _feasible_polytope()
    return [
        ("logsumexp", lse, rng.standard_normal(8) * 0.3),
        ("rosenbrock", RosenbrockOracle(a=1.0, b=100.0), np.array([-0.7, 0.4])),
        (
            "nleq_rosenbrock",
            NonlinearEquationsRosenbrockOracle(p=4, a=1.0, b=100.0),
            np.array([-0.7, 0.4]),
        ),
        ("nonlinear_equations", nleq, rng.standard_normal(6) * 0.5),
        ("chebyshev", ChebyshevOracle(n=5, p=4), rng.standard_normal(5) * 0.3),
        ("polytope", poly, rng.standard_normal(5) * 0.5),
        (
            "curvature",
            SeparableCurvatureOracle(c=np.geomspace(1.0, 50.0, 5), kappa=2.0),
            rng.standard_normal(5) * 0.3,
        ),
    ]


_CASES = _oracle_cases()
_CASE_IDS = [c[0] for c in _CASES]


# ------------------------------------------------------------- derivative checks


@pytest.mark.parametrize("name, oracle, x", _CASES, ids=_CASE_IDS)
def test_grad_matches_finite_difference(name, oracle, x):
    """Analytic gradient equals the central finite difference of ``func``."""
    analytic = oracle.grad(x)
    numeric = finite_diff_grad(oracle.func, x, h=1e-6)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name, oracle, x", _CASES, ids=_CASE_IDS)
def test_hess_matches_finite_difference(name, oracle, x):
    """Analytic Hessian equals the central finite difference of ``grad``."""
    analytic = oracle.hess(x)
    numeric = finite_diff_hess(oracle.grad, x, h=1e-5)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name, oracle, x", _CASES, ids=_CASE_IDS)
def test_hess_is_symmetric(name, oracle, x):
    """Every analytic Hessian is symmetric."""
    H = oracle.hess(x)
    np.testing.assert_allclose(H, H.T, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------ known optima


def test_logsumexp_optimum_at_origin():
    """The shifted log-sum-exp oracle is stationary at the origin (its ``f_star`` point)."""
    problem = make_logsumexp_problem(n=10, m=18)
    g0 = problem.oracle.grad(np.zeros(problem.n))
    assert np.linalg.norm(g0) < 1e-10
    assert problem.oracle.func(np.zeros(problem.n)) == pytest.approx(problem.f_star)


def test_rosenbrock_optimum():
    """Rosenbrock attains ``f = 0`` with zero gradient at ``(a, a^2)``."""
    oracle = RosenbrockOracle(a=1.0, b=100.0)
    x_star = np.array([1.0, 1.0])
    assert oracle.func(x_star) == pytest.approx(0.0, abs=1e-14)
    assert np.linalg.norm(oracle.grad(x_star)) < 1e-12


def test_nleq_rosenbrock_optimum():
    """The residual-form Rosenbrock attains ``f = 0`` at ``(a, a^2)``."""
    oracle = NonlinearEquationsRosenbrockOracle(p=4, a=1.0, b=100.0)
    assert oracle.func(np.array([1.0, 1.0])) == pytest.approx(0.0, abs=1e-14)


def test_nonlinear_equations_optimum():
    """The affine nonlinear-equations oracle attains ``f = 0`` at ``x* = A^{-1} b``."""
    oracle, x_star = _well_conditioned_nleq()
    assert oracle.func(x_star) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(oracle.grad(x_star)) < 1e-8


def test_chebyshev_optimum_all_ones():
    """The Chebyshev objective attains ``f = 0`` with zero gradient at ``x = (1,...,1)``."""
    oracle = ChebyshevOracle(n=6, p=4)
    x_star = np.ones(6)
    assert oracle.func(x_star) == pytest.approx(0.0, abs=1e-14)
    assert np.linalg.norm(oracle.grad(x_star)) < 1e-12


def test_polytope_optimum_interior():
    """The polytope penalty is ``0`` with zero gradient at a strictly feasible point."""
    oracle, x_feas = _feasible_polytope()
    assert oracle.func(x_feas) == pytest.approx(0.0, abs=1e-14)
    assert np.linalg.norm(oracle.grad(x_feas)) < 1e-12


def test_curvature_oracle_optimum_at_origin():
    """The separable curvature oracle attains ``f = 0`` with zero gradient at the origin."""
    oracle = SeparableCurvatureOracle(c=np.geomspace(1.0, 100.0, 8), kappa=5.0)
    assert oracle.func(np.zeros(8)) == pytest.approx(0.0, abs=1e-14)
    assert np.linalg.norm(oracle.grad(np.zeros(8))) < 1e-14


def test_curvature_oracle_conditioning_fixed_across_kappa():
    """Conditioning at the optimum is set by ``c`` and is independent of ``kappa``.

    This is the property the RQ1 mechanism experiment relies on: sweeping ``kappa`` changes
    curvature variation without touching the condition number at the solution.
    """
    c = np.geomspace(1.0, 100.0, 8)
    conds = []
    for kappa in [0.0, 1.0, 10.0, 100.0]:
        H0 = SeparableCurvatureOracle(c=c, kappa=kappa).hess(np.zeros(8))
        eig = np.linalg.eigvalsh(H0)
        conds.append(eig.max() / eig.min())
    np.testing.assert_allclose(conds, 100.0, rtol=1e-12)


def test_curvature_oracle_quadratic_when_kappa_zero():
    """At ``kappa = 0`` the Hessian is the constant ``diag(c)`` everywhere (pure quadratic)."""
    c = np.geomspace(1.0, 10.0, 6)
    oracle = SeparableCurvatureOracle(c=c, kappa=0.0)
    rng = np.random.RandomState(0)
    for _ in range(3):
        x = rng.standard_normal(6)
        np.testing.assert_allclose(oracle.hess(x), np.diag(c), rtol=1e-12, atol=1e-12)


# ----------------------------------------------------- approximate-Hessian builders


def _assert_psd(H: np.ndarray, *, tol: float = 1e-8) -> None:
    """Assert ``H`` is symmetric positive semi-definite up to a scaled tolerance."""
    assert H.shape[0] == H.shape[1]
    np.testing.assert_allclose(H, H.T, rtol=1e-10, atol=1e-10)
    min_eig = float(np.linalg.eigvalsh(0.5 * (H + H.T)).min())
    scale = max(1.0, float(np.abs(H).max()))
    assert min_eig >= -tol * scale


def test_approx_hess_logsumexp_psd():
    """Weighted Gauss-Newton ``A^T diag(pi) A / mu`` is symmetric PSD with the right shape."""
    problem = make_logsumexp_problem(n=8, m=14)
    x = np.random.RandomState(2).standard_normal(problem.n) * 0.3
    H = approx_hess_fn_logsumexp(problem.oracle, x)
    assert H.shape == (problem.n, problem.n)
    _assert_psd(H)


def test_approx_hess_fisher_term_psd():
    """The Fisher-term approximation (needs ``func_u``/``jac_u``) is symmetric PSD."""
    oracle, _ = _well_conditioned_nleq(n=6, p=4)
    x = np.random.RandomState(5).standard_normal(6) * 0.5
    H = approx_hess_fn_fisher_term(oracle, x)
    assert H.shape == (6, 6)
    _assert_psd(H)


def test_approx_hess_nonlinear_equations_psd():
    """The Gauss-Newton + Fisher approximation (needs ``_compute_u_and_jac``) is PSD."""
    oracle = NonlinearEquationsRosenbrockOracle(p=4, a=1.0, b=100.0)
    x = np.array([-0.7, 0.4])
    H = approx_hess_nonlinear_equations(oracle, x)
    assert H.shape == (2, 2)
    _assert_psd(H)


def test_approx_hess_chebyshev_psd():
    """The Chebyshev Gauss-Newton + Fisher approximation is symmetric PSD."""
    oracle = ChebyshevOracle(n=5, p=4)
    x = np.random.RandomState(3).standard_normal(5) * 0.3
    H = approx_hess_fn_chebyshev(oracle, x)
    assert H.shape == (5, 5)
    _assert_psd(H)
