"""Tests for :mod:`agns.approximations`."""

from __future__ import annotations

import numpy as np

from agns.approximations import (
    approx_hess_fn_chebyshev,
    approx_hess_fn_fisher_term,
    approx_hess_fn_logsumexp,
    approx_hess_nonlinear_equations,
)
from agns.oracles import (
    ChebyshevOracle,
    NonlinearEquationsOracle,
    create_log_sum_exp_oracle,
)


def test_logsumexp_approx_matches_diag_form() -> None:
    rng = np.random.default_rng(0)
    m, n = 8, 5
    A = rng.uniform(-1.0, 1.0, (m, n))
    b = rng.uniform(-1.0, 1.0, m)
    oracle = create_log_sum_exp_oracle(A, b, mu=1.0)
    x = rng.standard_normal(n)
    H_approx = approx_hess_fn_logsumexp(oracle, x)
    # The approximation drops the rank-one (A^T pi)(A^T pi)^T / mu term
    # from the exact Hessian.
    oracle._update_a_and_pi(x)
    H_exact = oracle.matmat_ATsA(oracle.pi) / oracle.mu
    np.testing.assert_allclose(H_approx, H_exact, rtol=1e-12)


def test_fisher_term_zero_when_p_equals_two() -> None:
    rng = np.random.default_rng(0)
    n, m = 4, 5
    A = rng.uniform(-1.0, 1.0, (n, m))
    b = rng.uniform(-1.0, 1.0, n)
    oracle = NonlinearEquationsOracle(p=2.0, A=A, b=b)
    x = rng.standard_normal(m)
    H = approx_hess_fn_fisher_term(oracle, x)
    assert np.allclose(H, 0.0)


def test_nonlinear_equations_approx_is_symmetric() -> None:
    rng = np.random.default_rng(0)
    n, m = 4, 5
    A = rng.uniform(-1.0, 1.0, (n, m))
    b = rng.uniform(-1.0, 1.0, n)
    oracle = NonlinearEquationsOracle(p=4.0, A=A, b=b)
    x = rng.standard_normal(m)
    H = approx_hess_nonlinear_equations(oracle, x)
    np.testing.assert_allclose(H, H.T, atol=1e-12)


def test_chebyshev_approx_is_symmetric() -> None:
    oracle = ChebyshevOracle(n=4, p=3)
    H = approx_hess_fn_chebyshev(oracle, np.array([0.5, 0.5, 0.5, 0.5]))
    np.testing.assert_allclose(H, H.T, atol=1e-12)


def test_chebyshev_approx_zero_at_optimum() -> None:
    """At the minimiser ``x = (1, ..., 1)``, ``u(x) = 0`` and the
    approximation reduces to ``J^T J`` only for ``p == 2``."""
    n = 4
    p = 2
    oracle = ChebyshevOracle(n=n, p=p)
    H = approx_hess_fn_chebyshev(oracle, np.ones(n))
    # u == 0 at the optimum; for p == 2 we get J^T J.
    J = oracle.jac_u(np.ones(n))
    np.testing.assert_allclose(H, J.T.dot(J), atol=1e-12)
