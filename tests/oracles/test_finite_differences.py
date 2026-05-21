"""Finite-difference cross-checks for every oracle's grad and Hessian.

Each oracle is exercised at a random point inside its natural domain.
The gradient is checked against a central-difference estimate; the
Hessian (when materialised) is checked for symmetry and against an
``H v`` central difference along a random direction.
"""

from __future__ import annotations

import numpy as np

from tests.oracles._fd import grad_fd, hess_vec_fd

# ---------------------------------------------------------------------------
# Ridge regression (constant Hessian)
# ---------------------------------------------------------------------------


def test_ridge_grad_matches_fd(ridge_problem, rng) -> None:
    oracle, _, _, _, _ = ridge_problem
    x = rng.standard_normal(oracle.n)
    np.testing.assert_allclose(oracle.grad(x), grad_fd(oracle, x), atol=1e-6, rtol=1e-5)


def test_ridge_hess_is_symmetric_and_constant(ridge_problem) -> None:
    oracle, _, _, _, _ = ridge_problem
    H0 = oracle.hess(np.zeros(oracle.n))
    H1 = oracle.hess(np.ones(oracle.n))
    np.testing.assert_allclose(H0, H1, atol=1e-12)
    np.testing.assert_allclose(H0, H0.T, atol=1e-12)


def test_ridge_hess_vec_matches_dense(ridge_problem, rng) -> None:
    oracle, _, _, _, _ = ridge_problem
    x = rng.standard_normal(oracle.n)
    v = rng.standard_normal(oracle.n)
    np.testing.assert_allclose(oracle.hess_vec(x, v), oracle.hess(x).dot(v), atol=1e-12)


# ---------------------------------------------------------------------------
# Logistic regression (non-constant Hessian)
# ---------------------------------------------------------------------------


def test_logistic_grad_matches_fd(logistic_problem, rng) -> None:
    oracle, _, _, _ = logistic_problem
    x = 0.1 * rng.standard_normal(oracle.n)
    np.testing.assert_allclose(oracle.grad(x), grad_fd(oracle, x), atol=1e-6, rtol=1e-5)


def test_logistic_hess_is_symmetric(logistic_problem, rng) -> None:
    oracle, _, _, _ = logistic_problem
    H = oracle.hess(0.1 * rng.standard_normal(oracle.n))
    np.testing.assert_allclose(H, H.T, atol=1e-10)


def test_logistic_hess_vec_matches_fd(logistic_problem, rng) -> None:
    oracle, _, _, _ = logistic_problem
    x = 0.1 * rng.standard_normal(oracle.n)
    v = rng.standard_normal(oracle.n)
    np.testing.assert_allclose(
        oracle.hess_vec(x, v), hess_vec_fd(oracle, x, v), atol=1e-5, rtol=1e-3
    )


# ---------------------------------------------------------------------------
# Soft-margin SVM
# ---------------------------------------------------------------------------


def test_softsvm_grad_matches_fd(soft_svm_problem, rng) -> None:
    oracle, _, _, _ = soft_svm_problem
    x = 0.5 * rng.standard_normal(oracle.n)
    np.testing.assert_allclose(oracle.grad(x), grad_fd(oracle, x), atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Smoothed LASSO
# ---------------------------------------------------------------------------


def test_smoothed_lasso_grad_matches_fd(smoothed_lasso_problem, rng) -> None:
    oracle, _, _, _ = smoothed_lasso_problem
    x = 0.1 * rng.standard_normal(oracle.n)
    np.testing.assert_allclose(oracle.grad(x), grad_fd(oracle, x), atol=1e-5, rtol=1e-4)


def test_smoothed_lasso_hess_is_symmetric(smoothed_lasso_problem, rng) -> None:
    oracle, _, _, _ = smoothed_lasso_problem
    H = oracle.hess(rng.standard_normal(oracle.n))
    np.testing.assert_allclose(H, H.T, atol=1e-12)


# ---------------------------------------------------------------------------
# Softmax regression
# ---------------------------------------------------------------------------


def test_softmax_grad_matches_fd(softmax_problem, rng) -> None:
    oracle, _ = softmax_problem
    n_dim = oracle.n * oracle.K
    x = 0.1 * rng.standard_normal(n_dim)
    np.testing.assert_allclose(oracle.grad(x), grad_fd(oracle, x), atol=1e-5, rtol=1e-3)


def test_softmax_hess_vec_matches_fd(softmax_problem, rng) -> None:
    oracle, _ = softmax_problem
    n_dim = oracle.n * oracle.K
    x = 0.1 * rng.standard_normal(n_dim)
    v = rng.standard_normal(n_dim)
    np.testing.assert_allclose(
        oracle.hess_vec(x, v), hess_vec_fd(oracle, x, v), atol=1e-4, rtol=1e-3
    )


# ---------------------------------------------------------------------------
# LogSumExp
# ---------------------------------------------------------------------------


def test_logsumexp_grad_matches_fd(logsumexp_raw) -> None:
    oracle, x = logsumexp_raw
    np.testing.assert_allclose(oracle.grad(x), grad_fd(oracle, x), atol=1e-6, rtol=1e-5)


def test_logsumexp_hess_is_symmetric(logsumexp_raw) -> None:
    oracle, x = logsumexp_raw
    H = oracle.hess(x)
    np.testing.assert_allclose(H, H.T, atol=1e-10)


def test_logsumexp_hess_vec_matches_dense(logsumexp_raw) -> None:
    oracle, x = logsumexp_raw
    v = np.ones_like(x)
    np.testing.assert_allclose(oracle.hess_vec(x, v), oracle.hess(x).dot(v), atol=1e-10)


# ---------------------------------------------------------------------------
# Nonlinear equations (linear residuals)
# ---------------------------------------------------------------------------


def test_nle_grad_matches_fd(nle_problem, rng) -> None:
    oracle, _, _, _, _ = nle_problem
    x = 0.5 * rng.standard_normal(10)
    np.testing.assert_allclose(oracle.grad(x), grad_fd(oracle, x), atol=1e-5, rtol=1e-4)


def test_nle_hess_is_symmetric(nle_problem, rng) -> None:
    oracle, _, _, _, _ = nle_problem
    H = oracle.hess(0.5 * rng.standard_normal(10))
    np.testing.assert_allclose(H, H.T, atol=1e-10)


# ---------------------------------------------------------------------------
# Chebyshev
# ---------------------------------------------------------------------------


def test_chebyshev_grad_matches_fd(chebyshev_problem) -> None:
    oracle, x_0, _ = chebyshev_problem
    np.testing.assert_allclose(oracle.grad(x_0), grad_fd(oracle, x_0), atol=1e-5, rtol=1e-3)


def test_chebyshev_zero_at_optimum(chebyshev_problem) -> None:
    oracle, _, _ = chebyshev_problem
    x_star = np.ones(oracle.n)
    assert oracle.func(x_star) == 0.0
    np.testing.assert_allclose(oracle.grad(x_star), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Rosenbrock 2D
# ---------------------------------------------------------------------------


def test_rosenbrock_2d_grad_matches_fd(rosenbrock_2d_problem) -> None:
    oracle, x = rosenbrock_2d_problem
    np.testing.assert_allclose(oracle.grad(x), grad_fd(oracle, x), atol=1e-4, rtol=1e-4)


def test_rosenbrock_2d_optimum_is_one() -> None:
    from agns.oracles import RosenbrockOracle

    o = RosenbrockOracle(a=1.0, b=100.0)
    assert o.func(np.array([1.0, 1.0])) == 0.0
    np.testing.assert_allclose(o.grad(np.array([1.0, 1.0])), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Rosenbrock NLE
# ---------------------------------------------------------------------------


def test_rosenbrock_nle_grad_matches_fd(rosenbrock_nle_problem) -> None:
    oracle, x = rosenbrock_nle_problem
    np.testing.assert_allclose(oracle.grad(x), grad_fd(oracle, x), atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# Matrix completion
# ---------------------------------------------------------------------------


def test_matrix_completion_grad_matches_fd(matrix_completion_problem) -> None:
    oracle, x = matrix_completion_problem
    np.testing.assert_allclose(oracle.grad(x), grad_fd(oracle, x, h=1e-5), atol=1e-4, rtol=1e-3)


def test_matrix_completion_hess_vec_matches_fd(matrix_completion_problem, rng) -> None:
    oracle, x = matrix_completion_problem
    v = rng.standard_normal(x.shape)
    np.testing.assert_allclose(
        oracle.hess_vec(x, v), hess_vec_fd(oracle, x, v), atol=1e-4, rtol=1e-3
    )


# ---------------------------------------------------------------------------
# Phase retrieval
# ---------------------------------------------------------------------------


def test_phase_retrieval_grad_matches_fd(phase_retrieval_problem) -> None:
    oracle, x = phase_retrieval_problem
    np.testing.assert_allclose(oracle.grad(x), grad_fd(oracle, x), atol=1e-4, rtol=1e-3)


def test_phase_retrieval_hess_is_symmetric(phase_retrieval_problem) -> None:
    oracle, x = phase_retrieval_problem
    H = oracle.hess(x)
    np.testing.assert_allclose(H, H.T, atol=1e-10)
