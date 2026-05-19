"""Tests for :mod:`agns.linalg`."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.linalg import LinAlgError

from agns.linalg import safe_cho_solve, wsm_rank_one_solve


@pytest.fixture()
def spd_matrix() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 8
    A = rng.standard_normal((n, n))
    return A.T @ A + n * np.eye(n), rng.standard_normal(n)


def test_safe_cho_solve_matches_numpy(spd_matrix: tuple[np.ndarray, np.ndarray]) -> None:
    A, b = spd_matrix
    x = safe_cho_solve(A, b)
    expected = np.linalg.solve(A, b)
    np.testing.assert_allclose(x, expected, rtol=1e-10)


def test_safe_cho_solve_indefinite_raises() -> None:
    A = np.array([[1.0, 0.0], [0.0, -1.0]])
    with pytest.raises((LinAlgError, ValueError)):
        safe_cho_solve(A, np.ones(2))


def test_wsm_rank_one_alpha_zero(spd_matrix: tuple[np.ndarray, np.ndarray]) -> None:
    """When alpha == 0 the WSM solve degenerates to -Binv g / lambda."""
    B, g = spd_matrix
    Binv = np.linalg.inv(B)
    lam = 2.5
    delta = wsm_rank_one_solve(g, lam, Binv, alpha=0.0, g0=np.zeros_like(g))
    expected = -Binv.dot(g) / lam
    np.testing.assert_allclose(delta, expected, rtol=1e-12)


def test_wsm_rank_one_matches_dense_inverse(spd_matrix: tuple[np.ndarray, np.ndarray]) -> None:
    """``(lambda B + alpha g0 g0^T)^{-1} (-g)`` matches dense solve."""
    B, g = spd_matrix
    Binv = np.linalg.inv(B)
    lam = 1.7
    alpha = 3.2
    g0 = np.linspace(0.1, 0.9, B.shape[0])
    A = lam * B + alpha * np.outer(g0, g0)
    expected = np.linalg.solve(A, -g)
    delta = wsm_rank_one_solve(g, lam, Binv, alpha, g0)
    np.testing.assert_allclose(delta, expected, rtol=1e-9, atol=1e-12)


def test_wsm_rank_one_rejects_nonpositive_lambda(spd_matrix: tuple[np.ndarray, np.ndarray]) -> None:
    B, g = spd_matrix
    Binv = np.linalg.inv(B)
    with pytest.raises(ValueError):
        wsm_rank_one_solve(g, 0.0, Binv, alpha=1.0, g0=np.ones_like(g))
    with pytest.raises(ValueError):
        wsm_rank_one_solve(g, -1.0, Binv, alpha=1.0, g0=np.ones_like(g))
