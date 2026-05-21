"""Chebyshev polynomial-chain oracle and factory.

Objective ``f(x) = (1/p) ||u(x)||^p`` where the residuals follow the
Chebyshev chain

    u_1(x) = 1/2 (1 - x_1),    u_i(x) = x_i - (2 x_{i-1}^2 - 1)  for i = 2..n.

Global minimiser ``x* = (1, ..., 1)`` with ``f^* = 0``.  This is a
non-convex test problem — the exact Hessian is indefinite far from the
optimum, which is why the inexact Gauss-Newton + Fisher approximation
matters for robust convergence (see ``approx_hess_fn_chebyshev`` in
``agns.approximations``).

"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from agns.approximations import approx_hess_fn_chebyshev
from agns.oracles.base import BaseSmoothOracle


class ChebyshevOracle(BaseSmoothOracle):
    """Chebyshev-chain oracle for ``f(x) = (1/p) ||u(x)||^p``."""

    def __init__(self, n: int, p: int) -> None:
        assert p >= 2, "p must be >= 2."
        self.n = n
        self.p = p

        def _func_u(x: NDArray[np.float64]) -> NDArray[np.float64]:
            xx = np.asarray(x).ravel()
            assert xx.shape[0] == self.n
            u = np.empty(self.n, dtype=float)
            u[0] = 0.5 * (1.0 - xx[0])
            for i in range(1, self.n):
                u[i] = xx[i] - (2.0 * xx[i - 1] ** 2 - 1.0)
            return u

        def _jac_u(x: NDArray[np.float64]) -> NDArray[np.float64]:
            xx = np.asarray(x).ravel()
            assert xx.shape[0] == self.n
            J = np.zeros((self.n, self.n), dtype=float)
            J[0, 0] = -0.5
            for i in range(1, self.n):
                J[i, i] = 1.0
                J[i, i - 1] = -4.0 * xx[i - 1]
            return J

        H_list: list[NDArray[np.float64]] = []
        for i in range(self.n):
            H_i = np.zeros((self.n, self.n), dtype=float)
            if i >= 1:
                H_i[i - 1, i - 1] = -4.0
            H_list.append(H_i)

        self.func_u = _func_u
        self.jac_u = _jac_u
        self.hess_u_list = H_list

    def func(self, x: NDArray[np.float64]) -> float:
        u = self.func_u(x)
        norm_u = np.linalg.norm(u)
        return float((1.0 / self.p) * (norm_u**self.p))

    def grad(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        u = self.func_u(x)
        norm_u = np.linalg.norm(u)
        J = self.jac_u(x)
        if norm_u == 0:
            return np.zeros(self.n, dtype=float)
        return (norm_u ** (self.p - 2)) * (J.T.dot(u))

    def hess(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        u = self.func_u(x)
        norm_u = np.linalg.norm(u)
        J = self.jac_u(x)
        if norm_u == 0:
            if self.p == 2:
                return cast("NDArray[np.float64]", J.T.dot(J))
            return np.zeros((self.n, self.n), dtype=float)

        H = (norm_u ** (self.p - 2)) * (J.T.dot(J))

        if self.p != 2:
            g_vec = J.T.dot(u)
            H += (self.p - 2) * (norm_u ** (self.p - 4)) * np.outer(g_vec, g_vec)

        coeff = norm_u ** (self.p - 2)
        for i in range(self.n):
            if u[i] != 0.0:
                H += coeff * u[i] * self.hess_u_list[i]

        return cast("NDArray[np.float64]", H)

    def hess_vec(self, x: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        H = self.hess(x)
        return cast("NDArray[np.float64]", H.dot(v))

    def third_vec_vec(self, x: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError("third_vec_vec is not implemented for ChebyshevOracle.")


# ---------------------------------------------------------------------------
# Problem factory
# ---------------------------------------------------------------------------


def make_chebyshev(n: int = 1000, p: int = 4, seed: int = 42) -> dict[str, Any]:
    """``min_x (1/p) ||u(x)||^p`` with Chebyshev-chain residuals.

    No natural non-trivial metric exists for this problem, so ``B = None``
    (Euclidean).  The WSM backend is therefore not applicable here.
    """
    rng = np.random.default_rng(seed)
    oracle = ChebyshevOracle(n, p)
    x_0 = rng.uniform(0.0, 1.0, n)
    f_star = oracle.func(np.ones(n))

    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": f_star,
        "B": None,
        "Binv": None,
        "approx_hess_fn": approx_hess_fn_chebyshev,
        "approx_hess_fn_wsm": None,
        "meta": {"type": "chebyshev", "n": n, "p": p, "seed": seed},
    }


__all__ = ["ChebyshevOracle", "make_chebyshev"]
