"""LogSumExp (softmax) oracle -- the primary ML problem.

Ported verbatim from ``epfml/grad-norm-smooth/src/oracles.py`` (Apache-2.0, commit
9f4ca00). This is the smooth softmax objective the logistic-regression / softmax
experiment is built on, and where the approximate-Hessian (Weighted Gauss-Newton) story
lives (see :func:`agns.oracles.approximations.approx_hess_fn_logsumexp`).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse
from numpy.typing import NDArray
from scipy.special import logsumexp, softmax

from agns.oracles.base import BaseSmoothOracle

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


class LogSumExpOracle(BaseSmoothOracle):
    r"""Oracle for ``f(x) = mu * log sum_i exp((<a_i, x> - b_i) / mu)``.

    Parameters
    ----------
    matvec_Ax:
        Linear map ``x -> A x``.
    matvec_ATx:
        Adjoint map ``x -> A^T x``.
    matmat_ATsA:
        Map ``s -> A^T diag(s) A`` (materialized lazily by the factory).
    b:
        Offset vector.
    mu:
        Smoothing parameter.
    """

    def __init__(
        self,
        matvec_Ax: Callable[[Vector], Vector],
        matvec_ATx: Callable[[Vector], Vector],
        matmat_ATsA: Callable[[Vector], Matrix],
        b: Vector,
        mu: float,
    ) -> None:
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = np.copy(b)
        self.mu = mu
        self.mu_inv = 1.0 / mu
        self.last_x: Vector | None = None
        self.last_x_pi: Vector | None = None
        self.last_v: Vector | None = None

    def func(self, x: Vector) -> float:
        """Return the smoothed log-sum-exp value at ``x``."""
        self._update_a(x)
        return self.mu * logsumexp(self.a)

    def grad(self, x: Vector) -> Vector:
        """Return the gradient ``A^T pi`` at ``x``."""
        self._update_a_and_pi(x)
        return self.AT_pi

    def hess(self, x: Vector) -> Matrix:
        """Return the Hessian ``(A^T diag(pi) A - (A^T pi)(A^T pi)^T) / mu`` at ``x``."""
        self._update_a_and_pi(x)
        return self.mu_inv * (
            self.matmat_ATsA(self.pi) - np.outer(self.AT_pi, self.AT_pi.T)
        )

    def hess_vec(self, x: Vector, v: Vector) -> Vector:
        """Return the Hessian-vector product at ``x`` with ``v``."""
        self._update_hess_vec(x, v)
        return self._hess_vec

    def third_vec_vec(self, x: Vector, v: Vector) -> Vector:
        """Return the third-derivative product ``D^3 f(x)[v, v]``."""
        self._update_hess_vec(x, v)
        return self.mu_inv * (
            self.mu_inv
            * self.matvec_ATx(self.pi * (self.Av * (self.Av - self.AT_pi_v)))
            - self.AT_pi_v * self._hess_vec
            - self._hess_vec.dot(v) * self.AT_pi
        )

    def _update_a(self, x: Vector) -> None:
        if not np.array_equal(self.last_x, x):
            self.last_x = np.copy(x)
            self.a = self.mu_inv * (self.matvec_Ax(x) - self.b)

    def _update_a_and_pi(self, x: Vector) -> None:
        self._update_a(x)
        if not np.array_equal(self.last_x_pi, x):
            self.last_x_pi = np.copy(x)
            self.pi = softmax(self.a)
            self.AT_pi = self.matvec_ATx(self.pi)

    def _update_hess_vec(self, x: Vector, v: Vector) -> None:
        if not np.array_equal(self.last_x, x) or not np.array_equal(self.last_v, v):
            self._update_a_and_pi(x)
            self.last_v = np.copy(v)
            self.Av = self.matvec_Ax(v)
            self.AT_pi_v = self.AT_pi.dot(v)
            self._hess_vec = self.mu_inv * (
                self.matvec_ATx(self.pi * self.Av) - self.AT_pi_v * self.AT_pi
            )


def create_log_sum_exp_oracle(A: Matrix, b: Vector, mu: float) -> LogSumExpOracle:
    """Create a LogSumExp oracle from a (possibly sparse) data matrix ``A``."""

    def matvec_Ax(x: Vector) -> Vector:
        return A.dot(x)

    def matvec_ATx(x: Vector) -> Vector:
        return A.T.dot(x)

    cache: dict[str, Matrix] = {}

    def matmat_ATsA(s: Vector) -> Matrix:
        if "B" not in cache:
            cache["B"] = A.toarray() if scipy.sparse.issparse(A) else A
        B = cache["B"]
        return B.T.dot(B * s.reshape(-1, 1))

    return LogSumExpOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, mu)


def create_log_sum_exp_zero_oracle(A: Matrix, b: Vector, mu: float) -> LogSumExpOracle:
    """Create a LogSumExp oracle shifted so its optimum sits at the origin."""
    oracle_0 = create_log_sum_exp_oracle(A, b, mu)
    g = oracle_0.grad(np.zeros(A.shape[1]))
    A_new = A - g

    def matvec_Ax(x: Vector) -> Vector:
        return A_new.dot(x)

    def matvec_ATx(x: Vector) -> Vector:
        return A_new.T.dot(x)

    cache: dict[str, Matrix] = {}

    def matmat_ATsA(s: Vector) -> Matrix:
        if "B" not in cache:
            cache["B"] = A_new.toarray() if scipy.sparse.issparse(A_new) else A_new
        B = cache["B"]
        return B.T.dot(B * s.reshape(-1, 1))

    return LogSumExpOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, mu)
