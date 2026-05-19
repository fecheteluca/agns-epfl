r"""LogSumExp oracle and problem factories.

The objective is

.. math::

    f(x) = \mu \, \log \sum_{i=1}^{m} \exp\bigl((\langle a_i, x\rangle
                                                  - b_i) / \mu\bigr),

with rows ``a_i`` of an ``m x n`` matrix ``A`` and a vector ``b`` in
``R^m``.  The oracle exposes ``func / grad / hess / hess_vec /
third_vec_vec`` plus an auxiliary ``_update_a_and_pi`` used by the
Hessian-approximation builder in :mod:`agns.approximations`.

Two factories are provided: ``make_logsumexp_synthetic`` for random
Gaussian ``A`` and ``make_logsumexp_real`` for LIBSVM datasets.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.special import logsumexp, softmax

from agns.approximations import approx_hess_fn_logsumexp
from agns.oracles.base import BaseSmoothOracle


def _arrays_equal(a: NDArray[np.float64] | None, b: NDArray[np.float64]) -> bool:
    """``np.array_equal`` lifted to the ``None``-comparisons we use for caching."""
    if a is None:
        return False
    return bool(np.array_equal(a, b))


class LogSumExpOracle(BaseSmoothOracle):
    """Oracle for ``f(x) = mu * log sum exp((<a_i, x> - b_i) / mu)``.

    Accepts callable ``matvec_Ax``, ``matvec_ATx``, ``matmat_ATsA`` so the
    same code works for dense and sparse ``A``.  Caches softmax weights
    across calls at the same point to amortise the dominant
    ``logsumexp`` cost.
    """

    def __init__(
        self,
        matvec_Ax: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        matvec_ATx: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        matmat_ATsA: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        b: NDArray[np.float64],
        mu: float,
    ) -> None:
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = np.copy(b)
        self.mu = mu
        self.mu_inv = 1.0 / mu
        self.last_x: NDArray[np.float64] | None = None
        self.last_x_pi: NDArray[np.float64] | None = None
        self.last_v: NDArray[np.float64] | None = None

    def func(self, x: NDArray[np.float64]) -> float:
        self._update_a(x)
        return float(self.mu * logsumexp(self.a))

    def grad(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        self._update_a_and_pi(x)
        return self.AT_pi

    def hess(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        self._update_a_and_pi(x)
        return self.mu_inv * (self.matmat_ATsA(self.pi) - np.outer(self.AT_pi, self.AT_pi.T))

    def hess_vec(self, x: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        self._update_hess_vec(x, v)
        return cast("NDArray[np.float64]", self._hess_vec)

    def third_vec_vec(self, x: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        self._update_hess_vec(x, v)
        result = self.mu_inv * (
            self.mu_inv * self.matvec_ATx(self.pi * (self.Av * (self.Av - self.AT_pi_v)))
            - self.AT_pi_v * self._hess_vec
            - self._hess_vec.dot(v) * self.AT_pi
        )
        return cast("NDArray[np.float64]", result)

    def _update_a(self, x: NDArray[np.float64]) -> None:
        if not _arrays_equal(self.last_x, x):
            self.last_x = np.copy(x)
            self.a = self.mu_inv * (self.matvec_Ax(x) - self.b)

    def _update_a_and_pi(self, x: NDArray[np.float64]) -> None:
        self._update_a(x)
        if not _arrays_equal(self.last_x_pi, x):
            self.last_x_pi = np.copy(x)
            self.pi = softmax(self.a)
            self.AT_pi = self.matvec_ATx(self.pi)

    def _update_hess_vec(self, x: NDArray[np.float64], v: NDArray[np.float64]) -> None:
        if not _arrays_equal(self.last_x, x) or not _arrays_equal(self.last_v, v):
            self._update_a_and_pi(x)
            self.last_v = np.copy(v)
            self.Av = self.matvec_Ax(v)
            self.AT_pi_v = self.AT_pi.dot(v)
            self._hess_vec = self.mu_inv * (
                self.matvec_ATx(self.pi * self.Av) - self.AT_pi_v * self.AT_pi
            )


def _densify(A: Any) -> NDArray[np.float64]:
    """Return a dense ``ndarray`` view of ``A``, regardless of sparsity."""
    return cast("NDArray[np.float64]", A.toarray() if sp.issparse(A) else np.asarray(A))


def create_log_sum_exp_oracle(A: Any, b: NDArray[np.float64], mu: float) -> LogSumExpOracle:
    """Build a LogSumExp oracle from a dense or sparse ``A``."""

    def matvec_Ax(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return cast("NDArray[np.float64]", A.dot(x))

    def matvec_ATx(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return cast("NDArray[np.float64]", A.T.dot(x))

    cached_dense: NDArray[np.float64] | None = None

    def matmat_ATsA(s: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal cached_dense
        if cached_dense is None:
            cached_dense = _densify(A)
        return cast("NDArray[np.float64]", cached_dense.T.dot(cached_dense * s.reshape(-1, 1)))

    return LogSumExpOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, mu)


def create_log_sum_exp_zero_oracle(A: Any, b: NDArray[np.float64], mu: float) -> LogSumExpOracle:
    """LogSumExp oracle re-centred so that ``argmin f = 0``.

    Computes the gradient ``g`` of the un-shifted oracle at ``x = 0`` and
    re-uses the same construction with ``A - g`` so that the new oracle's
    minimiser is the origin by construction.
    """
    oracle_0 = create_log_sum_exp_oracle(A, b, mu)
    g = oracle_0.grad(np.zeros(A.shape[1]))
    A_new = A - g

    def matvec_Ax(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return cast("NDArray[np.float64]", A_new.dot(x))

    def matvec_ATx(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return cast("NDArray[np.float64]", A_new.T.dot(x))

    cached_dense: NDArray[np.float64] | None = None

    def matmat_ATsA(s: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal cached_dense
        if cached_dense is None:
            cached_dense = _densify(A_new)
        return cast("NDArray[np.float64]", cached_dense.T.dot(cached_dense * s.reshape(-1, 1)))

    return LogSumExpOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, mu)


# ---------------------------------------------------------------------------
# Problem factories
# ---------------------------------------------------------------------------


def make_logsumexp_synthetic(
    n: int = 1000, m: int = 1000, mu: float = 1.0, seed: int = 42
) -> dict[str, Any]:
    """``min_x mu log sum_i exp(<a_i, x>/mu)`` with random ``A`` and optimum at 0.

    The metric ``B = A A^T`` is the Weighted Gauss-Newton Gram matrix and
    is the natural choice for this problem class.
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, (n, m))
    b = rng.uniform(-1.0, 1.0, m)

    oracle = create_log_sum_exp_zero_oracle(A.T, b, mu)
    x_0 = np.ones(n)
    f_star = oracle.func(np.zeros(n))
    B = A.dot(A.T)
    Binv = np.linalg.inv(B)

    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": f_star,
        "B": B,
        "Binv": Binv,
        "approx_hess_fn": approx_hess_fn_logsumexp,
        "approx_hess_fn_wsm": None,
        "meta": {
            "type": "logsumexp_synthetic",
            "n": n,
            "m": m,
            "mu": mu,
            "seed": seed,
        },
    }


def make_logsumexp_real(
    path: str,
    mu: float = 1.0,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """LogSumExp on a LIBSVM dataset.  Optimum at ``x = 0`` by construction."""
    try:
        from sklearn.datasets import load_svmlight_file
    except ImportError as e:  # pragma: no cover
        raise ImportError("scikit-learn required for real datasets") from e

    X, _ = load_svmlight_file(path)
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float64)

    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]

    m, n = X.shape
    b = np.zeros(m)
    oracle = create_log_sum_exp_zero_oracle(X, b, mu)
    rng2 = np.random.default_rng(seed + 1)
    x_0 = rng2.standard_normal(n)
    f_star = oracle.func(np.zeros(n))

    gram = X.T.dot(X)
    reg = 1e-8 * np.eye(n)
    B = gram + reg
    Binv = np.linalg.inv(B)

    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": f_star,
        "B": B,
        "Binv": Binv,
        "approx_hess_fn": approx_hess_fn_logsumexp,
        "approx_hess_fn_wsm": None,
        "meta": {"type": "logsumexp_real", "path": str(path), "n": n, "m": m, "mu": mu},
    }


__all__ = [
    "LogSumExpOracle",
    "create_log_sum_exp_oracle",
    "create_log_sum_exp_zero_oracle",
    "make_logsumexp_real",
    "make_logsumexp_synthetic",
]
