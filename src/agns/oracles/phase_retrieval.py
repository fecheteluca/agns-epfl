r"""Phase retrieval (non-convex).

Canonical objective: recover ``x \in R^n`` from quartic measurements

.. math::

    y_i \;\approx\; (a_i^{\top} x)^{2},    \qquad i = 1, ..., m,

by minimising

.. math::

    f(x) \;=\; \frac{1}{2m} \sum_{i=1}^{m}
        \bigl((a_i^{\top} x)^{2} - y_i\bigr)^{2}.

This is the classical Wirtinger-flow loss landscape (Candes-Li-Soltanolkotabi
2015) -- non-convex, with multiple local minima and a single up-to-sign
global optimum.  Used as a non-convex stress test.

The Hessian is

.. math::

    \nabla^{2} f(x) \;=\;
      \frac{1}{m} \sum_{i=1}^{m}
        \Bigl(
          6 (a_i^{\top} x)^{2} - 2 y_i
        \Bigr) a_i a_i^{\top}.

Sparse-aware in the same sense as the LR / LASSO oracles.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from agns.oracles.base import BaseSmoothOracle


class PhaseRetrievalOracle(BaseSmoothOracle):
    """Wirtinger-flow / quartic-residual phase retrieval oracle."""

    def __init__(
        self,
        A: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> None:
        self.A = np.asarray(A, dtype=float)
        self.y = np.asarray(y, dtype=float).ravel()
        self.m, self.n = self.A.shape
        if self.y.shape[0] != self.m:
            raise ValueError(f"y has {self.y.shape[0]} entries; expected {self.m}")

    def _ax(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return cast("NDArray[np.float64]", self.A @ x)

    def func(self, x: NDArray[np.float64]) -> float:
        ax = self._ax(x)
        sq = ax * ax
        r = sq - self.y
        return float(0.5 * (r * r).sum() / self.m)

    def grad(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        ax = self._ax(x)
        sq = ax * ax
        r = sq - self.y  # (m,)
        # d/dx 0.5 r_i^2 = r_i * 2 (a_i^T x) a_i / m
        coeffs = 2.0 * r * ax / self.m  # (m,)
        return cast("NDArray[np.float64]", self.A.T.dot(coeffs))

    def hess(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        ax = self._ax(x)
        sq = ax * ax
        # d_i = (6 (a_i^T x)^2 - 2 y_i) / m
        d = (6.0 * sq - 2.0 * self.y) / self.m
        # H = A^T diag(d) A
        DA = d[:, None] * self.A
        return cast("NDArray[np.float64]", self.A.T @ DA)

    def hess_vec(self, x: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        ax = self._ax(x)
        sq = ax * ax
        d = (6.0 * sq - 2.0 * self.y) / self.m
        Av: NDArray[np.float64] = cast("NDArray[np.float64]", self.A @ v)
        DAv = d * Av
        return cast("NDArray[np.float64]", self.A.T.dot(DAv))


def make_phase_retrieval_synthetic(
    m: int = 600,
    n: int = 30,
    noise_std: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Random Gaussian measurements + quartic targets ``y = (A x_true)^2``."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n)
    y = (A @ x_true) ** 2 + (noise_std * rng.standard_normal(m) if noise_std > 0 else 0.0)

    oracle = PhaseRetrievalOracle(A, y)
    # Spectral initialisation (Candes et al.): leading eigenvector of
    # (1/m) sum_i y_i a_i a_i^T.
    Y = (y[:, None] * A).T @ A / m
    eigvals, eigvecs = np.linalg.eigh(Y)
    spec_init = eigvecs[:, -1] * np.sqrt(max(eigvals[-1], 0.0))
    x_0 = spec_init + 0.1 * rng.standard_normal(n)

    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": 0.0,
        "B": None,
        "Binv": None,
        "approx_hess_fn": None,
        "approx_hess_fn_wsm": None,
        "meta": {
            "type": "phase_retrieval_synthetic",
            "m": m,
            "n": n,
            "noise_std": noise_std,
            "seed": seed,
        },
    }


__all__ = ["PhaseRetrievalOracle", "make_phase_retrieval_synthetic"]
