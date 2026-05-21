r"""Multiclass softmax regression with optional L2 regularisation.

Canonical objective: with ``X \in R^{m x n}``, class labels
``y \in \{0, 1, ..., K-1\}^m``, and weights packed as
``W \in R^{n x K}``,

.. math::

    f(W) \;=\; \frac{1}{m} \sum_{i=1}^{m}
                \Bigl[
                  \log\sum_{c=0}^{K-1} \exp(\langle x_i, w_c\rangle)
                  \;-\; \langle x_i, w_{y_i}\rangle
                \Bigr]
            \;+\; \frac{\lambda}{2}\,\|W\|_{F}^{2}.

Convex, ``lambda``-strongly convex when ``lambda > 0``, with Lipschitz
Hessian (uniformly bounded by ``||X||^2 / m + lambda``).

This oracle operates on a *flattened* weight vector
``w = W.flatten()`` of size ``n * K`` so it slots into the same
:class:`agns.oracles.base.BaseSmoothOracle` interface as the binary
oracles.

History keys produced are the same as for binary LR; this is a clean
warm-up problem for NN experiments on MNIST.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp, softmax

from agns.oracles.base import BaseSmoothOracle


class SoftmaxRegressionOracle(BaseSmoothOracle):
    """Multinomial logistic regression oracle (flattened weights)."""

    def __init__(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        n_classes: int,
        reg: float = 0.0,
    ) -> None:
        if reg < 0.0:
            raise ValueError(f"reg must be non-negative, got {reg}")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=np.int64).ravel()
        if self.y.min() < 0 or self.y.max() >= n_classes:
            raise ValueError(
                f"y values must lie in [0, {n_classes - 1}], got [{self.y.min()}, {self.y.max()}]"
            )
        self.m, self.n = self.X.shape
        self.K = n_classes
        self.reg = reg

    def _unpack(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        return w.reshape(self.n, self.K)

    def _logits(self, W: NDArray[np.float64]) -> NDArray[np.float64]:
        return cast("NDArray[np.float64]", self.X @ W)  # (m, K)

    def func(self, w: NDArray[np.float64]) -> float:
        W = self._unpack(w)
        logits = self._logits(W)
        lse = logsumexp(logits, axis=1)  # (m,)
        true_score = logits[np.arange(self.m), self.y]
        cross_entropy = float(np.mean(lse - true_score))
        reg_term = 0.5 * self.reg * float(w.dot(w))
        return cross_entropy + reg_term

    def grad(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        W = self._unpack(w)
        logits = self._logits(W)
        P = softmax(logits, axis=1)  # (m, K)
        # Subtract one-hot at the true class.
        P[np.arange(self.m), self.y] -= 1.0
        # dL/dW = (1/m) X^T (P - Y_onehot), shape (n, K)
        dW = self.X.T.dot(P) / self.m
        return cast("NDArray[np.float64]", dW.flatten() + self.reg * w)

    def hess_vec(self, w: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Block-diagonal-ish softmax Hessian-vector product.

        Closed form (see Bouchard 2007, Appendix):

            d^2 L_i / dW^2 [V] = (1/m) x_i x_i^T V (diag(p_i) - p_i p_i^T)

        We aggregate over ``i`` matrix-free.
        """
        W = self._unpack(w)
        V = v.reshape(self.n, self.K)
        logits = self._logits(W)
        P = softmax(logits, axis=1)  # (m, K)
        # u_i = (X V)_i  -> (m, K)
        XV = self.X @ V
        # Per-row Hess action: P * XV - P (P, XV)
        pxv = (P * XV).sum(axis=1, keepdims=True)  # (m, 1) -- dot(P_i, XV_i)
        action = P * XV - P * pxv  # (m, K)
        # Sum into dW shape: (n, K) = X^T action / m
        dW = self.X.T.dot(action) / self.m
        return cast("NDArray[np.float64]", dW.flatten() + self.reg * v)

    def hess(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        """Materialise the full (nK) x (nK) Hessian.

        Only feasible for small problems; Newton-style methods that need
        the dense factorisation will pay an O((nK)^3) cost.  For large
        ``n*K``, prefer the Hessian-vector form via :meth:`hess_vec`.
        """
        nK = self.n * self.K
        H = np.zeros((nK, nK))
        ek = np.eye(nK)
        for k in range(nK):
            H[:, k] = self.hess_vec(w, ek[:, k])
        return 0.5 * (H + H.T)  # symmetrise


def make_softmax_regression_synthetic(
    m: int = 500,
    n: int = 20,
    K: int = 3,
    reg: float = 1e-3,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, n))
    W_true = rng.standard_normal((n, K))
    logits = X @ W_true + 0.1 * rng.standard_normal((m, K))
    y = np.argmax(logits, axis=1)

    oracle = SoftmaxRegressionOracle(X, y, n_classes=K, reg=reg)
    nK = n * K
    x_0 = np.zeros(nK)
    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": None,
        "B": None,
        "Binv": None,
        "approx_hess_fn": None,
        "approx_hess_fn_wsm": None,
        "meta": {
            "type": "softmax_regression_synthetic",
            "m": m,
            "n": n,
            "K": K,
            "reg": reg,
            "seed": seed,
        },
    }


__all__ = ["SoftmaxRegressionOracle", "make_softmax_regression_synthetic"]
