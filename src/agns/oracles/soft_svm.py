r"""Smoothed quadratic-hinge soft-margin SVM.

Canonical objective (squared-hinge variant, labels ``y \in {-1, +1}``):

.. math::

    f(w) \;=\; \frac{1}{2m} \sum_{i=1}^{m}
                \max\bigl(0, 1 - y_i\langle x_i, w\rangle\bigr)^{2}
            \;+\; \frac{\mu}{2}\,\|w\|^{2}.

The squared hinge ``max(0, 1 - z)^2`` is once-but-not-twice-Lipschitz
at the boundary ``z = 1`` (the second derivative jumps from 2 to 0).
The Hessian is

.. math::

    H(w) \;=\; \frac{1}{m} X^{\top} \mathrm{diag}(\mathbb{1}\{1 - y \odot Xw > 0\}) X
            \;+\; \mu I,

with the indicator-mask making it piecewise constant in ``w``.  This
matches the standard squared-hinge SVM analyses (e.g., Chapelle 2007,
*Training a support vector machine in the primal*).  It is convex,
``mu``-strongly convex for ``mu > 0``, and globally Lipschitz-Hessian
when the data are bounded.

Sparse-aware.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from agns.oracles.base import BaseSmoothOracle


class SoftSVMOracle(BaseSmoothOracle):
    """Squared-hinge soft-margin SVM (sparse-aware)."""

    def __init__(self, X: Any, y: NDArray[np.float64], reg: float = 1e-3) -> None:
        if reg < 0.0:
            raise ValueError(f"reg must be non-negative, got {reg}")
        self.X = X
        self.m, self.n = X.shape
        y_arr: NDArray[np.float64] = np.asarray(y, dtype=float).ravel()
        uniq = np.unique(y_arr)
        if set(uniq.tolist()) <= {0.0, 1.0}:
            y_arr = (2.0 * y_arr - 1.0).astype(np.float64)
        elif not set(uniq.tolist()) <= {-1.0, 1.0}:
            raise ValueError(f"y must be in {{-1, +1}} or {{0, 1}}; got {uniq}")
        self.y = y_arr
        self.reg = reg

    def _margin_residual(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``1 - y_i <x_i, w>`` (the *unclipped* margin slack)."""
        Xw: NDArray[np.float64] = cast("NDArray[np.float64]", self.X @ w)
        return 1.0 - self.y * Xw

    def func(self, w: NDArray[np.float64]) -> float:
        s = self._margin_residual(w)
        s_plus = np.maximum(s, 0.0)
        return float(0.5 * np.sum(s_plus * s_plus) / self.m + 0.5 * self.reg * w.dot(w))

    def grad(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        s = self._margin_residual(w)
        active = s > 0.0
        # d/dw [0.5 s_i^2] = -y_i s_i x_i  on the active set.
        coeffs = -self.y * np.where(active, s, 0.0) / self.m
        if sp.issparse(self.X):
            g_data: NDArray[np.float64] = cast("NDArray[np.float64]", self.X.T @ coeffs)
        else:
            g_data = self.X.T.dot(coeffs)
        return g_data + self.reg * w

    def hess(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        s = self._margin_residual(w)
        d = (s > 0.0).astype(float) / self.m
        if sp.issparse(self.X):
            X_dense = self.X.toarray()
            DX = d[:, None] * X_dense
            H = X_dense.T @ DX
        else:
            DX = d[:, None] * self.X
            H = self.X.T @ DX
        if self.reg > 0.0:
            H = H + self.reg * np.eye(self.n)
        return cast("NDArray[np.float64]", H)

    def hess_vec(self, w: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        s = self._margin_residual(w)
        d = (s > 0.0).astype(float) / self.m
        Xv: NDArray[np.float64] = cast("NDArray[np.float64]", self.X @ v)
        DXv = d * Xv
        if sp.issparse(self.X):
            res: NDArray[np.float64] = cast("NDArray[np.float64]", self.X.T @ DXv)
        else:
            res = self.X.T.dot(DXv)
        if self.reg > 0.0:
            res = res + self.reg * v
        return res


def make_soft_svm_synthetic(
    m: int = 500,
    n: int = 50,
    reg: float = 1e-2,
    seed: int = 0,
) -> dict[str, Any]:
    """Random Gaussian features + sign-of-projection labels (separable)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, n))
    w_true = rng.standard_normal(n)
    y = np.sign(X.dot(w_true) + 0.5 * rng.standard_normal(m))
    y[y == 0] = 1.0

    oracle = SoftSVMOracle(X, y, reg=reg)
    x_0 = np.zeros(n)
    gram = X.T.dot(X) / m + max(reg, 1e-8) * np.eye(n)
    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": None,
        "B": gram,
        "Binv": np.linalg.inv(gram),
        "approx_hess_fn": None,
        "approx_hess_fn_wsm": None,
        "meta": {
            "type": "soft_svm_synthetic",
            "m": m,
            "n": n,
            "reg": reg,
            "seed": seed,
        },
    }


def make_soft_svm_libsvm(
    path: str,
    reg: float = 1e-2,
    max_samples: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Soft-margin SVM on a LIBSVM dataset."""
    try:
        from sklearn.datasets import load_svmlight_file
    except ImportError as e:  # pragma: no cover
        raise ImportError("scikit-learn required") from e

    X, y = load_svmlight_file(path)
    if sp.issparse(X) and X.shape[1] <= 5000:
        X = X.toarray()
    X = X.astype(np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    uniq = np.unique(y)
    if uniq.size == 2 and not set(uniq.tolist()) <= {-1.0, 0.0, 1.0}:
        y = np.where(y == uniq.max(), 1.0, -1.0)
    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    m, n = X.shape
    oracle = SoftSVMOracle(X, y, reg=reg)
    x_0 = np.zeros(n)
    if sp.issparse(X):
        gram = (X.T @ X).toarray() / m + max(reg, 1e-8) * np.eye(n)
    else:
        gram = X.T.dot(X) / m + max(reg, 1e-8) * np.eye(n)

    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": None,
        "B": gram,
        "Binv": np.linalg.inv(gram),
        "approx_hess_fn": None,
        "approx_hess_fn_wsm": None,
        "meta": {
            "type": "soft_svm_libsvm",
            "path": str(path),
            "m": m,
            "n": n,
            "reg": reg,
        },
    }


__all__ = [
    "SoftSVMOracle",
    "make_soft_svm_libsvm",
    "make_soft_svm_synthetic",
]
