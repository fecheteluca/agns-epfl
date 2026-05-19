r"""Ridge regression (L2-regularised least squares).

Canonical objective:

.. math::

    f(w) \;=\; \frac{1}{2m}\,\|X w - y\|^{2}
            \;+\; \frac{\mu}{2}\,\|w\|^{2}.

The objective is ``mu``-strongly convex when ``mu > 0``.  The Hessian
``H = X^T X / m + mu I`` is constant in ``w``, so the Lipschitz
constant of the Hessian is ``L_2 = 0`` --- a useful degenerate test
case for strongly-convex methods.

Sparse-aware: handles ``scipy.sparse`` matrices in ``func / grad /
hess / hess_vec`` (the full Hessian densifies to ``(n, n)`` regardless
of ``X``'s sparsity because ``X^T X`` is generally dense).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from agns.oracles.base import BaseSmoothOracle


class RidgeRegressionOracle(BaseSmoothOracle):
    r"""L2-regularised least squares oracle.

    Parameters
    ----------
    X : array or sparse matrix of shape ``(m, n)``
        Feature matrix.
    y : array of shape ``(m,)``
        Targets.
    reg : float
        Regularisation strength ``mu``.  ``reg > 0`` gives strong
        convexity; ``reg = 0`` is plain least squares.
    """

    def __init__(self, X: Any, y: NDArray[np.float64], reg: float = 0.0) -> None:
        if reg < 0.0:
            raise ValueError(f"reg must be non-negative, got {reg}")
        self.X = X
        self.m, self.n = X.shape
        self.y: NDArray[np.float64] = np.asarray(y, dtype=float).ravel()
        self.reg = reg
        # Precompute constant Hessian so hess() is O(1) after the first call.
        self._cached_H: NDArray[np.float64] | None = None

    def _residual(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        return cast("NDArray[np.float64]", self.X @ w - self.y)

    def func(self, w: NDArray[np.float64]) -> float:
        r = self._residual(w)
        return float(0.5 * r.dot(r) / self.m + 0.5 * self.reg * w.dot(w))

    def grad(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        r = self._residual(w)
        if sp.issparse(self.X):
            g_data: NDArray[np.float64] = cast("NDArray[np.float64]", self.X.T @ r / self.m)
        else:
            g_data = self.X.T.dot(r) / self.m
        return g_data + self.reg * w

    def hess(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        # Constant Hessian H = X^T X / m + reg I.
        if self._cached_H is None:
            if sp.issparse(self.X):
                XtX_sparse = self.X.T @ self.X
                XtX = XtX_sparse.toarray() if sp.issparse(XtX_sparse) else XtX_sparse
            else:
                XtX = self.X.T.dot(self.X)
            self._cached_H = XtX / self.m + self.reg * np.eye(self.n)
        return self._cached_H

    def hess_vec(self, w: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        Xv: NDArray[np.float64] = cast("NDArray[np.float64]", self.X @ v)
        if sp.issparse(self.X):
            res: NDArray[np.float64] = cast("NDArray[np.float64]", self.X.T @ Xv / self.m)
        else:
            res = self.X.T.dot(Xv) / self.m
        return res + self.reg * v


def make_ridge_regression_synthetic(
    m: int = 500,
    n: int = 50,
    reg: float = 1e-2,
    noise_std: float = 0.1,
    seed: int = 0,
) -> dict[str, Any]:
    """Random Gaussian design ``X`` + linear-Gaussian-noise targets.

    The closed-form optimum is computed analytically and supplied as
    ``f_star`` so stopping tests can use it.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, n))
    w_true = rng.standard_normal(n)
    y = X @ w_true + noise_std * rng.standard_normal(m)

    oracle = RidgeRegressionOracle(X, y, reg=reg)
    x_0 = np.zeros(n)
    # Closed form: w* = (X^T X / m + reg I)^{-1} (X^T y / m)
    H = X.T.dot(X) / m + reg * np.eye(n)
    w_star = np.linalg.solve(H, X.T.dot(y) / m)
    f_star = float(oracle.func(w_star))

    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": f_star,
        "B": H,
        "Binv": np.linalg.inv(H),
        "approx_hess_fn": None,
        "approx_hess_fn_wsm": None,
        "meta": {
            "type": "ridge_regression_synthetic",
            "m": m,
            "n": n,
            "reg": reg,
            "noise_std": noise_std,
            "seed": seed,
        },
    }


def make_ridge_regression_libsvm(
    path: str,
    reg: float = 1e-2,
    max_samples: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a ridge problem from a LIBSVM file.

    Labels are used directly as continuous targets (LIBSVM regression
    datasets); for binary-classification LIBSVM datasets the targets
    are coerced to ``{-1, +1}`` reals -- the resulting ridge problem
    is still a valid convex objective, but the model is no longer
    a probabilistic classifier.
    """
    try:
        from sklearn.datasets import load_svmlight_file
    except ImportError as e:  # pragma: no cover
        raise ImportError("scikit-learn required for LIBSVM loaders") from e

    X, y = load_svmlight_file(path)
    if sp.issparse(X) and X.shape[1] <= 5000:
        X = X.toarray()
    X = X.astype(np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    m, n = X.shape
    oracle = RidgeRegressionOracle(X, y, reg=reg)
    x_0 = np.zeros(n)
    if sp.issparse(X):
        H = (X.T @ X).toarray() / m + reg * np.eye(n)
    else:
        H = X.T.dot(X) / m + reg * np.eye(n)
    # Closed-form f_star.
    Xt_y = X.T.dot(y) / m if not sp.issparse(X) else (X.T @ y) / m
    w_star = np.linalg.solve(H, Xt_y)
    f_star = float(oracle.func(w_star))

    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": f_star,
        "B": H,
        "Binv": np.linalg.inv(H),
        "approx_hess_fn": None,
        "approx_hess_fn_wsm": None,
        "meta": {
            "type": "ridge_regression_libsvm",
            "path": str(path),
            "m": m,
            "n": n,
            "reg": reg,
        },
    }


__all__ = [
    "RidgeRegressionOracle",
    "make_ridge_regression_libsvm",
    "make_ridge_regression_synthetic",
]
