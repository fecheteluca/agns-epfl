r"""L2-regularised binary logistic regression.

Canonical objective (with labels ``y \in {-1, +1}``):

.. math::

    f(w) \;=\; \frac{1}{m} \sum_{i=1}^{m}
                \log\bigl(1 + \exp(-y_i\, \langle x_i, w\rangle)\bigr)
            \;+\; \frac{\lambda}{2}\,\|w\|^{2}.

The objective is convex; ``mu``-strongly convex when ``lambda > 0``
(with ``mu = lambda``); and has a globally Lipschitz Hessian whose
constant depends on the data scale.  Specifically the Hessian is

.. math::

    \nabla^{2} f(w)
    \;=\; \frac{1}{m}\, X^{\top} D(w) X \;+\; \lambda I,

with ``D(w) = diag(sigma(z) (1 - sigma(z)))`` and
``z = -y \odot (Xw)`` the margin.

The oracle is sparse-aware: scipy-sparse ``X`` is handled efficiently
in all of ``func / grad / hess / hess_vec`` (the full Hessian matrix
is densified only at the final ``X.T @ DX`` step, which is unavoidable
when ``X`` has dense columns).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from agns.oracles.base import BaseSmoothOracle


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Element-wise stable logistic sigmoid."""
    # Numerically stable form: ``sigma(z) = 1/(1+exp(-z))`` for z >= 0,
    # ``exp(z)/(1+exp(z))`` for z < 0 (avoids overflow at large +z).
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg = ~pos
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def _log1pexp(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Element-wise stable ``log(1 + exp(z))``."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = z[pos] + np.log1p(np.exp(-z[pos]))
    neg = ~pos
    out[neg] = np.log1p(np.exp(z[neg]))
    return out


class LogisticRegressionOracle(BaseSmoothOracle):
    r"""Binary L2-regularised logistic regression.

    Parameters
    ----------
    X : ``ndarray`` or ``scipy.sparse`` matrix, shape ``(m, n)``
        Feature matrix.  Sparse formats are supported transparently.
    y : ``ndarray``, shape ``(m,)``
        Labels in ``{-1, +1}``.  Other label encodings (e.g. ``{0, 1}``)
        are detected and converted.
    reg : float, default ``0.0``
        L2 regularisation strength ``lambda``.  ``reg > 0`` makes the
        objective ``reg``-strongly convex.
    """

    def __init__(
        self,
        X: Any,
        y: NDArray[np.float64],
        reg: float = 0.0,
    ) -> None:
        if reg < 0.0:
            raise ValueError(f"reg must be non-negative, got {reg}")
        self.X = X
        self.m, self.n = X.shape
        # Normalise labels to {-1, +1}.
        y_arr: NDArray[np.float64] = np.asarray(y, dtype=float).ravel()
        uniq = np.unique(y_arr)
        if set(uniq.tolist()) <= {0.0, 1.0}:
            y_arr = (2.0 * y_arr - 1.0).astype(np.float64)
        elif not set(uniq.tolist()) <= {-1.0, 1.0}:
            raise ValueError(f"y must contain values in {{-1, +1}} or {{0, 1}}; got {uniq}")
        self.y = y_arr
        self.reg = reg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _margin(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute ``z_i = -y_i <x_i, w>``."""
        Xw: NDArray[np.float64] = cast("NDArray[np.float64]", self.X @ w)
        return -self.y * Xw

    # ------------------------------------------------------------------
    # Required oracle interface
    # ------------------------------------------------------------------
    def func(self, w: NDArray[np.float64]) -> float:
        z = self._margin(w)
        avg_loss = float(np.mean(_log1pexp(z)))
        reg_term = 0.5 * self.reg * float(w.dot(w))
        return avg_loss + reg_term

    def grad(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        z = self._margin(w)
        s = _sigmoid(z)
        # d/dw loss_i = -y_i sigma(z_i) x_i / m
        coeffs = -self.y * s / self.m
        if sp.issparse(self.X):
            g_data: NDArray[np.float64] = cast("NDArray[np.float64]", self.X.T @ coeffs)
        else:
            g_data = self.X.T.dot(coeffs)
        return g_data + self.reg * w

    def hess(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        z = self._margin(w)
        s = _sigmoid(z)
        d = s * (1.0 - s) / self.m  # shape (m,)
        # H = X^T diag(d) X + reg I
        if sp.issparse(self.X):
            # Sparse-aware: form (D X)^T X.
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
        """Hessian-vector product ``H(w) v`` without forming ``H``."""
        z = self._margin(w)
        s = _sigmoid(z)
        d = s * (1.0 - s) / self.m
        Xv: NDArray[np.float64] = cast("NDArray[np.float64]", self.X @ v)
        DXv = d * Xv
        if sp.issparse(self.X):
            res: NDArray[np.float64] = cast("NDArray[np.float64]", self.X.T @ DXv)
        else:
            res = self.X.T.dot(DXv)
        if self.reg > 0.0:
            res = res + self.reg * v
        return res


# ---------------------------------------------------------------------------
# Problem factories
# ---------------------------------------------------------------------------


def make_logistic_regression_synthetic(
    m: int = 1000,
    n: int = 100,
    reg: float = 1e-3,
    seed: int = 0,
) -> dict[str, Any]:
    """Random Gaussian features + balanced labels.

    Constructs ``X ~ N(0, I)``, ``w_true ~ N(0, I)``, and
    ``y = sign(X w_true + noise)``.  Returns the standard problem dict
    so :func:`agns.registry.run_method` consumes it directly.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, n))
    w_true = rng.standard_normal(n)
    logits = X.dot(w_true) + 0.1 * rng.standard_normal(m)
    y = np.sign(logits)
    y[y == 0] = 1.0

    oracle = LogisticRegressionOracle(X, y, reg=reg)
    x_0 = np.zeros(n)
    # B = X^T X / m + reg I  is a natural metric (Gauss-Newton kernel + reg).
    gram = X.T.dot(X) / m + max(reg, 1e-8) * np.eye(n)
    Binv = np.linalg.inv(gram)

    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": None,  # unknown analytically
        "B": gram,
        "Binv": Binv,
        "approx_hess_fn": None,
        "approx_hess_fn_wsm": None,
        "meta": {
            "type": "logistic_regression_synthetic",
            "m": m,
            "n": n,
            "reg": reg,
            "seed": seed,
        },
    }


def make_logistic_regression_libsvm(
    path: str,
    reg: float = 1e-3,
    max_samples: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Load a LIBSVM file and build an L2-regularised LR problem."""
    try:
        from sklearn.datasets import load_svmlight_file
    except ImportError as e:  # pragma: no cover
        raise ImportError("scikit-learn required for LIBSVM loaders") from e

    X, y = load_svmlight_file(path)
    if sp.issparse(X):
        # Heuristic: if n_features is small enough to densify, do so for
        # the Newton-style methods that need a full Hessian.  Otherwise
        # keep sparse and rely on hess_vec.
        if X.shape[1] <= 5000:
            X = X.toarray()
        X = X.astype(np.float64)
    else:  # pragma: no cover
        X = np.asarray(X, dtype=np.float64)

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
    oracle = LogisticRegressionOracle(X, y, reg=reg)
    x_0 = np.zeros(n)

    if sp.issparse(X):
        gram = (X.T @ X).toarray() / m + max(reg, 1e-8) * np.eye(n)
    else:
        gram = X.T.dot(X) / m + max(reg, 1e-8) * np.eye(n)
    Binv = np.linalg.inv(gram)

    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": None,
        "B": gram,
        "Binv": Binv,
        "approx_hess_fn": None,
        "approx_hess_fn_wsm": None,
        "meta": {
            "type": "logistic_regression_libsvm",
            "path": str(path),
            "m": m,
            "n": n,
            "reg": reg,
        },
    }


__all__ = [
    "LogisticRegressionOracle",
    "make_logistic_regression_libsvm",
    "make_logistic_regression_synthetic",
]
