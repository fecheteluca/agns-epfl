r"""Smoothed LASSO regression with H\"older Hessian.

Canonical objective:

.. math::

    f(w) \;=\; \frac{1}{2m}\, \|Xw - y\|^{2}
            \;+\; \lambda \sum_{j=1}^{n} \phi_{\varepsilon}(w_j),

with the smoothed absolute value

.. math::

    \phi_{\varepsilon}(t) \;=\; \sqrt{t^{2} + \varepsilon^{2}} \;-\; \varepsilon.

This is the so-called *Pseudo-Huber* smoothing of ``|t|``: as
``epsilon -> 0``, ``phi_eps(t) -> |t|`` pointwise.  The smoothed
``phi_eps`` is :math:`C^{\infty}`, with

.. math::

    \phi'_{\varepsilon}(t) = \frac{t}{\sqrt{t^{2} + \varepsilon^{2}}},
    \qquad
    \phi''_{\varepsilon}(t) = \frac{\varepsilon^{2}}{(t^{2} + \varepsilon^{2})^{3/2}}.

The Hessian of ``f`` is

.. math::

    \nabla^{2} f(w)
    \;=\; \frac{1}{m} X^{\top} X
    \;+\; \lambda\, \mathrm{diag}\!\left(
        \frac{\varepsilon^{2}}{(w_j^{2}+\varepsilon^{2})^{3/2}}
    \right).

The diagonal term is Holder-continuous of exponent ``nu = 1/2`` but is
**not** Lipschitz (the third derivative of ``phi_eps`` blows up as
``|t| -> infinity``).  This makes the smoothed LASSO a natural Holder
benchmark.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from agns.oracles.base import BaseSmoothOracle


def _phi(t: NDArray[np.float64], eps: float) -> NDArray[np.float64]:
    return np.sqrt(t * t + eps * eps) - eps


def _phi_prime(t: NDArray[np.float64], eps: float) -> NDArray[np.float64]:
    return t / np.sqrt(t * t + eps * eps)


def _phi_double_prime(t: NDArray[np.float64], eps: float) -> NDArray[np.float64]:
    denom = (t * t + eps * eps) ** 1.5
    return eps * eps / denom


class SmoothedLassoOracle(BaseSmoothOracle):
    r"""Smoothed-L1-regularised least squares with Pseudo-Huber penalty.

    Parameters
    ----------
    X, y : feature matrix and targets (shapes ``(m, n)`` and ``(m,)``).
    reg : float
        L1 regularisation strength ``lambda``.
    smoothing : float
        Pseudo-Huber smoothing parameter ``epsilon > 0``.  Smaller values
        approach the non-smooth L1; the Hessian's H\"older constant
        scales as ``L_{1/2} = O(1/epsilon^{1/2})``.
    """

    def __init__(
        self,
        X: Any,
        y: NDArray[np.float64],
        reg: float = 1e-2,
        smoothing: float = 1e-2,
    ) -> None:
        if reg < 0.0:
            raise ValueError(f"reg must be non-negative, got {reg}")
        if smoothing <= 0.0:
            raise ValueError(f"smoothing must be positive, got {smoothing}")
        self.X = X
        self.m, self.n = X.shape
        self.y: NDArray[np.float64] = np.asarray(y, dtype=float).ravel()
        self.reg = reg
        self.eps_sm = smoothing
        # Cache constant part of the Hessian: ``X^T X / m``.
        self._cached_XtX_over_m: NDArray[np.float64] | None = None

    def _residual(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        return cast("NDArray[np.float64]", self.X @ w - self.y)

    def _XtX_over_m(self) -> NDArray[np.float64]:
        if self._cached_XtX_over_m is None:
            if sp.issparse(self.X):
                M = self.X.T @ self.X
                M = M.toarray() if sp.issparse(M) else M
            else:
                M = self.X.T.dot(self.X)
            self._cached_XtX_over_m = M / self.m
        return self._cached_XtX_over_m

    def func(self, w: NDArray[np.float64]) -> float:
        r = self._residual(w)
        ls_term = 0.5 * float(r.dot(r)) / self.m
        reg_term = self.reg * float(np.sum(_phi(w, self.eps_sm)))
        return float(ls_term + reg_term)

    def grad(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        r = self._residual(w)
        if sp.issparse(self.X):
            g_data: NDArray[np.float64] = cast("NDArray[np.float64]", self.X.T @ r / self.m)
        else:
            g_data = self.X.T.dot(r) / self.m
        return g_data + self.reg * _phi_prime(w, self.eps_sm)

    def hess(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        XtX_m = self._XtX_over_m()
        diag = self.reg * _phi_double_prime(w, self.eps_sm)
        return XtX_m + np.diag(diag)

    def hess_vec(self, w: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        Xv: NDArray[np.float64] = cast("NDArray[np.float64]", self.X @ v)
        if sp.issparse(self.X):
            res: NDArray[np.float64] = cast("NDArray[np.float64]", self.X.T @ Xv / self.m)
        else:
            res = self.X.T.dot(Xv) / self.m
        return res + self.reg * _phi_double_prime(w, self.eps_sm) * v


def make_smoothed_lasso_synthetic(
    m: int = 500,
    n: int = 50,
    s: int = 5,
    reg: float = 5e-2,
    smoothing: float = 1e-2,
    noise_std: float = 0.1,
    seed: int = 0,
) -> dict[str, Any]:
    """Random Gaussian design + ``s``-sparse signal + Pseudo-Huber smoothing.

    The closed-form optimum is unavailable for the smoothed problem,
    so ``f_star`` is left ``None``.  Use a long enough budget for the
    methods to plateau.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, n))
    w_true = np.zeros(n)
    support = rng.choice(n, size=s, replace=False)
    w_true[support] = rng.standard_normal(s)
    y = X @ w_true + noise_std * rng.standard_normal(m)

    oracle = SmoothedLassoOracle(X, y, reg=reg, smoothing=smoothing)
    x_0 = np.zeros(n)
    gram = X.T.dot(X) / m + 1e-8 * np.eye(n)
    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": None,
        "B": gram,
        "Binv": np.linalg.inv(gram),
        "approx_hess_fn": None,
        "approx_hess_fn_wsm": None,
        "meta": {
            "type": "smoothed_lasso_synthetic",
            "m": m,
            "n": n,
            "s": s,
            "reg": reg,
            "smoothing": smoothing,
            "noise_std": noise_std,
            "seed": seed,
        },
    }


__all__ = ["SmoothedLassoOracle", "make_smoothed_lasso_synthetic"]
