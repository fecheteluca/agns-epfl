"""Dual-norm and regularized-system helpers shared by the Newton-type methods.

Factored from the identical metric-initialization blocks repeated across
``epfml/grad-norm-smooth`` (Apache-2.0, commit 9f4ca00). The branches reproduce
upstream exactly:

* ``B is None``  -> Euclidean metric, ``dual_norm_sqr(x) = x . x``;
* ``B`` given    -> ``dual_norm_sqr(x) = (Binv x) . x`` with ``Binv = inv(B)`` if absent.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

DualNormSqr = Callable[[NDArray[np.float64]], np.float64]


def build_dual_norm(
    n: int,
    B: NDArray[np.float64] | None,
    Binv: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], DualNormSqr]:
    """Build the effective metric matrices and the squared dual-norm functional.

    Parameters
    ----------
    n:
        Problem dimension (used to materialize identity defaults).
    B:
        Primal metric matrix, or ``None`` for the Euclidean metric.
    Binv:
        Inverse metric; computed from ``B`` when ``None`` and ``B`` is given.

    Returns
    -------
    B_eff:
        ``B`` if given else ``I_n`` (the matrix added in the regularized system).
    Binv_eff:
        ``Binv``/``inv(B)`` if ``B`` given, else ``I_n`` (used by the rank-one WSM path).
    dual_norm_sqr:
        Callable computing the squared dual norm, matching the upstream lambda exactly.
    """
    if B is None:
        B_eff = np.eye(n)
        Binv_eff = np.eye(n)

        def dual_norm_sqr(x: NDArray[np.float64]) -> np.float64:
            return x.dot(x)

    else:
        B_eff = B
        Binv_eff = Binv if Binv is not None else np.linalg.inv(B)

        def dual_norm_sqr(x: NDArray[np.float64]) -> np.float64:
            return Binv_eff.dot(x).dot(x)

    return B_eff, Binv_eff, dual_norm_sqr


def regularised_lhs(
    H: NDArray[np.float64], lam: float, B: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the regularized Newton system matrix ``H + lam * B`` (upstream form)."""
    return H + lam * B


Precond = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def build_first_order_metric(
    B: NDArray[np.float64] | None,
    Binv: NDArray[np.float64] | None,
) -> tuple[DualNormSqr, DualNormSqr, Precond]:
    """Build ``(l2_norm_sqr, dual_norm_sqr, precond)`` for the gradient methods.

    Reproduces the metric-initialization block shared by ``gradient_method``,
    ``fast_gradient_method`` and ``grad_norm_smooth_gradient_method`` upstream:

    * ``B is None``  -> primal/dual norms both Euclidean;
    * ``B`` given    -> ``l2 = (Bx).x``, ``dual = (Binv x).x`` with ``Binv = inv(B)`` if absent;
    * preconditioner is identity when ``Binv`` is ``None`` else ``g -> Binv g``.
    """
    if B is None:

        def l2_norm_sqr(x: NDArray[np.float64]) -> np.float64:
            return x.dot(x)

        def dual_norm_sqr(x: NDArray[np.float64]) -> np.float64:
            return x.dot(x)

    else:
        if Binv is None:
            Binv = np.linalg.inv(B)

        def l2_norm_sqr(x: NDArray[np.float64]) -> np.float64:
            return B.dot(x).dot(x)

        def dual_norm_sqr(x: NDArray[np.float64]) -> np.float64:
            return Binv.dot(x).dot(x)

    if Binv is None:

        def precond(g: NDArray[np.float64]) -> NDArray[np.float64]:
            return g

    else:
        Binv_local = Binv

        def precond(g: NDArray[np.float64]) -> NDArray[np.float64]:
            return Binv_local.dot(g)

    return l2_norm_sqr, dual_norm_sqr, precond
