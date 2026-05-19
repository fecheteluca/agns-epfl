"""Linear-algebra primitives shared by every optimisation method.

Two helpers are exported:

* :func:`safe_cho_solve` --- a Cholesky-based linear solve.  Thin wrapper
  around :func:`scipy.linalg.cho_factor` / :func:`scipy.linalg.cho_solve`
  that raises :class:`numpy.linalg.LinAlgError` on indefinite input, so
  callers can catch and tighten regularisation.
* :func:`wsm_rank_one_solve` --- the Woodbury / Sherman-Morrison closed
  form for systems of the shape ``(lambda * B + alpha * g0 g0^T) delta = -g``.
  ``O(m^2)`` given a precomputed ``Binv``; ``O(m)`` if ``Binv = I``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

__all__ = ["safe_cho_solve", "wsm_rank_one_solve"]


def safe_cho_solve(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    *,
    lower: bool = False,
) -> NDArray[np.float64]:
    """Cholesky-solve ``A x = b`` for symmetric positive-definite ``A``.

    Errors propagate as :class:`numpy.linalg.LinAlgError` (or ``ValueError``)
    so callers can catch and tighten regularisation.

    Parameters
    ----------
    A : ndarray
        Symmetric positive-definite matrix of shape ``(n, n)``.
    b : ndarray
        Right-hand side of shape ``(n,)`` or ``(n, k)``.
    lower : bool
        Use the lower-triangular Cholesky factor.  Default ``False``.
    """
    factor = scipy.linalg.cho_factor(A, lower=lower)
    return cast("NDArray[np.float64]", scipy.linalg.cho_solve(factor, b))


def wsm_rank_one_solve(
    g: NDArray[np.float64],
    lambda_k: float,
    Binv: NDArray[np.float64],
    alpha: float,
    g0: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Solve ``(lambda B + alpha g0 g0^T) delta = -g`` in closed form.

    Implements

    .. math::

        (\lambda B + \alpha g_0 g_0^\top)^{-1}
        = \frac{1}{\lambda} B^{-1}
        - \frac{\alpha/\lambda^2}{1 + (\alpha/\lambda)\, g_0^\top B^{-1} g_0}
          \, B^{-1} g_0\, g_0^\top B^{-1}.

    Cost: two ``Binv``-multiplies plus a scalar correction, ``O(m^2)``
    (or ``O(m)`` if ``Binv`` is identity).

    The returned vector is ``delta = -A^{-1} g`` where
    ``A = lambda B + alpha g0 g0^T``.

    Parameters
    ----------
    g : ndarray
        Gradient vector at the current iterate, shape ``(m,)``.
    lambda_k : float
        Strictly positive regulariser.
    Binv : ndarray
        Inverse of the metric matrix ``B``, shape ``(m, m)``.
    alpha : float
        Non-negative rank-1 weight.  When ``alpha == 0`` the formula
        degenerates to ``-Binv g / lambda``.
    g0 : ndarray
        Vector defining the rank-1 update, shape ``(m,)``.
    """
    if lambda_k <= 0.0:
        raise ValueError(f"lambda_k must be strictly positive, got {lambda_k}")

    inv_scale = 1.0 / lambda_k
    Pinv_g = inv_scale * Binv.dot(-g)
    if alpha == 0.0:
        return cast("NDArray[np.float64]", Pinv_g)

    Pinv_g0 = inv_scale * Binv.dot(g0)
    denom = 1.0 + alpha * g0.dot(Pinv_g0)
    delta = Pinv_g - (alpha / denom) * Pinv_g0 * g0.dot(Pinv_g)
    return cast("NDArray[np.float64]", delta)
