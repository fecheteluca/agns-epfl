"""Hessian-approximation builders used by the inexact-Hessian methods.

Each function returns a full ``(n, n)`` Hessian-approximation matrix for
the given oracle and iterate.  They are designed to be passed as the
``approx_hess_fn`` argument of the optimisation methods.

Four builders are provided:

* :func:`approx_hess_fn_logsumexp` --- weighted Gauss-Newton approximation
  for the LogSumExp oracle (drops the rank-one outer-product correction
  in the exact Hessian).
* :func:`approx_hess_fn_fisher_term` --- pure rank-1 Fisher term of
  ``f = (1/p) ||u(x)||^p``; used by the WSM (Woodbury) backend, which
  exploits the rank-1 structure to replace the dense Cholesky with a
  closed-form Sherman-Morrison inversion.
* :func:`approx_hess_nonlinear_equations` --- full Gauss-Newton + Fisher
  approximation for ``f = (1/p) ||u(x)||^p``.
* :func:`approx_hess_fn_chebyshev` --- Gauss-Newton + Fisher specialisation
  for the Chebyshev-chain oracle.

The Gauss-Newton + Fisher algebra shared by the last three builders is
factored into the private helpers :func:`_fisher_term` and
:func:`_gauss_newton_fisher`; the public builders differ only in how
they extract ``(u, J)`` from their oracle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "approx_hess_fn_chebyshev",
    "approx_hess_fn_fisher_term",
    "approx_hess_fn_logsumexp",
    "approx_hess_nonlinear_equations",
]


def _fisher_term(u: NDArray[np.float64], J: NDArray[np.float64], p: float) -> NDArray[np.float64]:
    """Rank-1 Fisher term ``(p-2) ||u||^{p-4} (J^T u)(J^T u)^T``.

    Vanishes (returns the zero matrix) when ``p == 2``.  No special
    handling of ``||u|| == 0``: callers that need it (see
    :func:`_gauss_newton_fisher`) guard the call themselves.
    """
    n = J.shape[1]
    H = np.zeros((n, n))
    if p != 2:
        norm_u = np.linalg.norm(u)
        Ju = J.T.dot(u)
        H += (p - 2) * (norm_u ** (p - 4)) * np.outer(Ju, Ju)
    return H


def _gauss_newton_fisher(
    u: NDArray[np.float64], J: NDArray[np.float64], p: float
) -> NDArray[np.float64]:
    """Gauss-Newton + rank-1 Fisher Hessian approximation of ``(1/p) ||u||^p``.

    ``H = ||u||^{p-2} J^T J + (p - 2) ||u||^{p-4} (J^T u)(J^T u)^T``.

    The Fisher term vanishes when ``p == 2``.  At ``||u|| == 0`` the
    Gauss-Newton term degenerates to ``J^T J`` (``p == 2``) or the zero
    matrix (``p > 2``), and the Fisher term is omitted to avoid a
    ``0 * inf`` indeterminacy.
    """
    norm_u = np.linalg.norm(u)
    H = (norm_u ** (p - 2)) * (J.T.dot(J))
    if norm_u != 0.0:
        H = H + _fisher_term(u, J, p)
    return H


def approx_hess_fn_logsumexp(approx_oracle: Any, x_k: NDArray[np.float64]) -> NDArray[np.float64]:
    """Weighted Gauss-Newton approximation for LogSumExp.

    Computes ``H = (A^T diag(pi) A) / mu`` from the cached softmax
    weights of the LogSumExp oracle, dropping the rank-one
    ``-(A^T pi)(A^T pi)^T / mu`` correction from the exact Hessian.
    """
    approx_oracle._update_a_and_pi(x_k)
    pi = approx_oracle.pi
    return approx_oracle.matmat_ATsA(pi) / approx_oracle.mu


def approx_hess_fn_fisher_term(oracle: Any, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pure rank-1 Fisher term ``(p-2) ||u||^{p-4} (J^T u)(J^T u)^T``.

    Vanishes when ``p == 2``.  Used by the WSM backend.
    """
    u = oracle.func_u(x)
    J = oracle.jac_u(x)
    return _fisher_term(u, J, oracle.p)


def approx_hess_nonlinear_equations(oracle: Any, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gauss-Newton + Fisher approximation for ``(1/p) ||u(x)||^p``.

    ``H = ||u||^{p-2} J^T J + (p-2) ||u||^{p-4} (J^T u)(J^T u)^T``.

    Drops only the curvature term in the second-derivative-of-residuals
    expansion, retaining both the Gauss-Newton kernel and the rank-1
    Fisher correction.
    """
    u, J = oracle._compute_u_and_jac(x)
    return _gauss_newton_fisher(u, J, oracle.p)


def approx_hess_fn_chebyshev(oracle: Any, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gauss-Newton + Fisher specialisation for the Chebyshev oracle.

    Identical algebra to :func:`approx_hess_nonlinear_equations`, but
    the Chebyshev oracle exposes ``func_u`` / ``jac_u`` as separate
    methods (not packed in ``_compute_u_and_jac``) so the call signature
    differs.
    """
    u = oracle.func_u(x)
    J = oracle.jac_u(x)
    return _gauss_newton_fisher(u, J, oracle.p)
