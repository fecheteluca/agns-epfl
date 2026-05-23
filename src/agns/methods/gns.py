"""Gradient-Normalised-Smoothness Newton (GNS).

At each iteration ``k`` the method solves the regularised Newton system

.. math::

    \\bigl(H(x_k) + \\lambda_k B\\bigr)\\, \\delta_k = -g_k,
    \\qquad
    \\lambda_k = \\| g_k \\|_{*} / \\gamma_k,

where ``H(x_k)`` is either the exact Hessian (``is_approx=False``) or a
problem-specific Gauss-Newton approximation (``is_approx=True``, supplied
via ``approx_hess_fn``).  ``gamma_k`` is adapted by a doubling /
halving search that enforces the progress predicate

.. math::

    f(x_k) - f(x_k + \\delta_k) \\;\\ge\\; \\frac{\\| g(x_k + \\delta_k) \\|_*^2}
                                                  {8 \\lambda_k}.

The doubling / halving search itself is :func:`agns.methods._helpers.gns_adaptive_search`,
shared with GNS-WSM and the AGNS variants.

Registry keys: ``gns_exact`` (``is_approx=False``) and ``gns_inexact``
(``is_approx=True``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods._helpers import (
    ADAPTIVE_SEARCH_MAX_ITER,
    build_dual_norm,
    gns_adaptive_search,
    new_history,
    record_trace,
    regularised_lhs,
)
from agns.methods.base import MethodResult
from agns.methods.linalg import safe_cho_solve
from agns.methods.stopping import Status, check_stop
from agns.oracles.base import OracleCallsCounter
from agns.utils.timing import Timer

__all__ = ["gns"]


def gns(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    gamma_0: float = 1.0,
    gamma_min: float = 1e-9,
    adaptive_search: bool = True,
    is_approx: bool = False,
    approx_oracle: Any = None,
    approx_hess_fn: Any = None,
    trace: bool = True,
    B: NDArray[np.float64] | None = None,
    Binv: NDArray[np.float64] | None = None,
    eps: float = 1e-8,
    f_star: float | None = None,
    grad_tol: float | None = None,
    warnings: bool = False,
) -> MethodResult:
    """Run GNS (exact-Hessian or approximate-Hessian variant).

    Parameters
    ----------
    oracle : BaseSmoothOracle
        Provides ``func`` / ``grad`` / ``hess`` (the last is used only
        when ``is_approx=False``).
    x_0 : ndarray
        Initial iterate as a ``float64`` array.
    n_iters : int
        Maximum number of outer iterations.
    gamma_0, gamma_min : float
        Initial value and lower clip on the adaptive search parameter.
    adaptive_search : bool
        When ``True`` (default), the progress predicate adapts
        ``gamma_k`` per iteration; when ``False`` the trial step is
        accepted unconditionally.
    is_approx, approx_oracle, approx_hess_fn :
        Selects an approximate Hessian.  When ``is_approx=True`` and the
        problem provides ``approx_hess_fn(approx_oracle, x_k)``, that
        callable replaces ``oracle.hess(x_k)``.
    trace : bool
        If ``True`` (default), populate the per-iteration history dict.
    B, Binv : optional ndarrays
        Metric and its inverse for the dual gradient norm.  ``None``
        defaults to the Euclidean metric.
    eps, f_star, grad_tol :
        Stopping criteria.  See :func:`agns.methods.stopping.check_stop`.
    warnings : bool
        Emit verbose warnings on indefinite linear systems.
    """
    counter = OracleCallsCounter(oracle)
    timer = Timer()

    B_eff, _, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)

    x_k = np.copy(x_0)
    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    gamma_k = gamma_0

    history = new_history() if trace else None
    matrix_inverses = 0
    status = ""

    for k in range(n_iters + 1):
        if trace and history is not None:
            record_trace(
                history,
                func=f_k,
                grad_norm=g_k_norm,
                gamma_k=gamma_k,
                grad_calls=counter.grad_calls,
                matrix_inverses=matrix_inverses,
                time=timer.elapsed(),
                x_k=x_k.copy(),
            )

        decision = check_stop(
            k=k,
            n_iters=n_iters,
            f_k=f_k,
            f_star=f_star,
            eps=eps,
            g_norm=g_k_norm,
            grad_tol=grad_tol,
        )
        if decision.stop:
            status = decision.status
            break

        if is_approx and approx_oracle is not None and callable(approx_hess_fn):
            Hess_k = approx_hess_fn(approx_oracle, x_k)
        else:
            Hess_k = counter.hess(x_k)

        # Default args bind the per-iteration ``Hess_k`` / ``g_k`` at
        # closure-creation time (the search calls ``solve`` immediately).
        def solve(
            lambda_k: float,
            Hess_k: NDArray[np.float64] = Hess_k,
            g_k: NDArray[np.float64] = g_k,
        ) -> NDArray[np.float64]:
            return safe_cho_solve(regularised_lhs(Hess_k, lambda_k, B_eff), -g_k)

        x_new, f_new, g_new, g_new_norm_sqr, gamma_k, matrix_inverses = gns_adaptive_search(
            counter=counter,
            base_x=x_k,
            f_base=f_k,
            g_base=g_k,
            g_base_norm=g_k_norm,
            dual_norm_sqr=dual_norm_sqr,
            solve=solve,
            gamma_k=gamma_k,
            gamma_min=gamma_min,
            adaptive_search=adaptive_search,
            max_iter=ADAPTIVE_SEARCH_MAX_ITER,
            matrix_inverses=matrix_inverses,
            warnings=warnings,
            k=k,
        )

        x_k, f_k, g_k = x_new, f_new, g_new
        g_k_norm = g_new_norm_sqr**0.5

        if not np.isfinite(f_k) or not np.isfinite(g_k_norm):
            status = Status.DIVERGED_NONFINITE.value
            break

    return x_k, status, history
