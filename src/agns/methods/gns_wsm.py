"""GNS with Woodbury / Sherman-Morrison rank-1 inversion.

Equivalent to :func:`agns.methods.gns.gns` with a rank-1 inexact Hessian,
but the linear solve is replaced by the closed-form Woodbury formula in
:func:`agns.methods.linalg.wsm_rank_one_solve`.  This brings the per-iteration
cost from ``O(m^3)`` (dense Cholesky) down to ``O(m^2)``.

Selects the backend via ``invert_backend``:

* ``"cho"`` --- dense Cholesky on ``Hess_k + lambda_k * B`` (same as
  :func:`gns`).
* ``"wsm"`` --- closed-form Woodbury, requires
  ``is_approx=True`` and the oracle to expose ``func_u``, ``jac_u``, ``p``.

The adaptive ``gamma`` search itself is the shared
:func:`agns.methods._helpers.gns_adaptive_search`; only the per-step
linear solve differs between the two backends.

Registry key: ``gns_wsm``.
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
    rank_one_fisher_factor,
    record_trace,
    regularised_lhs,
)
from agns.methods.base import MethodResult
from agns.methods.linalg import safe_cho_solve, wsm_rank_one_solve
from agns.methods.stopping import Status, check_stop
from agns.oracles.base import OracleCallsCounter
from agns.utils.timing import Timer

__all__ = ["gns_wsm"]


def gns_wsm(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    gamma_0: float = 1.0,
    gamma_min: float = 1e-9,
    adaptive_search: bool = True,
    is_approx: bool = True,
    approx_oracle: Any = None,
    approx_hess_fn: Any = None,
    trace: bool = True,
    B: NDArray[np.float64] | None = None,
    Binv: NDArray[np.float64] | None = None,
    eps: float = 1e-8,
    f_star: float | None = None,
    grad_tol: float | None = None,
    warnings: bool = False,
    invert_backend: str = "wsm",
) -> MethodResult:
    """GNS with selectable linear-solve backend (Cholesky or rank-1 WSM)."""
    if invert_backend not in {"cho", "wsm"}:
        raise ValueError(f"invert_backend must be 'cho' or 'wsm', got {invert_backend!r}")
    if invert_backend == "wsm" and not (
        is_approx and callable(approx_hess_fn) and approx_oracle is not None
    ):
        raise ValueError("wsm backend requires is_approx=True and a rank-one approx_oracle")

    counter = OracleCallsCounter(oracle)
    timer = Timer()

    B_eff, Binv_eff, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)

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

        # Near-stationary skip; mirrors the short-circuit in
        # :func:`agns.methods.gns.gns` and :func:`agns.methods.agns.agns`.
        # Without it, GNS keeps paying one Hessian + many adaptive-search
        # probes per outer iter long after the iterate has converged,
        # which inflates the grad-call / matrix-inverse axes.
        if g_k_norm < eps:
            continue

        # The cho backend pre-computes the Hessian; the wsm backend
        # leaves it ``None`` and rebuilds the rank-1 factor per probe.
        Hess_k: NDArray[np.float64] | None
        if invert_backend == "cho":
            if is_approx and approx_oracle is not None and callable(approx_hess_fn):
                Hess_k = approx_hess_fn(approx_oracle, x_k)
            else:
                Hess_k = counter.hess(x_k)
        else:
            Hess_k = None

        # Default args bind the per-iteration state at closure-creation
        # time (the search calls ``solve`` immediately).
        def solve(
            lambda_k: float,
            Hess_k: NDArray[np.float64] | None = Hess_k,
            x_k: NDArray[np.float64] = x_k,
            g_k: NDArray[np.float64] = g_k,
        ) -> NDArray[np.float64]:
            if Hess_k is not None:
                return safe_cho_solve(regularised_lhs(Hess_k, lambda_k, B_eff), -g_k)
            assert Binv_eff is not None  # wsm backend requires a metric Binv
            alpha, g0 = rank_one_fisher_factor(approx_oracle, x_k, g_k)
            return wsm_rank_one_solve(g_k, lambda_k, Binv_eff, alpha, g0)

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
