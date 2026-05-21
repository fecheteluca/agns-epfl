"""Super-Universal Newton.

Per-iteration step: ``(H(x_k) + lambda_k B) delta_k = -g_k`` with
``lambda_k = H_k * ||g_k||_*^alpha``.  The regulariser ``H_k`` is
multiplicatively updated by an Armijo-style progress test on the trial
gradient.

Registry key: ``super_newton``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray

from agns.linalg import safe_cho_solve
from agns.methods._helpers import (
    ADAPTIVE_SEARCH_MAX_ITER,
    build_dual_norm,
    new_history,
    record_trace,
)
from agns.methods.base import MethodResult
from agns.oracles.base import OracleCallsCounter
from agns.stopping import Status, check_stop
from agns.utils.timing import Timer

__all__ = ["super_newton"]


def super_newton(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    H_0: float = 1.0,
    H_min: float = 1e-5,
    alpha: float = 1.0,
    adaptive_search: bool = True,
    trace: bool = True,
    B: NDArray[np.float64] | None = None,
    Binv: NDArray[np.float64] | None = None,
    eps: float = 1e-8,
    f_star: float | None = None,
    grad_tol: float | None = None,
    warnings: bool = False,
) -> MethodResult:
    """Run Super-Universal Newton.

    Parameters
    ----------
    H_0, H_min : float
        Initial and minimum value of the regulariser.
    alpha : float
        Gradient-norm exponent in ``lambda_k = H_k ||g_k||^alpha``.
    """
    counter = OracleCallsCounter(oracle)
    timer = Timer()
    B_eff, _, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)

    x_k = np.copy(x_0)
    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    H_k = H_0

    history = new_history() if trace else None
    matrix_inverses = 0
    status = ""

    for k in range(n_iters + 1):
        if trace and history is not None:
            record_trace(
                history,
                func=f_k,
                grad_norm=g_k_norm,
                H_k=H_k,
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

        Hess_k = counter.hess(x_k)

        x_new = x_k.copy()
        f_new = f_k
        g_new = g_k
        g_new_norm_sqr = g_k_norm * g_k_norm

        for i in range(ADAPTIVE_SEARCH_MAX_ITER + 1):
            if i == ADAPTIVE_SEARCH_MAX_ITER:
                if warnings:
                    print(f"W: adaptive_iterations_exceeded, k = {k}", flush=True)
                break

            lambda_k = H_k * g_k_norm**alpha
            try:
                delta_x = safe_cho_solve(Hess_k + lambda_k * B_eff, -g_k)
                matrix_inverses += 1
            except (LinAlgError, ValueError):
                if warnings:
                    print("W: linalg_error -> tightening H", flush=True)
                H_k *= 4.0
                continue

            x_trial = x_k + delta_x
            f_trial = counter.func(x_trial)
            g_trial = counter.grad(x_trial)
            g_trial_norm_sqr = dual_norm_sqr(g_trial)

            if not adaptive_search:
                x_new, f_new, g_new, g_new_norm_sqr = (
                    x_trial,
                    f_trial,
                    g_trial,
                    g_trial_norm_sqr,
                )
                break

            if g_trial.dot(-delta_x) >= g_trial_norm_sqr / (4.0 * lambda_k):
                x_new, f_new, g_new, g_new_norm_sqr = (
                    x_trial,
                    f_trial,
                    g_trial,
                    g_trial_norm_sqr,
                )
                H_k = max(H_k * 0.25, H_min)
                break

            H_k *= 4.0

        x_k, f_k, g_k = x_new, f_new, g_new
        g_k_norm = g_new_norm_sqr**0.5

        if not np.isfinite(f_k) or not np.isfinite(g_k_norm):
            status = Status.DIVERGED_NONFINITE.value
            break

    return x_k, status, history
