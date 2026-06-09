"""Normalized Gradient Method with the GNS adaptive search (optional baseline).

Refactor of ``grad_norm_smooth_gradient_method`` from
``epfml/grad-norm-smooth/src/methods.py`` (Apache-2.0, commit 9f4ca00). It uses the same
gradient-normalized acceptance test as GNS but takes a (preconditioned) gradient step
``T = x_k - precond(g_k) / lambda_k`` instead of a Newton step, so it keeps its own loop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods.common.norms import build_first_order_metric
from agns.methods.common.result import MethodResult
from agns.methods.common.stopping import check_stop
from agns.methods.common.trace import new_history, record_trace
from agns.oracles.base import OracleCallsCounter
from agns.utils.timing import Timer

_ADAPTIVE_SEARCH_MAX_ITER = 40


def grad_norm_smooth_gradient_method(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    gamma_0: float = 1.0,
    adaptive_search: bool = True,
    trace: bool = True,
    is_precond: bool = False,
    precond_oracle: Any = None,
    precond_fn: Any = None,
    eps: float = 1e-8,
    B: NDArray[np.float64] | None = None,
    Binv: NDArray[np.float64] | None = None,
    gamma_min: float = 1e-9,
    f_star: float | None = None,
    grad_tol: float | None = None,
    warnings: bool = False,
) -> MethodResult:
    """Run the Normalized Gradient Method.

    Returns
    -------
    MethodResult
        ``(x_k, status, history)``.
    """
    counter = OracleCallsCounter(oracle)
    timer = Timer()
    _, dual_norm_sqr, precond = build_first_order_metric(B, Binv)

    x_k = np.copy(x_0)
    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    gamma_k = gamma_0

    history = new_history() if trace else None
    status = ""

    for k in range(n_iters + 1):
        if trace and history is not None:
            record_trace(
                history,
                func=f_k,
                grad_norm=g_k_norm,
                gamma_k=gamma_k,
                func_calls=counter.func_calls,
                grad_calls=counter.grad_calls,
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

        T = x_k
        f_new: Any = f_k
        g_new = g_k
        g_new_norm_sqr: Any = g_k_norm**2
        for i in range(_ADAPTIVE_SEARCH_MAX_ITER + 1):
            if i == _ADAPTIVE_SEARCH_MAX_ITER:
                if warnings:
                    print("W: adaptive_search_max_iter_exceeded", flush=True)
                break

            lambda_k = g_k_norm / (gamma_k + eps)

            if is_precond and precond_oracle is not None and callable(precond_fn):
                P_k = precond_fn(precond_oracle, x_k) + eps * np.eye(x_k.shape[0])
                P_k_inv = np.linalg.inv(P_k)

                def precond(  # noqa: F811  (per-iter preconditioner, upstream behavior)
                    g: NDArray[np.float64], P_k_inv: NDArray[np.float64] = P_k_inv
                ) -> NDArray[np.float64]:
                    return P_k_inv.dot(g)

            T = x_k - precond(g_k) / lambda_k
            f_new = counter.func(T)
            g_new = counter.grad(T)
            g_new_norm_sqr = dual_norm_sqr(g_new)

            if not adaptive_search:
                break

            if f_k - f_new >= g_new_norm_sqr / (8 * lambda_k):
                gamma_k = max(gamma_k * 2.0, gamma_min)
                break
            gamma_k *= 0.5

        x_k = T
        f_k = f_new
        g_k = g_new
        g_k_norm = g_new_norm_sqr**0.5

    return x_k, status, history
