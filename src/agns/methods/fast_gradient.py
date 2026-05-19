"""Nesterov's accelerated gradient with FISTA-style backtracking.

Maintains the standard Nesterov ``(x_k, v_k, A_k)`` estimating-sequence
triple and steps the prox iterate through a backtracking line search
on the local Lipschitz estimate ``L_k``.

Registry key: ``fast_gradient``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods._helpers import build_dual_norm, new_history, record_trace
from agns.methods.base import MethodResult
from agns.oracles.base import OracleCallsCounter
from agns.stopping import Status, check_stop
from agns.utils.timing import Timer

__all__ = ["fast_gradient"]


def fast_gradient(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    L_0: float = 1.0,
    L_min: float = 1e-5,
    line_search: bool = True,
    trace: bool = True,
    B: NDArray[np.float64] | None = None,
    Binv: NDArray[np.float64] | None = None,
    eps: float = 1e-8,
    f_star: float | None = None,
    grad_tol: float | None = None,
    warnings: bool = False,
) -> MethodResult:
    """Nesterov's accelerated method with FISTA-style line search."""
    counter = OracleCallsCounter(oracle)
    timer = Timer()
    B_eff, Binv_eff, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)

    def l2_norm_sqr(s: NDArray[np.float64]) -> float:
        return float(B_eff.dot(s).dot(s))

    def precond(g: NDArray[np.float64]) -> NDArray[np.float64]:
        return Binv_eff.dot(g) if B is not None else g

    x_k = np.copy(x_0)
    v_k = np.copy(x_0)
    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    L_k = L_0
    A_k = 0.0

    history = new_history() if trace else None
    status = ""

    for k in range(n_iters + 1):
        if trace and history is not None:
            record_trace(
                history,
                func=f_k,
                grad_norm=g_k_norm,
                L=L_k,
                func_calls=counter.func_calls,
                grad_calls=counter.grad_calls,
                time=timer.elapsed(),
                x_k=x_k.copy(),
            )

        decision = check_stop(
            k=k, n_iters=n_iters, f_k=f_k, f_star=f_star, eps=eps,
            g_norm=g_k_norm, grad_tol=grad_tol,
        )
        if decision.stop:
            status = decision.status
            break

        accepted = False
        T = x_k.copy()
        f_T = f_k
        a_new = 1.0 / L_k

        line_search_max_iter = 30
        for i in range(line_search_max_iter + 1):
            if i == line_search_max_iter:
                if warnings:
                    print("W: line_search_max_iter_reached", flush=True)
                break

            a_trial = (1.0 + (1.0 + 4.0 * A_k * L_k) ** 0.5) / (2.0 * L_k)
            y_trial = (a_trial * v_k + A_k * x_k) / (a_trial + A_k)
            g_y_trial = counter.grad(y_trial)
            f_y_trial = counter.func(y_trial)

            T_trial = y_trial - precond(g_y_trial) / L_k
            f_T_trial = counter.func(T_trial)

            if not (np.isfinite(f_y_trial) and np.isfinite(f_T_trial)):
                L_k *= 2.0
                continue

            if not line_search:
                a_new = a_trial
                T, f_T = T_trial, f_T_trial
                accepted = True
                break

            armijo_rhs = (
                f_y_trial
                + g_y_trial.dot(T_trial - y_trial)
                + 0.5 * L_k * l2_norm_sqr(T_trial - y_trial)
            )
            if f_T_trial <= armijo_rhs:
                a_new = a_trial
                T, f_T = T_trial, f_T_trial
                L_k = max(L_k * 0.5, L_min)
                accepted = True
                break
            L_k *= 2.0

        if not accepted:
            status = Status.LINE_SEARCH_DIVERGED.value
            break

        v_k = T + (A_k / a_new) * (T - x_k)
        A_k += a_new

        x_k = T
        f_k = f_T
        g_k = counter.grad(x_k)
        g_k_norm = dual_norm_sqr(g_k) ** 0.5

    return x_k, status, history
