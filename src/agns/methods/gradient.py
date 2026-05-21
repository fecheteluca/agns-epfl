"""Gradient descent with backtracking line search.

Per-iteration step: ``x_{k+1} = x_k - precond(g_k) / L_k`` with ``L_k``
adaptively tuned by an Armijo-style backtracking test.

Registry key: ``gradient``.
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

__all__ = ["gradient"]


def gradient(
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
    """Gradient descent with Armijo-style backtracking on a per-iter ``L_k``."""
    counter = OracleCallsCounter(oracle)
    timer = Timer()
    B_eff, Binv_eff, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)

    def l2_norm_sqr(s: NDArray[np.float64]) -> float:
        return float(B_eff.dot(s).dot(s))

    def precond(g: NDArray[np.float64]) -> NDArray[np.float64]:
        return Binv_eff.dot(g) if B is not None else g

    x_k = np.copy(x_0)
    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    L_k = L_0

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

        accepted = False
        T = x_k.copy()
        f_T = f_k

        line_search_max_iter = 30
        for i in range(line_search_max_iter + 1):
            if i == line_search_max_iter:
                if warnings:
                    print("W: line_search_max_iter_reached", flush=True)
                break

            T_trial = x_k - precond(g_k) / L_k
            f_trial = counter.func(T_trial)

            if not np.isfinite(f_trial):
                L_k *= 2.0
                continue

            if not line_search:
                T, f_T, accepted = T_trial, f_trial, True
                break

            armijo_rhs = f_k + g_k.dot(T_trial - x_k) + 0.5 * L_k * l2_norm_sqr(T_trial - x_k)
            if f_trial <= armijo_rhs:
                T, f_T = T_trial, f_trial
                L_k = max(L_k * 0.5, L_min)
                accepted = True
                break
            L_k *= 2.0

        if not accepted:
            status = Status.LINE_SEARCH_DIVERGED.value
            break

        x_k = T
        f_k = f_T
        g_k = counter.grad(x_k)
        g_k_norm = dual_norm_sqr(g_k) ** 0.5

    return x_k, status, history
