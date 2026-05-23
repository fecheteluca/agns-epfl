"""Polyak heavy-ball method.

Per-iteration step: ``x_{k+1} = x_k - alpha g_k + beta (x_k - x_{k-1})``.

When ``line_search=True`` the step size ``alpha`` is backtracked until
``f(x_trial) <= f(x_k)``; if backtracking stalls past half the budget
the momentum term is killed (the standard Polyak momentum-reset
recovery).  When ``line_search=False`` ``alpha`` stays at its initial
value.

Registry key: ``heavy_ball``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods._helpers import build_dual_norm, new_history, record_trace
from agns.methods.base import MethodResult
from agns.methods.stopping import Status, check_stop
from agns.oracles.base import OracleCallsCounter
from agns.utils.timing import Timer

__all__ = ["heavy_ball"]


def heavy_ball(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    alpha: float = 1.0,
    beta: float = 0.9,
    line_search: bool = True,
    L_min: float = 1e-5,
    trace: bool = True,
    B: NDArray[np.float64] | None = None,
    Binv: NDArray[np.float64] | None = None,
    f_star: float | None = None,
    eps: float = 1e-8,
    grad_tol: float | None = None,
    warnings: bool = False,
) -> MethodResult:
    """Polyak heavy-ball with optional backtracking on ``alpha``.

    Parameters
    ----------
    alpha : float
        Step size (initial value when ``line_search=True``).
    beta : float in [0, 1)
        Momentum coefficient.  ``beta = 0`` recovers vanilla gradient
        descent.
    line_search : bool
        Backtrack ``alpha`` until ``f`` decreases; otherwise hold
        ``alpha`` constant.
    L_min : float
        Caps ``alpha`` from above at ``1 / L_min`` during the grow phase.
    """
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must lie in [0, 1), got {beta}")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    counter = OracleCallsCounter(oracle)
    timer = Timer()
    _, _, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)

    x_k = np.copy(x_0)
    x_prev = np.copy(x_0)
    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    alpha_k = alpha

    history = new_history() if trace else None
    status = ""

    for k in range(n_iters + 1):
        if trace and history is not None:
            record_trace(
                history,
                func=f_k,
                grad_norm=g_k_norm,
                alpha=alpha_k,
                beta=beta,
                grad_calls=counter.grad_calls,
                func_calls=counter.func_calls,
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

        x_trial = x_k - alpha_k * g_k + beta * (x_k - x_prev)
        f_trial = counter.func(x_trial)

        if not line_search:
            if not np.isfinite(f_trial):
                status = Status.LINE_SEARCH_DIVERGED.value
                break
            x_prev = x_k
            x_k = x_trial
            f_k = f_trial
            g_k = counter.grad(x_k)
            g_k_norm = dual_norm_sqr(g_k) ** 0.5
            continue

        accepted = False
        beta_eff = beta
        for j in range(80):
            if not np.isfinite(f_trial):
                alpha_k *= 0.5
            elif f_trial <= f_k:
                accepted = True
                break
            else:
                alpha_k *= 0.5
            if j == 40:
                beta_eff = 0.0
            x_trial = x_k - alpha_k * g_k + beta_eff * (x_k - x_prev)
            f_trial = counter.func(x_trial)
        if not accepted:
            if warnings:
                print("W: heavy_ball line search diverged", flush=True)
            status = Status.LINE_SEARCH_DIVERGED.value
            break

        alpha_k = min(alpha_k * 1.1, 1.0 / L_min)

        x_prev = x_k
        x_k = x_trial
        f_k = f_trial
        g_k = counter.grad(x_k)
        g_k_norm = dual_norm_sqr(g_k) ** 0.5

    return x_k, status, history
