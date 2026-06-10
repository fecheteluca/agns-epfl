from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods.common.gns_search import gns_adaptive_search
from agns.methods.common.linalg import safe_cho_solve
from agns.methods.common.norms import build_dual_norm, regularised_lhs
from agns.methods.common.restart import should_restart, validate_restart_mode
from agns.methods.common.result import MethodResult
from agns.methods.common.stopping import Status, check_stop
from agns.methods.common.trace import new_history, record_trace
from agns.oracles.base import OracleCallsCounter
from agns.utils.timing import Timer


def agns(
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
    momentum_offset: float = 3.0,
    restart_mode: str = "gradient",
    adaptive_search_max_iter: int = 40,
) -> MethodResult:
    """Run Accelerated Gradient-Normalized Smooth Newton.

    Parameters
    ----------
    oracle, x_0, n_iters, gamma_0, gamma_min, adaptive_search, is_approx, approx_oracle,
    approx_hess_fn, trace, B, Binv, eps, f_star, grad_tol, warnings:
        As in :func:`agns.methods.gns.grad_norm_smooth`.
    momentum_offset:
        Offset in the momentum coefficient ``beta_k = local_k / (local_k + offset)``.
    restart_mode:
        ``"gradient"`` (scheme II), ``"function_value"`` (scheme I), or ``"none"``.
    adaptive_search_max_iter:
        Inner adaptive-search cap (upstream uses 40).

    Returns
    -------
    MethodResult
        ``(x_k, status, history)``; the history carries per-step ``local_k`` and
        ``n_restarts_so_far`` plus terminal ``n_restarts_total``/``restart_events``/
        ``restart_mode`` entries.
    """
    validate_restart_mode(restart_mode)
    counter = OracleCallsCounter(oracle)
    timer = Timer()
    B_eff, _, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)

    x_k = np.copy(x_0)
    x_prev = np.copy(x_0)
    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    gamma_k = gamma_0
    local_k = 0
    n_restarts = 0
    restart_events: list[int] = []

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
                local_k=local_k,
                n_restarts_so_far=n_restarts,
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

        beta_k = local_k / (local_k + momentum_offset)
        y_k = x_k + beta_k * (x_k - x_prev)

        g_y = counter.grad(y_k)
        g_y_norm = dual_norm_sqr(g_y) ** 0.5
        f_y = counter.func(y_k)

        x_trial = y_k.copy()
        f_trial: Any = f_y
        g_trial = g_y.copy()
        g_trial_norm_sqr: Any = g_y_norm * g_y_norm

        if g_y_norm < eps:
            # Near-stationary extrapolation. Reset the momentum anchor; only ADVANCE to
            # y_k when f(y_k) <= f(x_k), else HOLD x_k so the next iter steps from x_k.
            if f_trial <= f_k:
                x_prev = x_k.copy()
                x_k, f_k, g_k = x_trial, f_trial, g_trial
                g_k_norm = g_trial_norm_sqr**0.5
            else:
                x_prev = x_k.copy()
            local_k = 0
            continue

        if is_approx and approx_oracle is not None and callable(approx_hess_fn):
            Hess_k = approx_hess_fn(approx_oracle, y_k)
        else:
            Hess_k = counter.hess(y_k)

        def solve(
            lambda_k: float, Hess_k: Any = Hess_k, g_y: Any = g_y
        ) -> NDArray[np.float64]:
            return safe_cho_solve(regularised_lhs(Hess_k, lambda_k, B_eff), -g_y)

        (
            x_trial,
            f_trial,
            g_trial,
            g_trial_norm_sqr,
            gamma_k,
            matrix_inverses,
        ) = gns_adaptive_search(
            counter=counter,
            base_x=y_k,
            f_base=f_y,
            g_base=g_y,
            g_base_norm=g_y_norm,
            dual_norm_sqr=dual_norm_sqr,
            solve=solve,
            gamma_k=gamma_k,
            gamma_min=gamma_min,
            adaptive_search=adaptive_search,
            max_iter=adaptive_search_max_iter,
            matrix_inverses=matrix_inverses,
            eps=eps,
            warnings=warnings,
            k=k,
            keep_last_trial_on_exhaust=True,
        )

        if should_restart(restart_mode, g_trial, x_trial, x_k, f_trial, f_k):
            x_prev = x_trial.copy()
            local_k = 0
            n_restarts += 1
            restart_events.append(k)
        else:
            x_prev = x_k.copy()
            local_k += 1

        x_k = x_trial.copy()
        f_k = f_trial
        g_k = g_trial
        g_k_norm = g_trial_norm_sqr**0.5

        if not np.isfinite(f_k) or not np.isfinite(g_k_norm):
            status = Status.DIVERGED_NONFINITE.value
            break

    if history is not None:
        history["n_restarts_total"] = [n_restarts]
        history["restart_events"] = restart_events
        history["restart_mode"] = [restart_mode]

    return x_k, status, history
