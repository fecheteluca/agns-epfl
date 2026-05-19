"""Accelerated Cubic Newton (Nesterov, 2008).

Couples a Monteiro-Svaiter-style estimate sequence with a
Nesterov-Polyak cubic-regularised inner step.  Per outer iteration:

1. Solve ``L * sigma * a^2 = A + a`` for ``a > 0``.
2. ``y_k = (A_k x_k + a v_k) / (A_k + a)``.
3. Cubic inner step at ``y_k``, with adaptive ``M_k``.
4. ``v_{k+1} = v_k - a * B^{-1} g(x_{k+1})`` (sigma=1 specialisation).

The dual update uses ``B^{-1}`` so the estimate sequence is consistent
with whatever metric the cubic regulariser uses (``B = I`` recovers the
canonical Euclidean Nesterov 2008 form).

The optional ``monotone`` guard refuses iterates that increase ``f``,
holding ``x_k`` and bumping ``M_k`` instead.  Useful on objectives
whose Hessian is not globally Lipschitz.

Registry key: ``accelerated_cubic_newton``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods._helpers import build_dual_norm, new_history, record_trace
from agns.methods.base import MethodResult
from agns.methods.cubic_newton import cubic_newton_step
from agns.oracles.base import OracleCallsCounter
from agns.stopping import Status, check_stop
from agns.utils.timing import Timer

__all__ = ["accelerated_cubic_newton"]


def accelerated_cubic_newton(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    M_0: float = 1.0,
    M_min: float = 1e-5,
    M_max: float = 1e10,
    sigma: float = 1.0,
    adaptive_search: bool = True,
    monotone: bool = True,
    trace: bool = True,
    B: NDArray[np.float64] | None = None,
    Binv: NDArray[np.float64] | None = None,
    eps: float = 1e-8,
    f_star: float | None = None,
    grad_tol: float | None = None,
    warnings: bool = False,
) -> MethodResult:
    """Nesterov 2008 accelerated cubic Newton.

    Parameters
    ----------
    M_0, M_min, M_max : float
        Cubic regulariser bounds.
    sigma : float
        Estimate-sequence coupling constant.  ``sigma = 1`` is the
        canonical choice.
    monotone : bool
        Refuse iterates that increase ``f`` and bump ``M_k`` instead.
    """
    counter = OracleCallsCounter(oracle)
    timer = Timer()
    B_eff, Binv_eff, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)

    x_k = np.copy(x_0)
    v_k = np.copy(x_0)
    A_k = 0.0
    M_k = M_0

    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5

    history = new_history() if trace else None
    matrix_inverses = 0
    status = ""

    for k in range(n_iters + 1):
        if trace and history is not None:
            record_trace(
                history,
                func=f_k,
                grad_norm=g_k_norm,
                M=M_k,
                A_k=A_k,
                grad_calls=counter.grad_calls,
                hess_calls=counter.hess_calls,
                matrix_inverses=matrix_inverses,
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

        L_sigma = M_k * sigma
        a_kp1 = (1.0 + np.sqrt(1.0 + 4.0 * L_sigma * A_k)) / (2.0 * L_sigma)
        A_kp1 = A_k + a_kp1
        y_k = (A_k * x_k + a_kp1 * v_k) / A_kp1
        f_y = counter.func(y_k)
        g_y = counter.grad(y_k)
        H_y = counter.hess(y_k)

        accepted = False
        f_trial = f_k
        x_trial = x_k.copy()
        for _inner in range(50):
            x_delta, model_value, message = cubic_newton_step(g_y, H_y, 0.5 * M_k, B_eff, 1e-8)
            if message != "success":
                M_k = min(M_k * 2.0, M_max)
                if M_k >= M_max:
                    break
                continue
            matrix_inverses += 1

            x_trial = y_k + x_delta
            f_trial = counter.func(x_trial)
            if not np.isfinite(f_trial):
                M_k = min(M_k * 2.0, M_max)
                continue

            if not adaptive_search:
                accepted = True
                break

            if f_trial <= f_y + model_value:
                accepted = True
                M_k = max(M_k * 0.5, M_min)
                break
            M_k = min(M_k * 2.0, M_max)

        if not accepted:
            if warnings:
                print(f"W: ACN inner loop failed at k = {k}", flush=True)
            status = Status.ADAPTIVE_SEARCH_FAILED.value
            break

        # Monotonisation guard: hold x_k when the trial increased f.
        if monotone and f_trial > f_k:
            M_k = min(M_k * 2.0, M_max)
            A_k = A_kp1
            continue

        x_k = x_trial
        f_k = f_trial
        g_k = counter.grad(x_k)
        g_k_norm = dual_norm_sqr(g_k) ** 0.5
        v_k = v_k - a_kp1 * Binv_eff.dot(g_k)
        A_k = A_kp1

    return x_k, status, history
