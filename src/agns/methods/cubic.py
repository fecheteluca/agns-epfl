from __future__ import annotations

from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from agns.methods.common.result import MethodResult
from agns.methods.common.stopping import check_stop
from agns.methods.common.trace import new_history, record_trace
from agns.oracles.base import OracleCallsCounter
from agns.utils.timing import Timer

_ADAPTIVE_SEARCH_MAX_ITER = 40
_STEP_MAX_ITERS = 50


def cubic_newton_step(
    g: NDArray[np.float64],
    A: NDArray[np.float64],
    H: float,
    B: NDArray[np.float64] | None = None,
    eps: float = 1e-8,
) -> tuple[NDArray[np.float64], float, str]:
    """Minimize ``<g,x> + 1/2<Ax,x> + H/3 ||x||^3`` via the root of ``r - ||(A+HrB)^-1 g||``.

    Returns the minimizer, its model value, and a status string ("success",
    "linalg_error", or "iterations_exceeded"). Ported verbatim from upstream.
    """
    n = g.shape[0]
    if B is None:
        B = np.eye(n)

        def l2_norm_sqr(x: NDArray[np.float64]) -> np.float64:
            return x.dot(x)

    else:

        def l2_norm_sqr(x: NDArray[np.float64]) -> np.float64:
            return B.dot(x).dot(x)

    def f(T: NDArray[np.float64], T_norm: float) -> float:
        return g.dot(T) + 0.5 * A.dot(T).dot(T) + H * T_norm**3 / 3.0

    def h(r: float, der: bool = False) -> tuple[float, float, NDArray[np.float64], Any]:
        ArB_cho_factor = scipy.linalg.cho_factor(A + H * r * B, lower=False)
        T = scipy.linalg.cho_solve(ArB_cho_factor, -g)
        T_norm = l2_norm_sqr(T) ** 0.5
        h_r = r - T_norm
        if der:
            BT = B.dot(T)
            h_r_prime = 1 + H / T_norm * scipy.linalg.cho_solve(ArB_cho_factor, BT).dot(
                BT
            )
        else:
            h_r_prime = None
        return h_r, T_norm, T, h_r_prime

    try:
        max_r = 1.0
        # Find max_r such that h(max_r) is nonnegative.
        for _ in range(_STEP_MAX_ITERS):
            h_r, T_norm, T, _ = h(max_r)
            if h_r < -eps:
                max_r *= 2
            elif -eps <= h_r <= eps:
                return T, f(T, T_norm), "success"
            else:
                break

        # Univariate Newton iterations.
        r = max_r
        for _ in range(_STEP_MAX_ITERS):
            h_r, T_norm, T, h_r_prime = h(r, der=True)
            if -eps <= h_r <= eps:
                return T, f(T, T_norm), "success"
            r -= h_r / h_r_prime

    except (np.linalg.LinAlgError, ValueError):
        return np.zeros(n), 0.0, "linalg_error"

    return np.zeros(n), 0.0, "iterations_exceeded"


def cubic_newton(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    H_0: float = 1.0,
    adaptive_search: bool = True,
    trace: bool = True,
    B: NDArray[np.float64] | None = None,
    eps: float = 1e-8,
    H_min: float = 1e-5,
    f_star: float | None = None,
    warnings: bool = False,
) -> MethodResult:
    """Run the Cubic Newton Method.

    Returns
    -------
    MethodResult
        ``(x_k, status, history)`` (history keys: func, grad_calls, H_k, time, x_k).
    """
    counter = OracleCallsCounter(oracle)
    timer = Timer()

    x_k = np.copy(x_0)
    f_k = counter.func(x_k)
    H_k = H_0

    history = new_history() if trace else None
    status = ""

    for k in range(n_iters + 1):
        if trace and history is not None:
            record_trace(
                history,
                func=f_k,
                grad_calls=counter.grad_calls,
                H_k=H_k,
                time=timer.elapsed(),
                x_k=x_k.copy(),
            )

        decision = check_stop(
            k=k,
            n_iters=n_iters,
            f_k=f_k,
            f_star=f_star,
            eps=eps,
            g_norm=0.0,
            grad_tol=None,
        )
        if decision.stop:
            status = decision.status
            break

        g_k = counter.grad(x_k)
        Hess_k = counter.hess(x_k)

        x_delta = np.zeros_like(x_k)
        f_new: Any = f_k
        for i in range(_ADAPTIVE_SEARCH_MAX_ITER + 1):
            if i == _ADAPTIVE_SEARCH_MAX_ITER:
                if warnings:
                    print("W: adaptive_iterations_exceeded", flush=True)
                break

            x_delta, model_value, message = cubic_newton_step(
                g_k, Hess_k, 0.5 * H_k, B, 1e-8
            )
            if message != "success" and warnings:
                print(f"W: cubic_newton_step: {message}", end=" ", flush=True)
            f_new = counter.func(x_k + x_delta)

            if not adaptive_search:
                break

            if f_new <= f_k + model_value:
                H_k *= 0.5
                H_k = max(H_k, H_min)
                break
            H_k *= 2

        x_k = x_k + x_delta
        f_k = f_new

    return x_k, status, history
