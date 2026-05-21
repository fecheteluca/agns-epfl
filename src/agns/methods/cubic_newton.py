"""Cubic-regularised Newton (Nesterov-Polyak).

At each iteration solves the cubic subproblem

.. math::

    T_k = \\arg\\min_{T} \\, \\langle g_k, T\\rangle + \\tfrac{1}{2} \\langle A_k T, T\\rangle
            + \\tfrac{H_k}{3} \\|T\\|^3

via :func:`cubic_newton_step`, then accepts or rejects the trial via a
model-decrease test.

Registry key: ``cubic_newton``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray

from agns.linalg import safe_cho_solve
from agns.methods._helpers import ADAPTIVE_SEARCH_MAX_ITER, new_history, record_trace
from agns.methods.base import MethodResult
from agns.oracles.base import OracleCallsCounter
from agns.stopping import Status, check_stop
from agns.utils.timing import Timer

__all__ = ["cubic_newton", "cubic_newton_step"]


def cubic_newton_step(
    g: NDArray[np.float64],
    A: NDArray[np.float64],
    H: float,
    B: NDArray[np.float64] | None = None,
    eps: float = 1e-8,
) -> tuple[NDArray[np.float64], float, str]:
    """Minimise ``<g, T> + 0.5 <A T, T> + (H/3) ||T||^3`` via the secular equation.

    Returns ``(T, model_value, status)``.  ``status`` is ``"success"``,
    ``"linalg_error"``, or ``"iterations_exceeded"``.
    """
    n = g.shape[0]
    if B is None:
        B = np.eye(n)

    def l2_norm_sqr(x: NDArray[np.float64]) -> float:
        return float(B.dot(x).dot(x))

    def f(T: NDArray[np.float64], T_norm: float) -> float:
        return float(g.dot(T) + 0.5 * A.dot(T).dot(T) + H * T_norm**3 / 3.0)

    def h(r: float, der: bool = False) -> tuple[float, float, NDArray[np.float64], float | None]:
        T = safe_cho_solve(A + H * r * B, -g)
        T_norm = l2_norm_sqr(T) ** 0.5
        h_r = r - T_norm
        h_r_prime: float | None
        if der:
            BT = B.dot(T)
            h_r_prime = 1.0 + H / T_norm * safe_cho_solve(A + H * r * B, BT).dot(BT)
        else:
            h_r_prime = None
        return h_r, T_norm, T, h_r_prime

    try:
        max_r = 1.0
        max_iters = 50
        for _ in range(max_iters):
            h_r, T_norm, T, _ = h(max_r)
            if h_r < -eps:
                max_r *= 2
            elif -eps <= h_r <= eps:
                return T, f(T, T_norm), "success"
            else:
                break

        r = max_r
        for _ in range(max_iters):
            h_r, T_norm, T, h_r_prime = h(r, der=True)
            if -eps <= h_r <= eps:
                return T, f(T, T_norm), "success"
            assert h_r_prime is not None
            r -= h_r / h_r_prime

    except (LinAlgError, ValueError, ZeroDivisionError):
        # ``ZeroDivisionError`` arises in the secular-equation Newton step
        # when ``T_norm`` underflows to zero at a (near-)stationary point;
        # treat it like any other degenerate-solve failure.
        return np.zeros(n), 0.0, "linalg_error"

    return np.zeros(n), 0.0, "iterations_exceeded"


def cubic_newton(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    H_0: float = 1.0,
    H_min: float = 1e-5,
    adaptive_search: bool = True,
    trace: bool = True,
    B: NDArray[np.float64] | None = None,
    eps: float = 1e-8,
    f_star: float | None = None,
    grad_tol: float | None = None,
    warnings: bool = False,
) -> MethodResult:
    """Run Nesterov-Polyak Cubic Newton."""
    counter = OracleCallsCounter(oracle)
    timer = Timer()

    x_k = np.copy(x_0)
    f_k = counter.func(x_k)
    H_k = H_0

    history = new_history() if trace else None
    status = ""

    # Cubic-Newton does not natively track the dual-gradient norm in its
    # legacy trace; the function value is the canonical stopping signal.
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
            g_norm=None,
            grad_tol=grad_tol,
        )
        if decision.stop:
            status = decision.status
            break

        g_k = counter.grad(x_k)
        Hess_k = counter.hess(x_k)

        x_new = x_k.copy()
        f_new = f_k

        for i in range(ADAPTIVE_SEARCH_MAX_ITER + 1):
            if i == ADAPTIVE_SEARCH_MAX_ITER:
                if warnings:
                    print("W: adaptive_iterations_exceeded", flush=True)
                break

            x_delta, model_value, message = cubic_newton_step(g_k, Hess_k, 0.5 * H_k, B, 1e-8)
            if message != "success":
                if warnings:
                    print(f"W: cubic_newton_step: {message}", flush=True)
                H_k *= 2.0
                continue

            f_trial = counter.func(x_k + x_delta)

            if not adaptive_search:
                x_new = x_k + x_delta
                f_new = f_trial
                break

            if f_trial <= f_k + model_value:
                x_new = x_k + x_delta
                f_new = f_trial
                H_k = max(H_k * 0.5, H_min)
                break

            H_k *= 2.0

        x_k, f_k = x_new, f_new

        if not np.isfinite(f_k):
            status = Status.DIVERGED_NONFINITE.value
            break

    return x_k, status, history
