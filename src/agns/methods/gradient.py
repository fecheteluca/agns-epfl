from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods.common.norms import build_first_order_metric
from agns.methods.common.result import MethodResult
from agns.methods.common.stopping import Status
from agns.methods.common.trace import new_history, record_trace
from agns.oracles.base import OracleCallsCounter
from agns.utils.timing import Timer

_LINE_SEARCH_MAX_ITER = 30


def gradient_method(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    max_iter: int = 1000,
    L_0: float = 1.0,
    line_search: bool = True,
    trace: bool = True,
    B: NDArray[np.float64] | None = None,
    Binv: NDArray[np.float64] | None = None,
    L_min: float = 1e-5,
    warnings: bool = False,
) -> MethodResult:
    """Run the Gradient Method.

    Returns
    -------
    MethodResult
        ``(x_k, status, history)``.
    """
    counter = OracleCallsCounter(oracle)
    timer = Timer()
    l2_norm_sqr, dual_norm_sqr, precond = build_first_order_metric(B, Binv)

    x_k = np.copy(x_0)
    func_k = counter.func(x_k)
    grad_k = counter.grad(x_k)
    grad_k_norm_sqr = dual_norm_sqr(grad_k)
    L_k = L_0

    history = new_history() if trace else None
    status = ""

    for k in range(max_iter + 1):
        if trace and history is not None:
            record_trace(
                history,
                func=func_k,
                grad_sqr_norm=grad_k_norm_sqr,
                L=L_k,
                func_calls=counter.func_calls,
                grad_calls=counter.grad_calls,
                time=timer.elapsed(),
                x_k=x_k.copy(),
            )

        if k == max_iter:
            status = Status.ITERATIONS_EXCEEDED.value
            break

        T = x_k
        func_T: Any = func_k
        for i in range(_LINE_SEARCH_MAX_ITER + 1):
            if i == _LINE_SEARCH_MAX_ITER:
                if warnings:
                    print("W: line_search_max_iter_reached", flush=True)
                break

            T = x_k - precond(grad_k) / L_k
            func_T = counter.func(T)

            if not line_search:
                break

            if func_T <= func_k + grad_k.dot(T - x_k) + 0.5 * L_k * l2_norm_sqr(
                T - x_k
            ):
                L_k *= 0.5
                L_k = max(L_k, L_min)
                break
            L_k *= 2

        x_k = T
        func_k = func_T
        grad_k = counter.grad(T)
        grad_k_norm_sqr = dual_norm_sqr(grad_k)

    return x_k, status, history
