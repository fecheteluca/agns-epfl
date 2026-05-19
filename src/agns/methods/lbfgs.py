"""L-BFGS via :func:`scipy.optimize.minimize` with method='L-BFGS-B'.

The scipy implementation is Fortran-backed (Zhu, Byrd, Lu, Nocedal 1997)
with a rolling-window BFGS Hessian approximation of size ``m``.  Per
accepted iterate the project's stopping criteria (``f_star``, ``eps``,
``grad_tol``) are evaluated in a callback so the run can terminate
ahead of scipy's intrinsic stopping.

Registry key: ``lbfgs``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.optimize
from numpy.typing import NDArray

from agns.methods._helpers import build_dual_norm, new_history, record_trace
from agns.methods.base import MethodResult
from agns.oracles.base import OracleCallsCounter
from agns.stopping import Status, success_status
from agns.utils.timing import Timer

__all__ = ["lbfgs"]


def lbfgs(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    m: int = 10,
    pgtol: float = 1e-10,
    factr: float = 1.0,
    trace: bool = True,
    f_star: float | None = None,
    eps: float = 1e-8,
    grad_tol: float | None = None,
    B: NDArray[np.float64] | None = None,
    Binv: NDArray[np.float64] | None = None,
    warnings: bool = False,
) -> MethodResult:
    """L-BFGS via scipy.optimize.minimize with a callback-driven history."""
    counter = OracleCallsCounter(oracle)
    timer = Timer()
    _, _, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)

    history = new_history() if trace else None

    f_0 = counter.func(x_0)
    g_0 = counter.grad(x_0)
    g_0_norm = dual_norm_sqr(g_0) ** 0.5
    if trace and history is not None:
        record_trace(
            history,
            func=f_0,
            grad_norm=g_0_norm,
            grad_calls=counter.grad_calls,
            func_calls=counter.func_calls,
            time=timer.elapsed(),
            x_k=np.asarray(x_0, dtype=float).copy(),
        )

    def _fun(x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        f = counter.func(x)
        g = counter.grad(x)
        return float(f), np.asarray(g, dtype=float)

    early_stop = {"hit": False}

    def _callback(xk: NDArray[np.float64]) -> bool:
        do_trace = trace and history is not None
        do_stop_check = (f_star is not None) or (grad_tol is not None)
        if not (do_trace or do_stop_check):
            return False
        # One func + grad oracle call per accepted iterate, shared
        # between the trace record and the early-stop test.
        f = counter.func(xk)
        g = counter.grad(xk)
        g_norm = dual_norm_sqr(g) ** 0.5
        if do_trace:
            record_trace(
                history,
                func=float(f),
                grad_norm=g_norm,
                grad_calls=counter.grad_calls,
                func_calls=counter.func_calls,
                time=timer.elapsed(),
                x_k=np.asarray(xk, dtype=float).copy(),
            )
        if (f_star is not None and float(f) - f_star < eps) or (
            grad_tol is not None and g_norm < grad_tol
        ):
            early_stop["hit"] = True
            return True
        return False

    try:
        result = scipy.optimize.minimize(
            _fun,
            np.asarray(x_0, dtype=float),
            jac=True,
            method="L-BFGS-B",
            callback=_callback,
            options={
                "maxiter": n_iters,
                "maxcor": m,
                "gtol": pgtol,
                "ftol": factr * np.finfo(float).eps,
            },
        )
    except Exception as exc:
        if warnings:
            print(f"W: lbfgs failed: {exc}", flush=True)
        return np.asarray(x_0, dtype=float), f"failed: {exc}", history

    n_iters_done = int(result.get("nit", 0))
    if early_stop["hit"] or result.success:
        status = success_status(n_iters_done)
    elif n_iters_done >= n_iters:
        status = Status.ITERATIONS_EXCEEDED.value
    else:
        status = f"failed: {result.message}"

    return np.asarray(result.x, dtype=float), status, history
