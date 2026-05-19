"""Newton's method with Armijo backtracking line search.

Per-iteration step:

* Newton direction ``d_k = -H_k^{-1} g_k``, with the regulariser
  ``tau`` doubled per Cholesky failure (Nocedal-Wright Sec. 3.4).
* Armijo line search ``f(x_k + alpha d_k) <= f(x_k) + c1 alpha <g_k, d_k>``
  with geometric backoff ``alpha <- rho_backtrack * alpha``.

If the Newton direction is not a descent direction (indefinite ``H_k``),
the method falls back to the negative gradient for that iteration.

Registry key: ``newton``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray

from agns.linalg import safe_cho_solve
from agns.methods._helpers import build_dual_norm, new_history, record_trace
from agns.methods.base import MethodResult
from agns.oracles.base import OracleCallsCounter
from agns.stopping import Status, check_stop
from agns.utils.timing import Timer

__all__ = ["newton"]


def newton(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    c1: float = 1e-4,
    rho_backtrack: float = 0.5,
    alpha_0: float = 1.0,
    max_backtracks: int = 30,
    tau_0: float = 1e-6,
    tau_max: float = 1e8,
    trace: bool = True,
    f_star: float | None = None,
    eps: float = 1e-8,
    grad_tol: float | None = None,
    B: NDArray[np.float64] | None = None,
    Binv: NDArray[np.float64] | None = None,
    warnings: bool = False,
) -> MethodResult:
    """Newton's method with Armijo backtracking and PSD-regularisation guard.

    Parameters
    ----------
    c1 : float
        Armijo sufficient-decrease constant.
    rho_backtrack : float
        Backtracking factor.
    alpha_0 : float
        Initial step length.  Default 1.0 (the canonical Newton step).
    max_backtracks : int
        Cap on backtracking iterations per outer iter.
    tau_0, tau_max : float
        Smallest and largest Hessian regulariser used to recover from
        indefinite ``H_k``.  ``tau`` is doubled per Cholesky failure
        starting from ``tau_0`` and the iteration aborts if it exceeds
        ``tau_max``.
    B, Binv :
        Optional metric for the gradient-norm convergence test.
    """
    counter = OracleCallsCounter(oracle)
    timer = Timer()
    _, _, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)
    I_n = np.eye(x_0.shape[0])

    x_k = np.copy(x_0)
    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5

    history = new_history() if trace else None
    matrix_inverses = 0
    status = ""
    alpha_prev = alpha_0  # for the trace at iteration 0

    for k in range(n_iters + 1):
        if trace and history is not None:
            record_trace(
                history,
                func=f_k,
                grad_norm=g_k_norm,
                alpha=alpha_prev,
                grad_calls=counter.grad_calls,
                func_calls=counter.func_calls,
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

        H_k = counter.hess(x_k)

        # PSD-regularised solve.
        tau = 0.0
        d_k: NDArray[np.float64] | None = None
        for _ in range(50):
            try:
                M = H_k + tau * I_n if tau > 0.0 else H_k
                d_k = safe_cho_solve(M, -g_k)
                matrix_inverses += 1
                break
            except (LinAlgError, ValueError):
                tau = tau_0 if tau == 0.0 else tau * 2.0
                if tau > tau_max:
                    if warnings:
                        print(f"W: Newton: tau exceeded {tau_max} at k = {k}", flush=True)
                    break

        if d_k is None:
            status = Status.LINALG_ERROR.value
            break

        # Ensure a descent direction; otherwise fall back to -gradient.
        gd = float(g_k.dot(d_k))
        if gd >= 0.0:
            d_k = -g_k
            gd = float(g_k.dot(d_k))

        alpha = alpha_0
        accepted = False
        f_trial = f_k
        for _ in range(max_backtracks):
            x_trial = x_k + alpha * d_k
            f_trial = counter.func(x_trial)
            if np.isfinite(f_trial) and f_trial <= f_k + c1 * alpha * gd:
                accepted = True
                break
            alpha *= rho_backtrack
        if not accepted:
            status = Status.LINE_SEARCH_DIVERGED.value
            break

        alpha_prev = alpha
        x_k = x_k + alpha * d_k
        f_k = f_trial
        g_k = counter.grad(x_k)
        g_k_norm = dual_norm_sqr(g_k) ** 0.5

    return x_k, status, history
