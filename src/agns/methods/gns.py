from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods.common.gns_search import gns_adaptive_search
from agns.methods.common.linalg import safe_cho_solve
from agns.methods.common.norms import build_dual_norm, regularised_lhs
from agns.methods.common.result import MethodResult
from agns.methods.common.stopping import check_stop
from agns.methods.common.trace import new_history, record_trace
from agns.oracles.base import OracleCallsCounter
from agns.utils.timing import Timer

_ADAPTIVE_SEARCH_MAX_ITER = 40


def grad_norm_smooth(
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
) -> MethodResult:
    """Run the Gradient-Regularized Newton Method (Algorithm 1).

    Parameters
    ----------
    oracle:
        Objective oracle exposing ``func``/``grad``/``hess``.
    x_0:
        Starting point.
    n_iters:
        Iteration budget.
    gamma_0, gamma_min:
        Initial regularization parameter and its floor (applied on accept).
    adaptive_search:
        Toggle the gradient-normalized acceptance loop.
    is_approx, approx_oracle, approx_hess_fn:
        Use an approximate Hessian ``approx_hess_fn(approx_oracle, x_k)`` when enabled.
    trace:
        Record per-iteration history.
    B, Binv:
        Metric matrix and its inverse (Euclidean when ``B`` is ``None``).
    eps:
        Functional-residual tolerance and ``lambda_k`` stabilizer.
    f_star, grad_tol:
        Optional optimum and gradient-norm stopping thresholds.
    warnings:
        Print adaptive-search exhaustion warnings.

    Returns
    -------
    MethodResult
        ``(x_k, status, history)``.
    """
    counter = OracleCallsCounter(oracle)
    timer = Timer()
    B_eff, _, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)

    x_k = np.copy(x_0)
    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    gamma_k = gamma_0

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

        if is_approx and approx_oracle is not None and callable(approx_hess_fn):
            Hess_k = approx_hess_fn(approx_oracle, x_k)
        else:
            Hess_k = counter.hess(x_k)

        def solve(
            lambda_k: float, Hess_k: Any = Hess_k, g_k: Any = g_k
        ) -> NDArray[np.float64]:
            return safe_cho_solve(regularised_lhs(Hess_k, lambda_k, B_eff), -g_k)

        x_k, f_k, g_k, g_trial_norm_sqr, gamma_k, matrix_inverses = gns_adaptive_search(
            counter=counter,
            base_x=x_k,
            f_base=f_k,
            g_base=g_k,
            g_base_norm=g_k_norm,
            dual_norm_sqr=dual_norm_sqr,
            solve=solve,
            gamma_k=gamma_k,
            gamma_min=gamma_min,
            adaptive_search=adaptive_search,
            max_iter=_ADAPTIVE_SEARCH_MAX_ITER,
            matrix_inverses=matrix_inverses,
            eps=eps,
            warnings=warnings,
            k=k,
            keep_last_trial_on_exhaust=True,
        )
        g_k_norm = g_trial_norm_sqr**0.5

    return x_k, status, history
