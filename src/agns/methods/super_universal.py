"""Super-Universal Newton Method.

Refactor of ``super_newton`` from ``epfml/grad-norm-smooth/src/methods.py`` (Apache-2.0,
commit 9f4ca00). This is the key rival to AGNS: it accelerates *implicitly* through the
regularization schedule ``lambda_k = H_k ||g_k||^alpha`` rather than via explicit momentum.

Its acceptance test differs from GNS (it uses the step direction and a factor of ``4``),
so it keeps its own inner loop rather than the shared :mod:`agns.methods.common.gns_search`.

> Semenov, Jaggi, Doikov. ICLR 2026. arXiv:2506.13710
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods.common.linalg import safe_cho_solve
from agns.methods.common.norms import build_dual_norm, regularised_lhs
from agns.methods.common.result import MethodResult
from agns.methods.common.stopping import check_stop
from agns.methods.common.trace import new_history, record_trace
from agns.oracles.base import OracleCallsCounter
from agns.utils.timing import Timer

_ADAPTIVE_SEARCH_MAX_ITER = 40


def super_newton(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    H_0: float = 1.0,
    alpha: float = 1.0,
    adaptive_search: bool = True,
    trace: bool = True,
    B: NDArray[np.float64] | None = None,
    Binv: NDArray[np.float64] | None = None,
    eps: float = 1e-8,
    H_min: float = 1e-5,
    f_star: float | None = None,
    grad_tol: float | None = None,
    warnings: bool = False,
) -> MethodResult:
    """Run the Super-Universal Newton Method.

    Parameters mirror :func:`agns.methods.gns.grad_norm_smooth`, with ``H_0``/``alpha``/
    ``H_min`` controlling the implicit regularization schedule instead of ``gamma``.

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
    H_k = H_0

    history = new_history() if trace else None
    matrix_inverses = 0
    status = ""

    for k in range(n_iters + 1):
        if trace and history is not None:
            record_trace(
                history,
                func=f_k,
                grad_norm=g_k_norm,
                H_k=H_k,
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

        Hess_k = counter.hess(x_k)

        x_trial = x_k
        f_new: Any = f_k
        g_new = g_k
        g_new_norm_sqr: Any = g_k_norm**2
        for i in range(_ADAPTIVE_SEARCH_MAX_ITER + 1):
            if i == _ADAPTIVE_SEARCH_MAX_ITER:
                if warnings:
                    print(f"W: adaptive_iterations_exceeded, k = {k}", flush=True)
                break

            lambda_k = H_k * g_k_norm**alpha
            delta_x = safe_cho_solve(regularised_lhs(Hess_k, lambda_k, B_eff), -g_k)
            matrix_inverses += 1

            x_trial = x_k + delta_x
            f_new = counter.func(x_trial)
            g_new = counter.grad(x_trial)
            g_new_norm_sqr = dual_norm_sqr(g_new)

            if not adaptive_search:
                break

            if g_new.dot(-delta_x) >= g_new_norm_sqr / (4 * lambda_k):
                H_k *= 0.25
                H_k = max(H_k, H_min)
                break
            H_k *= 4

        x_k = x_trial
        f_k = f_new
        g_k = g_new
        g_k_norm = g_new_norm_sqr**0.5

    return x_k, status, history
