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


def _make_wsm_solve(
    approx_oracle: Any,
    x_k: NDArray[np.float64],
    g_k: NDArray[np.float64],
    Binv: NDArray[np.float64],
) -> Any:
    """Build the Sherman-Morrison solve callback for a rank-one approximate Hessian.

    Reproduces the ``invert_backend == "wsm"`` branch of upstream exactly.
    """

    def solve(lambda_k: float) -> NDArray[np.float64]:
        u = approx_oracle.func_u(x_k)
        norm_u = np.linalg.norm(u)
        J = approx_oracle.jac_u(x_k)
        if approx_oracle.p != 2:
            alpha = (approx_oracle.p - 2) * norm_u ** (approx_oracle.p - 4)
            g0 = J.T.dot(u)
        else:
            alpha = 0.0
            g0 = np.zeros_like(g_k)
        inv_scale = 1.0 / lambda_k
        Pinv_gk = inv_scale * Binv.dot(-g_k)
        Pinv_g0 = inv_scale * Binv.dot(g0)
        denom = 1.0 + alpha * (g0.dot(Pinv_g0))
        return Pinv_gk - (alpha / denom) * Pinv_g0 * (g0.dot(Pinv_gk))

    return solve


def grad_norm_smooth_for_rank_one(
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
    invert_backend: str = "cho",
) -> MethodResult:
    """Run GNS with a selectable dense/rank-one inverse backend.

    See :func:`agns.methods.gns.grad_norm_smooth` for the shared parameters. The extra
    ``invert_backend`` selects ``"cho"`` (dense) or ``"wsm"`` (rank-one Sherman-Morrison).

    Returns
    -------
    MethodResult
        ``(x_k, status, history)`` with a history that omits ``x_k`` (upstream schema).
    """
    counter = OracleCallsCounter(oracle)
    timer = Timer()
    B_eff, Binv_eff, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)

    x_k = np.copy(x_0)
    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = np.sqrt(dual_norm_sqr(g_k))
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

        if invert_backend == "cho":

            def solve(
                lambda_k: float, Hess_k: Any = Hess_k, g_k: Any = g_k
            ) -> NDArray[np.float64]:
                return safe_cho_solve(regularised_lhs(Hess_k, lambda_k, B_eff), -g_k)

        elif invert_backend == "wsm":
            if not (is_approx and callable(approx_hess_fn)):
                raise ValueError("wsm backend requires a rank-one approx_hess_fn")
            solve = _make_wsm_solve(approx_oracle, x_k, g_k, Binv_eff)
        else:
            raise ValueError(f"Unknown invert_backend {invert_backend!r}")

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
        g_k_norm = np.sqrt(g_trial_norm_sqr)

    return x_k, status, history
