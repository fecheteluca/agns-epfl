"""AGNS: heuristic-momentum + gradient-restart accelerated GNS.

Per-iteration structure (``k >= 0``):

1. **Extrapolate.**  ``y_k = x_k + beta_k * (x_k - x_{k-1})`` with
   ``beta_k = local_k / (local_k + momentum_offset)``.  ``local_k``
   resets to ``0`` on restart, so ``beta_k = 0`` immediately after a
   restart.
2. **Oracle.**  Evaluate ``g(y_k)``, ``f(y_k)``, and ``H(y_k)`` (or its
   approximation when ``is_approx=True``).
3. **Adaptive Newton step.**  Solve ``(H(y_k) + lambda_k B) delta = -g(y_k)``
   with ``lambda_k = ||g(y_k)||_* / gamma_k``; the adaptive search
   doubles / halves ``gamma_k`` to enforce the GNS progress predicate
   at ``y_k``.  The search itself is the shared
   :func:`agns.methods._helpers.gns_adaptive_search`.
4. **Restart test.**  ``restart_mode`` selects either the
   O'Donoghue-Candes "scheme II" test
   ``g(x_trial) . (x_trial - x_k) > 0`` (the function-decrease analogue
   evaluated at the accepted iterate, NOT the standard scheme-I test
   ``g(y_k) . (x_k - x_{k-1}) > 0``), the function-value test
   ``f(x_trial) > f(x_k)``, or no restart.

This file contains only the dense-Cholesky variant ``agns``; the
Sherman-Morrison rank-1 variant ``agns_wsm`` lives in
:mod:`agns.methods.agns_wsm`, mirroring the GNS / GNS-WSM split.

Registry keys: ``agns_inexact`` (``is_approx=True``),
``agns_exact`` (``is_approx=False``), plus the two no-restart aliases
``agns_inexact_norestart`` and ``agns_exact_norestart`` (same function,
``restart_mode='none'`` baked in via the base yaml).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods._helpers import (
    build_dual_norm,
    gns_adaptive_search,
    new_history,
    record_trace,
    regularised_lhs,
    should_restart,
    validate_restart_mode,
)
from agns.methods.base import MethodResult
from agns.methods.linalg import safe_cho_solve
from agns.methods.stopping import Status, check_stop
from agns.oracles.base import OracleCallsCounter
from agns.utils.timing import Timer

__all__ = ["agns"]


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
    """AGNS with a dense Cholesky inner solve.

    Parameters
    ----------
    momentum_offset : float
        The ``3`` in the Nesterov-style schedule
        ``beta_k = local_k / (local_k + momentum_offset)``.
    restart_mode : {'gradient', 'function_value', 'none'}
        Which test to use for resetting ``local_k`` and the momentum
        anchor.  See module docstring for the predicates.
    adaptive_search_max_iter : int
        Cap on the inner ``gamma_k`` bisection.
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
        f_trial = f_y
        g_trial = g_y.copy()
        g_trial_norm_sqr = g_y_norm * g_y_norm

        if g_y_norm < eps:
            # Near-stationary extrapolation: reset the momentum anchor so
            # we do not feed an effectively-zero gradient into the next
            # step's momentum accumulator.
            x_prev = x_k.copy()
            x_k, f_k, g_k = x_trial, f_trial, g_trial
            g_k_norm = g_trial_norm_sqr**0.5
            local_k = 0
            continue

        if is_approx and approx_oracle is not None and callable(approx_hess_fn):
            Hess_k = approx_hess_fn(approx_oracle, y_k)
        else:
            Hess_k = counter.hess(y_k)

        # Default args bind the per-iteration ``Hess_k`` / ``g_y`` at
        # closure-creation time (the search calls ``solve`` immediately).
        def solve(
            lambda_k: float,
            Hess_k: NDArray[np.float64] = Hess_k,
            g_y: NDArray[np.float64] = g_y,
        ) -> NDArray[np.float64]:
            return safe_cho_solve(regularised_lhs(Hess_k, lambda_k, B_eff), -g_y)

        x_trial, f_trial, g_trial, g_trial_norm_sqr, gamma_k, matrix_inverses = gns_adaptive_search(
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

    # Surface the restart summary on the history dict so the aggregator
    # can lift it into the per-method record without scanning the
    # iteration-level trace.
    if history is not None:
        history["n_restarts_total"] = [n_restarts]
        history["restart_events"] = restart_events
        history["restart_mode"] = [restart_mode]

    return x_k, status, history
