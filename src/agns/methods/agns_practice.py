"""AGNS-Practice: heuristic-momentum + gradient-restart accelerated GNS.

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
   at ``y_k``.
4. **Restart test.**  ``restart_mode`` selects either the
   O'Donoghue-Candes gradient test ``g(x_trial) . (x_trial - x_k) > 0``,
   the function-value test ``f(x_trial) > f(x_k)``, or no restart.

Two flavours:

* :func:`agns_practice` --- dense Cholesky solve on every inner step.
* :func:`agns_practice_wsm` --- Woodbury rank-1 inversion via
  :func:`agns.linalg.wsm_rank_one_solve`; requires a rank-1 Fisher
  ``approx_hess_fn``.

Registry keys: ``agns_practice`` (``is_approx=True``),
``agns_practice_exact`` (``is_approx=False``), ``agns_practice_wsm``
(WSM backend).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray

from agns.linalg import safe_cho_solve, wsm_rank_one_solve
from agns.methods._helpers import (
    build_dual_norm,
    new_history,
    rank_one_fisher_factor,
    record_trace,
)
from agns.methods.base import MethodResult
from agns.oracles.base import OracleCallsCounter
from agns.stopping import Status, check_stop
from agns.utils.timing import Timer

__all__ = ["agns_practice", "agns_practice_wsm"]

_RESTART_MODES = {"gradient", "function_value", "none"}


def _validate_restart_mode(mode: str) -> None:
    if mode not in _RESTART_MODES:
        raise ValueError(
            f"restart_mode must be one of {sorted(_RESTART_MODES)}, got {mode!r}"
        )


def _should_restart(
    mode: str,
    g_trial: NDArray[np.float64],
    x_trial: NDArray[np.float64],
    x_k: NDArray[np.float64],
    f_trial: float,
    f_k: float,
) -> bool:
    if mode == "gradient":
        return bool(g_trial.dot(x_trial - x_k) > 0)
    if mode == "function_value":
        return bool(f_trial > f_k)
    return False


def agns_practice(
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
    """AGNS-Practice with a dense Cholesky inner solve.

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
    _validate_restart_mode(restart_mode)
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
            k=k, n_iters=n_iters, f_k=f_k, f_star=f_star, eps=eps,
            g_norm=g_k_norm, grad_tol=grad_tol,
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
            x_prev = x_k.copy()
            x_k, f_k, g_k = x_trial, f_trial, g_trial
            g_k_norm = g_trial_norm_sqr**0.5
            local_k += 1
            continue

        if is_approx and approx_oracle is not None and callable(approx_hess_fn):
            Hess_k = approx_hess_fn(approx_oracle, y_k)
        else:
            Hess_k = counter.hess(y_k)

        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter:
                if warnings:
                    print(f"W: adaptive_iterations_exceeded, k = {k}", flush=True)
                break

            lambda_k = g_y_norm / gamma_k
            try:
                delta_y = safe_cho_solve(Hess_k + lambda_k * B_eff, -g_y)
                matrix_inverses += 1
            except (LinAlgError, ValueError):
                if warnings:
                    print("W: linalg_error -> tightening gamma", flush=True)
                gamma_k = max(gamma_k * 0.5, gamma_min)
                continue

            x_trial = y_k + delta_y
            f_trial = counter.func(x_trial)
            g_trial = counter.grad(x_trial)
            g_trial_norm_sqr = dual_norm_sqr(g_trial)

            if not adaptive_search:
                break

            if f_y - f_trial >= g_trial_norm_sqr / (8.0 * lambda_k):
                gamma_k *= 2.0
                break
            gamma_k = max(gamma_k * 0.5, gamma_min)

        if _should_restart(restart_mode, g_trial, x_trial, x_k, f_trial, f_k):
            x_prev = x_trial.copy()
            local_k = 0
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

    return x_k, status, history


def agns_practice_wsm(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    gamma_0: float = 1.0,
    gamma_min: float = 1e-9,
    adaptive_search: bool = True,
    is_approx: bool = True,
    approx_oracle: Any = None,
    approx_hess_fn: Any = None,
    trace: bool = True,
    B: NDArray[np.float64] | None = None,
    Binv: NDArray[np.float64] | None = None,
    eps: float = 1e-8,
    f_star: float | None = None,
    grad_tol: float | None = None,
    warnings: bool = False,
    invert_backend: str = "wsm",
    momentum_offset: float = 3.0,
    restart_mode: str = "gradient",
    adaptive_search_max_iter: int = 40,
) -> MethodResult:
    """AGNS-Practice with Sherman-Morrison rank-1 inversion (WSM backend).

    Same hyperparameters as :func:`agns_practice`, plus ``invert_backend``
    (``"cho"`` or ``"wsm"``).
    """
    if invert_backend not in {"cho", "wsm"}:
        raise ValueError(f"invert_backend must be 'cho' or 'wsm', got {invert_backend!r}")
    _validate_restart_mode(restart_mode)

    counter = OracleCallsCounter(oracle)
    timer = Timer()
    B_eff, Binv_eff, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)

    x_k = np.copy(x_0)
    x_prev = np.copy(x_0)
    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    gamma_k = gamma_0
    local_k = 0

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
            k=k, n_iters=n_iters, f_k=f_k, f_star=f_star, eps=eps,
            g_norm=g_k_norm, grad_tol=grad_tol,
        )
        if decision.stop:
            status = decision.status
            break

        beta_k = local_k / (local_k + momentum_offset)
        y_k = x_k + beta_k * (x_k - x_prev)

        g_y = counter.grad(y_k)
        g_y_norm_sqr = dual_norm_sqr(g_y)
        g_y_norm = g_y_norm_sqr**0.5
        f_y = counter.func(y_k)

        x_trial = y_k.copy()
        f_trial = f_y
        g_trial = g_y.copy()
        g_trial_norm_sqr = g_y_norm_sqr

        if g_y_norm < eps:
            x_prev = x_k.copy()
            x_k, f_k, g_k = x_trial, f_trial, g_trial
            g_k_norm = g_y_norm
            local_k += 1
            continue

        Hess_k: NDArray[np.float64] | None
        if invert_backend == "cho":
            if is_approx and approx_oracle is not None and callable(approx_hess_fn):
                Hess_k = approx_hess_fn(approx_oracle, y_k)
            else:
                Hess_k = counter.hess(y_k)
        else:
            if not (is_approx and callable(approx_hess_fn) and approx_oracle is not None):
                raise ValueError("wsm backend requires is_approx=True and a rank-one approx_oracle")
            Hess_k = None

        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter:
                if warnings:
                    print(f"W: adaptive_iterations_exceeded, k = {k}", flush=True)
                break

            lambda_k = g_y_norm / gamma_k
            try:
                if invert_backend == "cho":
                    delta_y = safe_cho_solve(Hess_k + lambda_k * B_eff, -g_y)  # type: ignore[operator]
                else:
                    alpha, g0 = rank_one_fisher_factor(approx_oracle, y_k, g_y)
                    delta_y = wsm_rank_one_solve(g_y, lambda_k, Binv_eff, alpha, g0)
                matrix_inverses += 1
            except (LinAlgError, ValueError) as exc:
                if warnings:
                    print(f"W: linear solve error ({exc}) -> tightening gamma", flush=True)
                gamma_k = max(gamma_k * 0.5, gamma_min)
                continue

            x_trial = y_k + delta_y
            f_trial = counter.func(x_trial)
            g_trial = counter.grad(x_trial)
            g_trial_norm_sqr = dual_norm_sqr(g_trial)

            if not adaptive_search:
                break

            if f_y - f_trial >= g_trial_norm_sqr / (8.0 * lambda_k):
                gamma_k *= 2.0
                break
            gamma_k = max(gamma_k * 0.5, gamma_min)

        if _should_restart(restart_mode, g_trial, x_trial, x_k, f_trial, f_k):
            x_prev = x_trial.copy()
            local_k = 0
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

    return x_k, status, history
