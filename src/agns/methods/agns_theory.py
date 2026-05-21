"""AGNS-Theory: rho-strict Monteiro-Svaiter-coupled accelerated GNS.

Implements the Monteiro-Svaiter estimating-sequence coupling with a
rho-strict adaptive search at the inner Newton step.  The outer Picard
fixed-point on ``(a_{k+1}, lambda_k)`` resolves the circular dependence
on ``y_k = (A_k x_k + a_{k+1} v_k) / (A_k + a_{k+1})``.

Constants:

* :data:`RHO_PREFACTOR_OPTIMAL` --- the default strict-acceptance
  contraction ``rho = 1 / sqrt(2)``.  This maximises
  ``rho * sqrt(1 - rho**2)`` on ``(0, 1)`` (elementary AM-GM on
  ``rho**2`` and ``1 - rho**2``), which is the prefactor that appears
  in the canonical rate analysis.  Pass ``rho`` explicitly to override.

Registry key: ``agns_theory``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray

from agns.linalg import safe_cho_solve
from agns.methods._helpers import build_dual_norm, new_history, record_trace
from agns.methods.base import MethodResult
from agns.oracles.base import OracleCallsCounter
from agns.stopping import Status, check_stop, success_status
from agns.utils.timing import Timer

__all__ = ["RHO_PREFACTOR_OPTIMAL", "agns_theory"]

#: Default strict-acceptance contraction, equal to ``1 / sqrt(2)``.
RHO_PREFACTOR_OPTIMAL: float = 1.0 / math.sqrt(2.0)


def agns_theory(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    gamma_0: float = 1.0,
    gamma_min: float = 1e-12,
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
    rho: float = RHO_PREFACTOR_OPTIMAL,
    theta: float = 0.5,
    probe_max: int = 60,
    adaptive_search: bool = True,
    fp_max: int = 3,
    fp_rel_tol: float = 0.10,
    lambda_warm_start: bool = True,
) -> MethodResult:
    """AGNS-Theory: the rho-strict, MS-coupled accelerated GNS variant.

    Parameters
    ----------
    rho : float in (0, 1)
        Strict-acceptance contraction.  Default ``1 / sqrt(2)``.
    theta : float in (0, 1)
        Geometric step factor for the inner gamma search.
    probe_max : int
        Cap on the inner rho-strict probe loop.
    fp_max : int
        Outer Picard fixed-point cap on ``(a_{k+1}, lambda_k)``.
    fp_rel_tol : float
        Relative-change termination tolerance for the Picard loop.
    lambda_warm_start : bool
        Reuse the previous outer iteration's ``lambda`` as the inner
        seed (the default).  Set ``False`` to cold-start every outer
        iteration from ``g_k_norm / gamma_prev``.
    """
    if not (0.0 < rho < 1.0):
        raise ValueError(f"rho must lie in (0, 1), got {rho}")
    if not (0.0 < theta < 1.0):
        raise ValueError(f"theta must lie in (0, 1), got {theta}")

    counter = OracleCallsCounter(oracle)
    timer = Timer()
    B_eff, Binv_eff, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)

    # Monteiro-Svaiter estimating-sequence state.
    x_k = np.copy(x_0)
    v_k = np.copy(x_0)
    A_k = 0.0

    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5

    lambda_pred = max(g_k_norm / gamma_0, 1e-12)
    gamma_prev = gamma_0

    history = new_history() if trace else None
    matrix_inverses = 0
    status = ""

    for k in range(n_iters + 1):
        if trace and history is not None:
            record_trace(
                history,
                func=f_k,
                grad_norm=g_k_norm,
                gamma_k=gamma_prev,
                A_k=A_k,
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
        if not (np.isfinite(f_k) and np.isfinite(g_k_norm)):
            status = Status.DIVERGED_NONFINITE.value
            break

        lambda_iter = (
            lambda_pred if lambda_warm_start else max(g_k_norm / max(gamma_prev, gamma_min), 1e-12)
        )
        accepted = False
        stationary = False
        last_good: (
            tuple[
                float,
                float,
                NDArray[np.float64],
                NDArray[np.float64],
                float,
                NDArray[np.float64],
                float,
                float,
            ]
            | None
        ) = None

        a_kp1_try = 0.0
        A_kp1_try = A_k
        y_k_try = x_k.copy()
        f_y = f_k
        g_y = g_k.copy()

        for _fp_iter in range(fp_max):
            if lambda_iter <= 0.0 or not np.isfinite(lambda_iter):
                lambda_iter = max(g_k_norm / max(gamma_prev, gamma_min), 1e-12)
            a_kp1_try = (1.0 + np.sqrt(1.0 + 4.0 * lambda_iter * A_k)) / (2.0 * lambda_iter)
            A_kp1_try = A_k + a_kp1_try
            y_k_try = (A_k * x_k + a_kp1_try * v_k) / A_kp1_try

            g_y = counter.grad(y_k_try)
            g_y_norm = dual_norm_sqr(g_y) ** 0.5
            f_y = counter.func(y_k_try)

            if g_y_norm < eps:
                stationary = True
                last_good = (
                    a_kp1_try,
                    A_kp1_try,
                    y_k_try.copy(),
                    y_k_try.copy(),
                    f_y,
                    g_y.copy(),
                    gamma_prev,
                    lambda_iter,
                )
                break

            if is_approx and approx_oracle is not None and callable(approx_hess_fn):
                Hess_k = approx_hess_fn(approx_oracle, y_k_try)
            else:
                Hess_k = counter.hess(y_k_try)

            gamma_init = gamma_prev / theta
            inner_accepted = False
            for j in range(probe_max):
                gamma_probe = (theta**j) * gamma_init
                if gamma_probe < gamma_min:
                    break
                lambda_probe = g_y_norm / gamma_probe
                try:
                    h_probe = safe_cho_solve(Hess_k + lambda_probe * B_eff, -g_y)
                    matrix_inverses += 1
                except (LinAlgError, ValueError):
                    continue

                x_probe = y_k_try + h_probe
                f_probe = counter.func(x_probe)
                if not np.isfinite(f_probe):
                    continue
                g_probe = counter.grad(x_probe)
                g_probe_norm_sqr = dual_norm_sqr(g_probe)
                if not np.isfinite(g_probe_norm_sqr):
                    continue
                r_probe = g_probe + lambda_probe * B_eff.dot(h_probe)
                r_probe_dnorm = (max(dual_norm_sqr(r_probe), 0.0)) ** 0.5
                h_probe_norm = (max(h_probe.dot(B_eff).dot(h_probe), 0.0)) ** 0.5

                P1 = (f_y - f_probe) * 8.0 * g_y_norm >= gamma_probe * g_probe_norm_sqr - 1e-15
                P2 = r_probe_dnorm <= rho * lambda_probe * h_probe_norm + 1e-15

                if (not adaptive_search) or (P1 and P2):
                    last_good = (
                        a_kp1_try,
                        A_kp1_try,
                        y_k_try.copy(),
                        x_probe.copy(),
                        f_probe,
                        g_probe.copy(),
                        gamma_probe,
                        lambda_probe,
                    )
                    inner_accepted = True
                    break

            if not inner_accepted:
                break

            accepted = True
            lambda_k_try = last_good[7]  # type: ignore[index]
            rel = abs(lambda_k_try - lambda_iter) / max(lambda_iter, 1e-30)
            if rel < fp_rel_tol:
                break
            lambda_iter = lambda_k_try

        if last_good is not None:
            a_kp1, A_kp1, y_k, x_plus, f_plus, g_plus, gamma_k, lambda_k = last_good
        else:
            a_kp1, A_kp1 = a_kp1_try, A_kp1_try
            y_k = y_k_try
            x_plus = y_k.copy()
            f_plus = f_y
            g_plus = g_y.copy()
            gamma_k = gamma_prev
            lambda_k = lambda_pred

        if stationary:
            x_k = y_k.copy()
            A_k = A_kp1
            f_k = f_plus
            g_k = g_plus
            g_k_norm = (max(dual_norm_sqr(g_plus), 0.0)) ** 0.5
            # Stationary termination: confirmed convergence at y_k.
            status = success_status(k)
            break

        if not accepted:
            if warnings:
                print(
                    f"W: rho-strict search exhausted {probe_max} probes at k={k}",
                    flush=True,
                )
            x_plus = y_k.copy()
            f_plus = f_y
            g_plus = g_y.copy()
            v_k = v_k - a_kp1 * Binv_eff.dot(g_plus)
            x_k = x_plus
            A_k = A_kp1
            f_k = f_plus
            g_k = g_plus
            g_k_norm = (max(dual_norm_sqr(g_plus), 0.0)) ** 0.5
            continue

        # Monteiro-Svaiter dual update.
        v_k = v_k - a_kp1 * Binv_eff.dot(g_plus)
        x_k = x_plus
        A_k = A_kp1
        f_k = f_plus
        g_k = g_plus
        g_k_norm = (max(dual_norm_sqr(g_plus), 0.0)) ** 0.5
        gamma_prev = gamma_k
        lambda_pred = lambda_k

    return x_k, status, history
