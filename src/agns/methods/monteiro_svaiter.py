"""Monteiro-Svaiter Accelerated Cubic Newton (MS-ACN, 2013).

Couples the Monteiro-Svaiter estimating-sequence framework (same outer
skeleton as AGNS-Theory) with a Nesterov-Polyak cubic-regularised inner
step.

Per outer iteration:

1. Picard fixed point on ``(a_{k+1}, lambda)`` using the canonical
   coupling ``A_{k+1} = a_{k+1}^2 lambda``, ``A_{k+1} = A_k + a_{k+1}``.
2. ``y_k = (A_k x_k + a_{k+1} v_k) / A_{k+1}``.
3. Cubic inner step at ``y_k`` (cubic_newton_step).
4. Accept iff the Nesterov 2008 cubic-model decrease test holds.
5. Dual update ``v_{k+1} = v_k - a_{k+1} B^{-1} g(x_{k+1})``.

Registry key: ``monteiro_svaiter_acn``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods._helpers import build_dual_norm, new_history, record_trace
from agns.methods.base import MethodResult
from agns.methods.cubic_newton import cubic_newton_step
from agns.oracles.base import OracleCallsCounter
from agns.stopping import Status, check_stop
from agns.utils.timing import Timer

__all__ = ["monteiro_svaiter_acn"]


def monteiro_svaiter_acn(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    M_0: float = 1.0,
    M_min: float = 1e-5,
    M_max: float = 1e10,
    sigma_max: float = 0.9,
    probe_max: int = 30,
    monotone: bool = True,
    trace: bool = True,
    B: NDArray[np.float64] | None = None,
    Binv: NDArray[np.float64] | None = None,
    eps: float = 1e-8,
    f_star: float | None = None,
    grad_tol: float | None = None,
    warnings: bool = False,
) -> MethodResult:
    """Monteiro-Svaiter Accelerated Cubic Newton.

    Parameters
    ----------
    M_0, M_min, M_max : float
        Cubic regulariser bounds.
    sigma_max : float
        Strict A-HPE contraction (kept in the signature for API parity;
        the canonical cubic-prox specialisation collapses ``sigma -> 1``).
    probe_max : int
        Max inner probes per outer Picard iterate.
    monotone : bool
        Accept ``x_{k+1}`` only when ``f(x_{k+1}) <= f(x_k)``.
    """
    del sigma_max  # accepted for API parity; not used in this specialisation
    counter = OracleCallsCounter(oracle)
    timer = Timer()
    B_eff, Binv_eff, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)

    x_k = np.copy(x_0)
    v_k = np.copy(x_0)
    A_k = 0.0
    M_k = M_0

    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    lambda_pred = max(M_0, 1e-12)

    history = new_history() if trace else None
    matrix_inverses = 0
    status = ""

    fp_max = 3
    fp_rel_tol = 0.10

    for k in range(n_iters + 1):
        if trace and history is not None:
            record_trace(
                history,
                func=f_k,
                grad_norm=g_k_norm,
                lambda_k=lambda_pred,
                A_k=A_k,
                M=M_k,
                grad_calls=counter.grad_calls,
                hess_calls=counter.hess_calls,
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

        lambda_iter = lambda_pred
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

        for _fp_iter in range(fp_max):
            if lambda_iter <= 0.0 or not np.isfinite(lambda_iter):
                lambda_iter = max(M_k, 1e-12)
            a_kp1_try = (1.0 + np.sqrt(1.0 + 4.0 * lambda_iter * A_k)) / (2.0 * lambda_iter)
            A_kp1_try = A_k + a_kp1_try
            y_k_try = (A_k * x_k + a_kp1_try * v_k) / A_kp1_try

            g_y = counter.grad(y_k_try)
            f_y = counter.func(y_k_try)
            H_y = counter.hess(y_k_try)

            inner_accepted = False
            for _probe in range(probe_max):
                x_delta, model_value, message = cubic_newton_step(
                    g_y,
                    H_y,
                    0.5 * M_k,
                    B_eff,
                    1e-8,
                )
                if message != "success":
                    M_k = min(M_k * 2.0, M_max)
                    if M_k >= M_max:
                        break
                    continue
                matrix_inverses += 1

                x_probe = y_k_try + x_delta
                f_probe = counter.func(x_probe)
                if not np.isfinite(f_probe):
                    M_k = min(M_k * 2.0, M_max)
                    continue
                g_probe = counter.grad(x_probe)
                h_norm = float(np.linalg.norm(x_delta))
                lambda_eff = 0.5 * M_k * max(h_norm, 1e-15)

                if f_probe <= f_y + model_value:
                    last_good = (
                        a_kp1_try,
                        A_kp1_try,
                        y_k_try.copy(),
                        x_probe.copy(),
                        float(f_probe),
                        g_probe.copy(),
                        lambda_eff,
                        1.0,
                    )
                    inner_accepted = True
                    break
                M_k = min(M_k * 2.0, M_max)

            if not inner_accepted:
                break

            lambda_k_try = last_good[6]  # type: ignore[index]
            rel = abs(lambda_k_try - lambda_iter) / max(lambda_iter, 1e-30)
            if rel < fp_rel_tol:
                break
            lambda_iter = lambda_k_try

        if last_good is None:
            if warnings:
                print(f"W: MS-ACN inner loop failed at k = {k}", flush=True)
            status = Status.ADAPTIVE_SEARCH_FAILED.value
            break

        a_kp1, A_kp1, _y_k, x_plus, f_plus, g_plus, lambda_eff, _ = last_good

        if monotone and f_plus > f_k:
            A_k = A_kp1
            M_k = min(M_k * 2.0, M_max)
            continue

        x_k = x_plus
        f_k = float(f_plus)
        g_k = g_plus
        g_k_norm = dual_norm_sqr(g_k) ** 0.5
        v_k = v_k - a_kp1 * Binv_eff.dot(g_k)
        A_k = A_kp1
        lambda_pred = lambda_eff
        M_k = max(M_k * 0.5, M_min)

    return x_k, status, history
