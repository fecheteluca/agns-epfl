"""AdaHessian (Yao et al., 2021), NumPy variant with Hutchinson diagonal.

Adam-like update that replaces ``v_t = E[g_t^2]`` with the Hessian
diagonal estimate ``D_t = E[z (H z)]`` for Rademacher ``z`` (Hutchinson
trace estimator).  Per-iteration update:

    g_t  = oracle.grad(x_t)
    z_t  = sign(uniform(-1, 1))           (Rademacher)
    Hz_t = oracle.hess_vec(x_t, z_t)
    D_t  = z_t * Hz_t                     (element-wise)
    m_t  = beta1 m_{t-1} + (1 - beta1) g_t
    v_t  = beta2 v_{t-1} + (1 - beta2) D_t**2
    m_hat = m_t / (1 - beta1**t)
    v_hat = v_t / (1 - beta2**t)
    x_{t+1} = x_t - lr * m_hat / (sqrt(v_hat) + eps_num)

Defaults (lr = 0.1, beta1 = 0.9, beta2 = 0.999, eps_num = 1e-4) follow
Yao et al. (2021).

Registry key: ``adahessian``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods._helpers import build_dual_norm, new_history, record_trace
from agns.methods.base import MethodResult
from agns.oracles.base import OracleCallsCounter
from agns.stopping import Status, check_stop
from agns.utils.timing import Timer

__all__ = ["adahessian"]


def adahessian(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    lr: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps_num: float = 1e-4,
    n_hutchinson_probes: int = 1,
    seed: int = 0,
    trace: bool = True,
    B: NDArray[np.float64] | None = None,
    Binv: NDArray[np.float64] | None = None,
    f_star: float | None = None,
    eps: float = 1e-8,
    grad_tol: float | None = None,
    warnings: bool = False,
) -> MethodResult:
    """AdaHessian (Yao et al. 2021) on a NumPy oracle.

    Parameters
    ----------
    lr : float
        Step size.
    beta1, beta2 : float
        Adam-style moment decays.
    eps_num : float
        Numerical-stability constant (larger than Adam's because ``v_t``
        estimates Hessian-diagonal magnitudes).
    n_hutchinson_probes : int
        Number of independent Rademacher probes averaged per iteration.
    seed : int
        Seed for the Rademacher probe generator.
    """
    if not (0.0 <= beta1 < 1.0):
        raise ValueError(f"beta1 must lie in [0, 1), got {beta1}")
    if not (0.0 <= beta2 < 1.0):
        raise ValueError(f"beta2 must lie in [0, 1), got {beta2}")
    if lr <= 0.0:
        raise ValueError(f"lr must be positive, got {lr}")
    if n_hutchinson_probes < 1:
        raise ValueError(f"n_hutchinson_probes must be >= 1, got {n_hutchinson_probes}")

    counter = OracleCallsCounter(oracle)
    timer = Timer()
    _, _, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)
    rng = np.random.default_rng(seed)

    n = x_0.shape[0]
    x_k = np.copy(x_0)
    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    m: NDArray[np.float64] = np.zeros(n)
    v: NDArray[np.float64] = np.zeros(n)

    history = new_history() if trace else None
    status = ""

    for t in range(1, n_iters + 2):
        if trace and history is not None:
            record_trace(
                history,
                func=f_k,
                grad_norm=g_k_norm,
                lr_eff=lr,
                grad_calls=counter.grad_calls,
                hess_vec_calls=counter.hess_vec_calls,
                func_calls=counter.func_calls,
                time=timer.elapsed(),
                x_k=x_k.copy(),
            )

        k = t - 1
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

        D_sum: NDArray[np.float64] = np.zeros(n)
        for _ in range(n_hutchinson_probes):
            z = rng.choice([-1.0, 1.0], size=n)
            Hz = counter.hess_vec(x_k, z)
            D_sum = D_sum + z * Hz
        D = D_sum / n_hutchinson_probes

        m = beta1 * m + (1.0 - beta1) * g_k
        v = beta2 * v + (1.0 - beta2) * (D * D)
        bc1 = 1.0 - beta1**t
        bc2 = 1.0 - beta2**t
        m_hat = m / bc1
        v_hat = v / bc2

        x_k = x_k - lr * m_hat / (np.sqrt(v_hat) + eps_num)
        f_k = counter.func(x_k)
        if not np.isfinite(f_k):
            if warnings:
                print(f"W: adahessian diverged at t = {t}", flush=True)
            status = Status.DIVERGED_NONFINITE.value
            break
        g_k = counter.grad(x_k)
        g_k_norm = dual_norm_sqr(g_k) ** 0.5

    return x_k, status, history
