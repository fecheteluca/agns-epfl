"""Adam optimiser (Kingma & Ba, 2015), NumPy variant.

Per-iteration update:

    m_t = beta1 m_{t-1} + (1 - beta1) g_t
    v_t = beta2 v_{t-1} + (1 - beta2) g_t**2          (element-wise square)
    m_hat = m_t / (1 - beta1**t)
    v_hat = v_t / (1 - beta2**t)
    x_{t+1} = x_t - lr * m_hat / (sqrt(v_hat) + eps_num)

Default hyperparameters (lr = 1e-3, beta1 = 0.9, beta2 = 0.999,
eps_num = 1e-8) follow Kingma & Ba (2015).  ``g_t = oracle.grad(x_t)``
is deterministic in this variant; pointing at a stochastic-gradient
oracle would recover the mini-batch behaviour.

Registry key: ``adam``.
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

__all__ = ["adam"]


def adam(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps_num: float = 1e-8,
    trace: bool = True,
    B: NDArray[np.float64] | None = None,
    Binv: NDArray[np.float64] | None = None,
    f_star: float | None = None,
    eps: float = 1e-8,
    grad_tol: float | None = None,
    warnings: bool = False,
) -> MethodResult:
    """Adam (Kingma-Ba 2015) on a NumPy oracle.

    Parameters
    ----------
    lr : float
        Step size (Kingma-Ba's ``alpha``).
    beta1, beta2 : float
        First/second moment decay.
    eps_num : float
        Numerical-stability constant added inside the sqrt.
    B, Binv :
        Optional metric for the gradient-norm convergence test.  Adam
        itself operates in the Euclidean metric.
    """
    if not (0.0 <= beta1 < 1.0):
        raise ValueError(f"beta1 must lie in [0, 1), got {beta1}")
    if not (0.0 <= beta2 < 1.0):
        raise ValueError(f"beta2 must lie in [0, 1), got {beta2}")
    if lr <= 0.0:
        raise ValueError(f"lr must be positive, got {lr}")

    counter = OracleCallsCounter(oracle)
    timer = Timer()
    _, _, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)

    x_k = np.copy(x_0)
    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    m = np.zeros_like(x_0)
    v = np.zeros_like(x_0)

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
                func_calls=counter.func_calls,
                time=timer.elapsed(),
                x_k=x_k.copy(),
            )

        k = t - 1
        decision = check_stop(
            k=k, n_iters=n_iters, f_k=f_k, f_star=f_star, eps=eps,
            g_norm=g_k_norm, grad_tol=grad_tol,
        )
        if decision.stop:
            status = decision.status
            break

        m = beta1 * m + (1.0 - beta1) * g_k
        v = beta2 * v + (1.0 - beta2) * (g_k * g_k)
        bc1 = 1.0 - beta1**t
        bc2 = 1.0 - beta2**t
        m_hat = m / bc1
        v_hat = v / bc2

        x_k = x_k - lr * m_hat / (np.sqrt(v_hat) + eps_num)
        f_k = counter.func(x_k)
        if not np.isfinite(f_k):
            if warnings:
                print(f"W: adam diverged at t = {t}", flush=True)
            status = Status.DIVERGED_NONFINITE.value
            break
        g_k = counter.grad(x_k)
        g_k_norm = dual_norm_sqr(g_k) ** 0.5

    return x_k, status, history
