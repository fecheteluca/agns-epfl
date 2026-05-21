"""Trust-region Newton with Steihaug-CG subproblem solver.

At each outer iteration approximately solve

    min_d   m_k(d) := f_k + <g_k, d> + 0.5 <H_k d, d>
    s.t.    ||d|| <= Delta_k

via Steihaug's truncated CG, then update ``Delta_k`` based on the
actual-to-predicted reduction ratio (Nocedal-Wright, Ch. 4).

Registry key: ``trust_region``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods._helpers import build_dual_norm, new_history, record_trace
from agns.methods.base import MethodResult
from agns.oracles.base import OracleCallsCounter
from agns.stopping import check_stop
from agns.utils.timing import Timer

__all__ = ["trust_region"]


def _steihaug_cg(
    g: NDArray[np.float64],
    H: NDArray[np.float64],
    Delta: float,
    *,
    tol_rel: float = 1e-8,
    max_iter: int = 200,
) -> tuple[NDArray[np.float64], str]:
    """Steihaug truncated CG for ``min <g, d> + 0.5 d^T H d``, ``||d|| <= Delta``."""
    n = g.shape[0]
    p: NDArray[np.float64] = np.zeros(n)
    r: NDArray[np.float64] = g.copy()
    d: NDArray[np.float64] = -r.copy()
    rr = float(r.dot(r))
    g_norm = float(np.sqrt(rr))
    tol = max(tol_rel, tol_rel * g_norm)
    tol_sq = tol * tol

    if rr <= tol_sq:
        return p, "tol"

    for _ in range(max_iter):
        Hd = H.dot(d)
        dHd = float(d.dot(Hd))
        if dHd <= 0.0:
            tau = _solve_boundary_tau(p, d, Delta)
            return p + tau * d, "negative_curvature"

        alpha = rr / dHd
        p_next = p + alpha * d
        if float(np.linalg.norm(p_next)) >= Delta:
            tau = _solve_boundary_tau(p, d, Delta)
            return p + tau * d, "boundary"

        r_next = r + alpha * Hd
        rr_next = float(r_next.dot(r_next))
        if rr_next <= tol_sq:
            return p_next, "tol"

        beta = rr_next / rr
        d = -r_next + beta * d
        p = p_next
        r = r_next
        rr = rr_next

    return p, "iter_budget"


def _solve_boundary_tau(
    p: NDArray[np.float64],
    d: NDArray[np.float64],
    Delta: float,
) -> float:
    """Largest tau >= 0 with ``||p + tau d|| = Delta``."""
    a = float(d.dot(d))
    b = 2.0 * float(p.dot(d))
    c = float(p.dot(p)) - Delta * Delta
    disc = max(b * b - 4.0 * a * c, 0.0)
    return (-b + float(np.sqrt(disc))) / (2.0 * a)


def trust_region(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    n_iters: int = 1000,
    Delta_0: float = 1.0,
    Delta_max: float = 1e8,
    eta1: float = 0.25,
    eta2: float = 0.75,
    shrink_factor: float = 0.25,
    expand_factor: float = 2.0,
    cg_tol_rel: float = 1e-8,
    cg_max_iter: int = 200,
    trace: bool = True,
    B: NDArray[np.float64] | None = None,
    Binv: NDArray[np.float64] | None = None,
    f_star: float | None = None,
    eps: float = 1e-8,
    grad_tol: float | None = None,
    warnings: bool = False,
) -> MethodResult:
    """Trust-region Newton with Steihaug-CG subproblem solver.

    Parameters
    ----------
    Delta_0, Delta_max : float
        Initial and maximum trust-region radius.
    eta1, eta2 : float
        Acceptance / expansion thresholds (Nocedal-Wright defaults).
    shrink_factor, expand_factor : float
        Geometric update factors for the radius.
    cg_tol_rel, cg_max_iter :
        Subproblem CG tolerance and budget.
    """
    counter = OracleCallsCounter(oracle)
    timer = Timer()
    _, _, dual_norm_sqr = build_dual_norm(x_0.shape[0], B, Binv)

    x_k = np.copy(x_0)
    f_k = counter.func(x_k)
    g_k = counter.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    Delta = Delta_0

    history = new_history() if trace else None
    status = ""

    for k in range(n_iters + 1):
        if trace and history is not None:
            record_trace(
                history,
                func=f_k,
                grad_norm=g_k_norm,
                Delta=Delta,
                grad_calls=counter.grad_calls,
                hess_calls=counter.hess_calls,
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

        H_k = counter.hess(x_k)
        d, _cg_exit = _steihaug_cg(g_k, H_k, Delta, tol_rel=cg_tol_rel, max_iter=cg_max_iter)

        Hd = H_k.dot(d)
        m_pred = -(float(g_k.dot(d)) + 0.5 * float(d.dot(Hd)))
        if m_pred <= 0.0:
            Delta *= shrink_factor
            if warnings:
                print(f"W: TR: non-positive model reduction at k={k}", flush=True)
            continue

        x_trial = x_k + d
        f_trial = counter.func(x_trial)
        if not np.isfinite(f_trial):
            Delta *= shrink_factor
            continue
        actual = f_k - f_trial
        ratio = actual / m_pred

        d_norm = float(np.linalg.norm(d))
        if ratio < eta1:
            Delta *= shrink_factor
        elif ratio > eta2 and d_norm >= 0.99 * Delta:
            Delta = min(Delta * expand_factor, Delta_max)

        if ratio > 0.0:
            x_k = x_trial
            f_k = f_trial
            g_k = counter.grad(x_k)
            g_k_norm = dual_norm_sqr(g_k) ** 0.5

    return x_k, status, history
