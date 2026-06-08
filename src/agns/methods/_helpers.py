"""Internal helpers shared by the method implementations.

This module is private (leading underscore): it factors out duplicated
prologue/epilogue patterns from the optimisation methods so the methods
themselves stay focused on their per-iteration update rule.

Pieces of behaviour factored here:

* :data:`ADAPTIVE_SEARCH_MAX_ITER` --- canonical cap on the GNS / SUN
  inner adaptive-search loop, shared across methods that do not expose
  it as a kwarg.

* :func:`build_dual_norm` --- chooses the squared dual-norm closure
  used to measure ``g_k`` in the configured metric ``B``.  When ``B``
  is ``None`` the Euclidean metric is used and ``Binv`` defaults to the
  identity.

* :func:`gns_adaptive_search` --- the shared GNS adaptive ``gamma``
  search inner loop, used by GNS, GNS-WSM, AGNS and AGNS-WSM via a
  pluggable linear-solve callback.

* :func:`new_history` / :func:`record_trace` --- create and populate the
  :class:`agns.methods.base.HistoryDict` so trace-key naming and value
  shapes stay consistent across every method.

* :func:`rank_one_fisher_factor` --- compute ``(alpha, g0)`` of the
  rank-1 Fisher term ``alpha * g0 g0^T`` of ``(1/p) ||u(x)||^p`` for
  oracles that expose ``func_u`` / ``jac_u`` / ``p``.  Used by the WSM
  backend of GNS-WSM and AGNS-WSM.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray

from agns.methods.base import HistoryDict
from agns.oracles.base import OracleCallsCounter

__all__ = [
    "ADAPTIVE_SEARCH_MAX_ITER",
    "RESTART_MODES",
    "build_dual_norm",
    "gns_adaptive_search",
    "new_history",
    "rank_one_fisher_factor",
    "record_trace",
    "regularised_lhs",
    "should_restart",
    "validate_restart_mode",
]


#: Permitted AGNS ``restart_mode`` values; see :func:`should_restart`.
RESTART_MODES: frozenset[str] = frozenset({"gradient", "function_value", "none"})


def validate_restart_mode(mode: str) -> None:
    """Raise ``ValueError`` if ``mode`` is not one of :data:`RESTART_MODES`."""
    if mode not in RESTART_MODES:
        raise ValueError(f"restart_mode must be one of {sorted(RESTART_MODES)}, got {mode!r}")


def should_restart(
    mode: str,
    g_trial: NDArray[np.float64],
    x_trial: NDArray[np.float64],
    x_k: NDArray[np.float64],
    f_trial: float,
    f_k: float,
) -> bool:
    """Return ``True`` if the AGNS restart predicate fires under ``mode``.

    ``gradient``        : O'Donoghue-Candes "scheme II" gradient test
                          ``g(x_trial) . (x_trial - x_k) > 0``.
    ``function_value``  : function-decrease test ``f(x_trial) > f(x_k)``.
    ``none``            : never restart.
    """
    if mode == "gradient":
        return bool(g_trial.dot(x_trial - x_k) > 0)
    if mode == "function_value":
        return bool(f_trial > f_k)
    return False


def regularised_lhs(
    Hess: NDArray[np.float64],
    lambda_k: float,
    B_eff: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Return ``Hess + lambda_k * B_eff`` without mutating ``Hess``.

    When ``B_eff is None`` (the no-metric case for problems whose
    factory leaves the metric Euclidean to avoid an ``n x n``
    allocation), the regulariser collapses to ``lambda_k * I`` and is
    added in-place to the diagonal of a fresh copy.
    """
    A = Hess.copy()
    if B_eff is None:
        n = A.shape[0]
        idx = np.arange(n)
        A[idx, idx] += lambda_k
    else:
        A += lambda_k * B_eff
    return A


#: Default cap on the inner adaptive-search loop of GNS / Super-Newton.
ADAPTIVE_SEARCH_MAX_ITER: int = 40


def build_dual_norm(
    x_dim: int,
    B: NDArray[np.float64] | None,
    Binv: NDArray[np.float64] | None,
) -> tuple[
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    Callable[[NDArray[np.float64]], float],
]:
    """Return ``(B, Binv, dual_norm_sqr)`` for a method's prologue.

    When ``B`` is ``None``, both ``B`` and ``Binv`` are returned as
    ``None`` (the dual norm collapses to Euclidean ``g.dot(g)``).
    Callers that need a concrete metric matrix (the GNS / Newton
    families) only ever pass a non-None ``B`` and so always get a
    materialised pair back; the ``None`` sentinel is for first-order
    methods on high-dimensional problems where allocating a dense
    ``n x n`` identity would be wasteful.

    When ``B`` is given but ``Binv`` is ``None``, ``Binv`` is computed
    once.
    """
    if B is None:

        def euclidean_dual_norm_sqr(g: NDArray[np.float64]) -> float:
            return float(g.dot(g))

        return None, None, euclidean_dual_norm_sqr

    if Binv is None:
        Binv = cast("NDArray[np.float64]", np.linalg.inv(B))
    Binv_local: NDArray[np.float64] = Binv

    def metric_dual_norm_sqr(g: NDArray[np.float64]) -> float:
        return float(Binv_local.dot(g).dot(g))

    return B, Binv_local, metric_dual_norm_sqr


def gns_adaptive_search(
    *,
    counter: OracleCallsCounter,
    base_x: NDArray[np.float64],
    f_base: float,
    g_base: NDArray[np.float64],
    g_base_norm: float,
    dual_norm_sqr: Callable[[NDArray[np.float64]], float],
    solve: Callable[[float], NDArray[np.float64]],
    gamma_k: float,
    gamma_min: float,
    adaptive_search: bool,
    max_iter: int,
    matrix_inverses: int,
    warnings: bool,
    k: int,
    keep_last_trial_on_exhaust: bool = False,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64], float, float, int]:
    """Shared GNS adaptive ``gamma`` search inner loop.

    Probes the regularised Newton step at ``base_x`` for a sequence of
    ``gamma`` values: it doubles ``gamma`` on a step that satisfies the
    GNS progress predicate and halves it (clipped at ``gamma_min``)
    otherwise.

    ``solve(lambda_k)`` returns the trial step ``delta`` for the
    regulariser ``lambda_k = ||g_base||_* / gamma``; it raises
    :class:`numpy.linalg.LinAlgError` or :class:`ValueError` on an
    indefinite or otherwise unsolvable system, which the loop treats as
    a signal to tighten ``gamma``.

    Returns ``(x_new, f_new, g_new, g_new_norm_sqr, gamma_k,
    matrix_inverses)``.  On an accepted step (or unconditionally, when
    ``adaptive_search`` is ``False``) the trial state is returned.  When
    every probe is rejected the fallback depends on
    ``keep_last_trial_on_exhaust``: ``False`` (GNS / GNS-WSM) falls back
    to ``base_x``; ``True`` (the AGNS variants) returns the last probe
    actually computed, or ``base_x`` if no solve succeeded.
    """
    x_new = base_x.copy()
    f_new = f_base
    g_new = g_base
    g_new_norm_sqr = g_base_norm * g_base_norm

    for i in range(max_iter + 1):
        if i == max_iter:
            if warnings:
                print(f"W: adaptive_iterations_exceeded, k = {k}", flush=True)
            break

        lambda_k = g_base_norm / gamma_k
        try:
            delta = solve(lambda_k)
            matrix_inverses += 1
        except (LinAlgError, ValueError):
            if warnings:
                print("W: linalg_error -> tightening gamma", flush=True)
            gamma_k = max(gamma_k * 0.5, gamma_min)
            continue

        x_trial = base_x + delta
        f_trial = counter.func(x_trial)
        g_trial = counter.grad(x_trial)
        g_trial_norm_sqr = dual_norm_sqr(g_trial)

        if keep_last_trial_on_exhaust:
            x_new, f_new, g_new, g_new_norm_sqr = (
                x_trial,
                f_trial,
                g_trial,
                g_trial_norm_sqr,
            )

        if not adaptive_search:
            return x_trial, f_trial, g_trial, g_trial_norm_sqr, gamma_k, matrix_inverses

        # GNS progress predicate (Lemma 1 of the GNS analysis).
        if f_base - f_trial >= g_trial_norm_sqr / (8.0 * lambda_k):
            return (
                x_trial,
                f_trial,
                g_trial,
                g_trial_norm_sqr,
                gamma_k * 2.0,
                matrix_inverses,
            )
        gamma_k = max(gamma_k * 0.5, gamma_min)

    return x_new, f_new, g_new, g_new_norm_sqr, gamma_k, matrix_inverses


def new_history() -> HistoryDict:
    """Return an empty :class:`HistoryDict`."""
    return defaultdict(list)


def record_trace(history: HistoryDict, **fields: Any) -> None:
    """Append one row of values to a trace.

    Each keyword argument names a trace key; its value is appended to
    that key's list.  ``None`` values are skipped --- callers can pass
    keys conditionally without an ``if`` ladder.
    """
    for key, value in fields.items():
        if value is None:
            continue
        history[key].append(value)


def rank_one_fisher_factor(
    approx_oracle: Any,
    x: NDArray[np.float64],
    g_template: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Return ``(alpha, g0)`` for the rank-1 Fisher term of ``(1/p) ||u(x)||^p``.

    The full rank-1 Fisher Hessian-approximation factor is
    ``alpha * g0 g0^T`` with

    .. math::

        \\alpha \\;=\\; (p-2)\\,\\|u(x)\\|^{p-4},
        \\qquad
        g_0 \\;=\\; J(x)^{\\top} u(x).

    Vanishes (``alpha == 0``, ``g0 == 0``) when ``p == 2``.  Used by the
    Woodbury / Sherman-Morrison solve in :func:`agns.methods.linalg.wsm_rank_one_solve`.

    Parameters
    ----------
    approx_oracle :
        Oracle exposing ``p``, ``func_u(x)``, ``jac_u(x)``.
    x :
        Current iterate.
    g_template :
        A gradient-shaped array; only its dtype/shape is used to build
        the zero ``g0`` when ``p == 2``.
    """
    if approx_oracle.p == 2:
        return 0.0, np.zeros_like(g_template)
    u = approx_oracle.func_u(x)
    norm_u = float(np.linalg.norm(u))
    J = approx_oracle.jac_u(x)
    alpha = (approx_oracle.p - 2) * norm_u ** (approx_oracle.p - 4)
    g0 = J.T.dot(u)
    return alpha, g0
