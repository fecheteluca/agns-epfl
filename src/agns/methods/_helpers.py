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

* :func:`new_history` / :func:`record_trace` --- create and populate the
  :class:`agns.methods.base.HistoryDict` so trace-key naming and value
  shapes stay consistent across every method.

* :func:`rank_one_fisher_factor` --- compute ``(alpha, g0)`` of the
  rank-1 Fisher term ``alpha * g0 g0^T`` of ``(1/p) ||u(x)||^p`` for
  oracles that expose ``func_u`` / ``jac_u`` / ``p``.  Used by the WSM
  backend of GNS-WSM and AGNS-Practice-WSM.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from agns.methods.base import HistoryDict

__all__ = [
    "ADAPTIVE_SEARCH_MAX_ITER",
    "build_dual_norm",
    "new_history",
    "rank_one_fisher_factor",
    "record_trace",
]

#: Default cap on the inner adaptive-search loop of GNS / Super-Newton.
ADAPTIVE_SEARCH_MAX_ITER: int = 40


def build_dual_norm(
    x_dim: int,
    B: NDArray[np.float64] | None,
    Binv: NDArray[np.float64] | None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    Callable[[NDArray[np.float64]], float],
]:
    """Return ``(B, Binv, dual_norm_sqr)`` for a method's prologue.

    When ``B`` is ``None``, both ``B`` and ``Binv`` default to the
    identity and the dual norm collapses to Euclidean.  When ``B`` is
    given but ``Binv`` is ``None``, ``Binv`` is computed once.
    """
    if B is None:
        B_eff = np.eye(x_dim)
        Binv_eff = np.eye(x_dim)

        def dual_norm_sqr(g: NDArray[np.float64]) -> float:
            return float(g.dot(g))

        return B_eff, Binv_eff, dual_norm_sqr

    if Binv is None:
        Binv = cast("NDArray[np.float64]", np.linalg.inv(B))
    Binv_local: NDArray[np.float64] = Binv

    def dual_norm_sqr(g: NDArray[np.float64]) -> float:
        return float(Binv_local.dot(g).dot(g))

    return B, Binv_local, dual_norm_sqr


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
    Woodbury / Sherman-Morrison solve in :func:`agns.linalg.wsm_rank_one_solve`.

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
