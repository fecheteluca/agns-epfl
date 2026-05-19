"""Stopping-criterion helpers shared by every optimisation method.

Every method in :mod:`agns.methods` checks three termination conditions:

1. ``f_k - f_star < eps``    (function-value gap; only when ``f_star`` known)
2. ``||g_k||_* < grad_tol``  (dual gradient norm; only when ``grad_tol`` given)
3. ``k >= n_iters``          (iteration budget exhausted)

The :func:`check_stop` helper evaluates all three uniformly and returns a
:class:`StopDecision` carrying a status string that downstream tooling
(``summary.json``, the aggregator, the table renderer) parses.

The :class:`Status` enum captures the canonical non-success status
strings emitted by the methods (one entry per distinct terminal state).
A successful run uses the format produced by :func:`success_status`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["Status", "StopDecision", "check_stop", "success_status"]


class Status(str, Enum):
    """Terminal status of an optimisation run (non-success cases).

    Subclasses :class:`str` so legacy comparisons like
    ``status == "iterations_exceeded"`` continue to work.  Members
    enumerate every non-success status string actually emitted by the
    methods in :mod:`agns.methods`.  A successful termination is the
    string returned by :func:`success_status` (``"success, K iters"``).
    """

    ITERATIONS_EXCEEDED = "iterations_exceeded"
    DIVERGED_NONFINITE = "diverged_nonfinite"
    LINE_SEARCH_DIVERGED = "line_search_diverged"
    ADAPTIVE_SEARCH_FAILED = "adaptive_search_failed"
    LINALG_ERROR = "linalg_error"


@dataclass(frozen=True)
class StopDecision:
    """Outcome of a single stopping-criterion evaluation."""

    stop: bool
    status: str = ""

    @property
    def should_stop(self) -> bool:
        return self.stop


def success_status(k: int) -> str:
    """Canonical success status string at iteration ``k``."""
    return f"success, {k} iters"


def check_stop(
    *,
    k: int,
    n_iters: int,
    f_k: float,
    f_star: float | None,
    eps: float,
    g_norm: float | None = None,
    grad_tol: float | None = None,
) -> StopDecision:
    """Return whether the current iterate satisfies any stopping criterion.

    Parameters
    ----------
    k : int
        Current iteration index (0-based).
    n_iters : int
        Maximum allowed iterations.
    f_k : float
        Current function value.
    f_star : float or None
        Known optimal value, or ``None`` to skip the function-gap check.
    eps : float
        Function-value gap tolerance: stop when ``f_k - f_star < eps``.
    g_norm : float or None
        Current gradient norm (dual norm).  ``None`` skips the check.
    grad_tol : float or None
        Gradient-norm tolerance.  ``None`` skips the check.

    Returns
    -------
    StopDecision
        ``stop=True`` if any criterion fires; ``stop=False`` otherwise.
    """
    if f_star is not None and f_k - f_star < eps:
        return StopDecision(True, success_status(k))
    if grad_tol is not None and g_norm is not None and g_norm < grad_tol:
        return StopDecision(True, success_status(k))
    if k >= n_iters:
        return StopDecision(True, Status.ITERATIONS_EXCEEDED.value)
    return StopDecision(False)
