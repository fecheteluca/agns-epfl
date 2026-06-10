from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Status(StrEnum):
    """Terminal status strings (success carries its own iteration count)."""

    ITERATIONS_EXCEEDED = "iterations_exceeded"
    DIVERGED_NONFINITE = "diverged_nonfinite"


@dataclass(frozen=True)
class StopDecision:
    """Whether to stop and the status string to record."""

    stop: bool
    status: str


def check_stop(
    *,
    k: int,
    n_iters: int,
    f_k: float,
    f_star: float | None,
    eps: float,
    g_norm: float,
    grad_tol: float | None,
) -> StopDecision:
    """Evaluate the upstream stopping conditions at iteration ``k``.

    Parameters
    ----------
    k:
        Current iteration index.
    n_iters:
        Iteration budget (upstream loops over ``range(n_iters + 1)``).
    f_k:
        Current function value.
    f_star:
        Known optimum, or ``None`` to disable the functional-residual test.
    eps:
        Functional-residual tolerance.
    g_norm:
        Current dual gradient norm.
    grad_tol:
        Gradient-norm tolerance, or ``None`` to disable that test.

    Returns
    -------
    StopDecision
        ``stop=True`` with the matching status string, else ``stop=False``.
    """
    if (f_star is not None and f_k - f_star < eps) or (
        grad_tol is not None and g_norm < grad_tol
    ):
        return StopDecision(True, f"success, {k} iters")
    if k == n_iters:
        return StopDecision(True, Status.ITERATIONS_EXCEEDED.value)
    return StopDecision(False, "")
