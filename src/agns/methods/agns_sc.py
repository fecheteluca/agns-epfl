"""AGNS-SC: restarted AGNS-Theory for strongly convex objectives.

Wraps :func:`agns_theory` in an outer restart loop.  Each segment runs
``K_seg`` AGNS-Theory iterations starting from the previous segment's
final iterate; the per-segment trace is concatenated into a single
global :class:`HistoryDict`, with the segment boundaries recorded under
``history['segment_boundaries']``.

When ``K_seg`` is not supplied explicitly, the helper :func:`_default_k_seg`
chooses it from the strong-convexity constant ``mu``, an upper bound on
the Lipschitz Hessian ``L_2``, and an upper bound on the initial residual
``Delta_0 = f(x_0) - f^star``.

Registry key: ``agns_sc``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods.agns_theory import RHO_PREFACTOR_OPTIMAL, agns_theory
from agns.methods.base import MethodResult
from agns.stopping import success_status

__all__ = ["agns_sc"]


def _default_k_seg(
    L_2: float,
    mu: float,
    Delta_0: float,
    *,
    rho: float = RHO_PREFACTOR_OPTIMAL,
    theta: float = 0.5,
) -> int:
    """Closed-form choice of segment length ``K_seg``.

    Returns

    .. math::

        K_{seg}
        = \\bigl\\lceil
            3 \\cdot 2^{1/6}\\,
            (L_2/\\mu)^{1/3}\\,
            (\\Delta_0/\\mu)^{1/6}\\,
            (\\rho \\theta^2 \\sqrt{1 - \\rho^2})^{-1/3}
          \\bigr\\rceil.

    The ``(Delta_0/mu)^{1/6}`` factor is the dimensional cost of the
    strong-convexity substitution ``D_s^3 <= (2 Delta_s/mu)^{3/2}`` in
    the per-segment recursion.
    """
    if mu <= 0.0:
        raise ValueError(f"mu must be strictly positive, got {mu}")
    if Delta_0 < 0.0:
        raise ValueError(f"Delta_0 must be non-negative, got {Delta_0}")
    cond = L_2 / mu
    ratio = Delta_0 / mu
    decay = rho * theta**2 * math.sqrt(1.0 - rho**2)
    k = math.ceil(
        3.0 * 2.0 ** (1.0 / 6.0) * cond ** (1.0 / 3.0) * ratio ** (1.0 / 6.0) / decay ** (1.0 / 3.0)
    )
    return max(1, int(k))


def agns_sc(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    mu: float,
    L_2: float | None = None,
    Delta_0: float | None = None,
    K_seg: int | None = None,
    n_segments: int = 50,
    eps: float = 1e-12,
    f_star: float | None = None,
    rho: float = RHO_PREFACTOR_OPTIMAL,
    theta: float = 0.5,
    n_iters: int | None = None,
    **agns_theory_kwargs: Any,
) -> MethodResult:
    """Restarted AGNS-Theory for strongly convex objectives.

    Parameters
    ----------
    mu : float
        Strong-convexity constant.  Must be strictly positive.
    L_2 : float, optional
        Upper bound on the Lipschitz Hessian.  Used to pick ``K_seg``
        when ``K_seg`` is not supplied.
    Delta_0 : float, optional
        Upper bound on the initial residual ``f(x_0) - f^star``.  When
        omitted and ``f_star`` is given, this is computed as
        ``oracle.func(x_0) - f_star``.
    K_seg : int, optional
        Segment length.  Defaults to :func:`_default_k_seg`.
    n_segments : int
        Maximum number of restart segments.
    eps : float
        Function-value tolerance for global termination.
    f_star : float, optional
        Known optimum; if supplied the loop exits when the running
        residual drops below ``eps``.
    rho, theta : float
        AGNS-Theory hyperparameters (forwarded).
    **agns_theory_kwargs :
        Other AGNS-Theory kwargs.

    Returns
    -------
    (x_final, status, history) : tuple
        ``history`` is a concatenated trace across all segments with
        ``segment_boundaries`` (integer offsets) and ``segment_residuals``
        (``f(z_s) - f^star`` per segment).
    """
    if mu <= 0.0:
        raise ValueError(f"mu must be strictly positive, got {mu}")
    # ``n_iters`` is accepted for API parity with every other method
    # (the registry threads it through generically) but is ignored: the
    # restart loop uses ``n_segments`` and ``K_seg`` instead.
    _ = n_iters
    if K_seg is None:
        if L_2 is None:
            raise ValueError("Either K_seg or L_2 must be supplied")
        if Delta_0 is None:
            if f_star is None:
                raise ValueError("K_seg requires Delta_0 (or f_star to infer it from x_0)")
            Delta_0 = max(0.0, float(oracle.func(x_0)) - float(f_star))
        K_seg = _default_k_seg(L_2, mu, Delta_0, rho=rho, theta=theta)

    agg_history: defaultdict[str, list[Any]] = defaultdict(list)
    agg_history["segment_boundaries"].append(0)
    z = np.copy(x_0)
    status = ""

    for s in range(n_segments):
        z_next, seg_status, seg_history = agns_theory(
            oracle,
            z,
            n_iters=K_seg,
            f_star=f_star,
            eps=eps,
            rho=rho,
            theta=theta,
            **agns_theory_kwargs,
        )

        if seg_history is not None:
            for k, v in seg_history.items():
                agg_history[k].extend(v)
            agg_history["segment_boundaries"].append(len(agg_history["func"]))
            seg_func = seg_history.get("func", [])
            if seg_func:
                f_z = float(seg_func[-1])
                agg_history["segment_residuals"].append(
                    f_z - f_star if f_star is not None else float("nan")
                )

        z = np.copy(z_next)
        status = f"sc_segment_{s}: {seg_status}"

        if f_star is not None:
            f_z = (
                float(seg_history["func"][-1])
                if seg_history and seg_history.get("func")
                else float(oracle.func(z))
            )
            if f_z - f_star < eps:
                status = success_status(s + 1).replace("iters", "segments")
                break

    return z, status, agg_history
