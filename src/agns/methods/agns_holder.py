"""AGNS-Holder: AGNS-Theory annotated with the Hessian Holder exponent ``nu``.

The rho-strict adaptive search of :func:`agns_theory` is sufficient for
universal Holder smoothness; ``nu`` enters only the rate analysis and is
recorded in the trace so the figure pipeline can plot the empirical
slope alongside the theoretical exponent.

Registry key: ``agns_holder``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods.agns_theory import agns_theory
from agns.methods.base import MethodResult

__all__ = ["agns_holder"]


def agns_holder(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    nu: float = 1.0,
    L_holder: float | None = None,
    **agns_theory_kwargs: Any,
) -> MethodResult:
    """AGNS-Theory + per-run Holder metadata.

    Parameters
    ----------
    nu : float in [0, 1]
        Hessian Holder exponent.  ``nu = 1`` recovers Lipschitz-Hessian;
        ``nu = 0`` recovers bounded-Hessian.  The algorithm itself does
        not depend on ``nu``; the value is appended to
        ``history['nu']``.
    L_holder : float or None
        Best known upper bound on the Holder constant ``L_nu``.  Stored
        in ``history['L_holder']`` when provided.
    **agns_theory_kwargs :
        Forwarded verbatim to :func:`agns_theory`.
    """
    if not (0.0 <= nu <= 1.0):
        raise ValueError(f"nu must lie in [0, 1], got {nu}")

    x_final, status, history = agns_theory(oracle, x_0, **agns_theory_kwargs)

    if history is not None:
        history["nu"].append(float(nu))
        if L_holder is not None:
            history["L_holder"].append(float(L_holder))

    return x_final, status, history
