"""AGNS-Theory under bounded inexact Hessian.

When ``delta_budget > 0`` this wrapper wraps the oracle in
:class:`agns.oracles.inexact_wrapper.InexactHessianOracle`, which adds a
deterministic symmetric perturbation of spectral norm exactly
``delta_budget`` to every ``hess(x)`` call.  When ``delta_budget == 0``
the wrapper is a no-op and the behaviour is identical to
:func:`agns_theory`.

Registry key: ``agns_inexact``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods.agns_theory import agns_theory
from agns.methods.base import MethodResult
from agns.oracles.inexact_wrapper import InexactHessianOracle

__all__ = ["agns_inexact"]


def agns_inexact(
    oracle: Any,
    x_0: NDArray[np.float64],
    *,
    delta_budget: float = 0.0,
    noise_seed: int = 0,
    **agns_theory_kwargs: Any,
) -> MethodResult:
    """AGNS-Theory with a bounded Hessian perturbation.

    Parameters
    ----------
    delta_budget : non-negative float
        Worst-case operator-norm bound on the Hessian approximation
        error.  ``0.0`` is a no-op and recovers :func:`agns_theory`.
    noise_seed : int
        Seed for the perturbation RNG inside
        :class:`InexactHessianOracle`.  Two runs with the same seed
        produce identical perturbation sequences.
    **agns_theory_kwargs :
        Forwarded verbatim to :func:`agns_theory`.
    """
    if delta_budget < 0.0:
        raise ValueError(f"delta_budget must be non-negative, got {delta_budget}")

    run_oracle: Any
    if delta_budget > 0.0:
        run_oracle = InexactHessianOracle(oracle, delta_budget, noise_seed=noise_seed)
    else:
        run_oracle = oracle

    x_final, status, history = agns_theory(run_oracle, x_0, **agns_theory_kwargs)

    if history is not None:
        history["delta_budget"].append(float(delta_budget))

    return x_final, status, history
