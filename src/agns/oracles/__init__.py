"""Problem oracles and factories.

Re-exports the smooth-oracle base class and the call-counter wrapper.
Concrete problem oracles (logistic regression, least squares) and their
``make_*`` factories are added as they are implemented; the mapping from
the ``problem.type`` string used in YAML configs to a factory callable
lives in :data:`PROBLEM_REGISTRY`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agns.oracles.base import BaseSmoothOracle, OracleCallsCounter

#: Problem-type string -> factory callable.  Populated as oracles land.
PROBLEM_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {}

__all__ = [
    "PROBLEM_REGISTRY",
    "BaseSmoothOracle",
    "OracleCallsCounter",
]
