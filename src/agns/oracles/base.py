"""Smooth-oracle abstract base class and the call-counter wrapper.

Every concrete oracle in :mod:`agns.oracles` subclasses
:class:`BaseSmoothOracle` and overrides whichever methods are relevant
to its order of smoothness.

The wrapper :class:`OracleCallsCounter` exposes the same interface but
increments per-method counters on every call.  Optimisation methods wrap
their oracle in this counter so per-iteration histories can report
``func_calls`` / ``grad_calls`` / ``hess_calls`` / ``hess_vec_calls``.
The counter is transparent --- it forwards every call without altering
the numerical result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["BaseSmoothOracle", "OracleCallsCounter"]


class BaseSmoothOracle:
    """Abstract base class for smooth-function oracles.

    Subclasses override the subset of methods their problem class
    requires.  Methods that have not been overridden raise
    :class:`NotImplementedError`.
    """

    def func(self, x: NDArray[np.float64]) -> float:
        """Function value at ``x``."""
        raise NotImplementedError("func oracle is not implemented.")

    def grad(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Gradient at ``x``."""
        raise NotImplementedError("grad oracle is not implemented.")

    def hess(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Hessian at ``x``."""
        raise NotImplementedError("hess oracle is not implemented.")

    def hess_vec(self, x: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Hessian-vector product ``H(x) v``.

        Default implementation forms the full Hessian.  Override for
        problems where matrix-free ``H(x) v`` is asymptotically cheaper.
        """
        return self.hess(x).dot(v)  # type: ignore[no-any-return]

    def third_vec_vec(self, x: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Tensor-vector-vector contraction ``D^3 f(x)[v, v]``."""
        raise NotImplementedError("third_vec_vec oracle is not implemented.")


class OracleCallsCounter(BaseSmoothOracle):
    """Transparent wrapper that counts oracle calls.

    Forwards every call to the wrapped oracle and increments a per-method
    counter.  Optimisation methods read the counts back as
    ``oracle.func_calls`` / ``grad_calls`` / etc. and store them in the
    iteration history.

    Attribute access other than the counted methods falls through to the
    wrapped oracle via :meth:`__getattr__`, so problem-specific helpers
    (``func_u``, ``jac_u``, ``p``, ...) survive the wrapping.
    """

    def __init__(self, oracle: BaseSmoothOracle) -> None:
        self.oracle = oracle
        self.func_calls: int = 0
        self.grad_calls: int = 0
        self.hess_calls: int = 0
        self.hess_vec_calls: int = 0
        self.third_vec_vec_calls: int = 0

    def func(self, x: NDArray[np.float64]) -> float:
        self.func_calls += 1
        return self.oracle.func(x)

    def grad(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        self.grad_calls += 1
        return self.oracle.grad(x)

    def hess(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        self.hess_calls += 1
        return self.oracle.hess(x)

    def hess_vec(self, x: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        self.hess_vec_calls += 1
        return self.oracle.hess_vec(x, v)

    def third_vec_vec(self, x: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        self.third_vec_vec_calls += 1
        return self.oracle.third_vec_vec(x, v)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.oracle, name)
