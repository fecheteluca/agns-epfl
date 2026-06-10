from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


class BaseSmoothOracle:
    """Base class for smooth-function oracles."""

    def func(self, x: Vector) -> float:
        """Return the function value at ``x``."""
        raise NotImplementedError("Func oracle is not implemented.")

    def grad(self, x: Vector) -> Vector:
        """Return the gradient at ``x``."""
        raise NotImplementedError("Grad oracle is not implemented.")

    def hess(self, x: Vector) -> Matrix:
        """Return the Hessian matrix at ``x``."""
        raise NotImplementedError("Hessian oracle is not implemented.")

    def hess_vec(self, x: Vector, v: Vector) -> Vector:
        """Return the Hessian-vector product ``f''(x) v``."""
        return self.hess(x).dot(v)

    def third_vec_vec(self, x: Vector, v: Vector) -> Vector:
        """Return the third-derivative tensor product ``D^3 f(x)[v, v]``."""
        raise NotImplementedError("Third derivative oracle is not implemented.")


class OracleCallsCounter(BaseSmoothOracle):
    """Wrapper that counts oracle calls of the wrapped oracle."""

    def __init__(self, oracle: Any) -> None:
        self.oracle = oracle
        self.func_calls = 0
        self.grad_calls = 0
        self.hess_calls = 0
        self.hess_vec_calls = 0
        self.third_vec_vec_calls = 0

    def func(self, x: Vector) -> float:
        self.func_calls += 1
        return self.oracle.func(x)

    def grad(self, x: Vector) -> Vector:
        self.grad_calls += 1
        return self.oracle.grad(x)

    def hess(self, x: Vector) -> Matrix:
        self.hess_calls += 1
        return self.oracle.hess(x)

    def hess_vec(self, x: Vector, v: Vector) -> Vector:
        self.hess_vec_calls += 1
        return self.oracle.hess_vec(x, v)

    def third_vec_vec(self, x: Vector, v: Vector) -> Vector:
        self.third_vec_vec_calls += 1
        return self.oracle.third_vec_vec(x, v)
