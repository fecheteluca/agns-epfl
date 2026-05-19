"""Finite-difference helpers for oracle correctness tests.

Computes central-difference approximations to the gradient and the
Hessian-vector product so tests can check ``oracle.grad`` / ``hess`` /
``hess_vec`` against the function value.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from agns.oracles.base import BaseSmoothOracle


def grad_fd(
    oracle: BaseSmoothOracle, x: NDArray[np.float64], h: float = 1e-6
) -> NDArray[np.float64]:
    """Central-difference approximation of the gradient at ``x``."""
    n = x.shape[0]
    g = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        g[i] = (oracle.func(x_plus) - oracle.func(x_minus)) / (2.0 * h)
    return g


def hess_vec_fd(
    oracle: BaseSmoothOracle,
    x: NDArray[np.float64],
    v: NDArray[np.float64],
    h: float = 1e-5,
) -> NDArray[np.float64]:
    """Central-difference approximation of ``H(x) v``."""
    return (oracle.grad(x + h * v) - oracle.grad(x - h * v)) / (2.0 * h)
