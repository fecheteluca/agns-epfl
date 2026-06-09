r"""Separable oracle that isolates curvature *variation* from conditioning.

This family exists for the RQ1 mechanism experiment. The argument is: Nesterov momentum's
classical benefit is the :math:`\kappa \to \sqrt{\kappa}` conditioning improvement, but a
Newton step already neutralizes conditioning through the Hessian, so the only channel left
for momentum to exploit is how much the curvature *varies* along the trajectory. To test
that, we need a knob that changes curvature variation while holding the conditioning at the
optimum fixed.

The objective is separable and convex:

.. math::

    f(x) = \sum_i \tfrac{c_i}{2} x_i^2 + \kappa \,(\cosh x_i - 1 - \tfrac{x_i^2}{2}),

with optimum :math:`x^\star = 0`, :math:`f^\star = 0`. The Hessian is diagonal,

.. math::  \nabla^2 f(x) = \mathrm{diag}\big(c_i + \kappa(\cosh x_i - 1)\big),

so at the optimum it is exactly :math:`\mathrm{diag}(c_i)` for **every** :math:`\kappa`
(because :math:`\cosh 0 - 1 = 0`). Thus the spread of ``c_i`` fixes the conditioning at the
solution, while :math:`\kappa` alone controls the third-derivative magnitude
(:math:`f''' = \kappa \sinh x_i`) and hence the curvature variation away from it. Sweeping
:math:`\kappa` at fixed ``c_i`` isolates the mechanism.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from agns.oracles.base import BaseSmoothOracle

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


class SeparableCurvatureOracle(BaseSmoothOracle):
    r"""Separable convex oracle with fixed optimum-conditioning and tunable curvature.

    Parameters
    ----------
    c:
        Per-coordinate quadratic coefficients (positive). Their max/min ratio is the
        condition number of the Hessian at the optimum, held fixed across a ``kappa`` sweep.
    kappa:
        Non-negative curvature-variation strength. ``kappa = 0`` gives a pure quadratic
        (Newton is exact in one step); larger ``kappa`` makes the curvature vary more along
        the path.
    """

    def __init__(self, c: Vector, kappa: float) -> None:
        self.c = np.asarray(c, dtype=float)
        if np.any(self.c <= 0.0):
            raise ValueError("c must be positive")
        if kappa < 0.0:
            raise ValueError("kappa must be non-negative")
        self.kappa = float(kappa)
        self.n = self.c.shape[0]

    def func(self, x: Vector) -> float:
        x = np.asarray(x, dtype=float)
        quad = 0.5 * np.sum(self.c * x**2)
        curv = self.kappa * np.sum(np.cosh(x) - 1.0 - 0.5 * x**2)
        return float(quad + curv)

    def grad(self, x: Vector) -> Vector:
        x = np.asarray(x, dtype=float)
        return self.c * x + self.kappa * (np.sinh(x) - x)

    def hess(self, x: Vector) -> Matrix:
        x = np.asarray(x, dtype=float)
        return np.diag(self.c + self.kappa * (np.cosh(x) - 1.0))
