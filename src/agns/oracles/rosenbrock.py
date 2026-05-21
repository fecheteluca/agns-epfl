"""Rosenbrock oracles and factories.

Two flavours:

- :class:`RosenbrockOracle` — the canonical 2-D Rosenbrock
  ``f(x, y) = (a - x)^2 + b(y - x^2)^2`` exposed as a single-objective.
- :class:`NonlinearEquationsRosenbrockOracle` — Rosenbrock as a 2-residual
  nonlinear-equations problem ``f(x) = (1/p) ||u(x)||^p`` with
  ``u(x) = (a - x_1, sqrt(b) (x_2 - x_1^2))``.

The NLE variant is the canonical small non-convex benchmark exposed by
this package.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from agns.approximations import approx_hess_nonlinear_equations
from agns.oracles.base import BaseSmoothOracle


class RosenbrockOracle(BaseSmoothOracle):
    """Canonical 2-D Rosenbrock ``f(x, y) = (a - x)^2 + b(y - x^2)^2``."""

    def __init__(self, a: float = 1.0, b: float = 100.0) -> None:
        self.a = a
        self.b = b

    def func(self, x: NDArray[np.float64]) -> float:
        xx = x[0]
        yy = x[1]
        return float((self.a - xx) ** 2 + self.b * (yy - xx**2) ** 2)

    def grad(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        xx = x[0]
        yy = x[1]
        dfdx = 2.0 * (xx - self.a) - 4.0 * self.b * xx * (yy - xx**2)
        dfdy = 2.0 * self.b * (yy - xx**2)
        return np.array([dfdx, dfdy], dtype=float)

    def hess(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        xx = x[0]
        yy = x[1]
        d2fdx2 = 2.0 - 4.0 * self.b * (yy - 3.0 * xx**2)
        d2fdxdy = -4.0 * self.b * xx
        d2fdy2 = 2.0 * self.b
        return np.array([[d2fdx2, d2fdxdy], [d2fdxdy, d2fdy2]], dtype=float)


class NonlinearEquationsRosenbrockOracle(BaseSmoothOracle):
    """Rosenbrock as a nonlinear-equations objective ``(1/p) ||u(x)||^p``."""

    def __init__(self, p: int, a: float = 1.0, b: float = 100.0) -> None:
        assert p >= 2, "p must be >= 2"
        self.p = p
        self.a = a
        self.b = b

    def _compute_u_and_jac(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        xx = x[0]
        yy = x[1]
        u1 = self.a - xx
        u2 = np.sqrt(self.b) * (yy - xx**2)
        J = np.array([[-1.0, 0.0], [-2.0 * np.sqrt(self.b) * xx, np.sqrt(self.b)]], dtype=float)
        return np.array([u1, u2], dtype=float), J

    def func_u(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        u, _ = self._compute_u_and_jac(x)
        return u

    def jac_u(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        _, J = self._compute_u_and_jac(x)
        return J

    def func(self, x: NDArray[np.float64]) -> float:
        u, _ = self._compute_u_and_jac(x)
        norm_u = np.linalg.norm(u)
        return float((1.0 / self.p) * (norm_u**self.p))

    def grad(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        u, J = self._compute_u_and_jac(x)
        norm_u = np.linalg.norm(u)
        if norm_u == 0:
            return np.zeros(2, dtype=float)
        factor = norm_u ** (self.p - 2)
        return factor * (J.T.dot(u))

    def hess(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        u, J = self._compute_u_and_jac(x)
        norm_u = np.linalg.norm(u)
        p = self.p

        if norm_u == 0:
            return (norm_u ** (p - 2)) * (J.T.dot(J))

        H = (norm_u ** (p - 2)) * (J.T.dot(J))

        if p != 2:
            Ju = J.T.dot(u)
            outer_term = np.outer(Ju, Ju)
            H += (p - 2) * (norm_u ** (p - 4)) * outer_term

        sqrt_b = np.sqrt(self.b)
        u2 = u[1]
        Hess_u2 = np.array([[-2.0 * sqrt_b, 0.0], [0.0, 0.0]], dtype=float)
        H += u2 * (norm_u ** (p - 2)) * Hess_u2

        return cast("NDArray[np.float64]", H)


# ---------------------------------------------------------------------------
# Problem factories
# ---------------------------------------------------------------------------


def make_rosenbrock_nle(p: int = 5) -> dict[str, Any]:
    """Rosenbrock as a 2-residual nonlinear-equations problem.  Non-convex."""
    oracle = NonlinearEquationsRosenbrockOracle(p=p, a=1.0, b=100.0)
    x_0 = np.array([-2.0, 2.0])
    f_star = oracle.func(np.array([1.0, 1.0]))

    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": f_star,
        "B": None,
        "Binv": None,
        "approx_hess_fn": approx_hess_nonlinear_equations,
        "approx_hess_fn_wsm": None,
        "meta": {"type": "rosenbrock_nle", "p": p},
    }


def make_rosenbrock_2d() -> dict[str, Any]:
    """Canonical 2-D Rosenbrock with ``a = 1, b = 100``.  Non-convex."""
    oracle = RosenbrockOracle(a=1.0, b=100.0)
    x_0 = np.array([-2.0, 2.0])
    f_star = oracle.func(np.array([1.0, 1.0]))

    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": f_star,
        "B": None,
        "Binv": None,
        "approx_hess_fn": None,
        "approx_hess_fn_wsm": None,
        "meta": {"type": "rosenbrock_2d"},
    }


__all__ = [
    "NonlinearEquationsRosenbrockOracle",
    "RosenbrockOracle",
    "make_rosenbrock_2d",
    "make_rosenbrock_nle",
]
