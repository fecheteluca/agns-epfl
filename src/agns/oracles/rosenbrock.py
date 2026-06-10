from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from agns.oracles.base import BaseSmoothOracle

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


class RosenbrockOracle(BaseSmoothOracle):
    r"""Oracle for ``f(x, y) = (a - x)^2 + b (y - x^2)^2``."""

    def __init__(self, a: float = 1.0, b: float = 100.0) -> None:
        """Initialize the Rosenbrock oracle.

        Parameters
        ----------
        a : float, optional
            First Rosenbrock parameter.
        b : float, optional
            Second Rosenbrock parameter.
        """
        self.a = a
        self.b = b

    def func(self, x: Vector) -> float:
        """Return the function value at ``x``.

        Parameters
        ----------
        x : Vector
            Point of shape ``(2,)``.

        Returns
        -------
        float
            The Rosenbrock function value.
        """
        xx = x[0]
        yy = x[1]
        a = self.a
        b = self.b
        return (a - xx) ** 2 + b * (yy - xx**2) ** 2

    def grad(self, x: Vector) -> Vector:
        """Return the gradient at ``x``.

        Parameters
        ----------
        x : Vector
            Point of shape ``(2,)``.

        Returns
        -------
        Vector
            Gradient of shape ``(2,)``.
        """
        xx = x[0]
        yy = x[1]
        a = self.a
        b = self.b
        dfdx = 2.0 * (xx - a) - 4.0 * b * xx * (yy - xx**2)
        dfdy = 2.0 * b * (yy - xx**2)
        return np.array([dfdx, dfdy], dtype=float)

    def hess(self, x: Vector) -> Matrix:
        """Return the Hessian matrix at ``x``.

        Parameters
        ----------
        x : Vector
            Point of shape ``(2,)``.

        Returns
        -------
        Matrix
            Hessian of shape ``(2, 2)``.
        """
        xx = x[0]
        yy = x[1]
        b = self.b
        d2fdx2 = 2.0 - 4.0 * b * (yy - 3.0 * xx**2)
        d2fdxdy = -4.0 * b * xx
        d2fdy2 = 2.0 * b
        H = np.array([[d2fdx2, d2fdxdy], [d2fdxdy, d2fdy2]], dtype=float)
        return H


class NonlinearEquationsRosenbrockOracle(BaseSmoothOracle):
    r"""Oracle for ``f(x) = 1/p * \| u(x) \|^p``, ``p >= 2``.

    Here ``u(x)`` is the 2-vector of Rosenbrock residuals.
    """

    def __init__(self, p: float, a: float = 1.0, b: float = 100.0) -> None:
        """Initialize the nonlinear-equations Rosenbrock oracle.

        Parameters
        ----------
        p : float
            Power of the residual norm; must satisfy ``p >= 2``.
        a : float, optional
            First Rosenbrock parameter.
        b : float, optional
            Second Rosenbrock parameter.
        """
        assert p >= 2, "p must be >= 2"
        self.p = p
        self.a = a
        self.b = b

    def _compute_u_and_jac(self, x: Vector) -> tuple[Vector, Matrix]:
        """Return the residual vector and its Jacobian at ``x``.

        Parameters
        ----------
        x : Vector
            Point of shape ``(2,)``.

        Returns
        -------
        tuple of (Vector, Matrix)
            Residuals ``u(x)`` and Jacobian ``J(x)``.
        """
        xx = x[0]
        yy = x[1]
        a = self.a
        b = self.b
        u1 = a - xx
        u2 = np.sqrt(b) * (yy - xx**2)
        J = np.array([[-1.0, 0.0], [-2.0 * np.sqrt(b) * xx, np.sqrt(b)]], dtype=float)

        return np.array([u1, u2], dtype=float), J

    def func(self, x: Vector) -> float:
        """Return the function value at ``x``.

        Parameters
        ----------
        x : Vector
            Point of shape ``(2,)``.

        Returns
        -------
        float
            The function value.
        """
        u, _ = self._compute_u_and_jac(x)
        norm_u = np.linalg.norm(u)
        return (1.0 / self.p) * (norm_u**self.p)

    def grad(self, x: Vector) -> Vector:
        """Return the gradient at ``x``.

        Parameters
        ----------
        x : Vector
            Point of shape ``(2,)``.

        Returns
        -------
        Vector
            Gradient of shape ``(2,)``.
        """
        u, J = self._compute_u_and_jac(x)
        norm_u = np.linalg.norm(u)
        if norm_u == 0:
            return np.zeros(2, dtype=float)

        factor = norm_u ** (self.p - 2)
        return factor * (J.T.dot(u))

    def hess(self, x: Vector) -> Matrix:
        """Return the Hessian matrix at ``x``.

        Parameters
        ----------
        x : Vector
            Point of shape ``(2,)``.

        Returns
        -------
        Matrix
            Hessian of shape ``(2, 2)``.
        """
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

        return H
