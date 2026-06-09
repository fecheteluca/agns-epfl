r"""Nonlinear-equations stress-test oracle.

Ported verbatim from epfml/grad-norm-smooth/src/oracles.py (Apache-2.0, commit 9f4ca00).
Adds type hints and NumPy-style docstrings; the numerics are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from agns.oracles.base import BaseSmoothOracle

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


class NonlinearEquationsOracle(BaseSmoothOracle):
    r"""Oracle for ``f(x) = 1/p * ||u(x)||^p``, ``p >= 2``."""

    def __init__(
        self,
        p: float,
        func_u: Callable[[Vector], Vector] | None = None,
        jac_u: Callable[[Vector], Matrix] | None = None,
        hess_u_list: list[Matrix] | None = None,
        A: Matrix | None = None,
        b: Vector | None = None,
    ) -> None:
        """Initialize the nonlinear-equations oracle.

        Parameters
        ----------
        p : float
            Power of the residual norm; must satisfy ``p >= 2``.
        func_u : callable, optional
            Residual function ``u(x)``.
        jac_u : callable, optional
            Jacobian function ``J(x)`` of the residuals.
        hess_u_list : list of Matrix, optional
            Per-residual Hessian matrices.
        A : Matrix, optional
            Matrix for the affine residual ``u(x) = A x - b``.
        b : Vector, optional
            Offset for the affine residual ``u(x) = A x - b``.
        """
        assert p >= 2, "p must be >= 2"
        self.p = p
        self.func_u = func_u
        self.jac_u = jac_u
        self.hess_u_list = hess_u_list or []
        self.A = A
        self.b = b
        if A is not None and b is not None:

            def func_u(x: Vector) -> Vector:
                return A.dot(x) - b

            def jac_u(x: Vector) -> Matrix:
                return A

            self.func_u = func_u
            self.jac_u = jac_u
            self.hess_u_list = []

    def func(self, x: Vector) -> float:
        """Return the function value at ``x``.

        Parameters
        ----------
        x : Vector
            Evaluation point.

        Returns
        -------
        float
            The function value.
        """
        u = self.func_u(x)
        norm_u = np.linalg.norm(u)
        return (1.0 / self.p) * norm_u**self.p

    def grad(self, x: Vector) -> Vector:
        """Return the gradient at ``x``.

        Parameters
        ----------
        x : Vector
            Evaluation point.

        Returns
        -------
        Vector
            The gradient.
        """
        u = self.func_u(x)
        norm_u = np.linalg.norm(u)
        J = self.jac_u(x)
        return (norm_u ** (self.p - 2)) * (J.T.dot(u))

    def hess(self, x: Vector) -> Matrix:
        """Return the Hessian matrix at ``x``.

        Parameters
        ----------
        x : Vector
            Evaluation point.

        Returns
        -------
        Matrix
            The Hessian matrix.
        """
        u = self.func_u(x)
        norm_u = np.linalg.norm(u)
        J = self.jac_u(x)
        H = (norm_u ** (self.p - 2)) * (J.T.dot(J))
        if self.p != 2:
            g = J.T.dot(u)
            H += (self.p - 2) * (norm_u ** (self.p - 4)) * np.outer(g, g)
        if self.hess_u_list:
            coeff = norm_u ** (self.p - 2)
            for i, H_ui in enumerate(self.hess_u_list):
                H += (self.p / norm_u**2) * u[i] * coeff * H_ui
        return H
