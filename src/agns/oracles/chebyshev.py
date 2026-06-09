r"""Chebyshev least-squares stress-test oracle.

Ported verbatim from epfml/grad-norm-smooth/src/oracles.py (Apache-2.0, commit 9f4ca00).
Adds type hints and NumPy-style docstrings; the numerics are unchanged.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from agns.oracles.base import BaseSmoothOracle

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


class ChebyshevOracle(BaseSmoothOracle):
    r"""Oracle for the Chebyshev-polynomial least-squares objective.

    The residuals are::

        u_1(x) = 1 / 2 * (1 - x_1)
        u_i(x) = x_i - (2 x_{i-1}^2 - 1) for i = 2,...,n

    and ``f(x) = 1 / p * \| u(x)\|^p``. The unique global minimizer is
    ``x = (1, ..., 1)``.
    """

    def __init__(self, n: int, p: float) -> None:
        """Initialize the Chebyshev oracle.

        Parameters
        ----------
        n : int
            Problem dimension.
        p : float
            Power of the residual norm; must satisfy ``p >= 2``.
        """
        assert p >= 2, "p must be >= 2."
        self.n = n
        self.p = p

        def _func_u(x: Vector) -> Vector:
            x = np.asarray(x).ravel()
            assert x.shape[0] == self.n
            u = np.empty(self.n, dtype=float)
            u[0] = 0.5 * (1.0 - x[0])
            for i in range(1, self.n):
                u[i] = x[i] - (2.0 * x[i - 1] ** 2 - 1.0)
            return u

        def _jac_u(x: Vector) -> Matrix:
            x = np.asarray(x).ravel()
            assert x.shape[0] == self.n

            J = np.zeros((self.n, self.n), dtype=float)
            J[0, 0] = -0.5
            for i in range(1, self.n):
                J[i, i] = 1.0
                J[i, i - 1] = -4.0 * x[i - 1]
            return J

        H_list = []
        for i in range(self.n):
            H_i = np.zeros((self.n, self.n), dtype=float)
            if i >= 1:
                H_i[i - 1, i - 1] = -4.0
            H_list.append(H_i)

        self.func_u = _func_u
        self.jac_u = _jac_u
        self.hess_u_list = H_list

    def func(self, x: Vector) -> float:
        """Return the function value at ``x``.

        Parameters
        ----------
        x : Vector
            Evaluation point of shape ``(n,)``.

        Returns
        -------
        float
            The function value.
        """
        u = self.func_u(x)
        norm_u = np.linalg.norm(u)
        return (1.0 / self.p) * (norm_u**self.p)

    def grad(self, x: Vector) -> Vector:
        """Return the gradient at ``x``.

        Parameters
        ----------
        x : Vector
            Evaluation point of shape ``(n,)``.

        Returns
        -------
        Vector
            The gradient.
        """
        u = self.func_u(x)
        norm_u = np.linalg.norm(u)
        J = self.jac_u(x)
        if norm_u == 0:
            return np.zeros(self.n, dtype=float)
        return (norm_u ** (self.p - 2)) * (J.T.dot(u))

    def hess(self, x: Vector) -> Matrix:
        """Return the Hessian matrix at ``x``.

        Parameters
        ----------
        x : Vector
            Evaluation point of shape ``(n,)``.

        Returns
        -------
        Matrix
            The Hessian matrix.
        """
        u = self.func_u(x)
        norm_u = np.linalg.norm(u)
        J = self.jac_u(x)
        if norm_u == 0:
            if self.p == 2:
                return J.T.dot(J)
            else:
                return np.zeros((self.n, self.n), dtype=float)

        H = (norm_u ** (self.p - 2)) * (J.T.dot(J))

        if self.p != 2:
            g_vec = J.T.dot(u)
            H += (self.p - 2) * (norm_u ** (self.p - 4)) * np.outer(g_vec, g_vec)

        coeff = norm_u ** (self.p - 2)
        for i in range(self.n):
            if u[i] != 0.0:
                H += coeff * u[i] * self.hess_u_list[i]

        return H

    def hess_vec(self, x: Vector, v: Vector) -> Vector:
        """Return the Hessian-vector product ``f''(x) v``.

        Parameters
        ----------
        x : Vector
            Evaluation point of shape ``(n,)``.
        v : Vector
            Vector of shape ``(n,)``.

        Returns
        -------
        Vector
            The Hessian-vector product.
        """
        H = self.hess(x)
        return H.dot(v)

    def third_vec_vec(self, x: Vector, v: Vector) -> Vector:
        """Raise ``NotImplementedError``; not supported for this oracle.

        Parameters
        ----------
        x : Vector
            Evaluation point.
        v : Vector
            Direction vector.

        Raises
        ------
        NotImplementedError
            Always, since the third-derivative product is not implemented.
        """
        raise NotImplementedError(
            "third_vec_vec is not implemented for ChebyshevOracle."
        )
