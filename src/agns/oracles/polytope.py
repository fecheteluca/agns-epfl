from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from agns.oracles.base import BaseSmoothOracle

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


class PolytopeFeasibility(BaseSmoothOracle):
    r"""Oracle for the polytope-feasibility penalty.

    The function is::

        func(x) = sum_{i = 1}^m (<a_i, x> - b_i)_+^p,

    where ``(t)_+ := max(0, t)``, ``a_1, ..., a_m`` are the rows of the
    ``(m x n)`` matrix ``A``, ``b`` is a given ``(m x 1)`` vector, and
    ``p >= 2`` is a parameter.
    """

    def __init__(self, A: Matrix, b: Vector, p: float) -> None:
        """Initialize the polytope-feasibility oracle.

        Parameters
        ----------
        A : Matrix
            Constraint matrix of shape ``(m, n)``.
        b : Vector
            Constraint offsets of shape ``(m,)``.
        p : float
            Penalty power; ``p >= 2``.
        """
        self.A = A
        self.b = b
        self.p = p
        self.last_x: Vector | None = None

    def func(self, x: Vector) -> float:
        """Return the function value at ``x``.

        Parameters
        ----------
        x : Vector
            Evaluation point of shape ``(n,)``.

        Returns
        -------
        float
            The penalty value.
        """
        self._update_a(x)
        return np.sum(self.a**self.p)

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
        self._update_a(x)
        return self.p * self.A.T.dot(self.a ** (self.p - 1))

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
        self._update_a(x)
        if self.p == 2:
            u = np.zeros_like(self.a)
            u[self.a > 0] = 1.0
            return 2 * self.A.T.dot(self.A * u.reshape(-1, 1))
        else:
            return (
                self.p
                * (self.p - 1)
                * self.A.T.dot(self.A * (self.a ** (self.p - 2)).reshape(-1, 1))
            )

    def _update_a(self, x: Vector) -> None:
        """Refresh the cached clipped residual ``a`` for point ``x``.

        Parameters
        ----------
        x : Vector
            Evaluation point of shape ``(n,)``.
        """
        if not np.array_equal(self.last_x, x):
            self.last_x = np.copy(x)
            self.a = self.A.dot(x) - self.b
            self.a = np.maximum(self.a, np.zeros_like(self.a))


class PDifferenceOracle(BaseSmoothOracle):
    r"""Oracle for the p-difference function.

    The function is::

        func(x) = 1 / (p + 1) * (|x_1|^{p + 1}
                                 + sum_{i = 1}^n |x_i - a * x_{i - 1}|^{p + 1})
                  - x_n.
    """

    def __init__(self, p: float, n: int, a: float = 1) -> None:
        """Initialize the p-difference oracle.

        Parameters
        ----------
        p : float
            Power parameter.
        n : int
            Problem dimension.
        a : float, optional
            Coupling coefficient between successive coordinates.
        """
        self.p = p
        self.n = n
        self.a = a

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
        return (
            1.0
            / (self.p + 1)
            * (
                np.abs(x[0]) ** (self.p + 1)
                + np.sum(np.abs(x[1:] - self.a * x[0:-1]) ** (self.p + 1))
            )
        )  # - \
        # x[self.n - 1]

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
        g = np.zeros_like(x)
        h = x[1:] - self.a * x[0:-1]
        g[1:] += (self.p + 1) * np.abs(h) ** (self.p - 1) * h
        g[0:-1] -= self.a * (self.p + 1) * np.abs(h) ** (self.p - 1) * h
        g[0] += (self.p + 1) * np.abs(x[0]) ** (self.p - 1) * x[0]
        g *= 1.0 / (self.p + 1)
        return g

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
        H = np.zeros((x.shape[0], x.shape[0]))
        v = np.abs(x[1:] - self.a * x[0:-1]) ** (self.p - 1)
        diag = np.zeros_like(x)
        diag[1:] += (self.p + 1) * self.p * v
        diag[0:-1] += self.a**2 * (self.p + 1) * self.p * v
        diag[0] += (self.p + 1) * self.p * np.abs(x[0]) ** (self.p - 1)
        shift_diag = -self.a * (self.p + 1) * self.p * v
        np.fill_diagonal(H, diag)
        np.fill_diagonal(H[1:], shift_diag)
        np.fill_diagonal(H[:, 1:], shift_diag)
        return 1.0 / (self.p + 1) * H
