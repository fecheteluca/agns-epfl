import math

import numpy as np
import scipy
from scipy.special import logsumexp, softmax


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of the function at point x.
        """
        raise NotImplementedError("Func oracle is not implemented.")

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError("Grad oracle is not implemented.")

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError("Hessian oracle is not implemented.")

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)

    def third_vec_vec(self, x, v):
        """
        Computes tensor-vector-vector product with the third derivative tensor
        D^3 f(x)[v, v].
        """
        raise NotImplementedError("Third derivative oracle is not implemented.")


class OracleCallsCounter(BaseSmoothOracle):
    """
    Wrapper to count oracle calls.
    """

    def __init__(self, oracle):
        self.oracle = oracle
        self.func_calls = 0
        self.grad_calls = 0
        self.hess_calls = 0
        self.hess_vec_calls = 0
        self.third_vec_vec_calls = 0

    def func(self, x):
        self.func_calls += 1
        return self.oracle.func(x)

    def grad(self, x):
        self.grad_calls += 1
        return self.oracle.grad(x)

    def hess(self, x):
        self.hess_calls += 1
        return self.oracle.hess(x)

    def hess_vec(self, x, v):
        self.hess_vec_calls += 1
        return self.oracle.hess_vec(x, v)

    def third_vec_vec(self, x, v):
        self.third_vec_vec_calls += 1
        return self.oracle.third_vec_vec(x, v)


class PolytopeFeasibility(BaseSmoothOracle):
    """
    Oracle for function:
        func(x) = sum_{i = 1}^m (<a_i, x> - b_i)_+^p,
        where (t)_+ := max(0, t).
        a_1, ..., a_m are rows of (m x n) matrix A.
        b is given (m x 1) vector.
        p >= 2 is a parameter.
    """

    def __init__(self, A, b, p):
        self.A = A
        self.b = b
        self.p = p
        self.last_x = None

    def func(self, x):
        self._update_a(x)
        return np.sum(self.a**self.p)

    def grad(self, x):
        self._update_a(x)
        return self.p * self.A.T.dot(self.a ** (self.p - 1))

    def hess(self, x):
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

    def _update_a(self, x):
        if not np.array_equal(self.last_x, x):
            self.last_x = np.copy(x)
            self.a = self.A.dot(x) - self.b
            self.a = np.maximum(self.a, np.zeros_like(self.a))


class LogSumExpOracle(BaseSmoothOracle):
    """
    Oracle for function:
        func(x) = mu log sum_{i=1}^m exp( (<a_i, x> - b_i) / mu )
        a_1, ..., a_m are rows of (m x n) matrix A.
        b is given (m x 1) vector.
        mu is a scalar value.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, mu):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = np.copy(b)
        self.mu = mu
        self.mu_inv = 1.0 / mu
        self.last_x = None
        self.last_x_pi = None
        self.last_v = None

    def func(self, x):
        self._update_a(x)
        return self.mu * logsumexp(self.a)

    def grad(self, x):
        self._update_a_and_pi(x)
        return self.AT_pi

    def hess(self, x):
        self._update_a_and_pi(x)
        return self.mu_inv * (
            self.matmat_ATsA(self.pi) - np.outer(self.AT_pi, self.AT_pi.T)
        )

    def hess_vec(self, x, v):
        self._update_hess_vec(x, v)
        return self._hess_vec

    def third_vec_vec(self, x, v):
        self._update_hess_vec(x, v)
        return self.mu_inv * (
            self.mu_inv
            * self.matvec_ATx(self.pi * (self.Av * (self.Av - self.AT_pi_v)))
            - self.AT_pi_v * self._hess_vec
            - self._hess_vec.dot(v) * self.AT_pi
        )

    def _update_a(self, x):
        if not np.array_equal(self.last_x, x):
            self.last_x = np.copy(x)
            self.a = self.mu_inv * (self.matvec_Ax(x) - self.b)

    def _update_a_and_pi(self, x):
        self._update_a(x)
        if not np.array_equal(self.last_x_pi, x):
            self.last_x_pi = np.copy(x)
            self.pi = softmax(self.a)
            self.AT_pi = self.matvec_ATx(self.pi)

    def _update_hess_vec(self, x, v):
        if not np.array_equal(self.last_x, x) or not np.array_equal(self.last_v, v):
            self._update_a_and_pi(x)
            self.last_v = np.copy(v)
            self.Av = self.matvec_Ax(v)
            self.AT_pi_v = self.AT_pi.dot(v)
            self._hess_vec = self.mu_inv * (
                self.matvec_ATx(self.pi * self.Av) - self.AT_pi_v * self.AT_pi
            )


def create_log_sum_exp_oracle(A, b, mu):
    """
    Auxiliary function for creating log-sum-exp oracle.
    """
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)

    B = None

    def matmat_ATsA(s):
        nonlocal B
        if B is None:
            B = A.toarray() if scipy.sparse.issparse(A) else A
        return B.T.dot(B * s.reshape(-1, 1))

    return LogSumExpOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, mu)


def create_log_sum_exp_zero_oracle(A, b, mu):
    """
    Creates log-sum-exp oracle with optimum at zero.
    """
    oracle_0 = create_log_sum_exp_oracle(A, b, mu)
    g = oracle_0.grad(np.zeros(A.shape[1]))
    A_new = A - g

    matvec_Ax = lambda x: A_new.dot(x)
    matvec_ATx = lambda x: A_new.T.dot(x)

    B = None

    def matmat_ATsA(s):
        nonlocal B
        if B is None:
            B = A_new.toarray() if scipy.sparse.issparse(A_new) else A_new
        return B.T.dot(B * s.reshape(-1, 1))

    return LogSumExpOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, mu)


class PDifferenceOracle(BaseSmoothOracle):
    """
    Oracle for the function:
        func(x) = 1 / (p + 1) * (|x_1|^{p + 1}
                                 + sum_{i = 1}^n |x_i - a * x_{i - 1}|^{p + 1})
                  - x_n.
    """

    def __init__(self, p, n, a=1):
        self.p = p
        self.n = n
        self.a = a

    def func(self, x):
        return (
            1.0
            / (self.p + 1)
            * (
                np.abs(x[0]) ** (self.p + 1)
                + np.sum(np.abs(x[1:] - self.a * x[0:-1]) ** (self.p + 1))
            )
        )  # - \
        # x[self.n - 1]

    def grad(self, x):
        g = np.zeros_like(x)
        h = x[1:] - self.a * x[0:-1]
        g[1:] += (self.p + 1) * np.abs(h) ** (self.p - 1) * h
        g[0:-1] -= self.a * (self.p + 1) * np.abs(h) ** (self.p - 1) * h
        g[0] += (self.p + 1) * np.abs(x[0]) ** (self.p - 1) * x[0]
        g *= 1.0 / (self.p + 1)
        return g

    def hess(self, x):
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


class NonlinearEquationsOracle(BaseSmoothOracle):
    """
    Oracle for f(x) = 1/p * ||u(x)||^p,    p >= 2.
    """

    def __init__(self, p, func_u=None, jac_u=None, hess_u_list=None, A=None, b=None):
        assert p >= 2, "p must be >= 2"
        self.p = p
        self.func_u = func_u
        self.jac_u = jac_u
        self.hess_u_list = hess_u_list or []
        self.A = A
        self.b = b
        if A is not None and b is not None:
            self.func_u = lambda x: A.dot(x) - b
            self.jac_u = lambda x: A
            self.hess_u_list = []

    def func(self, x):
        u = self.func_u(x)
        norm_u = np.linalg.norm(u)
        return (1.0 / self.p) * norm_u**self.p

    def grad(self, x):
        u = self.func_u(x)
        norm_u = np.linalg.norm(u)
        J = self.jac_u(x)
        return (norm_u ** (self.p - 2)) * (J.T.dot(u))

    def hess(self, x):
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


class ChebyshevOracle(BaseSmoothOracle):
    """
    Oracle for the Chebyshev‐polynomial least‐squares objective:
        u_1(x) = 1 / 2 * (1 - x_1)
        u_i(x) = x_i - (2 x_{i-1}^2 - 1) for i = 2,...,n
    f(x) = 1 / p * \| u(x)\|^p
    The unique global minimizer is x = (1, … ,1)
    """

    def __init__(self, n, p):
        assert p >= 2, "p must be >= 2."
        self.n = n
        self.p = p

        def _func_u(x):
            x = np.asarray(x).ravel()
            assert x.shape[0] == self.n
            u = np.empty(self.n, dtype=float)
            u[0] = 0.5 * (1.0 - x[0])
            for i in range(1, self.n):
                u[i] = x[i] - (2.0 * x[i - 1] ** 2 - 1.0)
            return u

        def _jac_u(x):
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

    def func(self, x):
        u = self.func_u(x)
        norm_u = np.linalg.norm(u)
        return (1.0 / self.p) * (norm_u**self.p)

    def grad(self, x):
        u = self.func_u(x)
        norm_u = np.linalg.norm(u)
        J = self.jac_u(x)
        if norm_u == 0:
            return np.zeros(self.n, dtype=float)
        return (norm_u ** (self.p - 2)) * (J.T.dot(u))

    def hess(self, x):
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

    def hess_vec(self, x, v):
        H = self.hess(x)
        return H.dot(v)

    def third_vec_vec(self, x, v):
        raise NotImplementedError(
            "third_vec_vec is not implemented for ChebyshevOracle."
        )


class RosenbrockOracle(BaseSmoothOracle):
    """
    Oracle for f(x,y) = (a - x)^2 + b (y - x^2)^2.
    """

    def __init__(self, a=1.0, b=100.0):
        self.a = a
        self.b = b

    def func(self, x):
        xx = x[0]
        yy = x[1]
        a = self.a
        b = self.b
        return (a - xx) ** 2 + b * (yy - xx**2) ** 2

    def grad(self, x):
        xx = x[0]
        yy = x[1]
        a = self.a
        b = self.b
        dfdx = 2.0 * (xx - a) - 4.0 * b * xx * (yy - xx**2)
        dfdy = 2.0 * b * (yy - xx**2)
        return np.array([dfdx, dfdy], dtype=float)

    def hess(self, x):
        xx = x[0]
        yy = x[1]
        b = self.b
        d2fdx2 = 2.0 - 4.0 * b * (yy - 3.0 * xx**2)
        d2fdxdy = -4.0 * b * xx
        d2fdy2 = 2.0 * b
        H = np.array([[d2fdx2, d2fdxdy], [d2fdxdy, d2fdy2]], dtype=float)
        return H


class NonlinearEquationsRosenbrockOracle(BaseSmoothOracle):
    """
    Oracle for f(x) = 1/p * \| u(x) \|^p, p >=2
    where u(x) is the 2-vector of Rosenbrock residuals
    """

    def __init__(self, p, a=1.0, b=100.0):
        assert p >= 2, "p must be >= 2"
        self.p = p
        self.a = a
        self.b = b

    def _compute_u_and_jac(self, x):
        xx = x[0]
        yy = x[1]
        a = self.a
        b = self.b
        u1 = a - xx
        u2 = np.sqrt(b) * (yy - xx**2)
        J = np.array([[-1.0, 0.0], [-2.0 * np.sqrt(b) * xx, np.sqrt(b)]], dtype=float)

        return np.array([u1, u2], dtype=float), J

    def func(self, x):
        u, _ = self._compute_u_and_jac(x)
        norm_u = np.linalg.norm(u)
        return (1.0 / self.p) * (norm_u**self.p)

    def grad(self, x):
        u, J = self._compute_u_and_jac(x)
        norm_u = np.linalg.norm(u)
        if norm_u == 0:
            return np.zeros(2, dtype=float)

        factor = norm_u ** (self.p - 2)
        return factor * (J.T.dot(u))

    def hess(self, x):
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
