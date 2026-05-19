"""Nonlinear-equations oracle and factory.

The objective is ``f(x) = (1/p) * ||u(x)||^p`` for ``p >= 2``.  For
linear residuals ``u(x) = A x - b`` the oracle is one-shot constructible
from ``(p, A, b)``; for general ``u`` the user must supply callables
``func_u`` and ``jac_u`` plus (optionally) per-residual Hessians.

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from agns.approximations import (
    approx_hess_fn_fisher_term,
    approx_hess_nonlinear_equations,
)
from agns.oracles.base import BaseSmoothOracle


class NonlinearEquationsOracle(BaseSmoothOracle):
    """Oracle for ``f(x) = (1/p) ||u(x)||^p``, ``p >= 2``.

    Either pass ``(A, b)`` for the linear-residual case (most common),
    or supply explicit ``func_u``, ``jac_u``, and optionally a list of
    per-residual Hessians ``hess_u_list``.
    """

    def __init__(
        self,
        p: float,
        func_u: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        jac_u: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        hess_u_list: list[NDArray[np.float64]] | None = None,
        A: NDArray[np.float64] | None = None,
        b: NDArray[np.float64] | None = None,
    ) -> None:
        assert p >= 2.0, "p must be >= 2"
        self.p = float(p)
        self.func_u = func_u
        self.jac_u = jac_u
        self.hess_u_list = hess_u_list or []
        self.A = A
        self.b = b
        if A is not None and b is not None:
            self.func_u = lambda x: A.dot(x) - b
            self.jac_u = lambda x: A
            self.hess_u_list = []

    def func(self, x: NDArray[np.float64]) -> float:
        assert self.func_u is not None
        u = self.func_u(x)
        norm_u = np.linalg.norm(u)
        return float((1.0 / self.p) * norm_u**self.p)

    def grad(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.func_u is not None and self.jac_u is not None
        u = self.func_u(x)
        norm_u = np.linalg.norm(u)
        J = self.jac_u(x)
        return (norm_u ** (self.p - 2)) * (J.T.dot(u))  # type: ignore[no-any-return]

    def hess(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.func_u is not None and self.jac_u is not None
        u = self.func_u(x)
        norm_u = np.linalg.norm(u)
        J = self.jac_u(x)
        H = (norm_u ** (self.p - 2)) * (J.T.dot(J))
        if self.p != 2:
            g = J.T.dot(u)
            H += (self.p - 2) * (norm_u ** (self.p - 4)) * np.outer(g, g)
        # Chain-rule "second-derivative-of-residuals" term:
        #     sum_i ||u||^(p-2) * u_i * nabla^2 u_i(x).
        # Vanishes when u is linear (linear-residuals case below).
        if self.hess_u_list:
            coeff = norm_u ** (self.p - 2)
            for i, H_ui in enumerate(self.hess_u_list):
                H += coeff * u[i] * H_ui
        return cast("NDArray[np.float64]", H)

    def _compute_u_and_jac(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convenience accessor used by the approximation builder."""
        assert self.func_u is not None and self.jac_u is not None
        return self.func_u(x), self.jac_u(x)


# ---------------------------------------------------------------------------
# Problem factory
# ---------------------------------------------------------------------------


def make_nonlinear_equations(
    n: int = 100, m: int = 200, p: float = 4.0, seed: int = 42
) -> dict[str, Any]:
    """``min_x (1/p) ||Ax - b||^p`` with random ``A in R^{n x m}``.

    When ``m > n`` (underdetermined), ``A`` has full row rank with
    probability 1 so ``Ax = b`` has an exact solution and ``f^* = 0``.

    The Hessian's Holder exponent depends on ``p``:

    * ``p in (2, 3]`` -- ``(L, nu)``-Holder with ``nu = p - 2``.
    * ``p >= 3``      -- globally Lipschitz Hessian (``nu = 1``).
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, (n, m))
    b = rng.uniform(-1.0, 1.0, n)
    eps_reg = 1e-8

    oracle = NonlinearEquationsOracle(p, A=A, b=b)
    x_0 = np.ones(m)
    B = A.T.dot(A) + eps_reg * np.eye(m)
    Binv = np.linalg.inv(B)

    f_star = 0.0 if m > n else None

    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": f_star,
        "B": B,
        "Binv": Binv,
        # ``approx_hess_fn`` (used by gns_inexact / agns_practice / agns_holder /
        # agns_inexact) returns the full GN + Fisher approximation, matching the
        # docstring of ``approx_hess_nonlinear_equations``.  The WSM backend
        # requires a rank-1 Hessian and so receives only the Fisher term.
        "approx_hess_fn": approx_hess_nonlinear_equations,
        "approx_hess_fn_wsm": approx_hess_fn_fisher_term,
        "meta": {"type": "nonlinear_equations", "n": n, "m": m, "p": p, "seed": seed},
    }


__all__ = ["NonlinearEquationsOracle", "make_nonlinear_equations"]
