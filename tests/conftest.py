"""Shared pytest fixtures and finite-difference helpers.

The helpers here are the verification backbone of the oracle tests: central-difference
gradients and Hessians let us check every analytic ``grad``/``hess`` against the
function it claims to differentiate, independently of the implementation. The small
SPD quadratic and small log-sum-exp builders give the method tests a fast,
well-conditioned problem on which every method must converge and ported methods must
reproduce the pinned upstream trajectories.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from agns.oracles.logsumexp import create_log_sum_exp_zero_oracle
from agns.oracles.nonlinear_equations import NonlinearEquationsOracle
from agns.oracles.problems import Problem

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


# --------------------------------------------------------------- finite differences


def finite_diff_grad(
    func: Callable[[Vector], float], x: Vector, *, h: float = 1e-6
) -> Vector:
    """Central-difference gradient of ``func`` at ``x``.

    Central differences have ``O(h^2)`` truncation error, so with ``h = 1e-6`` the
    result matches an analytic gradient to roughly ``1e-9`` away from round-off.
    """
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.shape[0]):
        step = np.zeros_like(x)
        step[i] = h
        g[i] = (func(x + step) - func(x - step)) / (2.0 * h)
    return g


def finite_diff_hess(
    grad: Callable[[Vector], Vector], x: Vector, *, h: float = 1e-5
) -> Matrix:
    """Central-difference Hessian (Jacobian of the analytic ``grad``) at ``x``.

    Symmetrized to remove the small asymmetry that finite differences introduce, which
    is the property the analytic Hessian is checked against.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    H = np.zeros((n, n))
    for i in range(n):
        step = np.zeros_like(x)
        step[i] = h
        H[:, i] = (grad(x + step) - grad(x - step)) / (2.0 * h)
    return 0.5 * (H + H.T)


# ------------------------------------------------------------------ small problems


def make_quadratic_problem(n: int = 8, *, cond: float = 25.0, seed: int = 0) -> Problem:
    """A genuine SPD quadratic ``f(x) = 1/2 ||A x - b||^2`` with optimum ``f* = 0``.

    Realized as a :class:`NonlinearEquationsOracle` with ``p = 2``: its gradient is
    ``A^T(A x - b)`` and its Hessian is the constant SPD matrix ``A^T A``. The design
    ``A`` has a controlled condition number so every method (first- and second-order)
    converges quickly on it.
    """
    rng = np.random.RandomState(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.geomspace(1.0, np.sqrt(cond), n)
    A = (q * s) @ q.T  # symmetric PD design; A^T A has condition number ``cond``
    b = rng.standard_normal(n)
    oracle = NonlinearEquationsOracle(p=2, A=A, b=b)
    x_star = np.linalg.solve(A, b)
    return Problem(
        name=f"quadratic_n{n}",
        label=f"SPD quadratic (n={n})",
        oracle=oracle,
        x_0=np.ones(n),
        f_star=0.0,
        eps=1e-10,
        B=None,
        Binv=None,
        n=n,
        meta={"family": "quadratic", "x_star": x_star, "cond": cond},
    )


def make_logsumexp_problem(
    n: int = 12, m: int = 20, *, mu: float = 1.0, seed: int = 3124
) -> Problem:
    """A small, well-conditioned log-sum-exp instance with optimum shifted to the origin.

    Built the same way as ``config/oracles/random_small.yaml`` but smaller, so the
    Newton solves are cheap and ``H + lambda B`` stays positive definite (the regime
    where the refactored Cholesky path is bit-identical to upstream).
    """
    rng = np.random.RandomState(seed)
    A = rng.rand(n, m) * 2 - 1
    b = np.random.RandomState(seed + 1).rand(m) * 2 - 1
    D = A.T  # (m, n): m linear pieces, n variables
    oracle = create_log_sum_exp_zero_oracle(D, b, mu)
    f_star = float(oracle.func(np.zeros(n)))
    metric = D.T.dot(D)
    return Problem(
        name=f"logsumexp_n{n}_m{m}",
        label=f"LogSumExp (n={n})",
        oracle=oracle,
        x_0=np.full(n, 1.0),
        f_star=f_star,
        eps=1e-8,
        B=metric,
        Binv=np.linalg.inv(metric),
        n=n,
        meta={"family": "logsumexp", "m": m, "mu": mu},
    )


@pytest.fixture
def quadratic_problem() -> Problem:
    """Module-friendly SPD quadratic fixture (optimum ``f* = 0``)."""
    return make_quadratic_problem()


@pytest.fixture
def logsumexp_problem() -> Problem:
    """Small well-conditioned log-sum-exp fixture (optimum at the origin)."""
    return make_logsumexp_problem()
