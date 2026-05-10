"""
Problem instance factory.

Each function returns a standardized dict:
    oracle       — BaseSmoothOracle instance (unwrapped, no counter)
    x_0          — initial point (np.ndarray)
    f_star       — known optimum value, or None if unknown
    B            — metric matrix (n*n) for dual norm, or None (Euclidean)
    Binv         — inverse of B, or None
    approx_hess_fn     — inexact Hessian callable for CHO-based methods
    approx_hess_fn_wsm — inexact Hessian callable for WSM-based methods (rank-1 Fisher)
    meta         — dict with human-readable problem metadata
"""

import numpy as np
import scipy.sparse as sp

from oracles import (
    ChebyshevOracle,
    NonlinearEquationsOracle,
    NonlinearEquationsRosenbrockOracle,
    RosenbrockOracle,
    create_log_sum_exp_zero_oracle,
)
from approximations import (
    approx_hess_fn_chebyshev,
    approx_hess_fn_fisher_term,
    approx_hess_fn_logsumexp,
    approx_hess_nonlinear_equations,
)


# ---------------------------------------------------------------------------
# LogSumExp
# ---------------------------------------------------------------------------

def make_logsumexp_synthetic(n=1000, m=1000, mu=1.0, seed=42):
    """
    min_x  mu * log sum_i exp(<a_i, x> / mu)   s.t. x in R^n
    A in R^{n x m}, optimum at zero by construction.
    B = A A^T is the natural metric (Weighted Gauss-Newton Gram matrix).
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, (n, m))   # n x m
    b = rng.uniform(-1.0, 1.0, m)

    # oracle works on x in R^n; internally A^T (m x n) acts as matvec
    oracle = create_log_sum_exp_zero_oracle(A.T, b, mu)
    x_0 = np.ones(n)
    f_star = oracle.func(np.zeros(n))    # construction ensures optimum at 0
    B = A.dot(A.T)                       # n x n positive definite
    Binv = np.linalg.inv(B)

    return dict(
        oracle=oracle, x_0=x_0, f_star=f_star, B=B, Binv=Binv,
        approx_hess_fn=approx_hess_fn_logsumexp,
        approx_hess_fn_wsm=None,          # WSM not applicable (not rank-1)
        meta=dict(type="logsumexp_synthetic", n=n, m=m, mu=mu, seed=seed),
    )


def make_logsumexp_real(path, mu=1.0, max_samples=None, seed=42):
    """
    LogSumExp objective built from a LIBSVM dataset.
    X in R^{m x n} (m samples, n features); oracle optimizes over x in R^n.
    B = X^T X is the n x n Gram matrix (natural metric for this problem).
    """
    try:
        from sklearn.datasets import load_svmlight_file
    except ImportError as e:
        raise ImportError("scikit-learn required for real datasets") from e

    X, _ = load_svmlight_file(path)
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float64)

    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]

    m, n = X.shape
    b = np.zeros(m)
    oracle = create_log_sum_exp_zero_oracle(X, b, mu)
    # Optimum is at x*=0 by construction.  Use a non-trivial starting point so
    # the methods have actual work to do.  x=ones is far enough from 0 for real
    # data without being pathological.
    rng2 = np.random.default_rng(seed + 1)
    x_0 = rng2.standard_normal(n)
    f_star = oracle.func(np.zeros(n))    # true optimum value

    gram = X.T.dot(X)
    reg = 1e-8 * np.eye(n)
    B = gram + reg
    Binv = np.linalg.inv(B)

    return dict(
        oracle=oracle, x_0=x_0, f_star=f_star, B=B, Binv=Binv,
        approx_hess_fn=approx_hess_fn_logsumexp,
        approx_hess_fn_wsm=None,
        meta=dict(type="logsumexp_real", path=str(path), n=n, m=m, mu=mu),
    )


# ---------------------------------------------------------------------------
# Nonlinear equations  f(x) = 1/p * ||Ax - b||^p
# ---------------------------------------------------------------------------

def make_nonlinear_equations(n=100, m=200, p=4, seed=42):
    """
    min_x  1/p * ||Ax - b||^p,   x in R^m,  A in R^{n x m}.
    B = A^T A + eps I  (positive definite metric in the solution space).
    When m > n (underdetermined, which is the default), A has full row rank
    with probability 1, so Ax=b has an exact solution and f_star = 0.
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, (n, m))
    b = rng.uniform(-1.0, 1.0, n)
    eps_reg = 1e-8

    oracle = NonlinearEquationsOracle(p, A=A, b=b)
    x_0 = np.ones(m)
    B = A.T.dot(A) + eps_reg * np.eye(m)
    Binv = np.linalg.inv(B)

    # f_star=0 when m > n (underdetermined system has an exact solution).
    f_star = 0.0 if m > n else None

    return dict(
        oracle=oracle, x_0=x_0, f_star=f_star, B=B, Binv=Binv,
        approx_hess_fn=approx_hess_fn_fisher_term,
        approx_hess_fn_wsm=approx_hess_fn_fisher_term,
        meta=dict(type="nonlinear_equations", n=n, m=m, p=p, seed=seed),
    )


# ---------------------------------------------------------------------------
# Chebyshev polynomial chain
# ---------------------------------------------------------------------------

def make_chebyshev(n=1000, p=4, seed=42):
    """
    min_x  1/p * ||u(x)||^p,  x in R^n.
    u_1(x) = 1/2 (1 - x_1),  u_i(x) = x_i - (2 x_{i-1}^2 - 1).
    Global minimizer is x* = (1,...,1); f* = 0.
    No natural non-trivial metric; B = None (Euclidean).
    """
    rng = np.random.default_rng(seed)
    oracle = ChebyshevOracle(n, p)
    x_0 = rng.uniform(0.0, 1.0, n)
    f_star = oracle.func(np.ones(n))     # = 0.0 by construction

    return dict(
        oracle=oracle, x_0=x_0, f_star=f_star, B=None, Binv=None,
        approx_hess_fn=approx_hess_fn_chebyshev,
        approx_hess_fn_wsm=approx_hess_fn_fisher_term,
        meta=dict(type="chebyshev", n=n, p=p, seed=seed),
    )


# ---------------------------------------------------------------------------
# Rosenbrock expressed as nonlinear equations  f(x) = 1/p * ||u(x)||^p
# ---------------------------------------------------------------------------

def make_rosenbrock_nle(p=5):
    """
    min_x  1/p * ||u(x)||^p,   u(x) = (1-x1, sqrt(100)*(x2-x1^2)).
    Global minimizer x* = (1,1); f* = 0.
    """
    oracle = NonlinearEquationsRosenbrockOracle(p=p, a=1.0, b=100.0)
    x_0 = np.array([-2.0, 2.0])
    f_star = oracle.func(np.array([1.0, 1.0]))

    return dict(
        oracle=oracle, x_0=x_0, f_star=f_star, B=None, Binv=None,
        approx_hess_fn=approx_hess_nonlinear_equations,
        approx_hess_fn_wsm=approx_hess_fn_fisher_term,
        meta=dict(type="rosenbrock_nle", p=p),
    )


# ---------------------------------------------------------------------------
# Standard 2-D Rosenbrock
# ---------------------------------------------------------------------------

def make_rosenbrock_2d():
    """
    min_x  (1-x1)^2 + 100*(x2-x1^2)^2.
    Global minimizer x* = (1,1); f* = 0.
    No inexact Hessian approximation available for this oracle.
    """
    oracle = RosenbrockOracle(a=1.0, b=100.0)
    x_0 = np.array([-2.0, 2.0])
    f_star = oracle.func(np.array([1.0, 1.0]))

    return dict(
        oracle=oracle, x_0=x_0, f_star=f_star, B=None, Binv=None,
        approx_hess_fn=None,
        approx_hess_fn_wsm=None,
        meta=dict(type="rosenbrock_2d"),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "logsumexp_synthetic": make_logsumexp_synthetic,
    "logsumexp_real": make_logsumexp_real,
    "nonlinear_equations": make_nonlinear_equations,
    "chebyshev": make_chebyshev,
    "rosenbrock_nle": make_rosenbrock_nle,
    "rosenbrock_2d": make_rosenbrock_2d,
}
