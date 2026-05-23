"""Shared test fixtures.

Provides small, deterministic problem instances that the per-method and
per-oracle suites consume.  All fixtures are seeded so that test
failures bisect to the offending change rather than to RNG drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from agns.oracles import (
    ChebyshevOracle,
    LogisticRegressionOracle,
    LogSumExpOracle,
    MatrixCompletionOracle,
    NonlinearEquationsOracle,
    NonlinearEquationsRosenbrockOracle,
    PhaseRetrievalOracle,
    RidgeRegressionOracle,
    RosenbrockOracle,
    SmoothedLassoOracle,
    SoftmaxRegressionOracle,
    SoftSVMOracle,
    make_logsumexp_zero_oracle,
)
from agns.oracles.base import BaseSmoothOracle

# ---------------------------------------------------------------------------
# Small problem fixtures: every oracle gets one.
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture()
def ridge_problem() -> tuple[BaseSmoothOracle, np.ndarray, float, np.ndarray, np.ndarray]:
    """Small ridge regression: ``(oracle, x_0, f_star, B, Binv)``."""
    rng = np.random.default_rng(0)
    m, n = 60, 8
    X = rng.standard_normal((m, n))
    w_true = rng.standard_normal(n)
    y = X.dot(w_true) + 0.05 * rng.standard_normal(m)
    oracle = RidgeRegressionOracle(X, y, reg=1e-2)
    x_0 = np.zeros(n)
    H = X.T.dot(X) / m + 1e-2 * np.eye(n)
    w_star = np.linalg.solve(H, X.T.dot(y) / m)
    f_star = float(oracle.func(w_star))
    return oracle, x_0, f_star, H, np.linalg.inv(H)


@pytest.fixture()
def logsumexp_problem() -> tuple[BaseSmoothOracle, np.ndarray, float, np.ndarray, np.ndarray]:
    """Small zero-centred LogSumExp."""
    rng = np.random.default_rng(0)
    n, m = 8, 12
    A = rng.uniform(-1.0, 1.0, (n, m))  # rows -> dim of x
    b = rng.uniform(-1.0, 1.0, m)
    oracle = make_logsumexp_zero_oracle(A.T, b, mu=1.0)
    x_0 = np.ones(n)
    f_star = float(oracle.func(np.zeros(n)))
    B = A.dot(A.T)
    return oracle, x_0, f_star, B, np.linalg.inv(B)


@pytest.fixture()
def nle_problem() -> tuple[BaseSmoothOracle, np.ndarray, float, np.ndarray, np.ndarray]:
    """Nonlinear equations ``(1/p) ||Ax - b||^p``."""
    rng = np.random.default_rng(0)
    n, m = 6, 10
    A = rng.uniform(-1.0, 1.0, (n, m))
    b = rng.uniform(-1.0, 1.0, n)
    oracle = NonlinearEquationsOracle(p=4, A=A, b=b)
    x_0 = np.ones(m)
    B = A.T.dot(A) + 1e-8 * np.eye(m)
    return oracle, x_0, 0.0, B, np.linalg.inv(B)


@pytest.fixture()
def logistic_problem() -> tuple[BaseSmoothOracle, np.ndarray, np.ndarray, np.ndarray]:
    """Small L2 logistic regression: ``(oracle, x_0, B, Binv)``."""
    rng = np.random.default_rng(0)
    m, n = 60, 5
    X = rng.standard_normal((m, n))
    w_true = rng.standard_normal(n)
    y = np.sign(X.dot(w_true))
    y[y == 0] = 1.0
    oracle = LogisticRegressionOracle(X, y, reg=1e-2)
    x_0 = np.zeros(n)
    B = X.T.dot(X) / m + 1e-2 * np.eye(n)
    return oracle, x_0, B, np.linalg.inv(B)


@pytest.fixture()
def soft_svm_problem() -> tuple[BaseSmoothOracle, np.ndarray, np.ndarray, np.ndarray]:
    """Small squared-hinge soft-margin SVM."""
    rng = np.random.default_rng(0)
    m, n = 60, 5
    X = rng.standard_normal((m, n))
    w_true = rng.standard_normal(n)
    y = np.sign(X.dot(w_true) + 0.1 * rng.standard_normal(m))
    y[y == 0] = 1.0
    oracle = SoftSVMOracle(X, y, reg=1e-2)
    B = X.T.dot(X) / m + 1e-2 * np.eye(n)
    return oracle, np.zeros(n), B, np.linalg.inv(B)


@pytest.fixture()
def smoothed_lasso_problem() -> tuple[BaseSmoothOracle, np.ndarray, np.ndarray, np.ndarray]:
    """Pseudo-Huber-smoothed LASSO."""
    rng = np.random.default_rng(0)
    m, n = 60, 6
    X = rng.standard_normal((m, n))
    w_true = np.zeros(n)
    w_true[:3] = rng.standard_normal(3)
    y = X.dot(w_true) + 0.05 * rng.standard_normal(m)
    oracle = SmoothedLassoOracle(X, y, reg=1e-2, smoothing=1e-2)
    B = X.T.dot(X) / m + 1e-8 * np.eye(n)
    return oracle, np.zeros(n), B, np.linalg.inv(B)


@pytest.fixture()
def softmax_problem() -> tuple[BaseSmoothOracle, np.ndarray]:
    """Multinomial softmax regression."""
    rng = np.random.default_rng(0)
    m, n, K = 60, 5, 3
    X = rng.standard_normal((m, n))
    W_true = rng.standard_normal((n, K))
    y = np.argmax(X @ W_true + 0.1 * rng.standard_normal((m, K)), axis=1)
    oracle = SoftmaxRegressionOracle(X, y, n_classes=K, reg=1e-3)
    return oracle, np.zeros(n * K)


@pytest.fixture()
def chebyshev_problem() -> tuple[BaseSmoothOracle, np.ndarray, float]:
    """Small Chebyshev-chain oracle."""
    rng = np.random.default_rng(0)
    n, p = 4, 3
    oracle = ChebyshevOracle(n, p)
    x_0 = rng.uniform(0.0, 1.0, n)
    f_star = float(oracle.func(np.ones(n)))
    return oracle, x_0, f_star


@pytest.fixture()
def rosenbrock_2d_problem() -> tuple[BaseSmoothOracle, np.ndarray]:
    return RosenbrockOracle(a=1.0, b=100.0), np.array([-1.2, 1.0])


@pytest.fixture()
def rosenbrock_nle_problem() -> tuple[BaseSmoothOracle, np.ndarray]:
    return NonlinearEquationsRosenbrockOracle(p=4, a=1.0, b=100.0), np.array([-1.2, 1.0])


@pytest.fixture()
def matrix_completion_problem() -> tuple[BaseSmoothOracle, np.ndarray]:
    rng = np.random.default_rng(0)
    n1, n2, r = 6, 6, 2
    U_true = rng.standard_normal((n1, r))
    V_true = rng.standard_normal((n2, r))
    M = U_true @ V_true.T
    obs_idx = rng.choice(n1 * n2, size=int(0.6 * n1 * n2), replace=False)
    obs_rows = obs_idx // n2
    obs_cols = obs_idx % n2
    oracle = MatrixCompletionOracle(
        n1,
        n2,
        r,
        obs_rows,
        obs_cols,
        M[obs_rows, obs_cols],
        reg=1e-3,
    )
    x_0 = 0.1 * rng.standard_normal((n1 + n2) * r)
    return oracle, x_0


@pytest.fixture()
def phase_retrieval_problem() -> tuple[BaseSmoothOracle, np.ndarray]:
    rng = np.random.default_rng(0)
    m, n = 80, 5
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n)
    y = (A @ x_true) ** 2
    oracle = PhaseRetrievalOracle(A, y)
    x_0 = x_true + 0.05 * rng.standard_normal(n)
    return oracle, x_0


@pytest.fixture()
def torch_mlp_problem() -> tuple[BaseSmoothOracle, np.ndarray]:
    """Tiny MLP-classifier TorchOracle for FD cross-checks.

    Small enough that the full Hessian is materialisable in milliseconds.
    """
    from agns.oracles import make_mlp_classifier_synthetic

    problem = make_mlp_classifier_synthetic(
        n_samples=24,
        n_features=4,
        n_classes=2,
        hidden_dims=(3,),
        reg=1e-3,
        seed=0,
    )
    return problem["oracle"], problem["x_0"]


@pytest.fixture()
def logsumexp_raw() -> tuple[LogSumExpOracle, np.ndarray]:
    """Plain LogSumExp (no recentering) for hess_vec tests."""
    rng = np.random.default_rng(0)
    n, m = 6, 8
    A = rng.uniform(-1.0, 1.0, (m, n))  # m rows, n cols
    b = rng.uniform(-1.0, 1.0, m)
    from agns.oracles.logsumexp import make_logsumexp_pair_oracle

    return make_logsumexp_pair_oracle(A, b, mu=1.0), rng.standard_normal(n)
