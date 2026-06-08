"""AGNS fidelity test (spec section 5.4).

Guards :func:`agns.methods.agns.agns` against behavioural drift by
running it against the frozen reference copy in
:mod:`tests._reference_agns` on a fixed regularised-logistic problem and
asserting the iterate trajectories agree to <=1e-10 with identical
restart events.

A small self-contained logistic oracle is used here so the fidelity
guard does not depend on the full oracle layer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from agns.methods.agns import agns as agns_new
from agns.oracles.base import BaseSmoothOracle
from tests._reference_agns import agns as agns_ref


class _RegularisedLogistic(BaseSmoothOracle):
    """L2-regularised binary logistic regression, mean over samples.

    ``f(w) = mean_i [ log(1 + exp(x_i.w)) - y_i (x_i.w) ] + 0.5 * reg * ||w||^2``
    with labels ``y in {0, 1}``.
    """

    def __init__(self, X: NDArray[np.float64], y: NDArray[np.float64], reg: float) -> None:
        self.X = X
        self.y = y
        self.reg = reg
        self.n = X.shape[0]

    def func(self, w: NDArray[np.float64]) -> float:
        z = self.X @ w
        data = float(np.mean(np.logaddexp(0.0, z) - self.y * z))
        return data + 0.5 * self.reg * float(w @ w)

    def grad(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        z = self.X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        return self.X.T @ (p - self.y) / self.n + self.reg * w

    def hess(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        z = self.X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        s = p * (1.0 - p)
        return (self.X.T * s) @ self.X / self.n + self.reg * np.eye(w.shape[0])


def _fixed_problem(seed: int = 0) -> tuple[_RegularisedLogistic, NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    n, d = 200, 12
    X = rng.standard_normal((n, d))
    w_true = rng.standard_normal(d)
    probs = 1.0 / (1.0 + np.exp(-X @ w_true))
    y = (rng.uniform(size=n) < probs).astype(np.float64)
    oracle = _RegularisedLogistic(X, y, reg=1e-2)
    x_0 = np.zeros(d)
    return oracle, x_0


def test_agns_matches_reference_trajectory() -> None:
    oracle, x_0 = _fixed_problem(seed=0)
    common = dict(n_iters=100, eps=1e-12, restart_mode="gradient")

    x_ref, status_ref, hist_ref = agns_ref(oracle, x_0.copy(), **common)
    x_new, status_new, hist_new = agns_new(oracle, x_0.copy(), **common)

    assert hist_ref is not None and hist_new is not None
    # Full iterate trajectory agrees to <=1e-10.
    traj_ref = np.array(hist_ref["x_k"])
    traj_new = np.array(hist_new["x_k"])
    assert traj_ref.shape == traj_new.shape
    assert np.max(np.abs(traj_ref - traj_new)) <= 1e-10
    # Final iterate.
    assert np.linalg.norm(x_new - x_ref) <= 1e-10
    # Identical restart events and status.
    assert hist_new["restart_events"] == hist_ref["restart_events"]
    assert status_new == status_ref


def test_agns_problem_actually_exercises_restarts() -> None:
    """Guard that the fixture is non-trivial: it must fire >=1 restart,
    otherwise the fidelity test would not exercise the restart path."""
    oracle, x_0 = _fixed_problem(seed=0)
    _, _, hist = agns_new(oracle, x_0.copy(), n_iters=100, eps=1e-12, restart_mode="gradient")
    assert hist is not None
    assert len(hist["restart_events"]) >= 1
