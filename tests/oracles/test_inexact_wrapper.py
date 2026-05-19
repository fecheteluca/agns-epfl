"""Tests for :class:`agns.oracles.inexact_wrapper.InexactHessianOracle`."""

from __future__ import annotations

import numpy as np
import pytest

from agns.oracles import InexactHessianOracle, RidgeRegressionOracle


@pytest.fixture()
def base_oracle() -> tuple[RidgeRegressionOracle, np.ndarray]:
    rng = np.random.default_rng(0)
    m, n = 40, 6
    X = rng.standard_normal((m, n))
    y = rng.standard_normal(m)
    return RidgeRegressionOracle(X, y, reg=1e-2), rng.standard_normal(n)


def test_zero_budget_is_noop(base_oracle) -> None:
    base, x = base_oracle
    wrapper = InexactHessianOracle(base, delta_budget=0.0)
    np.testing.assert_array_equal(wrapper.hess(x), base.hess(x))


def test_perturbation_has_correct_spectral_norm(base_oracle) -> None:
    base, x = base_oracle
    delta = 0.25
    wrapper = InexactHessianOracle(base, delta_budget=delta, noise_seed=42)
    E = wrapper.hess(x) - base.hess(x)
    # ``E`` is symmetric.
    np.testing.assert_allclose(E, E.T, atol=1e-10)
    # Spectral norm matches the requested budget.
    assert np.linalg.norm(E, ord=2) == pytest.approx(delta, rel=1e-10)


def test_same_seed_same_perturbation(base_oracle) -> None:
    base, x = base_oracle
    a = InexactHessianOracle(base, delta_budget=0.1, noise_seed=7)
    b = InexactHessianOracle(base, delta_budget=0.1, noise_seed=7)
    np.testing.assert_allclose(a.hess(x), b.hess(x), atol=1e-12)


def test_different_seeds_different_perturbations(base_oracle) -> None:
    base, x = base_oracle
    a = InexactHessianOracle(base, delta_budget=0.1, noise_seed=1)
    b = InexactHessianOracle(base, delta_budget=0.1, noise_seed=2)
    diff = a.hess(x) - b.hess(x)
    assert np.linalg.norm(diff) > 1e-3


def test_func_grad_forwarded_unchanged(base_oracle) -> None:
    base, x = base_oracle
    wrapper = InexactHessianOracle(base, delta_budget=0.5, noise_seed=0)
    assert wrapper.func(x) == base.func(x)
    np.testing.assert_array_equal(wrapper.grad(x), base.grad(x))


def test_negative_budget_rejected(base_oracle) -> None:
    base, _ = base_oracle
    with pytest.raises(ValueError):
        InexactHessianOracle(base, delta_budget=-0.1)
