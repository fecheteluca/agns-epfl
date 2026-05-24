"""Smoke tests for every oracle factory.

Each factory must return a dict with the canonical keys consumed by the
runner: ``oracle``, ``x_0``, ``f_star``, ``B``, ``Binv``, ``approx_hess_fn``,
``approx_hess_fn_wsm``, ``meta``.
"""

from __future__ import annotations

import numpy as np
import pytest

from agns.oracles import PROBLEM_REGISTRY

_REQUIRED_KEYS = {
    "oracle",
    "x_0",
    "f_star",
    "B",
    "Binv",
    "approx_hess_fn",
    "approx_hess_fn_wsm",
    "meta",
}

# These factories require external data (LIBSVM files or torch).
_SKIP_FACTORIES = {
    "logsumexp_real",
    "logistic_regression_libsvm",
    "ridge_regression_libsvm",
    "soft_svm_libsvm",
    "mlp_classifier_synthetic",
}


@pytest.mark.parametrize(
    "name",
    [k for k in sorted(PROBLEM_REGISTRY) if k not in _SKIP_FACTORIES],
)
def test_factory_returns_canonical_dict(name: str) -> None:
    factory = PROBLEM_REGISTRY[name]
    prob = factory()
    assert set(prob.keys()) >= _REQUIRED_KEYS, f"{name} missing keys: {_REQUIRED_KEYS - set(prob)}"
    assert isinstance(prob["x_0"], np.ndarray)
    assert prob["x_0"].ndim == 1
    if prob["B"] is not None:
        assert prob["B"].shape == (prob["x_0"].shape[0], prob["x_0"].shape[0])
    if prob["Binv"] is not None:
        assert prob["Binv"].shape == prob["B"].shape


@pytest.mark.parametrize(
    "name",
    [k for k in sorted(PROBLEM_REGISTRY) if k not in _SKIP_FACTORIES],
)
def test_factory_oracle_func_finite_at_x0(name: str) -> None:
    factory = PROBLEM_REGISTRY[name]
    prob = factory()
    f0 = prob["oracle"].func(prob["x_0"])
    assert np.isfinite(f0), f"{name}: func(x_0) is not finite"
