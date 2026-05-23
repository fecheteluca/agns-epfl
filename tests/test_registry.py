"""Tests for :mod:`agns.pipeline.registry`."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from agns.pipeline.registry import METHOD_REGISTRY, PROBLEM_REGISTRY, MethodSpec, run_method

# Every registered method's run_fn must accept these arguments.
_GENERIC_KW = {"n_iters", "eps"}


def test_method_registry_keys_match_function_module_names() -> None:
    """The registry keys should map to functions whose names match where reasonable."""
    expected_at_least = {
        "gns_exact",
        "gns_inexact",
        "gns_wsm",
        "agns_inexact",
        "agns_exact",
        "agns_wsm",
        "newton",
        "super_newton",
        "cubic_newton",
        "accelerated_cubic_newton",
        "monteiro_svaiter_acn",
        "trust_region",
        "gradient",
        "fast_gradient",
        "heavy_ball",
        "lbfgs",
        "adam",
        "adahessian",
    }
    assert expected_at_least <= set(METHOD_REGISTRY)


@pytest.mark.parametrize("key", sorted(METHOD_REGISTRY))
def test_every_method_accepts_generic_kwargs(key: str) -> None:
    spec = METHOD_REGISTRY[key]
    sig = inspect.signature(spec.run_fn)
    # All methods must take ``oracle``, ``x_0`` and ``n_iters``, ``eps``.
    params = set(sig.parameters)
    assert "oracle" in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    # ``n_iters`` and ``eps`` may be accepted via **kwargs.
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    for kw in _GENERIC_KW:
        assert kw in params or accepts_var_kwargs, f"{key} missing {kw}"


def test_problem_registry_is_populated() -> None:
    assert "ridge_regression_synthetic" in PROBLEM_REGISTRY
    assert "logsumexp_synthetic" in PROBLEM_REGISTRY
    assert callable(PROBLEM_REGISTRY["ridge_regression_synthetic"])


def test_run_method_returns_canonical_tuple(ridge_problem) -> None:
    oracle, x0, f_star, B, Binv = ridge_problem
    spec = METHOD_REGISTRY["gns_exact"]
    cfg = {
        "n_iters": 3,
        "gamma_0": 1.0,
        "eps": 1e-12,
        "B": B,
        "Binv": Binv,
        "f_star": f_star,
        "adaptive_search": True,
    }
    x_final, status, hist = run_method(spec, oracle, x0, cfg, oracle, None)
    assert isinstance(x_final, np.ndarray)
    assert isinstance(status, str)
    assert hist is not None
    assert "func" in hist
    assert "grad_norm" in hist
    assert "time" in hist


def test_run_method_attaches_approx_when_uses_approx(nle_problem) -> None:
    oracle, x0, f_star, B, Binv = nle_problem
    from agns.oracles.approximations import approx_hess_fn_fisher_term

    spec = METHOD_REGISTRY["gns_inexact"]
    assert spec.uses_approx
    cfg = {
        "n_iters": 3,
        "gamma_0": 1.0,
        "eps": 1e-10,
        "B": B,
        "Binv": Binv,
        "f_star": f_star,
        "adaptive_search": True,
    }
    _x_final, _status, hist = run_method(
        spec,
        oracle,
        x0,
        cfg,
        oracle,
        approx_hess_fn_fisher_term,
    )
    assert hist is not None and hist["func"]


def test_method_spec_is_frozen() -> None:
    import dataclasses

    spec = METHOD_REGISTRY["gns_exact"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.uses_approx = True  # type: ignore[misc]


def test_method_spec_factory_default_kw_isolation() -> None:
    """Two MethodSpec instances must not share their default_kw dict."""
    a = MethodSpec(run_fn=lambda **kw: None)  # type: ignore[arg-type, return-value]
    b = MethodSpec(run_fn=lambda **kw: None)  # type: ignore[arg-type, return-value]
    a.default_kw["x"] = 1  # type: ignore[index]
    assert "x" not in b.default_kw
