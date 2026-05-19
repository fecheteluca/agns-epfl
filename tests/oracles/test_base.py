"""Tests for :mod:`agns.oracles.base`."""

from __future__ import annotations

import numpy as np
import pytest

from agns.oracles.base import BaseSmoothOracle, OracleCallsCounter


def test_base_methods_raise_by_default() -> None:
    o = BaseSmoothOracle()
    x = np.zeros(3)
    with pytest.raises(NotImplementedError):
        o.func(x)
    with pytest.raises(NotImplementedError):
        o.grad(x)
    with pytest.raises(NotImplementedError):
        o.hess(x)
    with pytest.raises(NotImplementedError):
        o.third_vec_vec(x, x)


def test_default_hess_vec_uses_hess() -> None:
    class StubOracle(BaseSmoothOracle):
        def hess(self, x: np.ndarray) -> np.ndarray:
            return np.eye(x.shape[0]) * 3.0

    o = StubOracle()
    v = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(o.hess_vec(np.zeros(3), v), 3.0 * v)


def test_oracle_calls_counter_increments() -> None:
    class StubOracle(BaseSmoothOracle):
        def func(self, x: np.ndarray) -> float:
            return float(x.sum())

        def grad(self, x: np.ndarray) -> np.ndarray:
            return np.ones_like(x)

        def hess(self, x: np.ndarray) -> np.ndarray:
            return np.eye(x.shape[0])

        def hess_vec(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
            return v

    counter = OracleCallsCounter(StubOracle())
    x = np.zeros(3)
    counter.func(x)
    counter.func(x)
    counter.grad(x)
    counter.hess(x)
    counter.hess_vec(x, x)
    assert counter.func_calls == 2
    assert counter.grad_calls == 1
    assert counter.hess_calls == 1
    assert counter.hess_vec_calls == 1


def test_oracle_calls_counter_forwards_attributes() -> None:
    class StubOracle(BaseSmoothOracle):
        custom_attr = 42

    counter = OracleCallsCounter(StubOracle())
    # Counter does not define ``custom_attr``; __getattr__ delegates.
    assert counter.custom_attr == 42
