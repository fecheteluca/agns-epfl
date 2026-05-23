"""Direct tests for the public helpers in :mod:`agns.methods._helpers`.

Several of these helpers are exercised end-to-end through the method
smoke tests, but they each carry independent invariants worth pinning
down in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from agns.methods._helpers import (
    RESTART_MODES,
    rank_one_fisher_factor,
    regularised_lhs,
    should_restart,
    validate_restart_mode,
)

# ---------------------------------------------------------------------------
# AGNS restart predicate
# ---------------------------------------------------------------------------


class TestValidateRestartMode:
    @pytest.mark.parametrize("mode", sorted(RESTART_MODES))
    def test_accepts_canonical_modes(self, mode: str) -> None:
        validate_restart_mode(mode)  # no raise

    @pytest.mark.parametrize("bad", ["", "GRADIENT", "func_value", "unknown", "0"])
    def test_rejects_anything_else(self, bad: str) -> None:
        with pytest.raises(ValueError):
            validate_restart_mode(bad)


class TestShouldRestart:
    def test_gradient_mode_fires_on_positive_dot_product(self) -> None:
        # g_trial . (x_trial - x_k) > 0 means the step direction is
        # *against* the gradient at the new iterate -- restart.
        x_k = np.array([0.0, 0.0])
        x_trial = np.array([1.0, 0.0])
        g_trial = np.array([2.0, 5.0])  # dot with (1, 0) = +2 > 0
        assert should_restart("gradient", g_trial, x_trial, x_k, 1.0, 0.0) is True

    def test_gradient_mode_does_not_fire_on_negative_dot_product(self) -> None:
        x_k = np.array([0.0, 0.0])
        x_trial = np.array([1.0, 0.0])
        g_trial = np.array([-2.0, 5.0])  # dot with (1, 0) = -2 < 0
        assert should_restart("gradient", g_trial, x_trial, x_k, 1.0, 0.0) is False

    def test_function_value_mode_fires_when_f_increases(self) -> None:
        # f_trial > f_k -> restart.
        x = np.array([0.0])
        assert should_restart("function_value", x, x, x, f_trial=2.0, f_k=1.0) is True
        assert should_restart("function_value", x, x, x, f_trial=0.5, f_k=1.0) is False

    def test_none_mode_never_fires(self) -> None:
        x = np.array([0.0])
        g = np.array([1.0])
        assert should_restart("none", g, x, x, 1.0, 0.0) is False
        assert should_restart("none", g, x, x, 0.0, 1.0) is False


# ---------------------------------------------------------------------------
# Regularised LHS construction (B = None and B = matrix)
# ---------------------------------------------------------------------------


class TestRegularisedLhs:
    def test_none_metric_adds_lambda_to_diagonal(self) -> None:
        H = np.array([[1.0, 2.0], [2.0, 4.0]])
        out = regularised_lhs(H, lambda_k=0.5, B_eff=None)
        expected = H + 0.5 * np.eye(2)
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_explicit_metric_scales_by_lambda(self) -> None:
        H = np.eye(3)
        B = np.diag([1.0, 2.0, 3.0])
        out = regularised_lhs(H, lambda_k=2.0, B_eff=B)
        expected = np.eye(3) + 2.0 * B
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_does_not_mutate_input_hessian(self) -> None:
        H = np.array([[3.0, 0.0], [0.0, 3.0]])
        original = H.copy()
        regularised_lhs(H, lambda_k=1.0, B_eff=None)
        np.testing.assert_array_equal(H, original)


# ---------------------------------------------------------------------------
# Rank-1 Fisher factor for ``(1/p) ||u||^p`` problems
# ---------------------------------------------------------------------------


class _NleStub:
    """Tiny stand-in for :class:`NonlinearEquationsOracle` for unit tests."""

    def __init__(self, p: float, u: np.ndarray, J: np.ndarray) -> None:
        self.p = p
        self._u = u
        self._J = J

    def func_u(self, x: np.ndarray) -> np.ndarray:
        return self._u

    def jac_u(self, x: np.ndarray) -> np.ndarray:
        return self._J


class TestRankOneFisherFactor:
    def test_p_equals_two_vanishes(self) -> None:
        oracle = _NleStub(p=2.0, u=np.array([1.0, 1.0]), J=np.eye(2))
        g_template = np.zeros(2)
        alpha, g0 = rank_one_fisher_factor(oracle, np.zeros(2), g_template)
        assert alpha == 0.0
        np.testing.assert_array_equal(g0, np.zeros_like(g_template))

    def test_p_equals_four_returns_jt_u(self) -> None:
        # Shapes: J is d x n with d = 3 residuals, n = 2 parameters.
        # The helper computes ``J^T u`` so u must be length d = 3.
        u = np.array([3.0, 4.0, 0.0])  # ||u|| = 5
        J = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 3x2
        oracle = _NleStub(p=4.0, u=u, J=J)
        alpha, g0 = rank_one_fisher_factor(oracle, np.zeros(2), np.zeros(2))
        # alpha = (p - 2) * ||u||^(p - 4) = 2 * 5^0 = 2
        assert alpha == pytest.approx(2.0)
        np.testing.assert_allclose(g0, J.T @ u, atol=1e-12)
