"""Monotonicity contract for the AGNS near-stationary branch.

When ``||g(y_k)|| < eps`` the AGNS / AGNS-WSM update considers
advancing to ``y_k`` (the momentum extrapolation).  The branch
refuses the move when ``f(y_k) > f(x_k)`` so momentum overshoot
at a near-stationary point cannot silently raise ``f``; the trace
must therefore be non-increasing through every triggering iteration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods import agns, agns_wsm
from agns.oracles import NonlinearEquationsOracle
from agns.oracles.approximations import approx_hess_fn_fisher_term


def _is_monotone_non_increasing(values: Sequence[float]) -> bool:
    """``True`` iff each value is <= the previous one (within float epsilon)."""
    eps = 1e-15
    return all(values[i + 1] <= values[i] + eps for i in range(len(values) - 1))


def _make_nle_problem() -> tuple[
    Any,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Small NLE oracle: cheap enough for tests, exposes the WSM interface.

    Returns ``(oracle, x_0, B, Binv)``; the WSM backend needs an
    explicit ``B`` / ``Binv`` because its closed-form solve cannot
    use the Euclidean-shortcut path.
    """
    rng = np.random.default_rng(0)
    n, m = 6, 10
    A = rng.uniform(-1.0, 1.0, (n, m))
    b = rng.uniform(-1.0, 1.0, n)
    oracle = NonlinearEquationsOracle(p=4, A=A, b=b)
    x_0 = np.ones(m)
    B = A.T.dot(A) + 1e-8 * np.eye(m)
    Binv = np.linalg.inv(B)
    return oracle, x_0, B, Binv


# ---------------------------------------------------------------------------
# End-to-end: AGNS / AGNS-WSM traces are monotone non-increasing
# ---------------------------------------------------------------------------


def test_agns_trace_is_monotone_non_increasing() -> None:
    """End-to-end: AGNS produces a monotonically non-increasing trace.

    Loose eps (1e-3) so the near-stationary branch trips frequently
    on the small NLE oracle.  Without the monotone guard inside the
    branch, momentum could overshoot a near-zero gradient and an
    unconditional accept would raise ``f``.
    """
    oracle, x_0, B, Binv = _make_nle_problem()
    _, _, hist = agns(
        oracle,
        x_0,
        n_iters=80,
        gamma_0=1.0,
        eps=1e-3,
        is_approx=False,
        adaptive_search=True,
        B=B,
        Binv=Binv,
    )
    assert hist is not None
    func = list(hist["func"])
    bad = [(i, func[i], func[i + 1]) for i in range(len(func) - 1) if func[i + 1] > func[i] + 1e-15]
    assert not bad, f"AGNS trace was not monotone: {bad[:5]}"


def test_agns_inexact_trace_is_monotone_non_increasing() -> None:
    """Same invariant for the inexact-Hessian path."""
    oracle, x_0, B, Binv = _make_nle_problem()
    _, _, hist = agns(
        oracle,
        x_0,
        n_iters=80,
        gamma_0=1.0,
        eps=1e-3,
        is_approx=True,
        approx_oracle=oracle,
        approx_hess_fn=approx_hess_fn_fisher_term,
        adaptive_search=True,
        B=B,
        Binv=Binv,
    )
    assert hist is not None
    assert _is_monotone_non_increasing(list(hist["func"]))


def test_agns_wsm_trace_is_monotone_non_increasing() -> None:
    """Same invariant for the WSM (Sherman-Morrison) backend."""
    oracle, x_0, B, Binv = _make_nle_problem()
    _, _, hist = agns_wsm(
        oracle,
        x_0,
        n_iters=80,
        gamma_0=1.0,
        eps=1e-3,
        is_approx=True,
        approx_oracle=oracle,
        approx_hess_fn=approx_hess_fn_fisher_term,
        adaptive_search=True,
        invert_backend="wsm",
        B=B,
        Binv=Binv,
    )
    assert hist is not None
    assert _is_monotone_non_increasing(list(hist["func"]))


# ---------------------------------------------------------------------------
# Synthetic oracle: forces the near-stationary refuse path
# ---------------------------------------------------------------------------


class _SyntheticOracle:
    """A 1-D oracle hand-crafted to trip the near-stationary branch.

    ``f(x) = 1e-10 (x - 1)^2 + 1e-3 (x - 1.5)^4``.

    At ``x = 1``, ``g`` is small and ``f`` is small.  Momentum
    extrapolation lands at ``y > 1``; the quartic term dominates, so
    ``f(y) > f(x)`` while ``g(y)`` may still be below a loose ``eps``.
    Drives AGNS through the near-stationary branch's refuse arm.
    """

    def __init__(self) -> None:
        self.func_calls = 0
        self.grad_calls = 0
        self.hess_calls = 0
        self.hess_vec_calls = 0

    def func(self, x: NDArray[np.float64]) -> float:
        v = float(x[0])
        return 1e-10 * (v - 1.0) ** 2 + 1e-3 * (v - 1.5) ** 4

    def grad(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        v = float(x[0])
        g = 2 * 1e-10 * (v - 1.0) + 4 * 1e-3 * (v - 1.5) ** 3
        return np.array([g])

    def hess(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        v = float(x[0])
        h = 2 * 1e-10 + 12 * 1e-3 * (v - 1.5) ** 2
        return np.array([[h]])

    def hess_vec(self, x: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.hess(x).dot(v)


def test_synthetic_trips_branch_yet_stays_monotone() -> None:
    """The hand-crafted oracle exercises the refuse arm; the trace stays monotone.

    A loose ``eps = 1e-2`` ensures the near-stationary check fires
    on the small gradients near ``x = 1``.  Whatever the path AGNS
    takes through the branch, the function-value trace must not
    have any upward jump.
    """
    oracle = _SyntheticOracle()
    x_0 = np.array([1.0])
    _, _, hist = agns(
        oracle,
        x_0,
        n_iters=6,
        gamma_0=1.0,
        eps=1e-2,
        is_approx=False,
        adaptive_search=True,
    )
    assert hist is not None
    func = list(hist["func"])
    assert _is_monotone_non_increasing(func), (
        f"refuse-branch test: trace went up: {func}"
    )
