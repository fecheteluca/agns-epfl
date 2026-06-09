"""The production AGNS reproduces the frozen reference, including restart events.

``agns.methods.agns.agns`` is the object of study, so its numerics are locked here against
the independent, self-contained ``tests/_reference_agns.reference_agns``. We require the
whole trajectory (function value, gradient norm, solve count, iterates, per-step momentum
counter) to agree to ``1e-10`` AND the restart bookkeeping (events, totals) to be identical.
Cases span the convex log-sum-exp instance, a conditioned quadratic, and the non-convex
Rosenbrock valley (which forces the robust-Cholesky fallback the reference also implements),
plus all three restart schemes, so the lock covers the branches that matter.
"""

from __future__ import annotations

import numpy as np
import pytest

from agns.methods.agns import agns
from agns.oracles.rosenbrock import RosenbrockOracle
from tests._reference_agns import reference_agns
from tests.conftest import make_logsumexp_problem, make_quadratic_problem

_ATOL = 1e-10


def _case_logsumexp_gradient():
    p = make_logsumexp_problem(n=10, m=16)
    kw = dict(
        B=p.B,
        Binv=p.Binv,
        f_star=p.f_star,
        eps=p.eps,
        momentum_offset=3.0,
        restart_mode="gradient",
    )
    return p.oracle, p.x_0.copy(), kw


def _case_logsumexp_fvalue():
    p = make_logsumexp_problem(n=10, m=16)
    kw = dict(
        B=p.B,
        Binv=p.Binv,
        f_star=p.f_star,
        eps=p.eps,
        momentum_offset=1.0,
        restart_mode="function_value",
    )
    return p.oracle, p.x_0.copy(), kw


def _case_logsumexp_norestart():
    p = make_logsumexp_problem(n=10, m=16)
    kw = dict(
        B=p.B,
        Binv=p.Binv,
        f_star=p.f_star,
        eps=p.eps,
        momentum_offset=3.0,
        restart_mode="none",
    )
    return p.oracle, p.x_0.copy(), kw


def _case_quadratic_gradient():
    q = make_quadratic_problem(n=8, cond=50.0)
    kw = dict(f_star=q.f_star, eps=q.eps, momentum_offset=1.0, restart_mode="gradient")
    return q.oracle, q.x_0.copy(), kw


def _case_rosenbrock_gradient():
    kw = dict(f_star=0.0, eps=1e-8, momentum_offset=1.0, restart_mode="gradient")
    return RosenbrockOracle(a=1.0, b=100.0), np.array([-1.2, 1.0]), kw


_CASES = {
    "logsumexp_gradient": _case_logsumexp_gradient,
    "logsumexp_function_value": _case_logsumexp_fvalue,
    "logsumexp_norestart": _case_logsumexp_norestart,
    "quadratic_gradient": _case_quadratic_gradient,
    "rosenbrock_gradient": _case_rosenbrock_gradient,
}

_N_ITERS = 60


@pytest.mark.parametrize("case_name", list(_CASES))
def test_agns_matches_reference_trajectory(case_name):
    """Function value, gradient norm, solve count and iterates match the reference."""
    oracle, x0, kw = _CASES[case_name]()
    _, _, h = agns(oracle, x0.copy(), n_iters=_N_ITERS, **kw)
    oracle_ref, x0_ref, kw_ref = _CASES[case_name]()
    _, _, h_ref = reference_agns(oracle_ref, x0_ref.copy(), n_iters=_N_ITERS, **kw_ref)

    for key in ["func", "grad_norm", "matrix_inverses", "local_k"]:
        a = np.asarray(h[key], dtype=float)
        b = np.asarray(h_ref[key], dtype=float)
        assert a.shape == b.shape, f"{key}: {a.shape} vs {b.shape}"
        np.testing.assert_allclose(a, b, rtol=0.0, atol=_ATOL, err_msg=key)
    np.testing.assert_allclose(
        np.asarray(h["x_k"]), np.asarray(h_ref["x_k"]), rtol=0.0, atol=_ATOL
    )


@pytest.mark.parametrize("case_name", list(_CASES))
def test_agns_matches_reference_restart_events(case_name):
    """Restart events and totals are identical between production and reference."""
    oracle, x0, kw = _CASES[case_name]()
    _, _, h = agns(oracle, x0.copy(), n_iters=_N_ITERS, **kw)
    oracle_ref, x0_ref, kw_ref = _CASES[case_name]()
    _, _, h_ref = reference_agns(oracle_ref, x0_ref.copy(), n_iters=_N_ITERS, **kw_ref)

    assert h["restart_events"] == h_ref["restart_events"]
    assert h["n_restarts_total"][0] == h_ref["n_restarts_total"][0]
    assert h["restart_mode"][0] == h_ref["restart_mode"][0]


def test_reference_actually_exercises_restarts():
    """Guard: the restart-bearing cases really do restart (the lock would be vacuous otherwise)."""
    counts = {}
    for name in ["logsumexp_gradient", "quadratic_gradient", "rosenbrock_gradient"]:
        oracle, x0, kw = _CASES[name]()
        _, _, h_ref = reference_agns(oracle, x0.copy(), n_iters=_N_ITERS, **kw)
        counts[name] = h_ref["n_restarts_total"][0]
    assert all(c >= 1 for c in counts.values()), counts
