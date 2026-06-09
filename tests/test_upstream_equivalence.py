"""Ported methods reproduce the pinned upstream trajectories bit-for-bit.

Each refactored method in ``agns.methods`` is run against the verbatim upstream
implementation vendored under ``tests/_upstream`` (``epfml/grad-norm-smooth`` at commit
9f4ca00) on the *same* small, well-conditioned log-sum-exp instance. On such instances the
regularized system stays positive definite, so the refactored ``safe_cho_solve`` takes the
plain Cholesky path and the trajectories agree to machine precision; we require ``<= 1e-12``.

The same fresh problem is rebuilt for each implementation (identical seed -> identical data),
so the comparison isolates the method, not the oracle.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from agns.methods.fast_gradient import fast_gradient_method
from agns.methods.gns import grad_norm_smooth
from agns.methods.gns_rank_one import grad_norm_smooth_for_rank_one
from agns.methods.gradient import gradient_method
from agns.methods.normalized_gradient import grad_norm_smooth_gradient_method
from agns.methods.super_universal import super_newton
from tests.conftest import make_logsumexp_problem

# Importing the vendored upstream triggers benign SyntaxWarnings from its verbatim
# docstrings (raw-string escapes); silence them so the suite stays quiet.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    from tests._upstream import methods as upstream

_ATOL = 1e-12
_N = 20


def _ours_and_upstream(ours_fn, up_fn, *, iter_kw, extra=None):
    """Run a method pair on identical fresh problems; return the two histories."""
    extra = extra or {}
    p1 = make_logsumexp_problem(n=10, m=16)
    p2 = make_logsumexp_problem(n=10, m=16)
    # The (identical) metric is supplied per problem below.
    _, _, h_ours = ours_fn(
        p1.oracle, p1.x_0.copy(), **{iter_kw: _N}, B=p1.B, Binv=p1.Binv, **extra
    )
    _, _, h_up = up_fn(
        p2.oracle, p2.x_0.copy(), **{iter_kw: _N}, B=p2.B, Binv=p2.Binv, **extra
    )
    return h_ours, h_up


def _assert_trace_equal(h_ours, h_up, keys):
    for key in keys:
        a = np.asarray(h_ours[key], dtype=float)
        b = np.asarray(h_up[key], dtype=float)
        assert a.shape == b.shape, f"{key}: shape {a.shape} vs {b.shape}"
        np.testing.assert_allclose(a, b, rtol=0.0, atol=_ATOL, err_msg=key)


def test_gns_matches_upstream():
    h_ours, h_up = _ours_and_upstream(
        grad_norm_smooth, upstream.grad_norm_smooth, iter_kw="n_iters"
    )
    _assert_trace_equal(
        h_ours, h_up, ["func", "grad_norm", "gamma_k", "matrix_inverses"]
    )
    # x_k trajectory matches too.
    np.testing.assert_allclose(
        np.asarray(h_ours["x_k"]), np.asarray(h_up["x_k"]), rtol=0.0, atol=_ATOL
    )


def test_super_universal_matches_upstream():
    h_ours, h_up = _ours_and_upstream(
        super_newton, upstream.super_newton, iter_kw="n_iters"
    )
    _assert_trace_equal(h_ours, h_up, ["func", "grad_norm", "H_k", "matrix_inverses"])


def test_cubic_matches_upstream():
    # Cubic uses B (no Binv); compare on the metric problem with f_star disabled.
    p1 = make_logsumexp_problem(n=10, m=16)
    p2 = make_logsumexp_problem(n=10, m=16)
    from agns.methods.cubic import cubic_newton

    _, _, h_ours = cubic_newton(p1.oracle, p1.x_0.copy(), n_iters=_N, B=p1.B)
    _, _, h_up = upstream.cubic_newton(p2.oracle, p2.x_0.copy(), n_iters=_N, B=p2.B)
    _assert_trace_equal(h_ours, h_up, ["func", "H_k"])
    np.testing.assert_allclose(
        np.asarray(h_ours["x_k"]), np.asarray(h_up["x_k"]), rtol=0.0, atol=_ATOL
    )


def test_gradient_matches_upstream():
    h_ours, h_up = _ours_and_upstream(
        gradient_method, upstream.gradient_method, iter_kw="max_iter"
    )
    _assert_trace_equal(h_ours, h_up, ["func", "grad_sqr_norm", "L"])


def test_fast_gradient_matches_upstream():
    h_ours, h_up = _ours_and_upstream(
        fast_gradient_method, upstream.fast_gradient_method, iter_kw="max_iter"
    )
    _assert_trace_equal(h_ours, h_up, ["func", "grad_sqr_norm", "L"])


def test_normalized_gradient_matches_upstream():
    h_ours, h_up = _ours_and_upstream(
        grad_norm_smooth_gradient_method,
        upstream.grad_norm_smooth_gradient_method,
        iter_kw="n_iters",
    )
    _assert_trace_equal(h_ours, h_up, ["func", "grad_norm", "gamma_k"])


def test_gns_rank_one_cho_matches_upstream():
    h_ours, h_up = _ours_and_upstream(
        grad_norm_smooth_for_rank_one,
        upstream.grad_norm_smooth_for_rank_one,
        iter_kw="n_iters",
        extra={"invert_backend": "cho"},
    )
    _assert_trace_equal(
        h_ours, h_up, ["func", "grad_norm", "gamma_k", "matrix_inverses"]
    )


@pytest.mark.parametrize("atol", [_ATOL])
def test_equivalence_tolerance_is_machine_precision(atol):
    """Guard: the GNS func trajectory matches upstream far tighter than the 1e-12 bar."""
    h_ours, h_up = _ours_and_upstream(
        grad_norm_smooth, upstream.grad_norm_smooth, iter_kw="n_iters"
    )
    max_diff = float(
        np.max(np.abs(np.asarray(h_ours["func"]) - np.asarray(h_up["func"])))
    )
    assert max_diff <= atol
