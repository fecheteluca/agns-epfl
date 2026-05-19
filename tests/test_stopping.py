"""Tests for :mod:`agns.stopping`."""

from __future__ import annotations

import pytest

from agns.stopping import Status, StopDecision, check_stop, success_status


def test_success_status_format() -> None:
    assert success_status(0) == "success, 0 iters"
    assert success_status(42) == "success, 42 iters"


def test_status_enum_str_compatibility() -> None:
    assert Status.ITERATIONS_EXCEEDED == "iterations_exceeded"
    assert Status.DIVERGED_NONFINITE == "diverged_nonfinite"


def test_stop_decision_should_stop_property() -> None:
    assert StopDecision(False).should_stop is False
    assert StopDecision(True, "x").should_stop is True


def test_check_stop_f_star_gap() -> None:
    d = check_stop(k=3, n_iters=100, f_k=1.0, f_star=0.999, eps=1e-2)
    assert d.stop and d.status == "success, 3 iters"


def test_check_stop_grad_tol() -> None:
    d = check_stop(
        k=5, n_iters=100, f_k=1.0, f_star=None, eps=1e-8,
        g_norm=1e-9, grad_tol=1e-8,
    )
    assert d.stop and d.status == "success, 5 iters"


def test_check_stop_iteration_budget() -> None:
    d = check_stop(k=100, n_iters=100, f_k=1.0, f_star=None, eps=1e-8)
    assert d.stop and d.status == "iterations_exceeded"


def test_check_stop_no_trigger() -> None:
    d = check_stop(
        k=10, n_iters=100, f_k=1.0, f_star=0.0, eps=1e-8,
        g_norm=1e-3, grad_tol=1e-8,
    )
    assert not d.stop and d.status == ""


def test_check_stop_skips_when_none() -> None:
    # No f_star -> no function-gap check.
    d = check_stop(k=10, n_iters=100, f_k=1.0, f_star=None, eps=1e-8)
    assert not d.stop
    # No g_norm/grad_tol -> no gradient check.
    d = check_stop(k=10, n_iters=100, f_k=1.0, f_star=None, eps=1e-8, g_norm=1e-9)
    assert not d.stop


@pytest.mark.parametrize("k", [99, 1_000_000])
def test_check_stop_exceeds_budget(k: int) -> None:
    d = check_stop(k=k, n_iters=99, f_k=1.0, f_star=None, eps=1e-8)
    assert d.stop and d.status == "iterations_exceeded"
