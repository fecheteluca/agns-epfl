"""Per-method smoke + correctness tests.

For every entry in :data:`agns.registry.METHOD_REGISTRY`:

1. **Smoke.**  Runs for a handful of iterations on a small strongly
   convex problem; verifies the canonical 3-tuple shape and the trace
   keys.
2. **Correctness.**  Verifies the method makes monotone progress in the
   function value: ``func[-1] <= func[0]`` (a weak invariant that
   catches sign errors, broken stopping criteria, and reversed update
   directions without depending on per-method rates).

Methods that need bespoke setup (``agns_sc`` needs ``mu``; AdaHessian
needs ``hess_vec``) are handled with parametric overrides below.
"""

from __future__ import annotations

import numpy as np
import pytest

from agns.registry import METHOD_REGISTRY, run_method

_REQUIRED_TRACE_KEYS = {"func", "time"}

# Methods that require a rank-1 ``approx_hess_fn`` and an oracle
# exposing ``(p, func_u, jac_u)``.  Covered by ``test_wsm_backend_runs``.
_WSM_METHODS = {"gns_wsm", "agns_practice_wsm"}
_GENERIC_METHODS = sorted(set(METHOD_REGISTRY) - _WSM_METHODS)

# Per-method overrides for the small-problem run.  ``cfg_extra`` adds
# extra entries to the generic cfg dict.
_PER_METHOD_CFG: dict[str, dict[str, object]] = {
    "agns_sc": {"mu": 1e-2, "L_2": 1.0, "n_segments": 3, "K_seg": 5},
    # Heavy-ball: smaller initial step so the line search does not bail.
    "heavy_ball": {"alpha": 0.1, "beta": 0.5},
    # Cubic / accelerated cubic: lower H_0 / M_0 reduces wasted inner probes.
    "cubic_newton": {"H_0": 1.0},
    "super_newton": {"H_0": 1.0},
    "accelerated_cubic_newton": {"M_0": 1.0},
    "monteiro_svaiter_acn": {"M_0": 1.0},
    # Adam / AdaHessian: higher lr suits the small ridge problem.
    "adam": {"lr": 0.1},
    "adahessian": {"lr": 0.5, "n_hutchinson_probes": 4, "seed": 0},
    # Newton: shorter iter budget; default is already permissive.
}


def _cfg_for(method_key: str, B: np.ndarray, Binv: np.ndarray, f_star: float) -> dict[str, object]:
    cfg: dict[str, object] = {
        "n_iters": 10,
        "gamma_0": 1.0,
        "eps": 1e-14,
        "B": B,
        "Binv": Binv,
        "f_star": f_star,
        "adaptive_search": True,
    }
    cfg.update(_PER_METHOD_CFG.get(method_key, {}))
    return cfg


# ---------------------------------------------------------------------------
# Smoke: every method returns the canonical tuple.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_key", _GENERIC_METHODS)
def test_method_returns_canonical_tuple(method_key: str, ridge_problem) -> None:
    oracle, x_0, f_star, B, Binv = ridge_problem
    spec = METHOD_REGISTRY[method_key]
    cfg = _cfg_for(method_key, B, Binv, f_star)
    x_final, status, hist = run_method(spec, oracle, x_0, cfg, oracle, None)
    assert isinstance(x_final, np.ndarray)
    assert x_final.shape == x_0.shape
    assert isinstance(status, str) and status != ""
    assert hist is not None
    assert isinstance(hist, dict)
    assert set(hist) >= _REQUIRED_TRACE_KEYS, (
        f"{method_key}: missing trace keys {_REQUIRED_TRACE_KEYS - set(hist)}"
    )


# ---------------------------------------------------------------------------
# Correctness: each method makes progress on the convex problem.
# ---------------------------------------------------------------------------


_METHODS_THAT_DESCEND = sorted(
    set(_GENERIC_METHODS)
    # Adam / AdaHessian have no monotone-descent guarantee from one
    # warm-start iterate.  Covered by their own dedicated smoke tests.
    - {"adam", "adahessian"}
)


@pytest.mark.parametrize("method_key", _METHODS_THAT_DESCEND)
def test_method_makes_progress(method_key: str, ridge_problem) -> None:
    oracle, x_0, f_star, B, Binv = ridge_problem
    spec = METHOD_REGISTRY[method_key]
    cfg = _cfg_for(method_key, B, Binv, f_star)
    cfg["n_iters"] = 30
    _x_final, _status, hist = run_method(spec, oracle, x_0, cfg, oracle, None)
    assert hist is not None
    f0 = float(hist["func"][0])
    f_last = float(hist["func"][-1])
    assert f_last <= f0 + 1e-12, f"{method_key}: f_last={f_last:.4e} > f_0={f0:.4e}"


def test_adam_runs_and_traces(ridge_problem) -> None:
    oracle, x_0, f_star, B, Binv = ridge_problem
    spec = METHOD_REGISTRY["adam"]
    cfg = _cfg_for("adam", B, Binv, f_star)
    cfg["n_iters"] = 50
    _x, _status, hist = run_method(spec, oracle, x_0, cfg, oracle, None)
    assert hist is not None
    assert len(hist["func"]) >= 2


def test_adahessian_runs_and_traces(ridge_problem) -> None:
    oracle, x_0, f_star, B, Binv = ridge_problem
    spec = METHOD_REGISTRY["adahessian"]
    cfg = _cfg_for("adahessian", B, Binv, f_star)
    cfg["n_iters"] = 50
    _x, _status, hist = run_method(spec, oracle, x_0, cfg, oracle, None)
    assert hist is not None
    assert len(hist["func"]) >= 2


# ---------------------------------------------------------------------------
# Specific method invariants.
# ---------------------------------------------------------------------------


def test_gns_inexact_uses_approx_oracle(nle_problem) -> None:
    """The inexact variant must call ``approx_hess_fn`` rather than ``hess``."""
    from agns.approximations import approx_hess_fn_fisher_term

    oracle, x_0, f_star, B, Binv = nle_problem
    spec = METHOD_REGISTRY["gns_inexact"]
    cfg = _cfg_for("gns_inexact", B, Binv, f_star)
    cfg["n_iters"] = 5
    _x_final, _status, hist = run_method(
        spec,
        oracle,
        x_0,
        cfg,
        oracle,
        approx_hess_fn_fisher_term,
    )
    assert hist is not None
    assert hist["func"][-1] <= hist["func"][0] + 1e-12


def test_wsm_backend_runs(nle_problem) -> None:
    """gns_wsm should run with the WSM backend on a rank-1 Fisher problem."""
    from agns.approximations import approx_hess_fn_fisher_term

    oracle, x_0, f_star, B, Binv = nle_problem
    spec = METHOD_REGISTRY["gns_wsm"]
    cfg = _cfg_for("gns_wsm", B, Binv, f_star)
    cfg["n_iters"] = 5
    _x, _status, hist = run_method(spec, oracle, x_0, cfg, oracle, approx_hess_fn_fisher_term)
    assert hist is not None
    assert hist["func"][-1] <= hist["func"][0] + 1e-12


def test_agns_holder_records_nu(ridge_problem) -> None:
    oracle, x_0, f_star, B, Binv = ridge_problem
    spec = METHOD_REGISTRY["agns_holder"]
    cfg = _cfg_for("agns_holder", B, Binv, f_star)
    cfg["nu"] = 0.5
    cfg["n_iters"] = 3
    _x, _status, hist = run_method(spec, oracle, x_0, cfg, oracle, None)
    assert hist is not None
    assert hist["nu"][-1] == 0.5


def test_agns_inexact_records_delta_budget(ridge_problem) -> None:
    oracle, x_0, f_star, B, Binv = ridge_problem
    spec = METHOD_REGISTRY["agns_inexact"]
    cfg = _cfg_for("agns_inexact", B, Binv, f_star)
    cfg["delta_budget"] = 1e-3
    cfg["n_iters"] = 3
    _x, _status, hist = run_method(spec, oracle, x_0, cfg, oracle, None)
    assert hist is not None
    assert hist["delta_budget"][-1] == 1e-3


def test_agns_sc_records_segment_boundaries(ridge_problem) -> None:
    oracle, x_0, f_star, B, Binv = ridge_problem
    spec = METHOD_REGISTRY["agns_sc"]
    cfg = _cfg_for("agns_sc", B, Binv, f_star)
    _x, _status, hist = run_method(spec, oracle, x_0, cfg, oracle, None)
    assert hist is not None
    assert "segment_boundaries" in hist
    # The first boundary is at index 0; further boundaries grow.
    import itertools

    bounds = hist["segment_boundaries"]
    assert bounds[0] == 0
    assert all(b1 >= b0 for b0, b1 in itertools.pairwise(bounds))


# ---------------------------------------------------------------------------
# Trace consistency: grad_norm is non-negative wherever it appears.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_key", _GENERIC_METHODS)
def test_grad_norm_nonnegative_when_traced(method_key: str, ridge_problem) -> None:
    oracle, x_0, f_star, B, Binv = ridge_problem
    spec = METHOD_REGISTRY[method_key]
    cfg = _cfg_for(method_key, B, Binv, f_star)
    _x, _status, hist = run_method(spec, oracle, x_0, cfg, oracle, None)
    assert hist is not None
    if hist.get("grad_norm"):
        arr = np.asarray(hist["grad_norm"], dtype=float)
        assert np.all(arr >= 0.0), f"{method_key}: grad_norm has negative entries"
