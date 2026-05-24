"""Tests for the runner threading ``args.seed`` into AdaHessian's RNG.

AdaHessian's signature has its own ``seed: int = 0`` kwarg and
constructs ``np.random.default_rng(seed)`` internally -- it doesn't
share the global RNG state that ``set_global_seed`` pins.  The
runner therefore needs to thread the per-run ``args.seed`` into
``method_common["seed"]`` for AdaHessian (and any future method
that takes a ``seed`` kwarg) to actually vary across the
``--seeds 0 1 2 ...`` axis.

The contract pinned here:

* AdaHessian on two different ``--seeds`` produces different traces;
* AdaHessian on the same ``--seeds`` produces identical traces;
* a YAML ``seed: 42`` per-method override beats the runner's plumbing;
* methods without a ``seed`` kwarg (e.g. Newton) are unaffected.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from agns.cli import run_benchmark


def _new_args(seed: int) -> argparse.Namespace:
    return argparse.Namespace(seed=seed, methods=None, warnings=False)


def _adahessian_cfg(extra_method_kw: dict | None = None) -> dict:
    method: dict = {
        "name": "adahessian",
        "label": "AdaHessian",
        "n_iters": 8,
    }
    if extra_method_kw:
        method.update(extra_method_kw)
    return {
        "problem": {
            "type": "logistic_regression_synthetic",
            # Larger ``n`` so the Hutchinson probe (length-n Rademacher)
            # has enough room to actually move the trajectory between seeds.
            # Small enough that the test runs in well under a second.
            "params": {"m": 80, "n": 20, "reg": 1.0e-3},
        },
        "common": {"eps": 1.0e-10},
        "methods": [method],
    }


def _func_trace(save_dir: Path) -> list[float]:
    with open(save_dir / "adahessian_history.pkl", "rb") as fh:
        hist = pickle.load(fh)
    return [float(v) for v in hist["func"]]


# ---------------------------------------------------------------------------
# Main behaviour: probes vary across seeds, are deterministic given a seed.
# ---------------------------------------------------------------------------


def test_adahessian_traces_differ_across_seeds(tmp_path: Path) -> None:
    """Different ``--seeds`` values must produce different traces.

    Two different seeds should yield visibly different function-value
    traces.  Without the runner plumbing the Hutchinson RNG would
    sit at the registry default regardless of ``args.seed``, so
    traces would only differ because the *problem* differed -- not
    because the probes did.  With the plumbing in place,
    ``args.seed=0`` and ``args.seed=1`` produce different traces
    because both the problem data AND the Hutchinson probes vary.
    The probe-only contribution is isolated by the test immediately
    below.
    """
    cfg = _adahessian_cfg()
    save_dir_0 = tmp_path / "s0"
    save_dir_1 = tmp_path / "s1"

    run_benchmark.run_experiment(cfg, _new_args(seed=0), save_dir_0)
    run_benchmark.run_experiment(cfg, _new_args(seed=1), save_dir_1)

    trace_0 = _func_trace(save_dir_0)
    trace_1 = _func_trace(save_dir_1)
    assert trace_0 != trace_1, "AdaHessian traces did not vary across seeds"


def test_adahessian_probes_vary_with_seed_holding_problem_fixed(tmp_path: Path) -> None:
    """Isolate the probe contribution from data variance.

    The runner overwrites ``problem_params["seed"] = args.seed``
    whenever ``args.seed`` is given, so we can't trivially hold the
    problem data fixed by setting a YAML problem seed.  Instead we
    hold ``args.seed`` constant and vary the **method-side** seed
    via the YAML ``seed:`` override.  Same args.seed -> same data;
    different YAML seed -> different probes.  Pins the precedence
    contract: a YAML method-side seed varies the probes, and is
    NOT silently overridden by the runner's automatic plumbing.
    """
    cfg_a = _adahessian_cfg(extra_method_kw={"seed": 7})
    cfg_b = _adahessian_cfg(extra_method_kw={"seed": 42})

    save_a = tmp_path / "a"
    save_b = tmp_path / "b"
    run_benchmark.run_experiment(cfg_a, _new_args(seed=0), save_a)
    run_benchmark.run_experiment(cfg_b, _new_args(seed=0), save_b)

    trace_a = _func_trace(save_a)
    trace_b = _func_trace(save_b)
    # Same problem data (same args.seed -> same problem_params['seed']),
    # so f(x_0) matches; probes differ -> tails differ.
    assert trace_a[0] == trace_b[0]
    assert trace_a[1:] != trace_b[1:]


def test_adahessian_same_seed_gives_identical_traces(tmp_path: Path) -> None:
    """Determinism check: two identical-seed runs are bit-stable."""
    cfg = _adahessian_cfg()
    save_dir_a = tmp_path / "a"
    save_dir_b = tmp_path / "b"
    run_benchmark.run_experiment(cfg, _new_args(seed=3), save_dir_a)
    run_benchmark.run_experiment(cfg, _new_args(seed=3), save_dir_b)
    assert _func_trace(save_dir_a) == _func_trace(save_dir_b)


# ---------------------------------------------------------------------------
# Precedence: user-supplied seed in YAML beats the runner's automatic plumbing.
# ---------------------------------------------------------------------------


def test_yaml_seed_override_beats_runner_plumbing(tmp_path: Path) -> None:
    """When the YAML says ``seed: 42``, the runner does NOT overwrite it.

    Run twice at ``args.seed=0`` so the problem data is identical
    across both runs.  Run A omits the YAML method-seed (so the
    runner threads ``args.seed=0`` into the probe RNG); run B sets
    ``seed: 42`` in the YAML (so the override wins).  Probes must
    differ -> trace tails must differ.  A null fix where the YAML
    override is silently ignored would yield identical traces.
    """
    cfg_no_override = _adahessian_cfg()
    cfg_with_override = _adahessian_cfg(extra_method_kw={"seed": 42})

    save_no = tmp_path / "no_override"
    save_yes = tmp_path / "with_override"
    run_benchmark.run_experiment(cfg_no_override, _new_args(seed=0), save_no)
    run_benchmark.run_experiment(cfg_with_override, _new_args(seed=0), save_yes)

    trace_no = _func_trace(save_no)
    trace_yes = _func_trace(save_yes)
    assert trace_no[0] == trace_yes[0]  # same problem
    assert trace_no[1:] != trace_yes[1:], (
        "YAML 'seed: 42' override did not change the probe RNG; the runner's "
        "automatic plumbing leaked through and overrode the explicit value."
    )


# ---------------------------------------------------------------------------
# Methods without a seed kwarg are unaffected.
# ---------------------------------------------------------------------------


def test_seed_plumbing_does_not_leak_to_methods_without_seed_kwarg(tmp_path: Path) -> None:
    """Newton has no ``seed`` kwarg; the plumbing must not forward to it.

    An unguarded plumbing that set ``method_common["seed"]`` whenever
    ``args.seed`` was given would route ``seed`` through the
    registry's ``_OPTIONAL_KWARGS`` loop to every method, crashing
    Newton with ``TypeError: newton() got an unexpected keyword
    argument 'seed'``.  The gated plumbing
    (``"seed" in spec.default_kw``) restricts forwarding to methods
    that actually accept it.

    Verifies the negative: Newton runs to completion with no
    seed-related error and produces its history pickle.
    """
    cfg = {
        "problem": {
            "type": "logistic_regression_synthetic",
            "params": {"m": 40, "n": 6, "reg": 1.0e-2},
        },
        "common": {"eps": 1.0e-10},
        "methods": [{"name": "newton", "label": "Newton", "n_iters": 5}],
    }
    save_dir = tmp_path / "out"
    run_benchmark.run_experiment(cfg, _new_args(seed=99), save_dir)

    # Pickle exists -> Newton ran cleanly.  Failure JSON does NOT
    # exist -> no exception was caught and logged.
    assert (save_dir / "newton_history.pkl").is_file()
    assert not list(save_dir.glob("*_failure.json"))
