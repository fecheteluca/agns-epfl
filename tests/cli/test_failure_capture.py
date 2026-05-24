"""Tests for the runner's structured failure capture.

Contract pinned by these tests:

  - a failed method writes ``<save_dir>/<mkey>_failure.json`` carrying
    exception type / message / traceback / wall_time;
  - ``summary.json`` gains a parallel ``failures`` block summarising
    every failed method on that seed;
  - the aggregator reads both, surfaces failures via a per-method
    ``failures`` list, and **keeps methods with zero successes** in
    the aggregated record so the downstream table can render
    "failed" instead of dropping the row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from agns.cli import run_benchmark
from agns.cli.aggregate import aggregate
from agns.pipeline.registry import METHOD_REGISTRY, MethodSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _boom(*_args: object, **_kwargs: object) -> None:
    """Synthetic failing method used in monkeypatched registry entries."""
    raise ValueError("synthetic failure for test")


def _new_args(seed: int, *, warnings: bool = False) -> argparse.Namespace:
    """Build the minimal ``argparse.Namespace`` ``run_experiment`` expects."""
    return argparse.Namespace(seed=seed, methods=None, warnings=warnings)


def _tiny_cfg(method_names: list[str]) -> dict[str, object]:
    """One synthetic-LR problem + the given methods (defaults pulled from the registry)."""
    return {
        "problem": {
            "type": "logistic_regression_synthetic",
            "params": {"m": 30, "n": 3, "reg": 1.0e-2},
        },
        "common": {"gamma_0": 1.0, "eps": 1.0e-10, "adaptive_search": True},
        "methods": [
            {"name": name, "label": f"label-{name}", "n_iters": 3}
            for name in method_names
        ],
    }


# ---------------------------------------------------------------------------
# Runner-level capture
# ---------------------------------------------------------------------------


def test_failure_json_written_when_method_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising method must produce ``<mkey>_failure.json`` and NO pickle."""
    monkeypatch.setitem(METHOD_REGISTRY, "_test_boom", MethodSpec(run_fn=_boom))

    cfg = _tiny_cfg(["_test_boom", "newton"])
    save_dir = tmp_path / "seed_0"
    run_benchmark.run_experiment(cfg, _new_args(seed=0), save_dir)

    failure_path = save_dir / "_test_boom_failure.json"
    assert failure_path.is_file(), "failure JSON was not written"

    data = json.loads(failure_path.read_text())
    assert data["method_key"] == "_test_boom"
    assert data["method_label"] == "label-_test_boom"
    assert data["exception_type"] == "ValueError"
    assert "synthetic failure" in data["message"]
    assert "Traceback" in data["traceback"]
    assert data["wall_time_s_before_failure"] >= 0.0
    assert data["schema_version"] == 1

    # Working method still produces its history pickle.
    assert (save_dir / "newton_history.pkl").is_file()

    # CRITICAL invariant: pickle for the failed method MUST NOT exist
    # (the aggregator relies on "pickle exists <-> method succeeded").
    assert not (save_dir / "_test_boom_history.pkl").exists()


def test_successful_methods_do_not_write_failure_json(tmp_path: Path) -> None:
    """A clean run should produce no ``*_failure.json`` files at all."""
    cfg = _tiny_cfg(["newton"])
    save_dir = tmp_path / "seed_0"
    run_benchmark.run_experiment(cfg, _new_args(seed=0), save_dir)

    assert (save_dir / "newton_history.pkl").is_file()
    assert list(save_dir.glob("*_failure.json")) == []


def test_summary_json_carries_failures_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``summary.json["failures"]`` mirrors the on-disk failure JSONs."""
    monkeypatch.setitem(METHOD_REGISTRY, "_test_boom", MethodSpec(run_fn=_boom))

    cfg = _tiny_cfg(["_test_boom", "newton"])
    save_dir = tmp_path / "seed_0"
    run_benchmark.run_experiment(cfg, _new_args(seed=0), save_dir)

    summ = json.loads((save_dir / "summary.json").read_text())
    assert "failures" in summ
    assert "_test_boom" in summ["failures"]
    fail = summ["failures"]["_test_boom"]
    assert fail["exception_type"] == "ValueError"
    assert "synthetic failure" in fail["message"]
    # Working method still listed in the methods block.
    assert "newton" in summ["methods"]


def test_summary_written_even_when_every_method_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Previously: all-fail seeds wrote nothing.  Now: summary.json exists with failures."""
    monkeypatch.setitem(METHOD_REGISTRY, "_test_boom1", MethodSpec(run_fn=_boom))
    monkeypatch.setitem(METHOD_REGISTRY, "_test_boom2", MethodSpec(run_fn=_boom))

    cfg = _tiny_cfg(["_test_boom1", "_test_boom2"])
    save_dir = tmp_path / "seed_0"
    run_benchmark.run_experiment(cfg, _new_args(seed=0), save_dir)

    summary_path = save_dir / "summary.json"
    assert summary_path.is_file(), "summary.json must be written even for all-fail seeds"
    summ = json.loads(summary_path.read_text())
    assert summ["methods"] == {}
    assert set(summ["failures"].keys()) == {"_test_boom1", "_test_boom2"}


# ---------------------------------------------------------------------------
# Aggregator-level surfacing
# ---------------------------------------------------------------------------


def test_aggregator_records_failures_in_method_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aggregator pulls failures into the per-method record + campaign summary."""
    monkeypatch.setitem(METHOD_REGISTRY, "_test_boom", MethodSpec(run_fn=_boom))

    raw_dir = tmp_path / "raw"
    cfg = _tiny_cfg(["_test_boom", "newton"])
    # Two seeds, both failing for _test_boom and succeeding for newton.
    for seed in (0, 1):
        run_benchmark.run_experiment(cfg, _new_args(seed=seed), raw_dir / f"seed_{seed}")

    agg_path = tmp_path / "agg.json"
    aggregate(
        campaign="tiny",
        raw_dir=raw_dir,
        output=agg_path,
        eps=1e-8,
        rate_tail=0.5,
        reference_dir=tmp_path / "nocache",
        use_reference_cache=False,
    )
    agg = json.loads(agg_path.read_text())

    # Per-method record carries the failures list.
    assert "_test_boom" in agg["methods"]
    boom_rec = agg["methods"]["_test_boom"]
    assert boom_rec["n_seeds_succeeded"] == 0
    assert boom_rec["n_seeds_failed"] == 2
    assert boom_rec["failure_rate"] == 1.0
    assert len(boom_rec["failures"]) == 2
    assert {f["seed"] for f in boom_rec["failures"]} == {"seed_0", "seed_1"}
    for f in boom_rec["failures"]:
        assert f["exception_type"] == "ValueError"
        assert "synthetic failure" in f["message"]

    # Working method has an empty failures field.
    assert agg["methods"]["newton"]["failures"] == []

    # Campaign-level failures_summary lists boom but not newton.
    assert "_test_boom" in agg["failures_summary"]
    assert "newton" not in agg["failures_summary"]
    boom_sum = agg["failures_summary"]["_test_boom"]
    assert boom_sum["n_failed_seeds"] == 2
    assert boom_sum["modal_exception_type"] == "ValueError"


def test_aggregator_keeps_failure_only_method_in_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure-only methods must NOT be silently dropped from the aggregated record."""
    monkeypatch.setitem(METHOD_REGISTRY, "_test_boom", MethodSpec(run_fn=_boom))

    raw_dir = tmp_path / "raw"
    # Only the failing method -- no other methods cushion the per-seed argmin.
    # Pair it with one working method so the per-seed-min fallback has something
    # to chew on (otherwise the aggregator's residual machinery has no anchor).
    cfg = _tiny_cfg(["_test_boom", "newton"])
    for seed in (0, 1, 2):
        run_benchmark.run_experiment(cfg, _new_args(seed=seed), raw_dir / f"seed_{seed}")

    agg_path = tmp_path / "agg.json"
    aggregate(
        campaign="tiny",
        raw_dir=raw_dir,
        output=agg_path,
        eps=1e-8,
        rate_tail=0.5,
        reference_dir=tmp_path / "nocache",
        use_reference_cache=False,
    )
    agg = json.loads(agg_path.read_text())

    rec = agg["methods"]["_test_boom"]
    # Minimal record: no curves, no rate fit, but failures and counts present.
    assert rec["n_seeds_succeeded"] == 0
    assert "iter_curve" not in rec
    assert "rate_fit" not in rec
    # Label pulled from the failure record (no summary.methods entry to draw from).
    assert rec["label"] == "label-_test_boom"


def test_aggregator_label_for_method_failing_on_subset_of_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When some seeds succeed and others fail, label comes from a success seed."""
    raises = {"on": True}

    def sometimes_boom(
        oracle: object,
        x_0: NDArray[np.float64],
        **kwargs: object,
    ) -> tuple[NDArray[np.float64], str, dict[str, list[float]]]:
        # Fails on the first seed only.  Subsequent seeds succeed and
        # return a tiny synthetic history -- enough for the aggregator
        # to consider the method "succeeded on some seeds".
        if raises["on"]:
            raises["on"] = False
            raise RuntimeError("first-seed failure")
        return (
            np.asarray(x_0, dtype=float),
            "success, 0 iters",
            {
                "func": [0.0, 0.0],
                "time": [0.0, 0.0],
                "grad_calls": [0, 0],
                "grad_norm": [0.0, 0.0],
            },
        )

    monkeypatch.setitem(METHOD_REGISTRY, "_test_mixed", MethodSpec(run_fn=sometimes_boom))

    raw_dir = tmp_path / "raw"
    cfg = _tiny_cfg(["_test_mixed", "newton"])
    for seed in (0, 1):
        run_benchmark.run_experiment(cfg, _new_args(seed=seed), raw_dir / f"seed_{seed}")

    agg_path = tmp_path / "agg.json"
    aggregate(
        campaign="tiny",
        raw_dir=raw_dir,
        output=agg_path,
        eps=1e-8,
        rate_tail=0.5,
        reference_dir=tmp_path / "nocache",
        use_reference_cache=False,
    )
    agg = json.loads(agg_path.read_text())

    rec = agg["methods"]["_test_mixed"]
    assert rec["n_seeds_succeeded"] == 1
    assert rec["n_seeds_failed"] == 1
    assert len(rec["failures"]) == 1
    # Label picked up from the success-seed's summary.methods entry.
    assert rec["label"] == "label-_test_mixed"


def test_aggregator_uses_standalone_failure_json_when_summary_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupted runs (no summary.json) still surface failures from *_failure.json."""
    monkeypatch.setitem(METHOD_REGISTRY, "_test_boom", MethodSpec(run_fn=_boom))

    raw_dir = tmp_path / "raw"
    cfg = _tiny_cfg(["_test_boom"])
    run_benchmark.run_experiment(cfg, _new_args(seed=0), raw_dir / "seed_0")
    # Simulate an interrupted run: nuke summary.json.
    (raw_dir / "seed_0" / "summary.json").unlink()
    assert (raw_dir / "seed_0" / "_test_boom_failure.json").is_file()

    agg_path = tmp_path / "agg.json"
    aggregate(
        campaign="interrupted",
        raw_dir=raw_dir,
        output=agg_path,
        eps=1e-8,
        rate_tail=0.5,
        reference_dir=tmp_path / "nocache",
        use_reference_cache=False,
    )
    agg = json.loads(agg_path.read_text())
    assert "_test_boom" in agg["methods"]
    assert agg["methods"]["_test_boom"]["n_seeds_failed"] == 1
    # Label fell back to the failure JSON's method_label.
    assert agg["methods"]["_test_boom"]["label"] == "label-_test_boom"
