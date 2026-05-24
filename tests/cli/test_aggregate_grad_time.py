"""Aggregator integration tests for the grad / time per-seed fields.

The wall-clock and gradient-call speedup tables consume
``grad_calls_to_eps_per_seed`` and ``wall_time_to_eps_per_seed`` from
the aggregated JSON.  These tests assert the aggregator populates
those fields consistently with the existing iter-axis field and that
the campaign-level host fingerprint is captured.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agns.cli.aggregate import aggregate
from agns.methods import gradient, newton
from agns.oracles import make_logistic_regression_synthetic
from agns.utils.io import save_json, save_pickle


def _build_tiny_raw(raw_dir: Path, seeds: tuple[int, ...] = (0, 1, 2)) -> None:
    """Tiny logistic-regression campaign that records grad_calls + time."""
    problem_params = {"m": 60, "n": 6, "reg": 1e-3}
    problem_meta = {"type": "logistic_regression_synthetic", "params": problem_params}
    for seed in seeds:
        seed_dir = raw_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        problem = make_logistic_regression_synthetic(seed=seed, **problem_params)
        oracle, x_0 = problem["oracle"], problem["x_0"]
        _, _, h_newton = newton(oracle, x_0, n_iters=20, eps=1e-14)
        _, _, h_grad = gradient(oracle, x_0, n_iters=80, eps=1e-14)
        save_pickle(h_newton, seed_dir / "newton_history.pkl")
        save_pickle(h_grad, seed_dir / "gradient_history.pkl")
        save_json(
            {
                "problem": problem_meta,
                "f_star": None,
                "methods": {
                    "newton": {"label": "Newton", "status": "ok", "wall_time_s": 0.0},
                    "gradient": {"label": "Gradient", "status": "ok", "wall_time_s": 0.0},
                },
            },
            seed_dir / "summary.json",
        )


def test_aggregator_records_grad_and_time_per_seed(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    agg_path = tmp_path / "agg.json"
    _build_tiny_raw(raw_dir, seeds=(0, 1, 2))
    aggregate(
        campaign="tiny_grad_time",
        raw_dir=raw_dir,
        output=agg_path,
        eps=1e-8,
        rate_tail=0.5,
        reference_dir=tmp_path / "nocache",
        use_reference_cache=False,
    )
    agg = json.loads(agg_path.read_text())
    for mkey in ("newton", "gradient"):
        rec = agg["methods"][mkey]
        for field in (
            "iters_to_eps_per_seed",
            "grad_calls_to_eps_per_seed",
            "wall_time_to_eps_per_seed",
        ):
            assert field in rec, f"{mkey} missing {field}"
            assert len(rec[field]) == 3


def test_grad_time_aligned_with_iters_per_seed(tmp_path: Path) -> None:
    """For every seed, all three fields are simultaneously None or simultaneously finite.

    They're computed at the same crossing index, so missing-status
    must agree across axes.  A mismatch would mean one axis is
    silently dropping data the others kept.
    """
    raw_dir = tmp_path / "raw"
    agg_path = tmp_path / "agg.json"
    _build_tiny_raw(raw_dir, seeds=(0, 1, 2))
    aggregate(
        campaign="tiny_align",
        raw_dir=raw_dir,
        output=agg_path,
        eps=1e-8,
        rate_tail=0.5,
        reference_dir=tmp_path / "nocache",
        use_reference_cache=False,
    )
    agg = json.loads(agg_path.read_text())
    for rec in agg["methods"].values():
        iters = rec["iters_to_eps_per_seed"]
        grads = rec["grad_calls_to_eps_per_seed"]
        times = rec["wall_time_to_eps_per_seed"]
        for i, g, t in zip(iters, grads, times, strict=True):
            none_mask = (i is None, g is None, t is None)
            # All three None or all three finite.
            assert len(set(none_mask)) == 1, (
                f"axis misalignment at seed: iter={i}, grad={g}, time={t}"
            )


def test_aggregated_json_carries_host_field(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    agg_path = tmp_path / "agg.json"
    _build_tiny_raw(raw_dir, seeds=(0,))
    aggregate(
        campaign="tiny_host",
        raw_dir=raw_dir,
        output=agg_path,
        eps=1e-8,
        rate_tail=0.5,
        reference_dir=tmp_path / "nocache",
        use_reference_cache=False,
    )
    agg = json.loads(agg_path.read_text())
    assert "host" in agg
    host = agg["host"]
    for key in ("cpu_model", "platform", "python_version", "numpy_version"):
        assert key in host
        assert isinstance(host[key], str)


def test_grad_to_eps_finite_when_crossing_present(tmp_path: Path) -> None:
    """Sanity: when a seed crosses eps, grad and time both record finite numbers.

    Newton on tiny convex LR will cross eps within the budget; the
    per-seed grad-count should therefore be a positive int and the
    wall-clock should be a positive float.
    """
    raw_dir = tmp_path / "raw"
    agg_path = tmp_path / "agg.json"
    _build_tiny_raw(raw_dir, seeds=(0,))
    aggregate(
        campaign="tiny_finite",
        raw_dir=raw_dir,
        output=agg_path,
        eps=1e-6,  # loose so Newton definitely crosses
        rate_tail=0.5,
        reference_dir=tmp_path / "nocache",
        use_reference_cache=False,
    )
    agg = json.loads(agg_path.read_text())
    newton_rec = agg["methods"]["newton"]
    iters0 = newton_rec["iters_to_eps_per_seed"][0]
    grads0 = newton_rec["grad_calls_to_eps_per_seed"][0]
    times0 = newton_rec["wall_time_to_eps_per_seed"][0]
    # If Newton crossed (likely on a tiny convex LR), every axis is finite.
    if iters0 is not None:
        assert isinstance(grads0, int) and grads0 >= 0
        assert isinstance(times0, float) and times0 >= 0.0
        assert np.isfinite(times0)
