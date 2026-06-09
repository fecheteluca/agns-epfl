"""The seed-aware experiment layer samples fresh problems and reports per-seed f*.

Covers the seed-aware run path: ``run_spec_over_seeds`` builds a fresh problem per seed,
tags each history with its own ``f_star`` (distinct under instance sampling), and the
summary/cost helpers consume those per-seed optima. Also smoke-builds the oracle configs
for the RQ1 mechanism and RQ3 fairness experiments.
"""

from __future__ import annotations

import numpy as np

from agns.experiments import (
    ExperimentResult,
    cost_samples,
    run_spec_over_seeds,
    summarize_result,
)
from agns.utils.config import load_config


def _result(spec_name, methods, seeds, n_iters=60):
    spec = load_config(f"oracles/{spec_name}.yaml")
    runs, styles, problem = run_spec_over_seeds(
        spec, methods, n_iters=n_iters, seeds=seeds
    )
    return ExperimentResult(
        name=spec_name,
        problem=problem,
        method_runs=runs,
        styles=styles,
        cost_axes=["iterations", "matrix_inverses"],
        config={},
    )


def test_run_spec_over_seeds_injects_distinct_f_star():
    """Instance sampling gives one history per seed, each with its own (distinct) f*."""
    res = _result("random_small", ["agns_exact", "gns"], seeds=[0, 1, 2, 3])
    for label, runs in res.method_runs.items():
        assert len(runs) == 4, label
        f_stars = [h["f_star"] for h in runs]
        assert len({round(f, 9) for f in f_stars}) == 4, f"{label} f* not distinct"


def test_summarize_uses_per_seed_optimum():
    """The summary table has one finite row per method using each seed's own residual."""
    res = _result("random_small", ["agns_exact", "gns"], seeds=[0, 1, 2])
    df = summarize_result(res, tolerance=1e-6)
    assert set(df["Method"]) == {"AGNS (exact)", "GNS"}
    assert np.all(np.isfinite(df["Matrix inv."].to_numpy()))
    assert np.all(df["Reached"].to_numpy() == 1.0)


def test_cost_samples_length_matches_seeds():
    """``cost_samples`` returns one cost per seed (the input to the stats helpers)."""
    res = _result("random_small", ["agns_exact", "gns"], seeds=[0, 1, 2, 3, 4])
    costs = cost_samples(
        res, "AGNS (exact)", tolerance=1e-6, cost_key="matrix_inverses"
    )
    assert len(costs) == 5
    assert all(np.isfinite(c) for c in costs)


def test_new_curvature_config_builds_and_runs():
    """The RQ1 curvature-variation config runs through the seed-aware path."""
    res = _result("curvature_base", ["agns_exact", "gns"], seeds=[0, 1])
    assert all(h["f_star"] == 0.0 for runs in res.method_runs.values() for h in runs)


def test_new_highsmooth_config_builds_and_runs():
    """The RQ3 high-order-smoothness config runs through the seed-aware path."""
    res = _result("nleq_highsmooth", ["agns_exact", "super_universal"], seeds=[0, 1])
    df = summarize_result(res, tolerance=1e-6)
    assert len(df) == 2
