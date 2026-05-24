"""Tests for the Picard inner-loop kwargs exposed by ``picard_acn_2008``.

``fp_max`` and ``fp_rel_tol`` are exposed as kwargs so the sensitivity
sweep in ``configs/ablations/picard_fp_grid.yaml`` can vary them; these
tests pin the contract:

* defaults are ``fp_max=3`` and ``fp_rel_tol=0.10``;
* both kwargs flow through the registry's ``_OPTIONAL_KWARGS`` plumbing,
  so a YAML ``sweep`` block actually changes the inner loop;
* the sweep machinery expands the ``picard_fp_grid`` config into the
  expected number of clones (8 swept + 1 default + 1 GNS reference).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from agns.methods import picard_acn_2008
from agns.pipeline.config import load_config
from agns.pipeline.registry import _OPTIONAL_KWARGS, METHOD_REGISTRY, run_method
from agns.pipeline.sweeps import expand_sweeps


def test_picard_acn_2008_signature_has_fp_kwargs() -> None:
    """Both ``fp_max`` and ``fp_rel_tol`` are in the function signature."""
    sig = inspect.signature(picard_acn_2008)
    assert "fp_max" in sig.parameters
    assert "fp_rel_tol" in sig.parameters


def test_picard_acn_2008_default_kwarg_values() -> None:
    """Default values are ``fp_max = 3`` and ``fp_rel_tol = 0.10``.

    Any caller that does not override these must keep getting
    identical numerics.
    """
    sig = inspect.signature(picard_acn_2008)
    assert sig.parameters["fp_max"].default == 3
    assert sig.parameters["fp_rel_tol"].default == 0.10


def test_fp_kwargs_in_registry_optional_kwargs() -> None:
    """The registry forwards both kwargs from YAML cfg to the method."""
    assert "fp_max" in _OPTIONAL_KWARGS
    assert "fp_rel_tol" in _OPTIONAL_KWARGS


def test_fp_kwargs_flow_through_run_method() -> None:
    """End-to-end: a cfg with non-default fp_max actually changes the run.

    Construct two small ridge runs at non-defaults; if the kwargs
    didn't flow through, both runs would produce identical traces
    (the registry's default would override).  With the kwargs
    plumbed, ``fp_max=1`` (single Picard iter) vs ``fp_max=10`` (up
    to ten) yields visibly different trajectories.

    The ridge problem here is tiny -- this is a plumbing test, not
    a convergence test.
    """
    rng = np.random.default_rng(0)
    m, n = 40, 6
    X = rng.standard_normal((m, n))
    y = X @ rng.standard_normal(n) + 0.05 * rng.standard_normal(m)
    from agns.oracles import RidgeRegressionOracle

    oracle = RidgeRegressionOracle(X, y, reg=1e-2)
    x_0 = np.zeros(n)
    H = X.T @ X / m + 1e-2 * np.eye(n)
    Binv = np.linalg.inv(H)

    spec = METHOD_REGISTRY["picard_acn_2008"]
    base_cfg = {
        "n_iters": 5,
        "eps": 1e-14,
        "B": H,
        "Binv": Binv,
        "M_0": 1.0,
    }
    cfg_small_fp = {**base_cfg, "fp_max": 1}
    cfg_large_fp = {**base_cfg, "fp_max": 10}

    _, _, hist_small = run_method(spec, oracle, x_0, cfg_small_fp, oracle, None)
    _, _, hist_large = run_method(spec, oracle, x_0, cfg_large_fp, oracle, None)
    assert hist_small is not None and hist_large is not None
    # On a small ridge problem the two fp_max values produce visibly
    # different trajectories; require *some* divergence rather than a
    # specific tolerance, since the exact magnitude is problem-dependent.
    diff = np.abs(np.asarray(hist_small["func"]) - np.asarray(hist_large["func"]))
    assert diff.max() > 0.0, (
        "fp_max=1 vs fp_max=10 produced bit-identical traces; the kwarg is not flowing "
        "through to picard_acn_2008's inner loop."
    )


def test_picard_fp_grid_yaml_expands_to_expected_method_set() -> None:
    """The ablation yaml composes into the expected sweep clones.

    Four ``fp_max`` values + four ``fp_rel_tol`` values + one default
    Picard-ACN reference + one ``gns_exact`` anchor = 10 method
    entries after sweep expansion.  Cell counts on the swept clones
    follow the keys produced by ``agns.pipeline.sweeps`` (``__param=value``
    suffixed).
    """
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "configs" / "ablations" / "picard_fp_grid.yaml"
    cfg = load_config(cfg_path)
    expanded = expand_sweeps(cfg)
    method_keys = [m.get("key", m["name"]) for m in expanded["methods"]]

    # 4 fp_max + 4 fp_rel_tol + 1 default reference + 1 GNS anchor = 10.
    assert len(method_keys) == 10
    assert sum(k.startswith("picard_acn_2008__fp_max=") for k in method_keys) == 4
    assert sum(k.startswith("picard_acn_2008__fp_rel_tol=") for k in method_keys) == 4
    assert "picard_acn_2008" in method_keys  # the default-reference run
    assert "gns_exact" in method_keys
