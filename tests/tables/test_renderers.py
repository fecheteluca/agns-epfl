"""End-to-end renderer tests for every module under :mod:`agns.tables`.

Each test constructs a *minimal* aggregated-JSON fixture in a
``tmp_path`` directory, invokes the renderer, and asserts that:

1. the output file exists, and
2. the LaTeX it contains carries the canonical booktabs scaffold plus
   at least one substring unique to that renderer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agns.tables import per_campaign, real_world_summary, restart_density, speedup


def _agns_record(
    *,
    n_iters: int,
    n_restarts: int,
    restart_mode: str,
    first_restart: int | None = 5,
    max_lk: int = 4,
) -> dict:
    return {
        "method": "agns_inexact",
        "label": "AGNS",
        "n_seeds_succeeded": 1,
        "n_seeds_failed": 0,
        "failure_rate": 0.0,
        "final_residuals": [1.2e-10],
        "final_residual_summary": {"median": 1.2e-10, "p25": 1.0e-10, "p75": 1.4e-10},
        "eps_target": 1e-8,
        "above_eps_rate": 0.0,
        "statuses": {"seed_0": "success, 12 iters"},
        "iter_curve": {
            "x": list(range(n_iters)),
            "median": [10.0 / (k + 1) for k in range(n_iters)],
            "p25": [9.0 / (k + 1) for k in range(n_iters)],
            "p75": [11.0 / (k + 1) for k in range(n_iters)],
        },
        "time_curve": {"x": [0], "median": [0.01], "p25": [0.01], "p75": [0.01]},
        "grad_curve": {"x": [0], "median": [k * 3 for k in range(n_iters)],
                       "p25": [k * 3 for k in range(n_iters)],
                       "p75": [k * 3 for k in range(n_iters)]},
        "rate_fit": {"slope": -1.5, "intercept": 0.0, "r_squared": 0.95, "n_points": 5},
        "restart_summary": {
            "restart_mode": restart_mode,
            "n_restarts_per_seed": [n_restarts],
            "n_iters_per_seed": [n_iters],
            "first_restart_iter_per_seed": [first_restart],
            "max_local_k_per_seed": [max_lk],
            "n_restarts_summary": {"median": n_restarts, "p25": n_restarts, "p75": n_restarts},
            "restart_density_summary": {
                "median": n_restarts / max(n_iters, 1),
                "p25": n_restarts / max(n_iters, 1),
                "p75": n_restarts / max(n_iters, 1),
            },
        },
    }


def _gns_record(n_iters: int) -> dict:
    return {
        "method": "gns_exact",
        "label": "GNS",
        "n_seeds_succeeded": 1,
        "n_seeds_failed": 0,
        "failure_rate": 0.0,
        "final_residuals": [3.0e-10],
        "final_residual_summary": {"median": 3.0e-10, "p25": 2.8e-10, "p75": 3.2e-10},
        "eps_target": 1e-8,
        "above_eps_rate": 0.0,
        "iter_curve": {
            "x": list(range(n_iters)),
            "median": [20.0 / (k + 1) for k in range(n_iters)],
            "p25": [18.0 / (k + 1) for k in range(n_iters)],
            "p75": [22.0 / (k + 1) for k in range(n_iters)],
        },
        "rate_fit": {"slope": -1.4, "intercept": 0.0, "r_squared": 0.94, "n_points": 5},
    }


@pytest.fixture()
def smoke_campaign(tmp_path: Path) -> Path:
    """Write one aggregated JSON exercising the full method roster.

    Carries one matched ``(gns_exact, agns_exact, agns_exact_norestart)``
    triple so the speedup-table renderer has rows to emit, plus an
    inexact-Hessian AGNS to exercise the cross-variant pickup.
    """
    methods = {
        "gns_exact": _gns_record(n_iters=20),
        "agns_exact": _agns_record(n_iters=12, n_restarts=2, restart_mode="gradient"),
        "agns_exact_norestart": _agns_record(
            n_iters=15, n_restarts=0, restart_mode="none", first_restart=None
        ),
        "agns_inexact": _agns_record(n_iters=14, n_restarts=3, restart_mode="gradient"),
    }
    doc = {
        "campaign": "smoke_campaign",
        "n_seeds": 1,
        "seeds": [0],
        "eps_target": 1e-8,
        "f_star_global_min": 0.0,
        "per_seed_f_star": {"seed_0": 0.0},
        "problem": {"type": "ridge_regression_synthetic", "params": {"n": 10}},
        "methods": methods,
    }
    p = tmp_path / "smoke_campaign.json"
    p.write_text(json.dumps(doc))
    return p


def test_per_campaign_renders_booktabs(smoke_campaign: Path, tmp_path: Path) -> None:
    out = tmp_path / "per_campaign.tex"
    per_campaign.render(smoke_campaign, out)
    tex = out.read_text()
    assert r"\toprule" in tex
    assert r"\midrule" in tex
    assert r"\bottomrule" in tex
    assert "Method" in tex
    # AGNS label survives table escaping.
    assert "AGNS" in tex


def test_restart_density_filters_to_agns_family(smoke_campaign: Path, tmp_path: Path) -> None:
    out = tmp_path / "restart_density.tex"
    restart_density.render(smoke_campaign.parent, out)
    tex = out.read_text()
    assert r"\toprule" in tex
    assert "Density" in tex
    # AGNS rows are included; the GNS row is not (no restart_summary block).
    assert "gradient" in tex
    assert "none" in tex
    # AGNS-norestart row reports zero restarts.
    assert "0" in tex


def test_speedup_renders_matched_pair(smoke_campaign: Path, tmp_path: Path) -> None:
    out = tmp_path / "speedup.tex"
    speedup.render(smoke_campaign.parent, out, eps=1e-1)
    tex = out.read_text()
    assert r"\toprule" in tex
    assert "Speedup" in tex
    # The (gns_exact, agns_exact, agns_exact_norestart) triple yields one row.
    assert "smoke" in tex
    assert "exact H" in tex


def test_real_world_summary_skips_when_no_real_campaigns(tmp_path: Path) -> None:
    # Empty aggregated dir -> RuntimeError ("nothing to render").
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(RuntimeError):
        real_world_summary.render(empty_dir, tmp_path / "out.tex")


def test_real_world_summary_renders_minimum_row(tmp_path: Path) -> None:
    """Drop a single ``real_lr_a9a.json`` and verify the rollup picks it up."""
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    doc = {
        "campaign": "real_lr_a9a",
        "methods": {
            "agns_inexact": _agns_record(n_iters=10, n_restarts=1, restart_mode="gradient"),
            "accelerated_cubic_newton": _gns_record(n_iters=12),
            "lbfgs": _gns_record(n_iters=15),
        },
    }
    (agg_dir / "real_lr_a9a.json").write_text(json.dumps(doc))
    out = tmp_path / "summary.tex"
    real_world_summary.render(agg_dir, out)
    tex = out.read_text()
    assert r"\toprule" in tex
    assert "Dataset" in tex
    # The {\sc a9a} cell wraps the dataset name in small caps.
    assert "a9a" in tex


def test_per_campaign_raises_on_missing_json(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        per_campaign.render(tmp_path / "missing.json", tmp_path / "out.tex")
