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
    # Inject per-seed iter counts so the new bootstrap CI path has data
    # to chew on.  The smoke_campaign fixture pre-dates this requirement.
    with open(smoke_campaign) as fh:
        doc = json.load(fh)
    for mkey in ("gns_exact", "agns_exact", "agns_exact_norestart"):
        if mkey in doc["methods"]:
            doc["methods"][mkey]["iters_to_eps_per_seed"] = [5, 6, 7, 8, 9]
    smoke_campaign.write_text(json.dumps(doc))
    out = tmp_path / "speedup.tex"
    speedup.render(smoke_campaign.parent, out)
    tex = out.read_text()
    assert r"\toprule" in tex
    assert "Speedup" in tex
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


# ---------------------------------------------------------------------------
# Three-state cells + method-key fallback in real_world_summary, plus
# speedup hardening against failure-only records.
# ---------------------------------------------------------------------------


def _failure_only_record(n_failed: int = 5, n_seeds: int = 5) -> dict:
    """Mirrors what aggregate.py emits when a method failed on every seed.

    Minimal schema: counts + failures list, NO iter_curve / rate_fit /
    final_residual_summary curves.
    """
    return {
        "method": "agns_inexact",
        "label": "AGNS",
        "n_seeds_succeeded": 0,
        "n_seeds_failed": n_failed,
        "failure_rate": 1.0,
        "final_residuals": [float("inf")] * n_seeds,
        "final_residual_summary": {
            "median": float("inf"),
            "p25": float("inf"),
            "p75": float("inf"),
        },
        "eps_target": 1e-8,
        "above_eps_rate": 1.0,
        "statuses": {},
        "failures": [
            {
                "seed": f"seed_{i}",
                "label": "AGNS",
                "exception_type": "LinAlgError",
                "message": "synthetic",
                "wall_time_s_before_failure": 0.01,
            }
            for i in range(n_failed)
        ],
    }


def test_real_world_summary_renders_failed_cell_for_failure_only_record(
    tmp_path: Path,
) -> None:
    """Method with zero successful seeds renders as ``failed (n/N)``, not as ``--``."""
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    doc = {
        "campaign": "real_lr_a9a",
        "n_seeds": 5,
        "methods": {
            "agns_inexact": _failure_only_record(n_failed=5, n_seeds=5),
            "accelerated_cubic_newton": _gns_record(n_iters=10),
            "lbfgs": _gns_record(n_iters=12),
        },
        "failures_summary": {
            "agns_inexact": {
                "n_failed_seeds": 5,
                "seeds": [f"seed_{i}" for i in range(5)],
                "modal_exception_type": "LinAlgError",
            }
        },
    }
    (agg_dir / "real_lr_a9a.json").write_text(json.dumps(doc))
    out = tmp_path / "summary.tex"
    real_world_summary.render(agg_dir, out)
    tex = out.read_text()
    assert r"\textit{failed (5/5)}" in tex
    # Modal exception type surfaced in the caption footnote.
    assert "LinAlgError" in tex


def test_real_world_summary_renders_not_run_for_missing_method_record(
    tmp_path: Path,
) -> None:
    """Method whose record is absent renders as ``\\textsc{n/r}``."""
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    doc = {
        "campaign": "real_lr_real_sim",
        "n_seeds": 5,
        "methods": {
            # First-order-only campaign: no AGNS, no ACN.
            "lbfgs": _gns_record(n_iters=12),
        },
        "failures_summary": {},
    }
    (agg_dir / "real_lr_real_sim.json").write_text(json.dumps(doc))
    out = tmp_path / "summary.tex"
    real_world_summary.render(agg_dir, out)
    tex = out.read_text()
    assert r"\textsc{n/r}" in tex


def test_real_world_summary_uses_agns_exact_fallback_when_inexact_absent(
    tmp_path: Path,
) -> None:
    """AGNS column should fall back to ``agns_exact`` when ``agns_inexact`` is absent.

    Ridge campaigns only configure ``agns_exact`` (no Fisher
    approximation for ridge); the renderer must look it up via the
    fallback chain so those rows render numerically rather than as n/r.
    """
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    doc = {
        "campaign": "real_ridge_housing",
        "n_seeds": 5,
        "methods": {
            # No agns_inexact.
            "agns_exact": _agns_record(n_iters=8, n_restarts=1, restart_mode="gradient"),
            "accelerated_cubic_newton": _gns_record(n_iters=10),
            "lbfgs": _gns_record(n_iters=12),
        },
        "failures_summary": {},
    }
    (agg_dir / "real_ridge_housing.json").write_text(json.dumps(doc))
    out = tmp_path / "summary.tex"
    real_world_summary.render(agg_dir, out)
    tex = out.read_text()
    # housing row produced a numeric AGNS cell (not n/r, not failed).
    # The unique residual signature of _agns_record is final_residual_summary.median=1.2e-10.
    assert r"\num{1.20e-10}" in tex


def test_real_world_summary_marks_missing_campaign_row_as_not_run(tmp_path: Path) -> None:
    """An aggregated JSON missing entirely yields a row of ``n/r`` cells.

    A campaign with no aggregated JSON used to disappear silently from
    the rollup; now it shows up as an explicit n/r row.
    """
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    # Drop a single campaign that the renderer will pick up.
    (agg_dir / "real_lr_a9a.json").write_text(
        json.dumps(
            {
                "campaign": "real_lr_a9a",
                "n_seeds": 5,
                "methods": {
                    "agns_inexact": _agns_record(n_iters=8, n_restarts=1, restart_mode="gradient"),
                    "accelerated_cubic_newton": _gns_record(n_iters=10),
                    "lbfgs": _gns_record(n_iters=12),
                },
                "failures_summary": {},
            }
        )
    )
    out = tmp_path / "summary.tex"
    real_world_summary.render(agg_dir, out)
    tex = out.read_text()
    # Every dataset in REAL_WORLD_CAMPAIGNS should appear in the rendered
    # table; the ones without aggregated JSONs should now show n/r.
    assert "rcv1" in tex  # was previously absent from the rendered table
    assert r"\textsc{n/r}" in tex


def test_real_world_summary_carries_cell_state_legend(tmp_path: Path) -> None:
    """The caption must include the 3-state legend so cells are self-decoding."""
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    (agg_dir / "real_lr_a9a.json").write_text(
        json.dumps(
            {
                "campaign": "real_lr_a9a",
                "n_seeds": 5,
                "methods": {
                    "agns_inexact": _agns_record(n_iters=8, n_restarts=1, restart_mode="gradient"),
                    "accelerated_cubic_newton": _gns_record(n_iters=10),
                    "lbfgs": _gns_record(n_iters=12),
                },
                "failures_summary": {},
            }
        )
    )
    out = tmp_path / "summary.tex"
    real_world_summary.render(agg_dir, out)
    tex = out.read_text()
    assert "n/r" in tex
    assert "failed" in tex
    assert "Cell key" in tex


# ---------------------------------------------------------------------------
# Speedup table: must tolerate failure-only records (no iter_curve key).
# ---------------------------------------------------------------------------


def _campaign_with_cost_counts(
    *,
    gns_iters: list[int | None],
    agns_iters: list[int | None],
    agns_nr_iters: list[int | None] | None = None,
    gns_grads: list[int | None] | None = None,
    agns_grads: list[int | None] | None = None,
    gns_times: list[float | None] | None = None,
    agns_times: list[float | None] | None = None,
    campaign_name: str = "tiny",
    host: dict | None = None,
) -> dict:
    """Variant builder that populates grad/time per-seed fields too."""
    doc = _campaign_with_iter_counts(
        gns_iters=gns_iters,
        agns_iters=agns_iters,
        agns_nr_iters=agns_nr_iters,
        campaign_name=campaign_name,
    )
    if gns_grads is not None:
        doc["methods"]["gns_exact"]["grad_calls_to_eps_per_seed"] = gns_grads
    if agns_grads is not None:
        doc["methods"]["agns_exact"]["grad_calls_to_eps_per_seed"] = agns_grads
    if gns_times is not None:
        doc["methods"]["gns_exact"]["wall_time_to_eps_per_seed"] = gns_times
    if agns_times is not None:
        doc["methods"]["agns_exact"]["wall_time_to_eps_per_seed"] = agns_times
    if host is not None:
        doc["host"] = host
    return doc


def _campaign_with_iter_counts(
    *,
    gns_iters: list[int | None],
    agns_iters: list[int | None],
    agns_nr_iters: list[int | None] | None = None,
    campaign_name: str = "tiny",
) -> dict:
    """Build a minimal aggregated JSON with per-seed iter counts populated."""
    methods: dict[str, dict] = {}
    methods["gns_exact"] = {
        "method": "gns_exact",
        "label": "GNS",
        "n_seeds_succeeded": sum(1 for v in gns_iters if v is not None),
        "n_seeds_failed": 0,
        "iter_curve": {"x": [], "median": [], "p25": [], "p75": []},
        "iters_to_eps_per_seed": gns_iters,
    }
    methods["agns_exact"] = {
        "method": "agns_exact",
        "label": "AGNS",
        "n_seeds_succeeded": sum(1 for v in agns_iters if v is not None),
        "n_seeds_failed": 0,
        "iter_curve": {"x": [], "median": [], "p25": [], "p75": []},
        "iters_to_eps_per_seed": agns_iters,
    }
    if agns_nr_iters is not None:
        methods["agns_exact_norestart"] = {
            "method": "agns_exact_norestart",
            "label": "AGNS-noR",
            "n_seeds_succeeded": sum(1 for v in agns_nr_iters if v is not None),
            "n_seeds_failed": 0,
            "iter_curve": {"x": [], "median": [], "p25": [], "p75": []},
            "iters_to_eps_per_seed": agns_nr_iters,
        }
    return {
        "campaign": campaign_name,
        "n_seeds": len(gns_iters),
        "eps_target": 1e-8,
        "methods": methods,
    }


def test_speedup_table_renders_ci_and_wilcoxon_columns(tmp_path: Path) -> None:
    """The CI bracket and the p_W column are both in the rendered table."""
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    # AGNS clearly faster on 10 seeds.
    doc = _campaign_with_iter_counts(
        gns_iters=[20, 21, 19, 22, 18, 21, 20, 19, 22, 20],
        agns_iters=[10, 11, 9, 12, 8, 11, 10, 9, 12, 10],
    )
    (agg_dir / "tiny.json").write_text(json.dumps(doc))
    out = tmp_path / "speedup.tex"
    speedup.render(agg_dir, out)
    tex = out.read_text()
    # Header columns present.
    assert "[CI95]" in tex
    assert r"p_{W}" in tex
    # Point estimate around 2.0 (20/10).
    assert "2.00" in tex
    # CI bracket appears as ``[lo, hi]`` somewhere in the row.
    assert "[" in tex and ", " in tex
    # p_W < 0.05 -> rendered with three decimals or as ``<0.001``.
    assert "<0.001" in tex or "0.00" in tex


def test_speedup_table_flags_noise_when_ci_brackets_one(tmp_path: Path) -> None:
    """A row whose CI brackets 1.0 carries the ``\\approx`` flag."""
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    # Identical iter counts -> point estimate 1.0, CI = [1, 1], flag fires.
    doc = _campaign_with_iter_counts(
        gns_iters=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        agns_iters=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    )
    (agg_dir / "tiny.json").write_text(json.dumps(doc))
    out = tmp_path / "speedup.tex"
    speedup.render(agg_dir, out)
    tex = out.read_text()
    # \approx appears in the body (not just in the header / caption text).
    body_start = tex.index(r"\midrule")
    body_end = tex.index(r"\bottomrule")
    body = tex[body_start:body_end]
    assert r"\approx" in body


def test_speedup_table_does_not_flag_when_ci_clearly_excludes_one(tmp_path: Path) -> None:
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    doc = _campaign_with_iter_counts(
        gns_iters=[100] * 20,
        agns_iters=[10] * 20,
    )
    (agg_dir / "tiny.json").write_text(json.dumps(doc))
    out = tmp_path / "speedup.tex"
    speedup.render(agg_dir, out)
    tex = out.read_text()
    body_start = tex.index(r"\midrule")
    body_end = tex.index(r"\bottomrule")
    body = tex[body_start:body_end]
    # No \approx in the body cells (it's still in the caption legend).
    # We check the column-12 flag position: the body row should not
    # contain \approx in its data cells.  Easiest assertion: the flag
    # cell is empty in this row.
    assert r"\approx$ &" not in body  # noise flag cell is empty for this row


def test_speedup_table_renders_wilcoxon_na_when_too_few_paired(tmp_path: Path) -> None:
    """Fewer than 5 paired seeds -> p_W = n/a in the rendered row."""
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    doc = _campaign_with_iter_counts(
        gns_iters=[10, 11, 12],
        agns_iters=[5, 6, 7],
    )
    (agg_dir / "tiny.json").write_text(json.dumps(doc))
    out = tmp_path / "speedup.tex"
    speedup.render(agg_dir, out)
    tex = out.read_text()
    body_start = tex.index(r"\midrule")
    body_end = tex.index(r"\bottomrule")
    body = tex[body_start:body_end]
    assert "n/a" in body


# ---------------------------------------------------------------------------
# Grad-call and wall-clock speedup tables.
# ---------------------------------------------------------------------------


def test_speedup_grad_axis_uses_grad_per_seed_field(tmp_path: Path) -> None:
    """``--axis grad`` reads grad_calls_to_eps_per_seed, not iters_to_eps_per_seed.

    Construct a campaign where iter and grad counts give opposite
    speedup directions: GNS has fewer iters but more grad calls
    (mirroring the real per-iteration probing pattern).  The grad
    axis should report AGNS faster; the iter axis should report
    GNS faster.
    """
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    doc = _campaign_with_cost_counts(
        gns_iters=[8, 9, 8, 9, 8, 9, 8, 9, 8, 9],     # GNS slightly fewer iters
        agns_iters=[10, 11, 10, 11, 10, 11, 10, 11, 10, 11],
        gns_grads=[120, 130, 120, 130, 120, 130, 120, 130, 120, 130],  # but many more grad calls
        agns_grads=[30, 33, 30, 33, 30, 33, 30, 33, 30, 33],
    )
    (agg_dir / "tiny.json").write_text(json.dumps(doc))

    # Iter-axis: GNS faster.
    out_iter = tmp_path / "speedup_iter.tex"
    speedup.render(agg_dir, out_iter, axis="iter")
    iter_tex = out_iter.read_text()
    # Iter speedup < 1.0 (GNS iters / AGNS iters = 8.5/10.5 ≈ 0.81).
    assert "0.81" in iter_tex

    # Grad-axis: AGNS faster.
    out_grad = tmp_path / "speedup_grad.tex"
    speedup.render(agg_dir, out_grad, axis="grad")
    grad_tex = out_grad.read_text()
    # Grad speedup = 125/31.5 ≈ 3.97.
    assert "3.97" in grad_tex
    # Header says "grads".
    assert "grads" in grad_tex


def test_speedup_time_axis_uses_time_per_seed_field(tmp_path: Path) -> None:
    """``--axis time`` reads wall_time_to_eps_per_seed and renders seconds."""
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    doc = _campaign_with_cost_counts(
        gns_iters=[10] * 10,
        agns_iters=[10] * 10,
        gns_times=[0.5] * 10,
        agns_times=[0.1] * 10,
        host={"cpu_model": "Test CPU", "blas_threads_env": {"OMP_NUM_THREADS": "1"}},
    )
    (agg_dir / "tiny.json").write_text(json.dumps(doc))
    out = tmp_path / "speedup_time.tex"
    speedup.render(agg_dir, out, axis="time")
    tex = out.read_text()
    # Speedup = 0.5 / 0.1 = 5.0.
    assert "5.00" in tex
    # Time-axis header uses "s" (seconds).
    assert "[s]" in tex
    # Host fingerprint surfaces in the caption.
    assert "Test CPU" in tex


def test_speedup_time_axis_caption_warns_when_host_missing(tmp_path: Path) -> None:
    """Time table with no host fingerprint surfaces a re-aggregate hint."""
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    doc = _campaign_with_cost_counts(
        gns_iters=[10] * 10,
        agns_iters=[10] * 10,
        gns_times=[0.5] * 10,
        agns_times=[0.1] * 10,
    )
    # Deliberately no "host" key.
    (agg_dir / "tiny.json").write_text(json.dumps(doc))
    out = tmp_path / "speedup_time.tex"
    speedup.render(agg_dir, out, axis="time")
    tex = out.read_text()
    assert "fingerprint not recorded" in tex


def test_speedup_grad_axis_skips_rows_without_grad_data(tmp_path: Path) -> None:
    """Older aggregated JSONs without grad_calls_to_eps_per_seed are silently skipped."""
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    # No grad fields populated.
    doc = _campaign_with_iter_counts(
        gns_iters=[10] * 10,
        agns_iters=[5] * 10,
    )
    (agg_dir / "tiny.json").write_text(json.dumps(doc))
    out = tmp_path / "speedup_grad.tex"
    speedup.render(agg_dir, out, axis="grad")
    tex = out.read_text()
    # Body should carry no real rows.
    body = tex[tex.index(r"\midrule"):tex.index(r"\bottomrule")]
    assert "no campaigns" in body.lower()


def test_speedup_render_axis_default_is_iter(tmp_path: Path) -> None:
    """speedup.render(...) without --axis is backward-compatible (iter table)."""
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    doc = _campaign_with_iter_counts(gns_iters=[10] * 10, agns_iters=[5] * 10)
    (agg_dir / "tiny.json").write_text(json.dumps(doc))
    out = tmp_path / "speedup.tex"
    speedup.render(agg_dir, out)  # no axis kwarg
    tex = out.read_text()
    # Iter-axis header uses "iters".
    assert "iters" in tex


def test_speedup_invalid_axis_raises(tmp_path: Path) -> None:
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    (agg_dir / "tiny.json").write_text("{}")
    with pytest.raises(ValueError, match="axis must be one of"):
        speedup.render(agg_dir, tmp_path / "out.tex", axis="hess")  # type: ignore[arg-type]


def test_speedup_axis_wrappers_match_render(tmp_path: Path) -> None:
    """render_iter / render_grad / render_time produce the same output as render(..., axis=...)."""
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    doc = _campaign_with_cost_counts(
        gns_iters=[10] * 10,
        agns_iters=[5] * 10,
        gns_grads=[100] * 10,
        agns_grads=[20] * 10,
        gns_times=[0.5] * 10,
        agns_times=[0.1] * 10,
        host={"cpu_model": "Test", "blas_threads_env": {"OMP_NUM_THREADS": "1"}},
    )
    (agg_dir / "tiny.json").write_text(json.dumps(doc))
    for axis in ("iter", "grad", "time"):
        out_a = tmp_path / f"{axis}_dispatch.tex"
        out_b = tmp_path / f"{axis}_wrapper.tex"
        speedup.render(agg_dir, out_a, axis=axis)  # type: ignore[arg-type]
        getattr(speedup, f"render_{axis}")(agg_dir, out_b)
        assert out_a.read_text() == out_b.read_text(), f"axis={axis} differs"


def test_speedup_does_not_crash_on_failure_only_agns(tmp_path: Path) -> None:
    """A failure-only AGNS record (no iter_curve) must be skipped, not raise.

    On a campaign where AGNS failed on every seed the aggregator
    produces a failure-only record without the ``iter_curve`` key.
    The speedup renderer must skip such pairings rather than crashing
    on a KeyError.
    """
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    doc = {
        "campaign": "broken_campaign",
        "n_seeds": 5,
        "methods": {
            "gns_exact": _gns_record(n_iters=20),
            "agns_exact": _failure_only_record(n_failed=5, n_seeds=5),
        },
        "failures_summary": {},
    }
    (agg_dir / "broken_campaign.json").write_text(json.dumps(doc))
    out = tmp_path / "speedup.tex"
    # Must not raise.
    speedup.render(agg_dir, out)
    tex = out.read_text()
    # The pairing is silently skipped; the table renders with no data
    # rows (the placeholder "(no campaigns with matched ...)" appears).
    assert r"\toprule" in tex
    assert "no campaigns" in tex.lower()
