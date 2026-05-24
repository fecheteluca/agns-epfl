"""Integration tests for the aggregator's cache-aware f_star selection.

Builds a tiny synthetic campaign on a problem whose factory leaves
``f_star=None`` (logistic regression), runs two methods, then aggregates
under three regimes:

  1. Cache present and ``use_reference_cache=True``  → residuals subtract cached f_ref.
  2. Cache present but ``use_reference_cache=False`` → fall through to per-seed-min.
  3. No cache + no declared f_star → per-seed-min with warning.

The pipeline-level smoke test in :mod:`tests.cli.test_pipeline` is
unchanged (it uses ridge regression, which declares f_star, so the
cache path is not exercised there).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agns.cli.aggregate import (
    FSTAR_CACHED_REFERENCE,
    FSTAR_DECLARED,
    FSTAR_PER_SEED_MIN_FALLBACK,
    aggregate,
)
from agns.methods import gradient, newton
from agns.oracles import make_logistic_regression_synthetic
from agns.pipeline.reference_cache import ReferenceEntry, save_entry
from agns.utils.io import save_json, save_pickle


def _build_tiny_raw(raw_dir: Path, seeds: tuple[int, ...] = (0, 1)) -> dict[str, object]:
    """Produce raw/seed_*/{newton,gradient}_history.pkl + summary.json.

    Returns the shared problem-meta dict (problem.type + params) so the
    test can build a matching cache entry.
    """
    problem_params = {"m": 60, "n": 6, "reg": 1e-3}
    problem_meta = {"type": "logistic_regression_synthetic", "params": problem_params}

    for seed in seeds:
        seed_dir = raw_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        problem = make_logistic_regression_synthetic(seed=seed, **problem_params)
        oracle, x_0 = problem["oracle"], problem["x_0"]
        # Two methods at short budgets.
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

    return problem_meta


def _seed_mins(raw_dir: Path) -> list[float]:
    """Compute per-seed min(func across methods) for assertion."""
    import pickle as _pickle  # local import to keep test deps explicit

    mins: list[float] = []
    for seed_dir in sorted(raw_dir.glob("seed_*")):
        seed_min = float("inf")
        for pkl in seed_dir.glob("*_history.pkl"):
            with open(pkl, "rb") as fh:
                hist = _pickle.load(fh)
            vals = np.asarray(hist.get("func", []), dtype=np.float64)
            if vals.size:
                seed_min = min(seed_min, float(vals.min()))
        mins.append(seed_min)
    return mins


# ---------------------------------------------------------------------------
# Cache hit precedence
# ---------------------------------------------------------------------------


def test_aggregator_records_iters_to_eps_per_seed(tmp_path: Path) -> None:
    """Each per-method record carries a per-seed iter-to-eps list.

    The list has one entry per seed dir (None when the seed's residual
    never crossed eps).  The speedup table's bootstrap CI consumes
    this list directly; without it the CI machinery has nothing to
    resample.
    """
    raw_dir = tmp_path / "raw"
    agg_path = tmp_path / "agg.json"
    _build_tiny_raw(raw_dir, seeds=(0, 1, 2))

    aggregate(
        campaign="tiny_iters",
        raw_dir=raw_dir,
        output=agg_path,
        eps=1e-8,
        rate_tail=0.5,
        reference_dir=tmp_path / "nocache",
        use_reference_cache=False,
    )
    agg = json.loads(agg_path.read_text())
    for mkey in ("newton", "gradient"):
        assert mkey in agg["methods"]
        rec = agg["methods"][mkey]
        assert "iters_to_eps_per_seed" in rec
        assert len(rec["iters_to_eps_per_seed"]) == 3
        # Each entry is None or a non-negative int.
        for v in rec["iters_to_eps_per_seed"]:
            assert v is None or (isinstance(v, int) and v >= 0)


def test_aggregator_uses_cached_f_ref_when_available(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    agg_path = tmp_path / "agg.json"
    cache_dir = tmp_path / "refcache"

    meta = _build_tiny_raw(raw_dir, seeds=(0, 1))

    # Seed the cache with a *deliberately distinct* f_ref that the
    # aggregator cannot have derived from the histories.  Pick a value
    # below the true per-seed min (the cache acts as a tighter
    # reference): residuals = func - f_ref, so a smaller f_ref ->
    # uniformly larger residuals than the fallback would produce.
    per_seed_min = _seed_mins(raw_dir)
    sentinel_f_ref = min(per_seed_min) - 1.0  # comfortably distinct
    for seed in (0, 1):
        save_entry(
            cache_dir,
            ReferenceEntry(
                problem_type=meta["type"],
                params=meta["params"],
                seed=seed,
                f_ref=sentinel_f_ref,
                grad_norm_ref=None,
                solver="newton",
                n_iters_spent=1,
                eps_target=1e-16,
                grad_tol_target=None,
                status="ok",
                wall_time_s=0.0,
            ),
        )

    aggregate(
        campaign="tiny",
        raw_dir=raw_dir,
        output=agg_path,
        eps=1e-8,
        rate_tail=0.5,
        reference_dir=cache_dir,
        use_reference_cache=True,
    )
    agg = json.loads(agg_path.read_text())

    # Every seed should carry the cache source.
    assert agg["f_star_source"] == FSTAR_CACHED_REFERENCE
    assert all(v == FSTAR_CACHED_REFERENCE for v in agg["f_star_per_source"].values())
    # And the per_seed_f_star is the sentinel, not the per-seed min.
    assert all(v == sentinel_f_ref for v in agg["per_seed_f_star"].values())


# ---------------------------------------------------------------------------
# Fallback path + warning emission
# ---------------------------------------------------------------------------


def test_aggregator_falls_back_with_warning_when_no_cache(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    agg_path = tmp_path / "agg.json"
    cache_dir = tmp_path / "empty_refcache"  # exists but holds no entries

    _build_tiny_raw(raw_dir, seeds=(0,))

    aggregate(
        campaign="tiny",
        raw_dir=raw_dir,
        output=agg_path,
        eps=1e-8,
        rate_tail=0.5,
        reference_dir=cache_dir,
        use_reference_cache=True,
    )
    agg = json.loads(agg_path.read_text())
    assert agg["f_star_source"] == FSTAR_PER_SEED_MIN_FALLBACK
    assert agg["warnings"], "fallback path must emit a user-visible warning"
    assert any("per-seed minimum" in w for w in agg["warnings"])


def test_use_reference_cache_false_skips_lookup(tmp_path: Path) -> None:
    """``--no-reference-cache`` disables the cache even when entries exist."""
    raw_dir = tmp_path / "raw"
    agg_path = tmp_path / "agg.json"
    cache_dir = tmp_path / "refcache"

    meta = _build_tiny_raw(raw_dir, seeds=(0,))
    save_entry(
        cache_dir,
        ReferenceEntry(
            problem_type=meta["type"],
            params=meta["params"],
            seed=0,
            f_ref=-12345.0,  # would dominate if consulted
            grad_norm_ref=None,
            solver="newton",
            n_iters_spent=1,
            eps_target=1e-16,
            grad_tol_target=None,
            status="ok",
            wall_time_s=0.0,
        ),
    )

    aggregate(
        campaign="tiny",
        raw_dir=raw_dir,
        output=agg_path,
        eps=1e-8,
        rate_tail=0.5,
        reference_dir=cache_dir,
        use_reference_cache=False,
    )
    agg = json.loads(agg_path.read_text())
    assert agg["f_star_source"] == FSTAR_PER_SEED_MIN_FALLBACK
    assert agg["per_seed_f_star"]["seed_0"] != -12345.0


# ---------------------------------------------------------------------------
# Declared f_star wins regardless of cache contents
# ---------------------------------------------------------------------------


def test_declared_f_star_in_summary_overrides_cache(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    agg_path = tmp_path / "agg.json"
    cache_dir = tmp_path / "refcache"

    meta = _build_tiny_raw(raw_dir, seeds=(0,))
    # Now overwrite summary.json with a *declared* f_star.
    declared = 0.42
    summary_path = raw_dir / "seed_0" / "summary.json"
    summ = json.loads(summary_path.read_text())
    summ["f_star"] = declared
    save_json(summ, summary_path)

    # Cache also present, with a different value.
    save_entry(
        cache_dir,
        ReferenceEntry(
            problem_type=meta["type"],
            params=meta["params"],
            seed=0,
            f_ref=0.99,
            grad_norm_ref=None,
            solver="newton",
            n_iters_spent=1,
            eps_target=1e-16,
            grad_tol_target=None,
            status="ok",
            wall_time_s=0.0,
        ),
    )

    aggregate(
        campaign="tiny",
        raw_dir=raw_dir,
        output=agg_path,
        eps=1e-8,
        rate_tail=0.5,
        reference_dir=cache_dir,
        use_reference_cache=True,
    )
    agg = json.loads(agg_path.read_text())
    assert agg["f_star_source"] == FSTAR_DECLARED
    assert agg["per_seed_f_star"]["seed_0"] == declared


# ---------------------------------------------------------------------------
# Per-campaign table footnote rendering
# ---------------------------------------------------------------------------


def test_per_campaign_table_renders_fallback_warning(tmp_path: Path) -> None:
    from agns.tables import per_campaign

    raw_dir = tmp_path / "raw"
    agg_path = tmp_path / "agg.json"
    tex_path = tmp_path / "out.tex"

    _build_tiny_raw(raw_dir, seeds=(0,))
    aggregate(
        campaign="tiny_fallback",
        raw_dir=raw_dir,
        output=agg_path,
        eps=1e-8,
        rate_tail=0.5,
        reference_dir=tmp_path / "no_such_cache",
        use_reference_cache=True,
    )
    per_campaign.render(agg_path, tex_path)
    tex = tex_path.read_text()
    assert "WARNING" in tex
    assert "per-seed minimum" in tex


def test_runner_writes_null_f_star_for_undeclared_factory(tmp_path: Path) -> None:
    """The runner must not bake the per-seed-min into ``summary.json[f_star]``.

    Historically the runner resolved ``f_star=None`` (from the factory)
    to the per-seed min across method ``f_final`` values and wrote that
    into ``summary.json[f_star]``.  The aggregator's "declared"
    precedence then picked it up as if the problem itself had declared
    that value, which silently re-introduced the per-seed-min bias
    that the cache-based ``f_star`` precedence is meant to avoid.

    The fixed runner writes ``f_star: null`` when the factory did not
    declare one, leaving residual computation to the aggregator (which
    will then prefer the cached reference or, lacking that, the
    documented per-seed-min fallback path with its visible warning).
    """
    import subprocess
    import sys
    import textwrap

    # Tiny logistic-regression-synthetic campaign; factory declares f_star=None.
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        problem:
          type: logistic_regression_synthetic
          params: {{m: 50, n: 5, reg: 1.0e-3, seed: 0}}
        common: {{gamma_0: 1.0, eps: 1.0e-10, adaptive_search: true}}
        methods:
          - name: newton
            label: Newton
            n_iters: 10
        output:
          save_dir: {tmp_path}/raw
    """))
    raw_dir = tmp_path / "raw"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "agns.cli.run_benchmark",
            "--config",
            str(cfg),
            "--output-dir",
            str(raw_dir),
            "--seeds",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    summ = json.loads((raw_dir / "seed_0" / "summary.json").read_text())
    # Factory leaves f_star=None -> runner must persist null, NOT a numeric stand-in.
    assert summ["f_star"] is None


def test_per_campaign_table_renders_cache_note(tmp_path: Path) -> None:
    from agns.tables import per_campaign

    raw_dir = tmp_path / "raw"
    agg_path = tmp_path / "agg.json"
    tex_path = tmp_path / "out.tex"
    cache_dir = tmp_path / "refcache"

    meta = _build_tiny_raw(raw_dir, seeds=(0,))
    save_entry(
        cache_dir,
        ReferenceEntry(
            problem_type=meta["type"],
            params=meta["params"],
            seed=0,
            f_ref=_seed_mins(raw_dir)[0] - 0.5,
            grad_norm_ref=None,
            solver="newton",
            n_iters_spent=1,
            eps_target=1e-16,
            grad_tol_target=None,
            status="ok",
            wall_time_s=0.0,
        ),
    )
    aggregate(
        campaign="tiny_cached",
        raw_dir=raw_dir,
        output=agg_path,
        eps=1e-8,
        rate_tail=0.5,
        reference_dir=cache_dir,
        use_reference_cache=True,
    )
    per_campaign.render(agg_path, tex_path)
    tex = tex_path.read_text()
    assert "reference solution" in tex.lower()
    assert "WARNING" not in tex
