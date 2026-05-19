"""End-to-end pipeline smoke test.

Writes a tiny YAML config, runs the four CLI stages (run_benchmark ->
aggregate -> make_figures -> make_tables) on a temporary results
directory, and checks that the expected artefacts are produced.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_TINY_CFG = """\
problem:
  type: ridge_regression_synthetic
  params: {m: 60, n: 8, reg: 0.01, seed: 0}
common:
  gamma_0: 1.0
  eps: 1.0e-10
  adaptive_search: true
methods:
  - name: gns_exact
    label: GNS (exact H)
    n_iters: 10
  - name: agns_theory
    label: AGNS-Theory
    n_iters: 10
    rho: 0.7071067811865475
    theta: 0.5
  - name: gradient
    label: Gradient
    n_iters: 50
output:
  save_dir: __FILLED_IN_BY_TEST__
"""


def _run_module(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", module, *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    agg_path = tmp_path / "agg" / "smoke.json"
    figs_dir = tmp_path / "figures" / "smoke"
    table_path = tmp_path / "tables" / "smoke.tex"

    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(_TINY_CFG.replace("__FILLED_IN_BY_TEST__", str(raw_dir)))

    # 1. run_benchmark
    r = _run_module(
        "agns.cli.run_benchmark",
        "--config", str(cfg_path),
        "--output-dir", str(raw_dir),
        "--seeds", "0", "1",
    )
    assert r.returncode == 0, r.stderr
    assert (raw_dir / "seed_0" / "summary.json").is_file()
    assert (raw_dir / "seed_0" / "gns_exact_history.pkl").is_file()
    assert (raw_dir / "seed_1" / "summary.json").is_file()

    # 2. aggregate
    r = _run_module(
        "agns.cli.aggregate",
        "--campaign", "smoke",
        "--raw-dir", str(raw_dir),
        "--output", str(agg_path),
    )
    assert r.returncode == 0, r.stderr
    assert agg_path.is_file()
    with open(agg_path) as fh:
        agg = json.load(fh)
    assert agg["campaign"] == "smoke"
    assert "methods" in agg
    assert "gns_exact" in agg["methods"]

    # 3. make_figures
    r = _run_module(
        "agns.cli.make_figures",
        "--aggregated", str(agg_path),
        "--output-dir", str(figs_dir),
        "--no-title",
        "--types", "iter_convergence",
    )
    assert r.returncode == 0, r.stderr
    assert (figs_dir / "iter_convergence.pdf").is_file()
    assert (figs_dir / "iter_convergence.png").is_file()

    # 4. make_tables
    r = _run_module(
        "agns.cli.make_tables",
        "--aggregated", str(agg_path),
        "--output", str(table_path),
    )
    assert r.returncode == 0, r.stderr
    assert table_path.is_file()
    tex = table_path.read_text()
    assert r"\begin{table}" in tex
    assert r"\toprule" in tex


def test_run_benchmark_rejects_seed_and_seeds(tmp_path: Path) -> None:
    cfg_path = tmp_path / "x.yaml"
    cfg_path.write_text(_TINY_CFG.replace("__FILLED_IN_BY_TEST__", str(tmp_path / "out")))
    r = _run_module(
        "agns.cli.run_benchmark",
        "--config", str(cfg_path),
        "--seed", "0",
        "--seeds", "0", "1",
    )
    assert r.returncode != 0
    assert "mutually exclusive" in r.stderr


def test_aggregate_missing_raw_dir_fails(tmp_path: Path) -> None:
    r = _run_module(
        "agns.cli.aggregate",
        "--campaign", "x",
        "--raw-dir", str(tmp_path / "does_not_exist"),
        "--output", str(tmp_path / "out.json"),
    )
    assert r.returncode != 0


def test_make_figures_missing_agg_fails(tmp_path: Path) -> None:
    r = _run_module(
        "agns.cli.make_figures",
        "--aggregated", str(tmp_path / "missing.json"),
        "--output-dir", str(tmp_path / "figs"),
    )
    assert r.returncode != 0
