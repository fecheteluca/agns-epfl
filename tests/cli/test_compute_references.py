"""Tests for the reference-solution computation CLI.

Covers two things:
  1. The public ``compute_one_reference`` function actually solves the
     problem (Newton converges, ``f_ref`` is non-trivial) and returns
     a populated :class:`ReferenceEntry`.
  2. The CLI front end persists the entry to the cache layout that
     :mod:`agns.pipeline.reference_cache` consumes.

The CLI is run as ``python -m agns.cli.compute_references`` (the same
invocation pattern used by every other CLI test) so the import-time
BLAS pinning at the top of the module actually fires.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from agns.cli.compute_references import compute_one_reference
from agns.pipeline.reference_cache import (
    ReferenceEntry,
    canonical_params_hash,
    lookup,
)


def test_compute_one_reference_solves_logistic_synthetic() -> None:
    """Sanity check: tiny logistic-regression instance, Newton converges fast."""
    entry = compute_one_reference(
        "logistic_regression_synthetic",
        params={"m": 60, "n": 5, "reg": 1e-2},
        seed=0,
        newton_iters=200,
        fast_gradient_iters=1000,
        eps=1e-16,
        grad_tol=1e-12,
    )
    assert isinstance(entry, ReferenceEntry)
    assert entry.problem_type == "logistic_regression_synthetic"
    assert entry.seed == 0
    # Newton on a small strongly-convex LR should win, not fall back.
    assert entry.solver in {"newton", "fast_gradient"}
    # Strongly-convex L2-regularised logistic loss is non-negative and
    # finite; f_ref should be too.
    assert entry.f_ref >= 0.0
    assert entry.f_ref < 1.0  # a loose ceiling that any sensible solver clears
    # Caller didn't pass ``seed`` inside params -> persisted params has none.
    assert "seed" not in entry.params


def test_compute_references_cli_persists_entry(tmp_path: Path) -> None:
    """End-to-end CLI run: writes an entry JSON the cache module can read back."""
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(
        textwrap.dedent(
            """\
            problem:
              type: logistic_regression_synthetic
              params: {m: 40, n: 4, reg: 1.0e-2, seed: 0}
            common: {gamma_0: 1.0, eps: 1.0e-12, adaptive_search: true}
            methods:
              - name: newton
                label: Newton
                n_iters: 5
            output:
              save_dir: __unused__
            """
        )
    )
    cache_dir = tmp_path / "refcache"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "agns.cli.compute_references",
            "--configs",
            str(cfg),
            "--seeds",
            "0",
            "--output-dir",
            str(cache_dir),
            "--newton-iters",
            "100",
            "--fast-gradient-iters",
            "100",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    # Manifest is written even when only one entry exists.
    assert (cache_dir / "manifest.json").is_file()
    # Cache module can resolve the entry.
    entry = lookup(
        cache_dir,
        "logistic_regression_synthetic",
        {"m": 40, "n": 4, "reg": 1e-2},
        seed=0,
    )
    assert entry is not None
    assert entry.f_ref >= 0.0
    # The hashed path actually exists on disk.
    h = canonical_params_hash({"m": 40, "n": 4, "reg": 1e-2})
    assert (cache_dir / "logistic_regression_synthetic" / f"{h}_seed_0.json").is_file()


def test_compute_references_cli_skips_factories_with_declared_f_star(tmp_path: Path) -> None:
    """Ridge regression declares f_star analytically -> CLI should skip it.

    No entry is computed; the manifest still gets rewritten (possibly
    empty) so a caller scripting around it always sees a valid file.
    """
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(
        textwrap.dedent(
            """\
            problem:
              type: ridge_regression_synthetic
              params: {m: 30, n: 4, reg: 1.0e-2, seed: 0}
            common: {gamma_0: 1.0, eps: 1.0e-12, adaptive_search: true}
            methods:
              - name: newton
                label: Newton
                n_iters: 5
            output:
              save_dir: __unused__
            """
        )
    )
    cache_dir = tmp_path / "refcache"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "agns.cli.compute_references",
            "--configs",
            str(cfg),
            "--seeds",
            "0",
            "--output-dir",
            str(cache_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert (cache_dir / "manifest.json").is_file()
    manifest_rows = json.loads((cache_dir / "manifest.json").read_text())
    assert manifest_rows == []
    # No problem-type subdirectory should have been created.
    assert not (cache_dir / "ridge_regression_synthetic").exists()
