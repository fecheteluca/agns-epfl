# CS-439 Project - AGNS: Accelerated Gradient-Regularised Newton 

Reference implementation and benchmark suite for **Gradient-Normalized
Smoothness (GNS) Newton** (Semenov, Jaggi, Doikov, arXiv 2506.13710,
Jun 2025) and an experimental **accelerated variant (AGNS)**.

GNS is the regularised-Newton method

```text
x_{k+1} = x_k − (H(x_k) + ‖∇f(x_k)‖_* / γ_k · B)^{−1} ∇f(x_k)
```

with an adaptive step-size `γ_k` chosen by a doubling/halving search
that enforces the progress predicate

```text
f(x_k) − f(x_{k+1}) ≥ γ_k / 8 · ‖∇f(x_{k+1})‖²_* / ‖∇f(x_k)‖_*.
```

AGNS extends GNS with a Nesterov-style momentum schedule and an O'Donoghue–Candes gradient restart.  The acceleration is heuristic and not theoretically justified by the GNS analysis; this repo provides the empirical study.

## What's here

* `src/agns/methods/` — every method as a free function returning the
  canonical `(x_final, status, history)` tuple.  GNS family
  (`gns`, `gns_wsm`), AGNS family (`agns`, `agns_wsm`), plus baseline
  Newton-family (`newton`, `cubic_newton`, `accelerated_cubic_newton`,
  `picard_acn_2008`, `super_newton`, `trust_region`), first-order
  (`gradient`, `fast_gradient`, `heavy_ball`, `lbfgs`), and ML
  (`adam`, `adahessian`).
* `src/agns/oracles/` — problem oracles for logistic / softmax /
  ridge / smoothed-lasso / LogSumExp / nonlinear-equations / Chebyshev /
  Rosenbrock / phase-retrieval / matrix-completion / soft-SVM / MLP.
  Each exposes `func`, `grad`, `hess`, `hess_vec` and is cross-checked
  by `tests/oracles/test_finite_differences.py`.
* `src/agns/cli/` — the data → reference-cache → run → aggregate →
  figures → tables pipeline, callable both as `python -m agns.cli.<command>`
  and via the `agns-*` console scripts in `pyproject.toml`.
* `src/agns/pipeline/reference_cache.py` and
  `src/agns/cli/compute_references.py` — per-instance cache of reference
  `f_star` solves, used by the aggregator instead of the
  per-seed-minimum-across-methods fallback.
* `configs/` — every campaign as a YAML composition of `_base/methods/`
  and `_base/problems/` includes; `configs/README.md` documents the
  composition rules.
* `scripts/run_experiments.sh` — regenerate every benchmark artefact
  end to end (data download → reference cache → benchmark → aggregate →
  figures → per-campaign tables → cross-dataset rollup + speedup +
  noise-floor + restart-density tables → cross-campaign figures).
* `tests/` — full test suite covering every oracle's gradient / Hessian
  via finite-difference, every method's smoke + monotone-progress
  invariant, the config loader, the sweep expander, the aggregator
  (including the reference-cache precedence path), the renderers
  (per-campaign, real-world summary, restart density, speedup on three
  axes, noise floor), and the AGNS restart-mode coverage.

## Quick start

```bash
# Install + download data
uv pip install -e ".[dev]"
bash scripts/download_data.sh

# Smoke run (one seed, approx 30-40 minutes)
AGNS_SEEDS="0" bash scripts/run_experiments.sh

# Full run (10 seeds, several hours; the script is sequential across
# campaigns and seeds — wrap with GNU parallel or xargs -P to parallelise)
bash scripts/run_experiments.sh
```

Artefacts land under `results/`: raw pickles in `numerical/raw/`,
aggregated JSONs in `numerical/aggregated/`, figures in `figures/`,
LaTeX tables in `tables/` (per-campaign tables plus the headline
cross-cutting tables `real_world_summary.tex`, `restart_density.tex`,
`agns_speedup.tex` / `agns_speedup_grad.tex` / `agns_speedup_time.tex`,
and `noise_floor.tex`), and cached reference solutions in
`reference_solutions/`.

## Reproducibility

* All RNGs (numpy / python / torch) are pinned per-seed via
  `agns.utils.seed.set_global_seed` (`src/agns/cli/run_benchmark.py`).
* All BLAS / OpenMP libraries are pinned to 1 thread before any numpy
  import so reductions are bit-stable across re-runs.
* Every LIBSVM dataset is SHA-256-verified against
  `scripts/checksums.txt`.
* Aggregated JSONs under `results/numerical/aggregated/` are the
  campaign artefact of record.  Compare two runs with a recursive
  `diff` of that directory to surface any numerical drift.

## License

MIT.  See `pyproject.toml`.
