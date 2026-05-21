# CS-439: Project - Accelerated Gradient Normalized Smoothness (AGNS)

Empirical benchmarks for a family of gradient-regularised Newton
optimisation methods (GNS, AGNS-Practice, AGNS-Theory, AGNS-Holder,
AGNS-Inexact, AGNS-SC) alongside a broad set of baselines. GNS is the
Gradient-Normalized Smoothness method of Semenov, Jaggi & Doikov (2025);
the accelerated AGNS variants are this project's contribution (see
Acknowledgements).

The pipeline is config-driven: a YAML file under `configs/` describes a
problem (oracle factory + parameters) and a list of methods to run; the
runner produces per-seed history pickles, the aggregator collapses them
to a single JSON, and the figure / table renderers turn that JSON into
publication-ready artefacts.

## Installation

```bash
pip install -e ".[dev]"        # runtime + dev (pytest, ruff, mypy)
```

The package layout is `src/agns/`; after `pip install -e .` you can
import it as `agns` and invoke any CLI as `python -m agns.cli.<name>`.

## Layout

```text
optml_project/
├── pyproject.toml                Package metadata + entry points
├── scripts/                      Bash drivers
│   ├── run_experiments.sh        Full pipeline driver
│   ├── download_data.sh          LIBSVM fetch + SHA-256 verification
│   ├── datasets.txt              Dataset registry
│   └── checksums.txt             Pinned digests
├── configs/                      YAML campaign configs
│   ├── _base/methods/            Per-method defaults (label, colour, ...)
│   ├── _base/problems/           Per-problem defaults
│   └── <section>/<campaign>.yaml Leaf configs (one per campaign)
├── src/agns/
│   ├── config.py                 YAML loader with include composition
│   ├── sweeps.py                 sweep: directives -> method clones
│   ├── stopping.py               Shared stopping criteria
│   ├── linalg.py                 safe_cho_solve, wsm_rank_one_solve
│   ├── metrics.py                Log-log slope fit, median+IQR, Wilcoxon
│   ├── approximations.py         Hessian-approximation builders
│   ├── registry.py               METHOD_REGISTRY + run_method
│   ├── methods/                  One module per optimiser
│   ├── oracles/                  One module per problem family
│   ├── utils/                    io, logging, memory, seed, timing
│   └── cli/                      Command-line entry points
├── tests/                        pytest suite (228 tests)
└── results/                      Created by the pipeline
    ├── numerical/raw/            Per-seed pickles + summary.json
    ├── numerical/aggregated/     Median+IQR JSON per campaign
    ├── figures/                  PDF + PNG figures per campaign
    ├── paper_figures/            Cross-campaign report figures
    └── tables/                   LaTeX tables per campaign
```

All plotting lives in `agns.cli.make_figures`: per-campaign figures
(`--aggregated`) land in `results/figures/<campaign>/`, and the two
cross-campaign report figures (`--report-figures`) land in
`results/paper_figures/`.  `scripts/` holds only shell drivers.

## Quick start

Run the full reference campaign list end to end:

```bash
bash scripts/run_experiments.sh                  # all campaigns, 10 seeds
AGNS_SEEDS="0 1 2" bash scripts/run_experiments.sh  # 3-seed smoke pass
AGNS_SKIP_RUN=1 bash scripts/run_experiments.sh  # re-render figures+tables only
AGNS_DRY_RUN=1 bash scripts/run_experiments.sh   # print commands only
```

`AGNS_SKIP_RUN=1` skips the benchmark + aggregation steps and only
re-renders figures and tables from the existing aggregated JSONs --- the
fast path when you just want to reproduce the figures.

Run a single campaign manually:

```bash
python -m agns.cli.run_benchmark \
    --config configs/rate_verification/logsumexp.yaml \
    --seeds 0 1 2 3 4

python -m agns.cli.aggregate \
    --campaign rate_verification \
    --raw-dir results/numerical/raw/rate_verification \
    --output  results/numerical/aggregated/rate_verification.json

python -m agns.cli.make_figures \
    --aggregated results/numerical/aggregated/rate_verification.json \
    --output-dir results/figures/rate_verification \
    --no-title

python -m agns.cli.make_tables \
    --aggregated results/numerical/aggregated/rate_verification.json \
    --output     results/tables/rate_verification.tex
```

The shell driver pins BLAS / OpenMP threads to 1 before any Python
process starts so per-seed numerical fields are stable across re-runs.

## CLI entry points

| module                              | role                                            |
|-------------------------------------|-------------------------------------------------|
| `agns.cli.run_benchmark`            | run one campaign for one or more seeds          |
| `agns.cli.aggregate`                | per-seed pickles -> aggregated JSON             |
| `agns.cli.make_figures`             | aggregated JSON -> figures (+ report figures)   |
| `agns.cli.make_tables`              | aggregated JSON -> per-campaign LaTeX table     |
| `agns.cli.make_summary`             | cross-dataset rollup over real-world campaigns  |
| `agns.cli.manifest`                 | SHA-256 manifest of aggregated JSONs            |
| `agns.cli.diff`                     | diff two aggregated-results directories         |

Each module has a `main()` function and a `python -m agns.cli.<name>`
entry point.  Console scripts (`agns-run`, `agns-aggregate`, etc.) are
defined in `pyproject.toml`.

## Methods

The registry binds method keys (used in YAML configs) to free functions:

| registry key                | function                                 | notes                        |
|-----------------------------|------------------------------------------|------------------------------|
| `gns_exact`, `gns_inexact`  | `agns.methods.gns`                       | switch via `is_approx`       |
| `gns_wsm`                   | `agns.methods.gns_wsm`                   | Woodbury rank-1 backend      |
| `agns_practice`             | `agns.methods.agns_practice`             | momentum + restart           |
| `agns_practice_exact`       | `agns.methods.agns_practice`             | `is_approx=False`            |
| `agns_practice_wsm`         | `agns.methods.agns_practice`             | Woodbury rank-1 backend      |
| `agns_theory`               | `agns.methods.agns_theory`               | rho-strict MS-coupled        |
| `agns_holder`               | `agns.methods.agns_holder`               | annotates `nu` in the trace  |
| `agns_inexact`              | `agns.methods.agns_inexact`              | bounded Hessian perturbation |
| `agns_sc`                   | `agns.methods.agns_sc`                   | restarted AGNS-Theory        |
| `newton`                    | `agns.methods.newton`                    | Armijo + PSD regularisation  |
| `super_newton`              | `agns.methods.super_newton`              | Super-Universal Newton       |
| `cubic_newton`              | `agns.methods.cubic_newton`              | Nesterov-Polyak              |
| `accelerated_cubic_newton`  | `agns.methods.accelerated_cubic_newton`  | Nesterov 2008                |
| `monteiro_svaiter_acn`      | `agns.methods.monteiro_svaiter`          | MS-ACN (2013)                |
| `trust_region`              | `agns.methods.trust_region`              | Steihaug-CG sub-solver       |
| `gradient`, `fast_gradient` | `agns.methods.gradient`, `fast_gradient` | first-order baselines        |
| `heavy_ball`                | `agns.methods.heavy_ball`                | Polyak momentum              |
| `lbfgs`                     | `agns.methods.lbfgs`                     | scipy wrapper                |
| `adam`, `adahessian`        | `agns.methods.adam`, `adahessian`        | ML optimisers                |

Every method shares one call convention: it accepts the generic kwargs
(`n_iters`, `eps`, `f_star`, `grad_tol`, `B`, `Binv`, `trace`,
`warnings`) — a method omits only a dial that does not apply to it
(e.g. `cubic_newton` solves its subproblem matrix-free in `B` and so
takes no `Binv`) — and returns the canonical 3-tuple
`(x_final, status_string, history)`.  The core trace keys (`func`,
`time`, `grad_calls`, `x_k`) are consistent across the family; every
method except `cubic_newton` also records `grad_norm`, and individual
methods add their own keys (`gamma_k`, `H_k`, `L`, ...).

## Oracles

Every oracle subclasses `agns.oracles.BaseSmoothOracle`.  Factories in
each module return the canonical problem dict
`{oracle, x_0, f_star, B, Binv, approx_hess_fn, approx_hess_fn_wsm, meta}`.

The registry mapping problem-type string -> factory lives in
`agns.oracles.PROBLEM_REGISTRY`:

* `logsumexp_synthetic`, `logsumexp_real`
* `nonlinear_equations`, `chebyshev`, `rosenbrock_2d`, `rosenbrock_nle`
* `logistic_regression_synthetic`, `logistic_regression_libsvm`
* `ridge_regression_synthetic`, `ridge_regression_libsvm`
* `soft_svm_synthetic`, `soft_svm_libsvm`
* `smoothed_lasso_synthetic`, `softmax_regression_synthetic`
* `matrix_completion_synthetic`, `phase_retrieval_synthetic`
* `mlp_classifier_synthetic`, `cnn_classifier_synthetic` (torch-backed)

The `InexactHessianOracle` wrapper in `agns.oracles.inexact_wrapper`
adds a deterministic symmetric perturbation of exact spectral norm
`delta_budget` to every `hess(x)` call.

## Data

`scripts/download_data.sh` fetches every LIBSVM dataset listed in
`scripts/datasets.txt`, optionally bunzip2's, and SHA-256-verifies
each payload against `scripts/checksums.txt`.  Files land in
`data/<family>/<name>` with `family in {binary, regression}`.

## Testing

```bash
PYTHONPATH=src python3 -m pytest -q          # 228 tests, ~7 s
```

The suite covers:

* every shared module (`config`, `sweeps`, `stopping`, `linalg`,
  `metrics`, `approximations`, `utils`);
* every method (canonical-tuple shape, monotone descent on a small
  convex problem, trace-key consistency, per-method invariants);
* every oracle (finite-difference gradient checks, Hessian symmetry,
  factory dict shape);
* the four CLI stages end-to-end on a tiny ridge problem.

The LIBSVM- and torch-backed factories are excluded from the
parametrised factory tests: the LIBSVM ones need data fetched by
`download_data.sh`, and the MLP/CNN ones are held out to keep the suite
fast.  `torch` is a core dependency, so the MLP campaign itself still
runs in the full pipeline.

## Reproducibility

* `run_experiments.sh` pins `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`
  before any Python process starts.
* `agns.cli.manifest --write` records the SHA-256 of every aggregated
  JSON (with wall-clock fields stripped) into
  `scripts/aggregated_checksums.txt`; `--check` verifies the on-disk
  artefacts against that manifest.
* `agns.utils.io.save_pickle` / `save_json` use write-temp-then-rename
  so an interrupted run never leaves a half-written file on disk.

## Acknowledgements

The GNS method this project benchmarks and accelerates was introduced by
A. Semenov, M. Jaggi and N. Doikov, *Gradient-Normalized Smoothness for
Optimization with Approximate Hessians*, arXiv:2506.13710 (2025). This
benchmarking pipeline was inspired by the reference repository accompanying
that paper; the AGNS-Practice / AGNS-Theory accelerated variants and the
benchmark itself are contributions of this project.
