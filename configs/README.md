# Experiment configurations

YAML configs consumed by `python -m agns.cli.run_benchmark`.

Each leaf config declares one *problem* (oracle factory + parameters)
and one or more *methods* to run against it.  Defaults that are shared
across configs live in `_base/`; leaves compose them via `include:`
directives understood by `agns.config.load_config`.

## Layout

```text
configs/
├── _base/
│   ├── methods/        One YAML per method: label, colour, linestyle, defaults
│   └── problems/       One YAML per problem family: oracle factory + defaults
│
├── rate_verification/  Log-log slope sanity-checks per problem class
├── convex_lipschitz/   Convex objectives with globally Lipschitz Hessian
├── holder_hessian/     Convex objectives with Holder-continuous Hessian
├── strongly_convex/    Strongly-convex objectives
├── non_convex/         Non-convex stress tests
├── neural_networks/    Tiny torch MLP / CNN benchmarks
├── inexact_hessian/    Hessian-noise (`delta_budget`) sweep
├── scaling/            Problem-size scaling study
├── oracle_cost_regime/ Regime where gradient calls dominate cost
├── real_world/         LIBSVM-backed convex benchmarks
└── ablations/          AGNS hyperparameter ablations
```

## Composition model

Every leaf YAML looks like this:

```yaml
include:
  - ../_base/problems/<problem>.yaml   # pulls in problem: { type, params }

problem:
  params:
    <override>: <value>                # leaf wins on conflicts

common:
  gamma_0: 1.0
  eps: 1.0e-10
  adaptive_search: true

methods:
  - include: [../_base/methods/<method>.yaml]
    n_iters: 200                       # per-method override

output:
  save_dir: results/numerical/raw/<campaign>
  title: "..."
```

The loader merges deeply for dicts and replaces outright for lists.  A
list element may carry its own `include:` so each `methods:` entry picks
up the defaults from a base-method YAML.

## Sweeps

A method entry may declare a `sweep:` block to unroll into one method
clone per value:

```yaml
methods:
  - include: [../_base/methods/agns_theory.yaml]
    n_iters: 200
    sweep:
      param: rho
      values: [0.5, 0.6, 0.7071, 0.85, 0.95]
```

`agns.sweeps.expand_sweeps` assigns each clone a unique
`<name>__<param>=<value>` key so the resulting pickles do not collide on
disk and the figure pipeline groups them automatically.  See
`ablations/rho_sweep.yaml` for the canonical example.

## Adding a new campaign

1. Drop a leaf YAML under the relevant subdirectory.
2. Add a row to the heredoc in `scripts/run_experiments.sh` so the
   driver picks it up.  Optional third column: `nu_ref` (forwarded to
   `make_figures --nu-ref` for Holder rate overlays).
3. Verify the config loads:
   `python -c "from agns.config import load_config; load_config('configs/<path>.yaml')"`.

`tests/test_config.py::test_repo_configs_all_loadable` runs that check
automatically over every YAML in this tree.
