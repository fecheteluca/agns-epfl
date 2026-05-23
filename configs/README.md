# Campaign configs

Every benchmark campaign is a YAML composition of two base layers:

- `_base/problems/*.yaml` — one per problem type (the `problem.type`
  string resolves to a factory in `agns.oracles.PROBLEM_REGISTRY`).
- `_base/methods/*.yaml` — one per registered method (the `name` field
  resolves to a key in `agns.pipeline.registry.METHOD_REGISTRY`).

A campaign YAML at `configs/<family>/<name>.yaml` typically looks like:

```yaml
include:
  - ../_base/problems/<problem>.yaml

problem:
  params: { ... }              # per-campaign overrides

common:
  gamma_0: 1.0
  eps: 1.0e-10
  adaptive_search: true

methods:
  - include: [../_base/methods/gns_exact.yaml]
    n_iters: 150
  - include: [../_base/methods/agns_exact.yaml]
    n_iters: 150
  # ...

output:
  save_dir: results/numerical/raw/<campaign>
  title: "Human-readable title for figures"
```

## Composition rules (see `src/agns/pipeline/config.py`)

- `include:` is a list of paths resolved *relative to the file
  containing the include*.  Each include is loaded recursively and
  *deep-merged* underneath the current node.  Keys at the current node
  override keys from includes.
- Dicts merge deeply, key by key.  Lists are replaced outright.  This
  is the only sane behaviour for benchmark plotting: a leaf saying
  `methods: [A, B]` wants exactly those, not the union with whatever
  the base file defined.
- Method entries inside `methods:` are themselves dicts and may carry
  their own `include:` directive.  That lets a campaign pull
  `_base/methods/<name>.yaml` for default colour / linestyle / iter
  budget while supplying just the overrides.
- Cycles are detected and raise `ConfigError`.

## Sweeps

A method entry may carry a `sweep:` block to expand into multiple
clones, one per value:

```yaml
methods:
  - include: [../_base/methods/agns_inexact.yaml]
    n_iters: 150
    sweep:
      param: gamma_0
      values: [0.01, 0.1, 1.0, 10.0, 100.0]
```

Each clone gets `param` set to one element of `values` and a unique
`key = <name>__<param>=<value>` so the pickles do not collide and the
aggregator records them as distinct methods.

Multiple sweep entries on the same method list are independent (each
expands separately).  See `src/agns/pipeline/sweeps.py` and the
`configs/ablations/momentum_x_restart_grid.yaml` campaign for an
example of two independent sweeps in one `methods:` list.

## Adding a new campaign

1. Pick (or add) a `_base/problems/<problem>.yaml`.
2. Pick the method set from `_base/methods/`.  Every campaign in a
   given problem class should reuse the same canonical method set so
   the comparison axis is consistent across datasets.
3. Add the campaign to `scripts/run_experiments.sh` so a full pipeline
   refresh picks it up.
4. Verify by dry-running:
   `AGNS_SEEDS="0" python -m agns.cli.run_benchmark --config <path>`.
