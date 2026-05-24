# Statistical methodology

This document describes the confidence-interval and significance-test
machinery the benchmark uses, primarily in the AGNS-vs-GNS speedup
table.

## What the speedup table reports

For each campaign and each matched (GNS, AGNS) pair the table reports:

- `iters(GNS)` — median over seeds of per-seed first iteration where the GNS residual ≤ `eps`.
- `iters(AGNS)` — same, for AGNS.
- `Speedup` — `median(iters_GNS) / median(iters_AGNS)`.
- `[CI95]` — pair-preserving percentile bootstrap 95% CI around `Speedup`.
- `p_W` — two-sided paired Wilcoxon signed-rank test on per-seed iter counts; `n/a` when too few paired seeds (see *Dropping rule*).
- `≈` — flag fires when the CI brackets `1.0` **or** `p_W > 0.05`. The data do not support a directional speedup claim on this sample.

`eps` is whatever value the **aggregator** (`agns.cli.aggregate`) used
when it computed `iters_to_eps_per_seed` — set via the aggregator's
`--eps` flag (default `1e-8`). The speedup-table CLI no longer exposes
its own `--eps`, so the eps semantics are unambiguous.

## Bootstrap procedure

- **Statistic**: `median(a) / median(b)`, where `a` and `b` are the per-seed iter-to-eps arrays of the two methods.
- **Resampling**: pair-preserving. On each of `n_resamples` iterations a vector of `n_seeds` indices is drawn with replacement, and both arrays are indexed with the *same* draw. This matches the experimental design (same seed produces the same problem instance and is presented to both methods).
- **Number of resamples**: `n_resamples = 10_000`. The Monte Carlo error on the 2.5% / 97.5% quantiles at `n=10_000` is roughly 1% of the CI half-width — small enough that the headline number is stable across re-runs.
- **RNG seed**: pinned to `agns.utils.metrics.BOOTSTRAP_DEFAULT_RNG_SEED = 0`. The CI is therefore **deterministic** given the per-seed iter counts; two re-runs against the same aggregated JSON produce byte-identical CI endpoints.
- **Handling `None` entries**: a seed where the method never crossed `eps` within the iteration budget is recorded as `None` in the per-seed list. The bootstrap converts `None` to `+inf` before resampling, so a resample dominated by non-crossers reports an infinite ratio — the truthful answer when one side fails to converge.
- **CI endpoint when every resample is non-finite**: both `lo` and `hi` are `nan`. The renderer formats this as `--`.

The helper is `agns.utils.metrics.bootstrap_ratio_ci`; it returns a `BootstrapCI` dataclass with `point_estimate`, `lo`, `hi`, `n_resamples`, `n_paired_seeds`, `rng_seed`.

## Wilcoxon procedure

- **Statistic**: paired Wilcoxon signed-rank, via `scipy.stats.wilcoxon`.
- **Pairing**: same as the bootstrap — seed `i` of method A paired with seed `i` of method B.
- **Dropping rule**: pairs where either side is `None` / non-finite are dropped before calling scipy. After dropping, if fewer than 5 paired seeds remain the helper returns `None` and the renderer prints `n/a`. Wilcoxon at `n<5` is too underpowered to be informative.
- **Identical input short-circuit**: when the surviving pairs are bit-identical the helper returns `(0.0, 1.0, n)` — scipy raises in that case but the empirically meaningful answer is "no evidence of difference".
- **Tail**: two-sided. Helper is `agns.utils.metrics.paired_wilcoxon`.

## When the `≈` flag fires

A row carries `\(\approx\)` in the flag column when **either**:

- the bootstrap CI covers `1.0`, **or**
- the Wilcoxon p-value is `> 0.05`.

Both conditions mean "the data don't support a directional claim that AGNS is faster than GNS on this campaign at this sample size". A row without the flag has either a CI strictly above (or strictly below) 1.0 *and* a Wilcoxon p ≤ 0.05.

## Why median(per-seed ratios)-of-paired-counts (not ratio-of-medians-of-curves)

An alternative speedup statistic is `iters(GNS_median_curve crosses eps) / iters(AGNS_median_curve crosses eps)` — first crossing of the smoothed median curve. That statistic has two drawbacks:

1. It mixes seeds in a way that **breaks pairing**: the median curve is a per-iteration percentile, not a per-seed object, so a bootstrap CI around it would have to resample iterations, which is the wrong sampling unit.
2. The smoothed-median first-crossing can differ from the median of per-seed first-crossings (consider three seeds with crossings at iter 5, 10, 1000 — the per-seed median is 10, but the smoothed median curve crosses much later, dragged by the non-converger).

`median(per_seed_iters_GNS) / median(per_seed_iters_AGNS)` resolves both: the bootstrap naturally resamples seed indices, and the point estimate is robust to one badly-behaved seed.

## Why 10 seeds, not 5

Five seeds is the absolute minimum: with integer iter-to-eps counts in the 3–22 range (typical for real-world LIBSVM), the bootstrap CI is dominated by sampling noise -- a single seed at the median can shift the resample distribution by 20% per iteration. Ten seeds roughly halves the standard error of the median, which is why ten is the current default.

The runner's `AGNS_SEEDS` default is `"0 1 2 3 4 5 6 7 8 9"`. Override with the env var for shorter smoke runs (`AGNS_SEEDS="0"` for a sanity check) or longer runs (`AGNS_SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"` for 15 seeds).

## What this machinery does *not* do

- It does not correct for multiple comparisons across the ~50 rows of the speedup table. The Wilcoxon p-values are per-row. A reader who wants a family-wise error rate should apply a Bonferroni or Holm-Bonferroni adjustment themselves.
- It does not compute CIs for the per-campaign or real-world summary tables. Those tables already report IQR, which is a non-parametric spread measure; layering a CI on top adds complexity without changing the qualitative story.
- The grad-call and wall-clock variants of the speedup table
  (`agns_speedup_grad.tex`, `agns_speedup_time.tex`) reuse this same
  bootstrap and Wilcoxon machinery, only swapping the per-seed payload
  (`grad_calls_to_eps_per_seed` and `wall_time_to_eps_per_seed`,
  respectively) for the iter payload (`iters_to_eps_per_seed`).  The
  wall-clock table's caption stamps the host fingerprint so two runs
  produced on different machines are not silently compared.

## Versioning

The helper signatures are part of the package's public API. Any change to the default `n_resamples`, the default RNG seed, or the resampling scheme is a **behavioural** change that will move CI endpoints. Pin a specific package version if reproducible CI endpoints across deployments matter.
