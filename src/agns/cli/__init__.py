"""Command-line entry points.

Each module here exposes a single ``main()`` function plus a ``python -m
agns.cli.<name>`` entry point.  The pipeline runs in four stages:

1. :mod:`agns.cli.run_benchmark` --- execute one campaign for one or
   more seeds, writing raw history pickles to
   ``results/numerical/raw/<campaign>/seed_*/``.
2. :mod:`agns.cli.aggregate` --- align the per-seed pickles into a
   single ``results/numerical/aggregated/<campaign>.json`` carrying
   median+IQR curves, final-residual statistics, and a log-log slope
   fit per method.
3. :mod:`agns.cli.make_figures` --- render every campaign figure (PDF
   + PNG) under ``results/figures/<campaign>/``.
4. :mod:`agns.cli.make_tables` --- render the per-campaign LaTeX table
   under ``results/tables/<campaign>.tex``.

Auxiliary tools:

* :mod:`agns.cli.make_summary` --- cross-dataset rollup table over the
  real-world LIBSVM campaigns.
* :mod:`agns.cli.manifest` --- SHA-256 manifest of the aggregated JSON
  files, used by CI to detect silent drift.
* :mod:`agns.cli.diff` --- human-readable diff between two aggregated
  result directories.
"""

from __future__ import annotations

__all__: list[str] = []
