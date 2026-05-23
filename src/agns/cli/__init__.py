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
3. :mod:`agns.cli.make_plots` --- thin dispatcher over
   :mod:`agns.plots`; renders every per-campaign figure (PDF + PNG)
   under ``results/figures/<campaign>/`` and the cross-campaign
   ``paper_figures/`` set when ``--report-figures`` is given.
4. :mod:`agns.cli.make_tables` --- thin dispatcher over
   :mod:`agns.tables`; subcommands ``per_campaign``,
   ``real_world_summary``, ``restart_density``, and ``speedup`` cover
   the four LaTeX-table artefact types.
"""

from __future__ import annotations

__all__: list[str] = []
