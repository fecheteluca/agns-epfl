"""Render figures from an aggregated JSON (or several).

Thin dispatcher over :mod:`agns.plots`.  Two top-level modes:

* per-campaign (``--aggregated <campaign>.json --output-dir DIR``):
  reads one aggregated JSON and emits the panels selected by
  ``--types``: ``iter_convergence``, ``time_convergence``,
  ``grad_convergence``, ``rate_loglog``, ``sweep`` (when applicable).
* report figures (``--report-figures --results-dir DIR``): emits the
  cross-campaign panels (currently ``scaling_time``) into
  ``<results-dir>/paper_figures/``.

Per-plot implementations live in :mod:`agns.plots.convergence`,
:mod:`agns.plots.rate_loglog`, :mod:`agns.plots.sweep`, and
:mod:`agns.plots.scaling`.

Usage::

    python -m agns.cli.make_plots \\
        --aggregated results/numerical/aggregated/rate_verification.json \\
        --output-dir results/figures/rate_verification \\
        --no-title

    python -m agns.cli.make_plots --report-figures --results-dir results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agns.plots import convergence, rate_loglog, scaling, sweep

__all__ = ["main"]

_VALID_TYPES = ("iter_convergence", "time_convergence", "grad_convergence", "rate_loglog", "sweep")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--aggregated",
        default=None,
        help="Path to <campaign>.json (per-campaign mode; required unless "
        "--report-figures is given).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the per-campaign figures into.",
    )
    p.add_argument(
        "--report-figures",
        action="store_true",
        help="Render the cross-campaign report figures into "
        "<results-dir>/paper_figures/ instead.",
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Results root for --report-figures mode (default: results).",
    )
    p.add_argument(
        "--types",
        nargs="+",
        default=list(_VALID_TYPES),
        help="Figure types to produce.",
    )
    p.add_argument("--title", default=None, help="Override figure title.")
    p.add_argument(
        "--no-title",
        action="store_true",
        help="Suppress titles (captions carry the take-away).",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Limit to this subset of method keys (default: every method).",
    )
    p.add_argument(
        "--base-methods-dir",
        default="configs/_base/methods",
        help="Directory containing ``_base/methods/<name>.yaml`` for fallback labels.",
    )
    p.add_argument(
        "--nu-ref",
        type=float,
        default=None,
        help="Hölder exponent for the rate_loglog reference line ``C / K^{2+nu}``.",
    )
    p.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=None,
        metavar=("W", "H"),
        help="Override figure size in inches.",
    )
    p.add_argument(
        "--no-legend",
        action="store_true",
        help="Suppress the per-figure legend.",
    )
    return p.parse_args()


def _run_report_figures(results_dir: Path) -> None:
    """Render the cross-campaign report figures into ``paper_figures/``."""
    agg = results_dir / "numerical" / "aggregated"
    out = results_dir / "paper_figures"
    candidates = [agg / f"{c}.json" for c in ("scaling_lr", "scaling_lr_n200", "scaling_lr_n800")]
    present = [p for p in candidates if p.is_file()]
    if len(present) >= 2:
        scaling.render(present, out)
    else:
        print("  [figure] WARN: scaling campaigns missing; skipping scaling_time")


def main() -> None:
    args = parse_args()

    if args.report_figures:
        _run_report_figures(Path(args.results_dir).resolve())
        return

    if not args.aggregated or not args.output_dir:
        sys.exit(
            "per-campaign mode requires --aggregated and --output-dir "
            "(or pass --report-figures for the cross-campaign figures)"
        )
    agg_path = Path(args.aggregated).resolve()
    if not agg_path.is_file():
        sys.exit(f"aggregated JSON not found: {agg_path}")
    out_dir = Path(args.output_dir).resolve()
    base_dir = Path(args.base_methods_dir).resolve()

    with open(agg_path) as fh:
        agg = json.load(fh)

    methods: dict[str, dict[str, Any]] = dict(agg.get("methods", {}))
    if args.methods is not None:
        methods = {k: v for k, v in methods.items() if k in args.methods}
    if not methods:
        sys.exit("no methods to plot (filter excluded everything)")

    title = None if args.no_title else args.title
    rc_override: dict[str, Any] = {}
    if args.figsize:
        rc_override["figure.figsize"] = (float(args.figsize[0]), float(args.figsize[1]))
    show_legend = not args.no_legend

    common_kwargs: dict[str, Any] = {
        "base_methods_dir": base_dir,
        "title": title,
        "show_legend": show_legend,
        "rc_override": rc_override,
    }

    for ftype in args.types:
        if ftype in ("iter_convergence", "time_convergence", "grad_convergence"):
            convergence.render(ftype, methods, out_dir, **common_kwargs)
        elif ftype == "rate_loglog":
            rate_loglog.render(methods, out_dir, nu_ref=args.nu_ref, **common_kwargs)
        elif ftype == "sweep":
            if sweep.has_sweep_methods(methods):
                sweep.render(methods, out_dir, **common_kwargs)
        else:
            print(f"  [figure] WARN: unknown figure type {ftype!r}, skipping")


if __name__ == "__main__":
    main()
