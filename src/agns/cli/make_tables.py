"""Render LaTeX tables from aggregated JSONs.

Thin dispatcher over :mod:`agns.tables`.  One subcommand per table
type:

* ``per_campaign``       --- per-method residual+slope table for one
                             aggregated JSON (the default for
                             ``run_experiments.sh``).
* ``real_world_summary`` --- cross-dataset rollup over the
                             real-world LIBSVM campaigns.
* ``restart_density``    --- restart-event diagnostics across every
                             AGNS-family method.
* ``speedup``            --- iters(GNS) / iters(AGNS) with restart-role
                             column.

Usage::

    python -m agns.cli.make_tables per_campaign \\
        --aggregated results/numerical/aggregated/rate_verification.json \\
        --output results/tables/rate_verification.tex

    python -m agns.cli.make_tables real_world_summary \\
        --aggregated-dir results/numerical/aggregated \\
        --output results/tables/real_world_summary.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agns.tables import (
    noise_floor,
    per_campaign,
    real_world_summary,
    restart_density,
    speedup,
)

__all__ = ["main"]


def _per_campaign_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "per_campaign",
        help="Render the per-method residual+slope table for one aggregated JSON.",
    )
    p.add_argument("--aggregated", required=True, help="Path to <campaign>.json.")
    p.add_argument("--output", required=True, help="Path to write the .tex file.")
    p.add_argument(
        "--order",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="Method order (default: alphabetical).",
    )
    p.add_argument("--caption", default=None, help="Override the table caption.")
    p.add_argument("--label", default=None, help="Override the LaTeX label.")


def _summary_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "real_world_summary",
        help="Render the cross-dataset rollup over the real-world LIBSVM campaigns.",
    )
    p.add_argument(
        "--aggregated-dir",
        required=True,
        help="Directory of <campaign>.json files.",
    )
    p.add_argument("--output", required=True, help="Destination .tex path.")
    p.add_argument("--label", default="tab:real-world-summary", help="LaTeX label.")


def _restart_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "restart_density",
        help="Render the cross-campaign restart-density diagnostic table.",
    )
    p.add_argument("--aggregated-dir", required=True, help="Directory of aggregated JSONs.")
    p.add_argument("--output", required=True, help="Destination .tex path.")


def _speedup_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "speedup",
        help=(
            "Render an AGNS-vs-GNS speedup table.  Statistic is "
            "median(cost_GNS)/median(cost_AGNS) on the chosen --axis, "
            "with a paired bootstrap CI and Wilcoxon p-value.  Eps is "
            "taken from the aggregated JSON's eps_target."
        ),
    )
    p.add_argument("--aggregated-dir", required=True, help="Directory of aggregated JSONs.")
    p.add_argument("--output", required=True, help="Destination .tex path.")
    p.add_argument(
        "--axis",
        choices=("iter", "grad", "time"),
        default="iter",
        help=(
            "Cost axis: iter (iterations, default; the headline axis), "
            "grad (gradient-oracle calls), time (wall-clock seconds; "
            "host-dependent, fingerprint embedded in the caption)."
        ),
    )


def _noise_floor_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "noise_floor",
        help=(
            "Render the AGNS-vs-baseline noise-floor diagnostic table.  "
            "One row per (campaign, baseline) flagging cells where the "
            "speedup is indistinguishable from 1.00x on the available "
            "sample (CI brackets 1.0 OR Wilcoxon p > 0.05)."
        ),
    )
    p.add_argument("--aggregated-dir", required=True, help="Directory of aggregated JSONs.")
    p.add_argument("--output", required=True, help="Destination .tex path.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="kind", required=True)
    _per_campaign_parser(sub)
    _summary_parser(sub)
    _restart_parser(sub)
    _speedup_parser(sub)
    _noise_floor_parser(sub)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.kind == "per_campaign":
        try:
            per_campaign.render(
                Path(args.aggregated).resolve(),
                Path(args.output).resolve(),
                order=args.order,
                caption=args.caption,
                label=args.label,
            )
        except FileNotFoundError as e:
            sys.exit(str(e))
    elif args.kind == "real_world_summary":
        try:
            real_world_summary.render(
                Path(args.aggregated_dir).resolve(),
                Path(args.output).resolve(),
                label=args.label,
            )
        except (FileNotFoundError, RuntimeError) as e:
            sys.exit(str(e))
    elif args.kind == "restart_density":
        try:
            restart_density.render(
                Path(args.aggregated_dir).resolve(),
                Path(args.output).resolve(),
            )
        except FileNotFoundError as e:
            sys.exit(str(e))
    elif args.kind == "speedup":
        try:
            speedup.render(
                Path(args.aggregated_dir).resolve(),
                Path(args.output).resolve(),
                axis=args.axis,
            )
        except FileNotFoundError as e:
            sys.exit(str(e))
    elif args.kind == "noise_floor":
        try:
            noise_floor.render(
                Path(args.aggregated_dir).resolve(),
                Path(args.output).resolve(),
            )
        except FileNotFoundError as e:
            sys.exit(str(e))


if __name__ == "__main__":
    main()
