"""Diff two aggregated-results directories.

Diagnostic utility: quantify the numerical drift between two trees of
aggregated JSONs.  Computes per-campaign, per-method maximum relative
drift on the final-residual median and on the rate-fit slope.

Output is a Markdown table; methods whose final residual drifts beyond
``--threshold`` are flagged with a warning glyph and the process exits
non-zero (unless ``--no-fail``).

Usage::

    python -m agns.cli.diff \\
        --new results/numerical/aggregated \\
        --old results.prev/numerical/aggregated
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

__all__ = ["main"]


def _rel(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return float("nan")
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan") if (math.isnan(a) or math.isnan(b)) else 0.0
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom


def _load(p: Path) -> dict[str, Any]:
    with open(p) as f:
        return json.load(f)


def _diff_campaign(name: str, new_dir: Path, old_dir: Path) -> tuple[float, float, int, int]:
    new = _load(new_dir / f"{name}.json")
    old = _load(old_dir / f"{name}.json")

    methods_new = set(new.get("methods", {}))
    methods_old = set(old.get("methods", {}))
    common = methods_new & methods_old

    max_residual_drift = 0.0
    max_slope_drift = 0.0
    n_methods = 0
    n_methods_drifted = 0
    for m in common:
        a = new["methods"][m]
        b = old["methods"][m]
        n_methods += 1

        a_med = a.get("final_residual_summary", {}).get("median")
        b_med = b.get("final_residual_summary", {}).get("median")
        r1 = _rel(a_med, b_med)
        if r1 == r1:  # not NaN
            max_residual_drift = max(max_residual_drift, r1)

        a_sl = a.get("rate_fit", {}).get("slope")
        b_sl = b.get("rate_fit", {}).get("slope")
        if a_sl is not None and b_sl is not None:
            r2 = _rel(a_sl, b_sl)
            if r2 == r2:
                max_slope_drift = max(max_slope_drift, r2)

        if r1 == r1 and r1 > 0:
            n_methods_drifted += 1

    return max_residual_drift, max_slope_drift, n_methods, n_methods_drifted


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--new", required=True, help="Directory holding the new aggregated JSONs.")
    p.add_argument("--old", required=True, help="Directory holding the reference aggregated JSONs.")
    p.add_argument(
        "--threshold",
        type=float,
        default=5e-13,
        help="Drift threshold above which a campaign is flagged.",
    )
    p.add_argument(
        "--no-fail",
        action="store_true",
        help="Exit 0 even if drift exceeds the threshold.",
    )
    args = p.parse_args()

    new_dir = Path(args.new).resolve()
    old_dir = Path(args.old).resolve()
    if not new_dir.is_dir():
        sys.exit(f"new directory not found: {new_dir}")
    if not old_dir.is_dir():
        sys.exit(f"old directory not found: {old_dir}")

    new_campaigns = {p.stem for p in new_dir.glob("*.json")}
    old_campaigns = {p.stem for p in old_dir.glob("*.json")}
    only_new = new_campaigns - old_campaigns
    only_old = old_campaigns - new_campaigns
    common = sorted(new_campaigns & old_campaigns)

    print(f"# Aggregated-results diff: {new_dir} vs {old_dir}")
    print()
    print(f"- New only: {sorted(only_new) or 'none'}")
    print(f"- Old only: {sorted(only_old) or 'none'}")
    print(f"- Common campaigns: {len(common)}")
    print()
    print(f"threshold = {args.threshold:.0e}")
    print()
    print("| Campaign | Methods | Max residual drift | Max slope drift | Drifted |")
    print("|----------|---------|--------------------|-----------------|---------|")

    flag = 0
    for name in common:
        r, s, n, n_drift = _diff_campaign(name, new_dir, old_dir)
        warn = " :warning:" if r > args.threshold else ""
        if r > args.threshold:
            flag = 1
        print(f"| `{name}` | {n} | {r:.2e}{warn} | {s:.2e} | {n_drift}/{n} |")

    return 0 if (args.no_fail or flag == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
