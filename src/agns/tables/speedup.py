"""AGNS-vs-GNS iteration-budget speedup table (with restart-role column).

For every campaign that contains a matched (GNS, AGNS) pair, report the
iterations needed to reach the residual target ``eps`` (median trace,
first crossing) and the speedup ratio iters(GNS) / iters(AGNS).  The
no-restart column pairs against the matching AGNS variant
(exact↔exact, inexact↔inexact) so the restart contribution is
mathematically meaningful.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agns.tables._tex import build_booktabs, escape_label

__all__ = ["render"]


def _iters_to_eps(iter_curve_median: list[float], eps: float) -> float | None:
    """Median iteration index at which the residual first drops to ``eps``."""
    arr = np.asarray(iter_curve_median, dtype=float)
    crossings = np.where(arr <= eps)[0]
    if crossings.size == 0:
        return None
    return float(crossings[0])


def render(aggregated_dir: Path, output: Path, *, eps: float = 1e-8) -> None:
    """Render the AGNS-vs-GNS speedup table.

    Each row pairs a GNS variant with its matching AGNS variant
    (exact-exact, inexact-inexact).  The ``Restart contrib.`` column is
    (Speedup - Speedup-no-restart).
    """
    json_paths = sorted(aggregated_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"no aggregated JSONs under {aggregated_dir}")

    header = [
        "Campaign",
        "Variant",
        "iters(GNS)",
        "iters(AGNS)",
        "Speedup",
        "iters(no-r)",
        "Speedup no-r",
        "Restart contrib.",
    ]
    colspec = "llrrrrrr"

    pairings: list[tuple[str, str, str, str]] = [
        ("exact H",   "gns_exact",   "agns_exact",   "agns_exact_norestart"),
        ("inexact H", "gns_inexact", "agns_inexact", "agns_inexact_norestart"),
    ]

    rows: list[list[str]] = []
    for jp in json_paths:
        with open(jp) as fh:
            doc = json.load(fh)
        campaign = doc.get("campaign", jp.stem)
        methods = doc.get("methods", {})
        for variant_label, gns_key, agns_key, nr_key in pairings:
            gns = methods.get(gns_key)
            agns = methods.get(agns_key)
            if gns is None or agns is None:
                continue
            agns_nr = methods.get(nr_key)
            gns_iters = _iters_to_eps(gns["iter_curve"]["median"], eps)
            agns_iters = _iters_to_eps(agns["iter_curve"]["median"], eps)
            agns_nr_iters = (
                _iters_to_eps(agns_nr["iter_curve"]["median"], eps)
                if agns_nr is not None
                else None
            )
            speedup = gns_iters / agns_iters if (gns_iters and agns_iters) else float("nan")
            speedup_nr = (
                gns_iters / agns_nr_iters if (gns_iters and agns_nr_iters) else float("nan")
            )
            restart_contrib = (
                speedup - speedup_nr
                if (np.isfinite(speedup) and np.isfinite(speedup_nr))
                else float("nan")
            )
            rows.append([
                escape_label(campaign),
                variant_label,
                "--" if gns_iters is None else f"{int(gns_iters)}",
                "--" if agns_iters is None else f"{int(agns_iters)}",
                "--" if not np.isfinite(speedup) else f"{speedup:.2f}",
                "--" if agns_nr_iters is None else f"{int(agns_nr_iters)}",
                "--" if not np.isfinite(speedup_nr) else f"{speedup_nr:.2f}",
                "--" if not np.isfinite(restart_contrib) else f"{restart_contrib:+.2f}",
            ])

    if not rows:
        rows = [["(no campaigns with matched GNS / AGNS pairs)"] + [""] * (len(header) - 1)]

    caption = (
        f"AGNS-vs-GNS iteration-budget speedup at $\\varepsilon = {eps:.0e}$.  "
        "Each row pairs a GNS variant with the matching AGNS variant "
        "(exact$\\leftrightarrow$exact, inexact$\\leftrightarrow$inexact). "
        "``iters'' is the first median-trace iteration to reach the residual target; "
        "``Speedup'' is iters(GNS) / iters(AGNS); ``no-r'' columns use AGNS with "
        "restart\\_mode=none. ``Restart contrib.'' is (Speedup $-$ Speedup no-r); "
        "positive means restart contributes to the speedup."
    )
    tex = build_booktabs(colspec, header, rows, caption=caption, label="tab:agns-speedup")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex)
    print(f"  [speedup] wrote {output}")
