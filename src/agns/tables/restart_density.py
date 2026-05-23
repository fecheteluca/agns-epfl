"""Cross-campaign restart-density diagnostic table.

One row per (campaign, AGNS-family method) summarising the restart
event statistics emitted by :mod:`agns.methods.agns` and
:mod:`agns.methods.agns_wsm`.  Methods without restart instrumentation
(anything not in the AGNS family) are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agns.tables._tex import build_booktabs, escape_label

__all__ = ["render"]


_AGNS_FAMILY_PREFIXES: tuple[str, ...] = ("agns",)


def _is_agns_family(method_key: str) -> bool:
    return any(method_key.startswith(prefix) for prefix in _AGNS_FAMILY_PREFIXES)


def _fmt_int(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return str(round(float(x)))


def _fmt_density(x: float) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{x:.3f}"


def render(aggregated_dir: Path, output: Path) -> None:
    """Render the cross-campaign restart-density table.

    The aggregator emits a ``restart_summary`` block on every AGNS-family
    method (and only those); this table consumes those blocks and lays
    out one row per (campaign, method).  Methods that produced no
    restart instrumentation are silently skipped.
    """
    json_paths = sorted(aggregated_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"no aggregated JSONs under {aggregated_dir}")

    header = [
        "Campaign",
        "Method",
        "Mode",
        "Iters",
        "Restarts",
        "Density",
        "First",
        "Max $\\ell_k$",
    ]
    colspec = "llllrrrr"

    rows: list[list[str]] = []
    for jp in json_paths:
        with open(jp) as fh:
            doc = json.load(fh)
        campaign = doc.get("campaign", jp.stem)
        methods = doc.get("methods", {})
        for mkey in sorted(methods):
            if not _is_agns_family(mkey):
                continue
            rec = methods[mkey]
            label = rec.get("label", mkey)
            rs = rec.get("restart_summary")
            if not rs:
                continue
            mode = rs.get("restart_mode") or "?"
            iters_med = float(np.median(rs.get("n_iters_per_seed") or [float("nan")]))
            restarts_med = rs["n_restarts_summary"]["median"]
            density_med = rs["restart_density_summary"]["median"]
            first_iters = [
                v for v in (rs.get("first_restart_iter_per_seed") or []) if v is not None
            ]
            first_med = float(np.median(first_iters)) if first_iters else float("nan")
            max_lk_med = float(np.median(rs.get("max_local_k_per_seed") or [0]))
            rows.append([
                escape_label(campaign),
                escape_label(label),
                escape_label(mode),
                _fmt_int(iters_med),
                _fmt_int(restarts_med),
                _fmt_density(density_med),
                _fmt_int(first_med),
                _fmt_int(max_lk_med),
            ])

    if not rows:
        rows = [["(no AGNS-family runs found)"] + [""] * (len(header) - 1)]

    caption = (
        "Per-campaign restart-density diagnostics for every AGNS-family method. "
        "Density is the median across seeds of n\\_restarts / n\\_iters. "
        "``First'' is the median iteration at which the first restart fired "
        "(``--'' if no seed ever restarted)."
    )
    tex = build_booktabs(colspec, header, rows, caption=caption, label="tab:restart-density")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex)
    print(f"  [restart-density] wrote {output}")
