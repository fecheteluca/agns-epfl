"""Noise-floor diagnostic: AGNS vs each baseline, per campaign.

Companion to the headline ``agns_speedup.tex``.  Where the speedup
table answers "by how much does AGNS beat GNS at the iter axis?",
the noise-floor table answers a sharper question per baseline:

  Is the AGNS-vs-baseline speedup *statistically distinguishable*
  from 1.00x on the available sample?

A row carries:

* median per-seed iter-to-eps for AGNS and for the baseline;
* the point speedup ``median(iters_baseline) / median(iters_AGNS)``;
* a pair-preserving percentile bootstrap 95% CI on the speedup;
* the paired Wilcoxon p-value on the per-seed counts;
* a flag (``approx``) that fires when the CI brackets 1.0 OR p > 0.05.

The flag is the headline diagnostic: rows where it fires mean the
data does not support a directional speedup claim for that
(campaign, baseline) pair on the available sample.

Reuses the bootstrap helper + paired_wilcoxon from
:mod:`agns.utils.metrics`.  See ``docs/statistics.md`` for the
resampling design.  Sibling of :mod:`agns.tables.speedup` (whose
matched-pair iter-only headline this table generalises across the
full baseline roster).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agns.tables._tex import build_booktabs, escape_label
from agns.utils.metrics import (
    BOOTSTRAP_DEFAULT_RNG_SEED,
    bootstrap_ratio_ci,
    paired_wilcoxon,
)

__all__ = ["BASELINES", "render"]


# Column AGNS picks the first registry key it finds; ridge campaigns
# only carry ``agns_exact`` (no GN approximation for ridge), so the
# fallback chain mirrors the one in
# :mod:`agns.tables.real_world_summary`.
_AGNS_KEYS: list[str] = ["agns_inexact", "agns_exact"]

# Baselines compared against AGNS, one row per (campaign, baseline).
# Same display-name -> key-fallback-list shape as
# ``real_world_summary.SUMMARY_METHODS`` for cross-table consistency.
BASELINES: list[tuple[str, list[str]]] = [
    ("GNS", ["gns_inexact", "gns_exact"]),
    ("Newton", ["newton"]),
    ("Super-Newton", ["super_newton"]),
    ("Cubic-Newton", ["cubic_newton"]),
    ("ACN", ["accelerated_cubic_newton"]),
    ("Picard-ACN", ["picard_acn_2008"]),
    ("Trust-Region", ["trust_region"]),
    ("L-BFGS", ["lbfgs"]),
    ("Gradient", ["gradient"]),
    ("Fast-Gradient", ["fast_gradient"]),
]


def _per_seed_iters(rec: dict | None) -> list | None:
    """Return the per-seed iter-to-eps list, or ``None`` when unusable.

    Failure-only records and methods missing the field (older
    aggregated JSONs predating the schema additions) both yield
    ``None``; the renderer skips such rows.
    """
    if rec is None:
        return None
    series = rec.get("iters_to_eps_per_seed")
    if not series:
        return None
    return list(series)


def _resolve_record(
    methods: dict[str, dict[str, Any]],
    keys: list[str],
) -> tuple[str, dict[str, Any]] | None:
    """Pick the first ``keys`` entry whose record is in ``methods``."""
    for k in keys:
        if k in methods:
            return k, methods[k]
    return None


def _median_or_none(values: list) -> float | None:
    """Median treating ``None`` / non-finite as +inf."""
    if not values:
        return None
    arr = np.asarray(
        [
            np.inf if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)
            for v in values
        ],
        dtype=np.float64,
    )
    med = float(np.median(arr))
    return med if np.isfinite(med) else None


def _fmt_int(v: float | None) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return str(round(float(v)))


def _fmt_ratio(v: float) -> str:
    if not np.isfinite(v):
        return "--"
    return f"{v:.2f}"


def _fmt_ci(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "--"
    return f"[{lo:.2f}, {hi:.2f}]"


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "n/a"
    if p < 1e-3:
        return "<0.001"
    return f"{p:.3f}"


def _build_caption() -> str:
    return (
        r"AGNS-vs-baseline noise-floor diagnostic on the iter axis.  "
        r"Each row pairs AGNS (column ``AGNS variant'') with the named "
        r"baseline on a single campaign.  cost columns report the "
        r"across-seed median per-seed iters-to-$\varepsilon$ (eps from "
        r"the aggregated JSON).  Speedup is "
        r"$\text{median(iters\_baseline)} / \text{median(iters\_AGNS)}$ "
        r"with a pair-preserving percentile bootstrap 95\% CI "
        r"($n=10{,}000$ resamples, RNG seed pinned; see "
        r"\texttt{docs/statistics.md}).  $p_W$ is the paired Wilcoxon "
        r"p-value (n/a when fewer than five paired seeds crossed $\varepsilon$).  "
        r"$\approx$ flags rows where the CI brackets 1.0 or $p_W > 0.05$ "
        r"-- the data do not support a directional speedup claim on "
        r"this sample.  "
        r"\textit{Metric note:} AGNS / Super-Newton / Cubic-Newton / ACN / "
        r"Picard-ACN regularise with the problem-supplied $B$ "
        r"($H + \lambda B$); Newton, Trust-Region, and L-BFGS do not "
        r"(see README \S``Metric ($B$) asymmetry'').  On ill-conditioned "
        r"problems this favours the metric-aware family by an amount "
        r"that is independent of the AGNS momentum / restart structure."
    )


#: Marker appended to the Campaign cell of any row whose aggregated
#: JSON declared ``f_star_source = "per_seed_min_fallback"``.
FALLBACK_MARK = r"$^\dagger$"

_FALLBACK_FOOTNOTE = (
    r"  $\dagger$ marks campaigns whose $f^\star$ was the "
    r"per-seed minimum across methods (no cached reference solution).  "
    r"On such campaigns the per-seed winner's residual is mechanically "
    r"zero, so the speedup CI is biased toward the winner.  Compute "
    r"references with \texttt{agns-references --configs <yaml>} and "
    r"re-aggregate to remove the marker."
)


def render(aggregated_dir: Path, output: Path) -> None:
    """Render the AGNS-vs-baseline noise-floor table.

    For each ``aggregated_dir/<campaign>.json``, iterate the
    :data:`BASELINES` list and emit one row per (campaign, baseline)
    where both AGNS and the baseline carry per-seed iter-to-eps data.
    Rows without usable data on either side are silently skipped.
    """
    json_paths = sorted(aggregated_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"no aggregated JSONs under {aggregated_dir}")

    header = [
        "Campaign",
        "AGNS variant",
        "Baseline",
        "iters(AGNS)",
        "iters(base)",
        "Speedup",
        "[CI95]",
        "$p_W$",
        r"$\approx 1$",
    ]
    colspec = "l l l r r r l r c"

    rows: list[list[str]] = []
    any_fallback = False
    for jp in json_paths:
        with open(jp) as fh:
            doc = json.load(fh)
        campaign = doc.get("campaign", jp.stem)
        fstar_source = str(doc.get("f_star_source", "declared"))
        is_fallback = fstar_source == "per_seed_min_fallback"
        methods = doc.get("methods", {})

        agns_pick = _resolve_record(methods, _AGNS_KEYS)
        if agns_pick is None:
            continue
        agns_key, agns_rec = agns_pick
        agns_seeds = _per_seed_iters(agns_rec)
        if agns_seeds is None:
            continue

        for baseline_disp, baseline_keys in BASELINES:
            base_pick = _resolve_record(methods, baseline_keys)
            if base_pick is None:
                continue
            _, base_rec = base_pick
            base_seeds = _per_seed_iters(base_rec)
            if base_seeds is None or len(base_seeds) != len(agns_seeds):
                continue

            ci = bootstrap_ratio_ci(
                base_seeds, agns_seeds, rng_seed=BOOTSTRAP_DEFAULT_RNG_SEED
            )
            wilcoxon = paired_wilcoxon(base_seeds, agns_seeds)
            p_value = wilcoxon[1] if wilcoxon is not None else None

            agns_med = _median_or_none(agns_seeds)
            base_med = _median_or_none(base_seeds)

            noise = ci.covers_unity or (p_value is not None and p_value > 0.05)
            noise_cell = r"$\approx$" if noise else ""

            campaign_cell = escape_label(campaign)
            if is_fallback:
                campaign_cell = campaign_cell + FALLBACK_MARK
                any_fallback = True

            rows.append([
                campaign_cell,
                escape_label(agns_key),
                baseline_disp,
                _fmt_int(agns_med),
                _fmt_int(base_med),
                _fmt_ratio(ci.point_estimate),
                _fmt_ci(ci.lo, ci.hi),
                _fmt_p(p_value),
                noise_cell,
            ])

    if not rows:
        rows = [["(no campaigns with usable AGNS + baseline data)", *[""] * (len(header) - 1)]]

    caption = _build_caption()
    if any_fallback:
        caption = caption + _FALLBACK_FOOTNOTE
    tex = build_booktabs(
        colspec,
        header,
        rows,
        caption=caption,
        label="tab:noise-floor",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex)
    print(f"  [noise_floor] wrote {output} ({len(rows)} rows)")
