"""AGNS-vs-GNS speedup tables (iter / grad-call / wall-clock variants).

For every campaign with a matched (GNS, AGNS) pair, report:

* across-seed median per-seed cost-to-eps on the selected ``axis``;
* point speedup ``median(cost_GNS) / median(cost_AGNS)`` plus a
  pair-preserving percentile bootstrap 95% CI;
* paired Wilcoxon p-value on the per-seed costs;
* a ``noise`` flag that fires when the CI brackets 1.0 or p > 0.05;
* the no-restart pairing's matching speedup + CI, and the restart
  contribution (difference between the two point estimates).

Three axes are supported:

* ``"iter"`` (default) -- iterations to eps.  The headline axis.
* ``"grad"`` -- gradient-oracle calls to eps.  Surfaces AGNS's
  per-iteration probing cost that iter-count hides.
* ``"time"`` -- wall-clock seconds to eps.  Host-dependent; the
  caption embeds a host fingerprint.

All three statistics share the same bootstrap + Wilcoxon machinery
(see :mod:`agns.utils.metrics` and ``docs/statistics.md``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from agns.tables._tex import build_booktabs, escape_label
from agns.utils.host_info import fmt_host_fingerprint
from agns.utils.metrics import (
    BOOTSTRAP_DEFAULT_RNG_SEED,
    bootstrap_ratio_ci,
    paired_wilcoxon,
)

__all__ = ["render", "render_grad", "render_iter", "render_time"]


Axis = Literal["iter", "grad", "time"]


# Per-axis configuration.  Centralised so adding a fourth axis (e.g.
# Hessian-vector products) is one entry, not a sweep through the file.
_AXIS_CONFIG: dict[Axis, dict[str, Any]] = {
    "iter": {
        "per_seed_field": "iters_to_eps_per_seed",
        "unit_singular": "iteration",
        "unit_plural": "iterations",
        "unit_short": "iters",
        "fmt": "int",
    },
    "grad": {
        "per_seed_field": "grad_calls_to_eps_per_seed",
        "unit_singular": "gradient call",
        "unit_plural": "gradient calls",
        "unit_short": "grads",
        "fmt": "int",
    },
    "time": {
        "per_seed_field": "wall_time_to_eps_per_seed",
        "unit_singular": "second",
        "unit_plural": "seconds",
        "unit_short": "s",
        "fmt": "float3",
    },
}


def _per_seed_costs(rec: dict | None, field: str) -> list | None:
    """Return the per-seed cost list for ``field``, or ``None`` when unavailable."""
    if rec is None:
        return None
    series = rec.get(field)
    if not series:
        return None
    return list(series)


def _fmt_value(v: float | None, fmt: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if fmt == "int":
        return str(round(float(v)))
    if fmt == "float3":
        return f"{float(v):.3g}"
    raise ValueError(f"unknown fmt {fmt!r}")  # pragma: no cover


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


def _median_or_none(values: list) -> float | None:
    """Median of a list, treating ``None`` as infinity."""
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


#: Marker appended to the Campaign cell of any row whose aggregated
#: JSON declared ``f_star_source = "per_seed_min_fallback"``.  The
#: caption explains what the marker means.
FALLBACK_MARK = r"$^\dagger$"

_FALLBACK_FOOTNOTE = (
    r"  $\dagger$ marks campaigns whose $f^\star$ was the "
    r"per-seed minimum across methods (no cached reference solution).  "
    r"On such campaigns the per-seed winner's residual is mechanically "
    r"zero, so its $\text{iters-to-}\varepsilon$ is a lower bound and "
    r"the speedup CI is biased toward the winner.  Compute references "
    r"with \texttt{agns-references --configs <yaml>} and re-aggregate "
    r"to remove the marker."
)


def _caption(axis: Axis, host: dict | None) -> str:
    cfg = _AXIS_CONFIG[axis]
    unit_plural = cfg["unit_plural"]
    unit_short = cfg["unit_short"]
    axis_label = {
        "iter": "iteration-budget",
        "grad": "gradient-oracle-budget",
        "time": "wall-clock",
    }[axis]
    host_clause = ""
    if axis == "time":
        fp = fmt_host_fingerprint(host) if host else ""
        if fp:
            host_clause = f"  \\textit{{Host:}} {escape_label(fp)}."
        else:
            host_clause = (
                "  \\textit{Host:} fingerprint not recorded "
                "(re-aggregate with the current code to capture it)."
            )
    return (
        f"AGNS-vs-GNS {axis_label} speedup at the aggregator's "
        f"$\\varepsilon$ (measured in {unit_plural}).  "
        f"Each row pairs a GNS variant with the matching AGNS variant "
        f"(exact$\\leftrightarrow$exact, inexact$\\leftrightarrow$inexact).  "
        f"cost() columns report the across-seed median per-seed cost-to-$\\varepsilon$ "
        f"in {unit_short}.  "
        f"Speedup is $\\text{{median(cost\\_GNS)}} / \\text{{median(cost\\_AGNS)}}$ "
        f"with a pair-preserving percentile bootstrap 95\\% CI "
        f"($n=10{{,}}000$ resamples, RNG seed pinned).  "
        f"$p_{{W}}$ is the paired Wilcoxon p-value (n/a when fewer than five "
        f"seeds crossed $\\varepsilon$ on both sides).  "
        f"$\\approx$ flags rows where the CI brackets 1.0 or $p_{{W}} > 0.05$ -- "
        f"the speedup is indistinguishable from no effect on this sample.  "
        f"no-r columns use AGNS with \\texttt{{restart\\_mode=none}}; "
        f"Restart contrib. = Speedup $-$ Speedup no-r.  "
        f"\\textit{{Metric note:}} the GNS / AGNS / Super-Newton / Cubic-Newton "
        f"family regularises with the problem-supplied $B$ "
        f"($H + \\lambda B$); damped Newton, trust-region, and L-BFGS do not "
        f"(see README \\S\"Metric ($B$) asymmetry\").{host_clause}"
    )


def _label_for_axis(axis: Axis) -> str:
    return {
        "iter": "tab:agns-speedup",
        "grad": "tab:agns-speedup-grad",
        "time": "tab:agns-speedup-time",
    }[axis]


def _render_speedup_table(
    aggregated_dir: Path,
    output: Path,
    *,
    axis: Axis,
) -> None:
    """Shared implementation underneath the three axis-specific wrappers."""
    if axis not in _AXIS_CONFIG:
        raise ValueError(f"axis must be one of {sorted(_AXIS_CONFIG)}, got {axis!r}")
    json_paths = sorted(aggregated_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"no aggregated JSONs under {aggregated_dir}")

    cfg = _AXIS_CONFIG[axis]
    per_seed_field = cfg["per_seed_field"]
    value_fmt = cfg["fmt"]
    unit_short = cfg["unit_short"]

    header = [
        "Campaign",
        "Variant",
        f"cost(GNS) [{unit_short}]",
        f"cost(AGNS) [{unit_short}]",
        "Speedup",
        "[CI95]",
        "$p_{W}$",
        "$\\approx 1$",
        f"cost(no-r) [{unit_short}]",
        "Speedup no-r",
        "[CI95]",
        "Restart contrib.",
    ]
    colspec = "llrrr l r c r r l r"

    pairings: list[tuple[str, str, str, str]] = [
        ("exact H", "gns_exact", "agns_exact", "agns_exact_norestart"),
        ("inexact H", "gns_inexact", "agns_inexact", "agns_inexact_norestart"),
    ]

    rows: list[list[str]] = []
    # First non-empty host record we see across the campaigns; surfaced
    # in the caption for the wall-clock variant.  Most users aggregate
    # one machine at a time so a single fingerprint is the typical case.
    host: dict | None = None
    # Track whether any rendered row used the per-seed-min fallback;
    # toggles a footnote in the caption explaining the dagger marker.
    any_fallback = False
    for jp in json_paths:
        with open(jp) as fh:
            doc = json.load(fh)
        if host is None:
            cand = doc.get("host")
            if cand:
                host = cand
        campaign = doc.get("campaign", jp.stem)
        fstar_source = str(doc.get("f_star_source", "declared"))
        is_fallback = fstar_source == "per_seed_min_fallback"
        methods = doc.get("methods", {})
        for variant_label, gns_key, agns_key, nr_key in pairings:
            gns_seeds = _per_seed_costs(methods.get(gns_key), per_seed_field)
            agns_seeds = _per_seed_costs(methods.get(agns_key), per_seed_field)
            if gns_seeds is None or agns_seeds is None:
                continue
            if len(gns_seeds) != len(agns_seeds):
                continue

            ci = bootstrap_ratio_ci(gns_seeds, agns_seeds, rng_seed=BOOTSTRAP_DEFAULT_RNG_SEED)
            wilcoxon = paired_wilcoxon(gns_seeds, agns_seeds)
            p_value = wilcoxon[1] if wilcoxon is not None else None

            gns_med = _median_or_none(gns_seeds)
            agns_med = _median_or_none(agns_seeds)

            nr_seeds = _per_seed_costs(methods.get(nr_key), per_seed_field)
            if nr_seeds is not None and len(nr_seeds) == len(gns_seeds):
                ci_nr = bootstrap_ratio_ci(
                    gns_seeds, nr_seeds, rng_seed=BOOTSTRAP_DEFAULT_RNG_SEED
                )
                nr_med = _median_or_none(nr_seeds)
                restart_contrib = ci.point_estimate - ci_nr.point_estimate
            else:
                ci_nr = None
                nr_med = None
                restart_contrib = float("nan")

            noise = ci.covers_unity or (p_value is not None and p_value > 0.05)
            noise_cell = r"$\approx$" if noise else ""

            campaign_cell = escape_label(campaign)
            if is_fallback:
                campaign_cell = campaign_cell + FALLBACK_MARK
                any_fallback = True

            rows.append([
                campaign_cell,
                variant_label,
                _fmt_value(gns_med, value_fmt),
                _fmt_value(agns_med, value_fmt),
                _fmt_ratio(ci.point_estimate),
                _fmt_ci(ci.lo, ci.hi),
                _fmt_p(p_value),
                noise_cell,
                _fmt_value(nr_med, value_fmt),
                _fmt_ratio(ci_nr.point_estimate) if ci_nr is not None else "--",
                _fmt_ci(ci_nr.lo, ci_nr.hi) if ci_nr is not None else "--",
                f"{restart_contrib:+.2f}" if np.isfinite(restart_contrib) else "--",
            ])

    if not rows:
        rows = [["(no campaigns with matched GNS / AGNS pairs)", *[""] * (len(header) - 1)]]

    caption = _caption(axis, host)
    if any_fallback:
        caption = caption + _FALLBACK_FOOTNOTE
    tex = build_booktabs(
        colspec,
        header,
        rows,
        caption=caption,
        label=_label_for_axis(axis),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex)
    print(f"  [speedup-{axis}] wrote {output}")


def render_iter(aggregated_dir: Path, output: Path) -> None:
    """Iteration-budget speedup table (the headline axis)."""
    _render_speedup_table(aggregated_dir, output, axis="iter")


def render_grad(aggregated_dir: Path, output: Path) -> None:
    """Gradient-oracle-budget speedup table.

    Reads ``grad_calls_to_eps_per_seed`` from each aggregated JSON;
    older JSONs without this field will have their rows silently
    skipped (re-aggregate to refresh).
    """
    _render_speedup_table(aggregated_dir, output, axis="grad")


def render_time(aggregated_dir: Path, output: Path) -> None:
    """Wall-clock speedup table.

    Reads ``wall_time_to_eps_per_seed`` from each aggregated JSON.
    Host-dependent: the caption carries the host fingerprint captured
    at aggregate time.  Two re-runs on the same machine should produce
    similar wall-clock numbers; cross-machine comparisons require
    reading the fingerprint and judging carefully.
    """
    _render_speedup_table(aggregated_dir, output, axis="time")


def render(aggregated_dir: Path, output: Path, *, axis: Axis = "iter") -> None:
    """Backward-compatible dispatcher.

    Earlier callers invoked ``speedup.render(dir, out)`` which produced
    the iter-count table; that call still works.  Pass ``axis="grad"``
    or ``axis="time"`` to render the sibling variants.
    """
    _render_speedup_table(aggregated_dir, output, axis=axis)
