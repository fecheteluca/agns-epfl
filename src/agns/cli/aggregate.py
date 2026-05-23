"""Collect a multi-seed campaign into a single JSON.

Reads ``<raw_dir>/seed_*/<method>_history.pkl`` produced by
:mod:`agns.cli.run_benchmark`, aligns the per-seed traces, and writes a
single ``<campaign>.json`` under ``results/numerical/aggregated/`` with:

* per-method median + IQR curves on the three primary axes
  (iterations, wall-clock time, gradient-oracle calls);
* per-method final-residual distribution + summary statistics;
* per-method failure rate (fraction of seeds that did not reach
  ``--eps``);
* per-method log-log slope fit;
* the campaign metadata (problem spec, seed list, ``f_star``).

The aggregated JSON is the single input that
:mod:`agns.cli.make_plots` and :mod:`agns.cli.make_tables` consume.

Usage::

    python -m agns.cli.aggregate \\
        --campaign rate_verification \\
        --raw-dir results/numerical/raw/rate_verification \\
        --output results/numerical/aggregated/rate_verification.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from agns.utils.io import load_pickle, save_json
from agns.utils.metrics import fit_loglog_slope, median_iqr_curves

__all__ = ["aggregate", "main"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate a multi-seed campaign into a single JSON.",
    )
    p.add_argument("--campaign", required=True, help="Campaign name (JSON top-level field).")
    p.add_argument("--raw-dir", required=True, help="Directory containing seed_*/ subdirs.")
    p.add_argument("--output", required=True, help="Path of the aggregated JSON to write.")
    p.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Failure threshold (seeds with f_final - f_star > eps are flagged).",
    )
    p.add_argument(
        "--rate-tail-fraction",
        type=float,
        default=0.5,
        help="Fraction of the trailing trace used for the log-log rate fit.",
    )
    return p.parse_args()


def _seed_dirs(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("seed_"))


def _to_array(seq: Any) -> NDArray[np.float64]:
    return np.asarray(seq, dtype=np.float64).ravel()


def _aggregate_method(
    method_key: str,
    seed_dirs: list[Path],
    per_seed_f_star: dict[Path, float],
    fallback_f_star: float,
    eps: float,
    rate_tail: float,
) -> dict[str, Any] | None:
    """Build the per-method aggregated record, or ``None`` if every seed failed."""
    func_traces: list[NDArray[np.float64]] = []
    time_traces: list[NDArray[np.float64]] = []
    grad_traces: list[NDArray[np.float64]] = []
    final_residuals: list[float] = []
    label: str | None = None
    # ``statuses`` keys are the per-seed directory names ("seed_0", ...);
    # the dict guarantees alignment when some seeds lack a summary.json
    # while others have one.
    statuses: dict[str, str] = {}
    # Restart-event accounting (only meaningful for AGNS-family methods).
    n_restarts_per_seed: list[int] = []
    first_restart_iter_per_seed: list[int | None] = []
    n_iters_per_seed: list[int] = []
    max_local_k_per_seed: list[int] = []
    restart_mode_observed: str | None = None

    for sd in seed_dirs:
        pkl = sd / f"{method_key}_history.pkl"
        if not pkl.is_file():
            final_residuals.append(float("inf"))
            continue
        hist = cast("dict[str, Any]", load_pickle(pkl))
        if "func" not in hist or not hist["func"]:
            final_residuals.append(float("inf"))
            continue
        func = _to_array(hist["func"])
        fs_seed = per_seed_f_star.get(sd, fallback_f_star)
        residual = np.maximum(func - fs_seed, 0.0)
        func_traces.append(residual)
        final_residuals.append(float(residual[-1]))
        if "time" in hist:
            time_traces.append(_to_array(hist["time"]))
        if "grad_calls" in hist:
            grad_traces.append(_to_array(hist["grad_calls"]))

        # Restart instrumentation written by ``agns`` / ``agns_wsm``;
        # absent for non-AGNS methods (default to zero so the aggregated
        # record carries a uniform schema).
        nr_field = hist.get("n_restarts_total")
        nr = int(nr_field[0]) if nr_field else 0
        n_restarts_per_seed.append(nr)
        events = hist.get("restart_events") or []
        first_restart_iter_per_seed.append(int(events[0]) if events else None)
        n_iters_per_seed.append(int(len(hist.get("func", [])) - 1))
        local_k_trace = hist.get("local_k") or []
        max_local_k_per_seed.append(int(max(local_k_trace)) if local_k_trace else 0)
        rm_field = hist.get("restart_mode")
        if rm_field and restart_mode_observed is None:
            restart_mode_observed = str(rm_field[0])

        summ_path = sd / "summary.json"
        if summ_path.is_file():
            try:
                with open(summ_path) as fh:
                    summ = json.load(fh)
                m = summ.get("methods", {}).get(method_key, {})
                if label is None:
                    label = m.get("label", method_key)
                st = m.get("status")
                if st is not None:
                    statuses[sd.name] = str(st)
            except (OSError, json.JSONDecodeError):
                pass

    if not func_traces:
        return None

    iters_agg = median_iqr_curves(func_traces)
    finite = [r for r in final_residuals if np.isfinite(r)] or [np.inf]
    record: dict[str, Any] = {
        "method": method_key,
        "label": label or method_key,
        "n_seeds_succeeded": len(func_traces),
        "n_seeds_failed": len(seed_dirs) - len(func_traces),
        "failure_rate": 1.0 - len(func_traces) / max(len(seed_dirs), 1),
        "final_residuals": final_residuals,
        "final_residual_summary": {
            "median": float(np.median(finite)),
            "p25": float(np.percentile(finite, 25)),
            "p75": float(np.percentile(finite, 75)),
        },
        "eps_target": eps,
        "above_eps_rate": float(np.mean([1.0 if r > eps else 0.0 for r in final_residuals])),
        "statuses": statuses,
        "iter_curve": {
            "x": iters_agg.x.tolist(),
            "median": iters_agg.median.tolist(),
            "p25": iters_agg.lower.tolist(),
            "p75": iters_agg.upper.tolist(),
        },
    }

    # Restart-event summary -- ONLY emitted when the method actually
    # advertised a ``restart_mode`` field (i.e. is an AGNS-family run).
    # Non-AGNS methods would otherwise produce a degenerate
    # ``restart_mode: null`` record that confuses downstream consumers.
    if restart_mode_observed is not None and n_restarts_per_seed:
        densities = [
            (r / max(n, 1)) for r, n in zip(n_restarts_per_seed, n_iters_per_seed, strict=False)
        ]
        record["restart_summary"] = {
            "restart_mode": restart_mode_observed,
            "n_restarts_per_seed": n_restarts_per_seed,
            "n_iters_per_seed": n_iters_per_seed,
            "first_restart_iter_per_seed": first_restart_iter_per_seed,
            "max_local_k_per_seed": max_local_k_per_seed,
            "n_restarts_summary": {
                "median": float(np.median(n_restarts_per_seed)),
                "p25": float(np.percentile(n_restarts_per_seed, 25)),
                "p75": float(np.percentile(n_restarts_per_seed, 75)),
            },
            "restart_density_summary": {
                "median": float(np.median(densities)),
                "p25": float(np.percentile(densities, 25)),
                "p75": float(np.percentile(densities, 75)),
            },
        }

    if time_traces:
        time_agg = median_iqr_curves(time_traces)
        record["time_curve"] = {
            "x": time_agg.x.tolist(),
            "median": time_agg.median.tolist(),
            "p25": time_agg.lower.tolist(),
            "p75": time_agg.upper.tolist(),
        }

    if grad_traces:
        grad_agg = median_iqr_curves(grad_traces)
        record["grad_curve"] = {
            "x": grad_agg.x.tolist(),
            "median": grad_agg.median.tolist(),
            "p25": grad_agg.lower.tolist(),
            "p75": grad_agg.upper.tolist(),
        }

    try:
        k = np.arange(1, iters_agg.median.size, dtype=np.float64)
        y = iters_agg.median[1:]
        fit = fit_loglog_slope(k, y, tail_fraction=rate_tail)
        record["rate_fit"] = {
            "slope": fit.slope,
            "intercept": fit.intercept,
            "r_squared": fit.r_squared,
            "n_points": fit.n_points,
            "tail_fraction": rate_tail,
        }
    except ValueError as e:
        record["rate_fit"] = {"error": str(e)}

    return record


def _resolve_per_seed_f_star(
    seed_dirs: list[Path],
    method_keys: set[str],
) -> tuple[dict[Path, float], float]:
    """Determine the residual ``f_star`` to subtract per seed.

    Prefers the value recorded in that seed's ``summary.json``; otherwise
    falls back to the per-seed minimum across every method's final
    ``func`` value.  The second return is the campaign-wide minimum.
    """
    per_seed: dict[Path, float] = {}
    global_min = float("inf")

    for sd in seed_dirs:
        declared: float | None = None
        summ_path = sd / "summary.json"
        if summ_path.is_file():
            try:
                with open(summ_path) as fh:
                    summ = json.load(fh)
                fs = summ.get("f_star")
                if fs is not None:
                    declared = float(fs)
            except (OSError, json.JSONDecodeError):
                pass

        seed_min = float("inf")
        for mkey in method_keys:
            pkl = sd / f"{mkey}_history.pkl"
            if pkl.is_file():
                hist = cast("dict[str, Any]", load_pickle(pkl))
                vals = _to_array(hist.get("func", []))
                if vals.size:
                    seed_min = min(seed_min, float(vals.min()))

        chosen = declared if declared is not None else seed_min
        per_seed[sd] = chosen
        global_min = min(global_min, chosen)

    return per_seed, global_min


def aggregate(
    campaign: str,
    raw_dir: Path,
    output: Path,
    eps: float,
    rate_tail: float,
) -> dict[str, Any]:
    """Build the aggregated JSON for ``campaign`` and write it to ``output``."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw-dir does not exist: {raw_dir}")
    seed_dirs = _seed_dirs(raw_dir)
    if not seed_dirs:
        raise RuntimeError(
            f"no seed_*/ subdirectories under {raw_dir}; did you forget '--seeds' on run_benchmark?"
        )

    method_keys: set[str] = set()
    problem_meta: dict[str, Any] | None = None
    seeds_used: list[int] = []
    for sd in seed_dirs:
        with contextlib.suppress(ValueError):
            seeds_used.append(int(sd.name.split("_", 1)[1]))
        for pkl in sd.glob("*_history.pkl"):
            method_keys.add(pkl.stem[: -len("_history")])
        summ_path = sd / "summary.json"
        if problem_meta is None and summ_path.is_file():
            try:
                with open(summ_path) as fh:
                    summ = json.load(fh)
                problem_meta = summ.get("problem")
            except (OSError, json.JSONDecodeError):
                pass

    per_seed_f_star, f_star_global_min = _resolve_per_seed_f_star(seed_dirs, method_keys)
    print(
        f"  [aggregate] {len(seed_dirs)} seeds, {len(method_keys)} methods, "
        f"min(f_star)={f_star_global_min:.6e} across seeds"
    )

    method_records: dict[str, Any] = {}
    for mkey in sorted(method_keys):
        rec = _aggregate_method(
            mkey,
            seed_dirs,
            per_seed_f_star=per_seed_f_star,
            fallback_f_star=f_star_global_min,
            eps=eps,
            rate_tail=rate_tail,
        )
        if rec is None:
            print(f"  [aggregate] WARN: no usable history for method {mkey!r}")
            continue
        method_records[mkey] = rec
        fit = rec["rate_fit"]
        slope_str = f"{fit['slope']:+.3f}" if "slope" in fit else "n/a"
        print(
            f"    {mkey:<25s}  n_ok={rec['n_seeds_succeeded']:>2d}  "
            f"slope={slope_str:>7s}  "
            f"final={rec['final_residual_summary']['median']:.3e}"
        )

    out = {
        "campaign": campaign,
        "n_seeds": len(seed_dirs),
        "seeds": sorted(seeds_used),
        "eps_target": eps,
        "f_star_global_min": f_star_global_min,
        "per_seed_f_star": {sd.name: v for sd, v in per_seed_f_star.items()},
        "problem": problem_meta,
        "methods": method_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, output)
    print(f"  [aggregate] wrote {output}")
    return out


def main() -> None:
    args = parse_args()
    aggregate(
        campaign=args.campaign,
        raw_dir=Path(args.raw_dir).resolve(),
        output=Path(args.output).resolve(),
        eps=float(args.eps),
        rate_tail=float(args.rate_tail_fraction),
    )


if __name__ == "__main__":
    main()
