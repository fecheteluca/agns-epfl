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

from agns.pipeline.reference_cache import DEFAULT_REFERENCE_DIR
from agns.pipeline.reference_cache import lookup as cache_lookup
from agns.utils.host_info import capture_host_info
from agns.utils.io import load_pickle, save_json
from agns.utils.metrics import fit_loglog_slope, median_iqr_curves

__all__ = ["aggregate", "main"]


#: ``f_star`` provenance tags written into the aggregated JSON.
FSTAR_DECLARED = "declared"
FSTAR_CACHED_REFERENCE = "cached_reference"
FSTAR_PER_SEED_MIN_FALLBACK = "per_seed_min_fallback"

#: User-visible warning text injected into the aggregated JSON when the
#: per-seed-min fallback path fires.  The per-campaign table renderer
#: reads ``f_star_source`` and emits a matching footnote.
_FALLBACK_WARNING = (
    "Residual computed against per-seed minimum across methods "
    "(no cached reference solution available). "
    "Final-residual values may be biased toward the strongest method."
)
_CACHE_NOTE = (
    "Residual computed against cached reference solution under "
    "results/reference_solutions/."
)


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
    p.add_argument(
        "--reference-dir",
        default=str(DEFAULT_REFERENCE_DIR),
        help=(
            "Directory containing cached reference solutions; consulted only when "
            "the problem factory did not declare f_star.  Default: "
            "results/reference_solutions."
        ),
    )
    p.add_argument(
        "--no-reference-cache",
        action="store_true",
        help=(
            "Disable cached-reference lookup; fall straight through to the "
            "per-seed-min fallback.  Useful for debugging cache-vs-fallback drift."
        ),
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
    *,
    failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Build the per-method aggregated record.

    Returns ``None`` only when the method has neither successful seeds
    nor recorded failures (i.e. it isn't actually present in the raw
    output tree at all).  When the method has *only* failures, a
    minimal record is returned so downstream consumers can render
    "failed" instead of silently dropping the row.

    Eps-boundary convention: ``residual < eps`` is "crossed" / success
    (mirrors :func:`agns.methods.stopping.check_stop`).  A residual
    of exactly ``eps`` is treated as not-yet-success on both the
    ``iters_to_eps_per_seed`` side and the ``above_eps_rate`` side.
    """
    failures = list(failures or [])
    # Per-seed first crossing of ``eps`` recorded on three axes.  All
    # three are aligned by construction (computed at the same seed-and-
    # iteration as ``iters_to_eps``).  Consumed by the speedup table
    # for iter / grad-call / wall-clock CI variants.  ``None`` when
    # the seed never crossed eps, or when the underlying counter wasn't
    # traced (older history pickles without ``grad_calls`` / ``time``).
    iters_to_eps_per_seed: list[int | None] = []
    grad_calls_to_eps_per_seed: list[int | None] = []
    wall_time_to_eps_per_seed: list[float | None] = []
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
            iters_to_eps_per_seed.append(None)
            grad_calls_to_eps_per_seed.append(None)
            wall_time_to_eps_per_seed.append(None)
            continue
        hist = cast("dict[str, Any]", load_pickle(pkl))
        if "func" not in hist or not hist["func"]:
            final_residuals.append(float("inf"))
            iters_to_eps_per_seed.append(None)
            grad_calls_to_eps_per_seed.append(None)
            wall_time_to_eps_per_seed.append(None)
            continue
        func = _to_array(hist["func"])
        fs_seed = per_seed_f_star.get(sd, fallback_f_star)
        residual = np.maximum(func - fs_seed, 0.0)
        func_traces.append(residual)
        final_residuals.append(float(residual[-1]))
        # First iteration where this seed's residual drops *strictly*
        # below the aggregator's eps; ``None`` if it never does.
        # Strict less-than mirrors ``check_stop``'s success predicate
        # (``f - f_star < eps``) so a residual exactly at the boundary
        # is treated as not-yet-success on both sides.  Bootstrap CIs
        # in the speedup table reuse this list without re-loading the
        # raw pickles.  Mirror-fields for grad calls and wall-clock
        # look up the counter at the same crossing index so all three
        # axes report the cost of the same iterate.
        crossings = np.where(residual < eps)[0]
        if crossings.size:
            idx = int(crossings[0])
            iters_to_eps_per_seed.append(idx)
            grad_series = hist.get("grad_calls")
            if grad_series and idx < len(grad_series) and grad_series[idx] is not None:
                grad_calls_to_eps_per_seed.append(int(grad_series[idx]))
            else:
                grad_calls_to_eps_per_seed.append(None)
            time_series = hist.get("time")
            if time_series and idx < len(time_series) and time_series[idx] is not None:
                wall_time_to_eps_per_seed.append(float(time_series[idx]))
            else:
                wall_time_to_eps_per_seed.append(None)
        else:
            iters_to_eps_per_seed.append(None)
            grad_calls_to_eps_per_seed.append(None)
            wall_time_to_eps_per_seed.append(None)
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
        # Zero successful seeds.  If we also have no recorded failures,
        # the method isn't really in the raw tree -- skip it (so the
        # downstream table renders the method as missing).  If we *do*
        # have failures, emit a minimal record so the downstream table
        # can render "failed" instead of "missing".
        if not failures:
            return None
        # No successful seed could supply a label via summary.json["methods"];
        # fall back to a label stored on the failure record itself.
        if label is None:
            for f in failures:
                lab = f.get("label")
                if lab:
                    label = str(lab)
                    break
        return {
            "method": method_key,
            "label": label or method_key,
            "n_seeds_succeeded": 0,
            "n_seeds_failed": len(failures),
            "failure_rate": 1.0,
            "final_residuals": [float("inf")] * len(seed_dirs),
            "final_residual_summary": {
                "median": float("inf"),
                "p25": float("inf"),
                "p75": float("inf"),
            },
            "eps_target": eps,
            "above_eps_rate": 1.0,
            "statuses": statuses,
            # No seed produced a trace, so no seed crossed eps on any axis.
            "iters_to_eps_per_seed": [None] * len(seed_dirs),
            "grad_calls_to_eps_per_seed": [None] * len(seed_dirs),
            "wall_time_to_eps_per_seed": [None] * len(seed_dirs),
            "failures": failures,
        }

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
        # Fraction of seeds that did NOT cross eps.  Complement of the
        # ``check_stop`` success predicate: a seed succeeds iff its
        # final residual is strictly below eps, so it is "above"
        # whenever ``residual >= eps``.  Matches the strict-less-than
        # convention used by ``iters_to_eps_per_seed`` above.
        "above_eps_rate": float(np.mean([1.0 if r >= eps else 0.0 for r in final_residuals])),
        "statuses": statuses,
        "iters_to_eps_per_seed": iters_to_eps_per_seed,
        "grad_calls_to_eps_per_seed": grad_calls_to_eps_per_seed,
        "wall_time_to_eps_per_seed": wall_time_to_eps_per_seed,
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

    # Always attach the failures field (empty when the method ran cleanly
    # on every seed) so downstream consumers can switch on its truthiness.
    record["failures"] = failures

    return record


def _resolve_per_seed_f_star(
    seed_dirs: list[Path],
    method_keys: set[str],
    *,
    reference_dir: Path | None = None,
    use_cache: bool = True,
) -> tuple[dict[Path, float], float, dict[str, str]]:
    """Determine the residual ``f_star`` to subtract per seed.

    Precedence per seed:
      1. ``summary.json`` carries a declared ``f_star`` from the factory.
      2. The cached reference solution under ``reference_dir`` matches
         ``(problem.type, params_hash, seed)``.
      3. Per-seed minimum across every method's final ``func`` value
         (last-resort fallback).

    The returned ``per_seed_source`` dict maps each seed directory's
    basename (``seed_<i>``) to one of :data:`FSTAR_DECLARED` /
    :data:`FSTAR_CACHED_REFERENCE` / :data:`FSTAR_PER_SEED_MIN_FALLBACK`
    so downstream consumers (table renderer) can show the right footnote.
    """
    per_seed: dict[Path, float] = {}
    per_seed_source: dict[str, str] = {}
    global_min = float("inf")

    for sd in seed_dirs:
        declared: float | None = None
        cached: float | None = None
        problem_meta: dict[str, Any] | None = None
        seed_idx: int | None = None
        with contextlib.suppress(ValueError):
            seed_idx = int(sd.name.split("_", 1)[1])

        summ_path = sd / "summary.json"
        if summ_path.is_file():
            try:
                with open(summ_path) as fh:
                    summ = json.load(fh)
                fs = summ.get("f_star")
                if fs is not None:
                    declared = float(fs)
                problem_meta = summ.get("problem")
            except (OSError, json.JSONDecodeError):
                pass

        # Cache lookup only when (a) no declared f_star, (b) reference_dir
        # given, (c) we know the seed index, and (d) the seed's
        # summary.json carried problem.type + params.
        if (
            declared is None
            and use_cache
            and reference_dir is not None
            and seed_idx is not None
            and problem_meta
            and "type" in problem_meta
        ):
            entry = cache_lookup(
                reference_dir,
                str(problem_meta["type"]),
                dict(problem_meta.get("params", {}) or {}),
                int(seed_idx),
            )
            if entry is not None:
                cached = float(entry.f_ref)

        seed_min = float("inf")
        for mkey in method_keys:
            pkl = sd / f"{mkey}_history.pkl"
            if pkl.is_file():
                hist = cast("dict[str, Any]", load_pickle(pkl))
                vals = _to_array(hist.get("func", []))
                if vals.size:
                    seed_min = min(seed_min, float(vals.min()))

        if declared is not None:
            chosen = declared
            source = FSTAR_DECLARED
        elif cached is not None:
            chosen = cached
            source = FSTAR_CACHED_REFERENCE
        else:
            chosen = seed_min
            source = FSTAR_PER_SEED_MIN_FALLBACK
        per_seed[sd] = chosen
        per_seed_source[sd.name] = source
        global_min = min(global_min, chosen)

    return per_seed, global_min, per_seed_source


def aggregate(
    campaign: str,
    raw_dir: Path,
    output: Path,
    eps: float,
    rate_tail: float,
    *,
    reference_dir: Path | None = None,
    use_reference_cache: bool = True,
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
    # method_key -> list of failure dicts {"seed", "exception_type", "message"}.
    # Built from each seed's summary.json["failures"] block, with standalone
    # <mkey>_failure.json files used as a fallback when summary.json is
    # missing or malformed.
    method_failures: dict[str, list[dict[str, Any]]] = {}
    for sd in seed_dirs:
        with contextlib.suppress(ValueError):
            seeds_used.append(int(sd.name.split("_", 1)[1]))
        for pkl in sd.glob("*_history.pkl"):
            method_keys.add(pkl.stem[: -len("_history")])

        seed_failures_from_summary: dict[str, dict[str, Any]] | None = None
        summ_path = sd / "summary.json"
        if summ_path.is_file():
            try:
                with open(summ_path) as fh:
                    summ = json.load(fh)
                if problem_meta is None:
                    problem_meta = summ.get("problem")
                seed_failures_from_summary = summ.get("failures") or {}
            except (OSError, json.JSONDecodeError):
                seed_failures_from_summary = None

        if seed_failures_from_summary is not None:
            # ``summary.json["failures"][mkey]`` carries the slim summary
            # the runner wrote (label / exception_type / message /
            # wall_time_s_before_failure).
            for mkey, rec in seed_failures_from_summary.items():
                method_keys.add(mkey)
                method_failures.setdefault(mkey, []).append(
                    {
                        "seed": sd.name,
                        "label": str(rec.get("label", mkey)),
                        "exception_type": str(rec.get("exception_type", "")),
                        "message": str(rec.get("message", "")),
                        "wall_time_s_before_failure": float(
                            rec.get("wall_time_s_before_failure", 0.0)
                        ),
                    }
                )
        else:
            # Fallback: discover failures via standalone *_failure.json files
            # so an interrupted run (no summary.json written yet) still
            # surfaces what failed.  These files use ``method_label`` as
            # the label key (the full per-failure schema); normalise to
            # ``label`` so callers see one consistent field name.
            for fjson in sd.glob("*_failure.json"):
                mkey = fjson.stem[: -len("_failure")]
                try:
                    with open(fjson) as fh:
                        rec = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    continue
                method_keys.add(mkey)
                method_failures.setdefault(mkey, []).append(
                    {
                        "seed": sd.name,
                        "label": str(rec.get("method_label", rec.get("label", mkey))),
                        "exception_type": str(rec.get("exception_type", "")),
                        "message": str(rec.get("message", "")),
                        "wall_time_s_before_failure": float(
                            rec.get("wall_time_s_before_failure", 0.0)
                        ),
                    }
                )

    per_seed_f_star, f_star_global_min, f_star_per_source = _resolve_per_seed_f_star(
        seed_dirs,
        method_keys,
        reference_dir=reference_dir,
        use_cache=use_reference_cache,
    )
    # Roll the per-seed sources up to a single campaign-level tag.  Order
    # of preference: declared > cached_reference > per_seed_min_fallback,
    # but if seeds disagree we promote to the *weakest* source so the
    # rendered table warning fires whenever even one seed needed fallback.
    sources_observed = set(f_star_per_source.values())
    if FSTAR_PER_SEED_MIN_FALLBACK in sources_observed:
        f_star_source = FSTAR_PER_SEED_MIN_FALLBACK
    elif FSTAR_CACHED_REFERENCE in sources_observed:
        f_star_source = FSTAR_CACHED_REFERENCE
    else:
        f_star_source = FSTAR_DECLARED
    warnings_list: list[str] = []
    if f_star_source == FSTAR_PER_SEED_MIN_FALLBACK:
        warnings_list.append(_FALLBACK_WARNING)
    elif f_star_source == FSTAR_CACHED_REFERENCE:
        warnings_list.append(_CACHE_NOTE)

    print(
        f"  [aggregate] {len(seed_dirs)} seeds, {len(method_keys)} methods, "
        f"min(f_star)={f_star_global_min:.6e} across seeds  "
        f"[f_star_source={f_star_source}]"
    )
    if f_star_source == FSTAR_PER_SEED_MIN_FALLBACK:
        print(f"  [aggregate] WARNING: {_FALLBACK_WARNING}")

    method_records: dict[str, Any] = {}
    for mkey in sorted(method_keys):
        rec = _aggregate_method(
            mkey,
            seed_dirs,
            per_seed_f_star=per_seed_f_star,
            fallback_f_star=f_star_global_min,
            eps=eps,
            rate_tail=rate_tail,
            failures=method_failures.get(mkey),
        )
        if rec is None:
            print(f"  [aggregate] WARN: no usable history for method {mkey!r}")
            continue
        method_records[mkey] = rec
        # A failure-only record skips the rate fit (no curve to fit on).
        fit = rec.get("rate_fit") or {}
        slope_str = f"{fit['slope']:+.3f}" if "slope" in fit else "n/a"
        n_failed = rec.get("n_seeds_failed", 0)
        fail_tag = f"  failed={n_failed}" if n_failed else ""
        print(
            f"    {mkey:<25s}  n_ok={rec['n_seeds_succeeded']:>2d}{fail_tag}  "
            f"slope={slope_str:>7s}  "
            f"final={rec['final_residual_summary']['median']:.3e}"
        )

    # Campaign-level failure summary: one entry per method with at least
    # one failure, including the modal exception type.  Useful for tables
    # and for triage at the end of a long run.
    failures_summary: dict[str, dict[str, Any]] = {}
    for mkey, fails in method_failures.items():
        types = [f["exception_type"] for f in fails if f.get("exception_type")]
        modal = max(set(types), key=types.count) if types else ""
        failures_summary[mkey] = {
            "n_failed_seeds": len(fails),
            "seeds": [f["seed"] for f in fails],
            "modal_exception_type": modal,
        }

    out = {
        "campaign": campaign,
        "n_seeds": len(seed_dirs),
        "seeds": sorted(seeds_used),
        "eps_target": eps,
        "f_star_global_min": f_star_global_min,
        "f_star_source": f_star_source,
        "f_star_per_source": f_star_per_source,
        "per_seed_f_star": {sd.name: v for sd, v in per_seed_f_star.items()},
        "problem": problem_meta,
        "methods": method_records,
        "failures_summary": failures_summary,
        "warnings": warnings_list,
        # Captured at aggregate-time so the wall-clock speedup table
        # caption can carry a host fingerprint.  Assumes the user ran
        # the campaign on the same host they are aggregating on; an
        # extension could capture per-seed at run time.
        "host": capture_host_info(),
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
        reference_dir=Path(args.reference_dir).resolve(),
        use_reference_cache=not args.no_reference_cache,
    )


if __name__ == "__main__":
    main()
