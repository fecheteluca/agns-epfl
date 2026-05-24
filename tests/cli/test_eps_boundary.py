"""Eps-boundary alignment between the aggregator and ``check_stop``.

The optimisation methods stop iff ``f_k - f_star < eps`` (strict
less-than; see :func:`agns.methods.stopping.check_stop`).  The
aggregator's per-seed crossing detector and ``above_eps_rate`` must
mirror this convention so a downstream consumer reading both the
in-method stopping log and the aggregated JSON sees consistent
results at the boundary.

These tests pin the strict-less-than contract on both sides:

* ``iters_to_eps_per_seed``: residual strictly below eps -> crossed;
  residual exactly at eps -> NOT crossed.
* ``above_eps_rate``: residual strictly below eps -> succeeded;
  residual at or above eps -> failed.
"""

from __future__ import annotations

import json
from pathlib import Path

from agns.cli.aggregate import aggregate
from agns.methods.stopping import check_stop
from agns.utils.io import save_json, save_pickle


def _write_seed_history(
    raw_dir: Path,
    seed_idx: int,
    method_key: str,
    func_trace: list[float],
    f_star: float,
    *,
    problem_meta: dict | None = None,
) -> None:
    """Persist a hand-crafted history pickle + summary.json for one seed."""
    seed_dir = raw_dir / f"seed_{seed_idx}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    hist = {
        "func": list(func_trace),
        "time": [0.0] * len(func_trace),
        "grad_calls": list(range(len(func_trace))),
        "grad_norm": [0.0] * len(func_trace),
    }
    save_pickle(hist, seed_dir / f"{method_key}_history.pkl")
    save_json(
        {
            "problem": problem_meta or {"type": "synthetic", "params": {}},
            "f_star": f_star,
            "methods": {method_key: {"label": method_key, "status": "ok", "wall_time_s": 0.0}},
        },
        seed_dir / "summary.json",
    )


def _aggregate(raw_dir: Path, agg_path: Path, eps: float) -> dict:
    aggregate(
        campaign="boundary_test",
        raw_dir=raw_dir,
        output=agg_path,
        eps=eps,
        rate_tail=0.5,
        reference_dir=raw_dir.parent / "no_cache",
        use_reference_cache=False,
    )
    return json.loads(agg_path.read_text())


# ---------------------------------------------------------------------------
# iters_to_eps_per_seed: strict < eps
# ---------------------------------------------------------------------------


def test_residual_strictly_below_eps_counts_as_crossed(tmp_path: Path) -> None:
    """A residual just below ``eps`` is crossed at its first such iter."""
    raw_dir = tmp_path / "raw"
    eps = 1e-8
    # Trace decays: starts at 1.0, drops to 0.5*eps at iter 3.
    trace = [1.0, 0.5, 0.1, 0.5 * eps, 0.0]
    _write_seed_history(raw_dir, 0, "m", trace, f_star=0.0)

    agg = _aggregate(raw_dir, tmp_path / "agg.json", eps)
    assert agg["methods"]["m"]["iters_to_eps_per_seed"] == [3]


def test_residual_exactly_eps_does_NOT_count_as_crossed(tmp_path: Path) -> None:
    """Strict less-than: residual at exactly ``eps`` does not trigger a crossing.

    Mirrors :func:`check_stop`'s strict ``<`` predicate.  A trace
    that hits ``eps`` exactly and stays there has no crossing
    iter; the seed reports ``None``.
    """
    raw_dir = tmp_path / "raw"
    eps = 1e-8
    # Three iterations sitting *exactly* on eps -> never strictly below.
    trace = [1.0, 1e-4, eps, eps, eps]
    _write_seed_history(raw_dir, 0, "m", trace, f_star=0.0)

    agg = _aggregate(raw_dir, tmp_path / "agg.json", eps)
    # No iter is strictly below eps -> None.
    assert agg["methods"]["m"]["iters_to_eps_per_seed"] == [None]


def test_residual_crosses_after_exactly_eps_iter(tmp_path: Path) -> None:
    """Iter at-eps is skipped; the first strictly-below iter wins."""
    raw_dir = tmp_path / "raw"
    eps = 1e-8
    trace = [1.0, eps, 1e-9, 1e-10]  # at-eps at iter 1; crosses at iter 2
    _write_seed_history(raw_dir, 0, "m", trace, f_star=0.0)

    agg = _aggregate(raw_dir, tmp_path / "agg.json", eps)
    assert agg["methods"]["m"]["iters_to_eps_per_seed"] == [2]


# ---------------------------------------------------------------------------
# above_eps_rate: succeeded iff r < eps
# ---------------------------------------------------------------------------


def test_above_eps_rate_one_when_final_residual_exactly_eps(tmp_path: Path) -> None:
    """A final residual sitting on the boundary is "above" (not yet success)."""
    raw_dir = tmp_path / "raw"
    eps = 1e-8
    _write_seed_history(raw_dir, 0, "m", [1.0, eps], f_star=0.0)

    agg = _aggregate(raw_dir, tmp_path / "agg.json", eps)
    assert agg["methods"]["m"]["above_eps_rate"] == 1.0


def test_above_eps_rate_zero_when_final_residual_just_below_eps(tmp_path: Path) -> None:
    """A final residual strictly below the boundary is success."""
    raw_dir = tmp_path / "raw"
    eps = 1e-8
    _write_seed_history(raw_dir, 0, "m", [1.0, 0.5 * eps], f_star=0.0)

    agg = _aggregate(raw_dir, tmp_path / "agg.json", eps)
    assert agg["methods"]["m"]["above_eps_rate"] == 0.0


def test_above_eps_rate_proportional_across_seeds(tmp_path: Path) -> None:
    """3 of 4 seeds finish at/above eps -> rate = 0.75."""
    raw_dir = tmp_path / "raw"
    eps = 1e-8
    # Seed 0 succeeds (below); seeds 1-3 stay at/above.
    _write_seed_history(raw_dir, 0, "m", [1.0, 0.5 * eps], f_star=0.0)
    _write_seed_history(raw_dir, 1, "m", [1.0, eps], f_star=0.0)       # at -> above
    _write_seed_history(raw_dir, 2, "m", [1.0, 2 * eps], f_star=0.0)   # above
    _write_seed_history(raw_dir, 3, "m", [1.0, 1e-4], f_star=0.0)      # above

    agg = _aggregate(raw_dir, tmp_path / "agg.json", eps)
    assert agg["methods"]["m"]["above_eps_rate"] == 0.75


# ---------------------------------------------------------------------------
# Aggregator <-> check_stop consistency
# ---------------------------------------------------------------------------


def test_aggregator_matches_check_stop_at_boundary(tmp_path: Path) -> None:
    """A residual ``check_stop`` would refuse is also "not crossed" by the aggregator.

    Direct pair: feed the same ``(residual, eps)`` into ``check_stop``
    and into the aggregator.  Whatever ``check_stop`` says about
    "success", the aggregator must agree on "crossed".
    """
    raw_dir = tmp_path / "raw"
    eps = 1e-8

    # Boundary case: residual == eps.  check_stop says NOT success.
    decision = check_stop(k=0, n_iters=10, f_k=eps, f_star=0.0, eps=eps)
    assert not decision.stop, "check_stop incorrectly declared success at residual==eps"

    # Aggregator on the same value must also say "not crossed".
    _write_seed_history(raw_dir, 0, "m", [eps], f_star=0.0)
    agg = _aggregate(raw_dir, tmp_path / "agg.json", eps)
    assert agg["methods"]["m"]["iters_to_eps_per_seed"] == [None]

    # Just below: check_stop says success.
    decision_below = check_stop(
        k=0, n_iters=10, f_k=0.5 * eps, f_star=0.0, eps=eps
    )
    assert decision_below.stop
