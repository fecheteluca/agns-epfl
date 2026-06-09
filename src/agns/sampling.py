"""Per-seed problem sampling (the hybrid seed protocol).

The optimization methods are deterministic, so a run-seed only produces statistical spread
if it changes the *problem*. Each oracle config declares how a seed varies it via a ``vary``
key:

* ``"instance"`` -- draw a fresh random instance per seed (offset the generation ``seed``);
  used for families with random data (random log-sum-exp, nonlinear equations, polytope).
  The starting point is held fixed so the spread is purely instance-to-instance.
* ``"start"``    -- hold the objective fixed and draw a random starting point per seed
  (``x0_random`` with ``x0_seed = seed``); used for parametric objectives (Rosenbrock,
  Chebyshev), for controlled sweeps that must hold an attribute fixed (conditioning,
  curvature), and for real LIBSVM datasets that cannot be regenerated.

This is the single place that turns a base oracle spec + a seed into a concrete
:class:`~agns.oracles.problems.Problem`, so every experiment path samples seeds the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agns.oracles.problems import Problem, build_problem

VALID_VARY = frozenset({"instance", "start"})


def spec_for_seed(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return a per-seed copy of ``spec`` according to its ``vary`` mode.

    Parameters
    ----------
    spec:
        Base oracle config. ``spec["vary"]`` selects the protocol (default ``"instance"``).
    seed:
        Run-seed; offsets the generation seed (``"instance"``) or the start seed (``"start"``).

    Returns
    -------
    dict
        A shallow copy with the seed-dependent fields set.
    """
    vary = spec.get("vary", "instance")
    if vary not in VALID_VARY:
        raise ValueError(
            f"Unknown vary {vary!r} for oracle {spec.get('name', spec.get('family'))!r}; "
            f"expected one of {sorted(VALID_VARY)}."
        )
    out = dict(spec)
    if vary == "instance":
        # Fresh data instance; keep the start deterministic so spread is instance-only.
        out["seed"] = int(spec.get("seed", 0)) + int(seed)
        out["x0_random"] = False
    else:  # "start"
        # Fixed objective; random starting point per seed.
        out["x0_random"] = True
        out["x0_seed"] = int(seed)
    return out


def problem_for_seed(spec: dict[str, Any], seed: int, data_dir: str | Path) -> Problem:
    """Build the :class:`~agns.oracles.problems.Problem` for ``spec`` at run-seed ``seed``."""
    return build_problem(spec_for_seed(spec, seed), data_dir)
