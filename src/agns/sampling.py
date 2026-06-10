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
