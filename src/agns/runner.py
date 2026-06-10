from __future__ import annotations

import inspect
import warnings
from typing import Any

import numpy as np

from agns.oracles.problems import Problem
from agns.registry import get_approx, get_method
from agns.utils.seeding import set_seed


def _filtered_kwargs(func: Any, candidate: dict[str, Any]) -> dict[str, Any]:
    """Keep only the candidate kwargs that ``func`` actually accepts."""
    accepted = set(inspect.signature(func).parameters)
    return {k: v for k, v in candidate.items() if k in accepted}


def run_variant(
    variant: dict[str, Any],
    problem: Problem,
    *,
    n_iters: int,
    seed: int = 0,
    use_problem_metric: bool = True,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Run one method variant on a problem for a single seed.

    Parameters
    ----------
    variant:
        Variant config with at least ``method`` (a key in ``METHOD_FACTORY``), an optional
        ``params`` dict, and an optional ``approx`` (a key in ``APPROX_FACTORY``).
    problem:
        The LogSumExp instance providing oracle/start/metric/optimum.
    n_iters:
        Iteration budget (mapped to ``n_iters`` or ``max_iter`` per method).
    seed:
        Seed applied before the run (the deterministic methods are seed-independent given
        a fixed problem, but seeding keeps any randomness reproducible).
    use_problem_metric:
        If ``True`` pass the problem's ``B``/``Binv`` metric; otherwise run Euclidean.

    Returns
    -------
    tuple
        ``(x_k, status, history)`` from the method.
    """
    set_seed(seed)
    func = get_method(variant["method"])
    params = dict(variant.get("params", {}))

    candidate: dict[str, Any] = {
        "n_iters": n_iters,
        "max_iter": n_iters,
        "f_star": problem.f_star,
        "eps": problem.eps,
    }
    if use_problem_metric:
        candidate["B"] = problem.B
        candidate["Binv"] = problem.Binv

    approx_name = variant.get("approx")
    if approx_name is not None:
        candidate["is_approx"] = True
        candidate["approx_oracle"] = problem.oracle
        candidate["approx_hess_fn"] = get_approx(approx_name)

    # Variant params win over the defaults above; warn if a param is unused.
    accepted = set(inspect.signature(func).parameters)
    for key, value in params.items():
        if key not in accepted:
            warnings.warn(
                f"Variant {variant.get('label', variant['method'])!r}: param {key!r} "
                f"not accepted by method {variant['method']!r}; ignored.",
                stacklevel=2,
            )
        candidate[key] = value

    kwargs = _filtered_kwargs(func, candidate)
    return func(problem.oracle, problem.x_0, **kwargs)


def run_variant_seeds(
    variant: dict[str, Any],
    problem: Problem,
    *,
    n_iters: int,
    seeds: list[int],
    use_problem_metric: bool = True,
) -> list[dict[str, Any]]:
    """Run a variant across several seeds and return the list of histories."""
    histories = []
    for seed in seeds:
        _, _, history = run_variant(
            variant,
            problem,
            n_iters=n_iters,
            seed=seed,
            use_problem_metric=use_problem_metric,
        )
        histories.append(history)
    return histories
