"""Name -> factory registries for methods, approximate Hessians, and oracles.

Configs refer to components by string name; this module maps those names to the
callables. Method *variants* (e.g. ``agns_exact`` vs ``agns_inexact`` vs
``agns_norestart``) are not separate functions -- they are the same base method bound to
different parameters in ``config/methods/*.yaml`` and dispatched by :mod:`agns.runner`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agns.methods.agns import agns
from agns.methods.agns_rank_one import agns_rank_one
from agns.methods.cubic import cubic_newton
from agns.methods.fast_gradient import fast_gradient_method
from agns.methods.gns import grad_norm_smooth
from agns.methods.gns_rank_one import grad_norm_smooth_for_rank_one
from agns.methods.gradient import gradient_method
from agns.methods.normalized_gradient import grad_norm_smooth_gradient_method
from agns.methods.super_universal import super_newton
from agns.oracles.approximations import (
    approx_hess_fn_chebyshev,
    approx_hess_fn_fisher_term,
    approx_hess_fn_logsumexp,
    approx_hess_nonlinear_equations,
)

METHOD_FACTORY: dict[str, Callable[..., Any]] = {
    "gns": grad_norm_smooth,
    "gns_rank_one": grad_norm_smooth_for_rank_one,
    "super_universal": super_newton,
    "cubic": cubic_newton,
    "gradient": gradient_method,
    "fast_gradient": fast_gradient_method,
    "normalized_gradient": grad_norm_smooth_gradient_method,
    "agns": agns,
    "agns_rank_one": agns_rank_one,
}

APPROX_FACTORY: dict[str, Callable[..., Any]] = {
    "logsumexp": approx_hess_fn_logsumexp,
    "fisher_term": approx_hess_fn_fisher_term,
    "nonlinear_equations": approx_hess_nonlinear_equations,
    "chebyshev": approx_hess_fn_chebyshev,
}


def get_method(name: str) -> Callable[..., Any]:
    """Return the base method callable registered under ``name``."""
    try:
        return METHOD_FACTORY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown method {name!r}; available: {sorted(METHOD_FACTORY)}"
        ) from exc


def get_approx(name: str) -> Callable[..., Any]:
    """Return the approximate-Hessian callable registered under ``name``."""
    try:
        return APPROX_FACTORY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown approx_hess_fn {name!r}; available: {sorted(APPROX_FACTORY)}"
        ) from exc
