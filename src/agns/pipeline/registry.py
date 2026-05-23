"""Method registry + :func:`run_method` helper.

The registry binds method keys (used in YAML configs) to :class:`MethodSpec`
records describing how to call each method.  :func:`run_method` translates
a generic experiment ``cfg`` dict (with keys like ``n_iters``, ``gamma_0``,
``eps`` ...) into a concrete method call, attaching the approximate
Hessian whenever ``spec.uses_approx`` is set.

Because every method in :mod:`agns.methods` takes ``n_iters`` as the
iteration budget and uses the same generic kwargs, the registry no
longer needs a ``param_map`` indirection: the method signatures are the
single source of truth.

The :data:`METHOD_REGISTRY` and :data:`PROBLEM_REGISTRY` dicts are the
canonical name -> callable mappings consumed by the runner.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods import (
    MethodResult,
    accelerated_cubic_newton,
    adahessian,
    adam,
    agns,
    agns_wsm,
    cubic_newton,
    fast_gradient,
    gns,
    gns_wsm,
    gradient,
    heavy_ball,
    lbfgs,
    monteiro_svaiter_acn,
    newton,
    super_newton,
    trust_region,
)
from agns.oracles import PROBLEM_REGISTRY

__all__ = ["METHOD_REGISTRY", "PROBLEM_REGISTRY", "MethodSpec", "run_method"]


@dataclass(frozen=True)
class MethodSpec:
    """Registry entry for a single optimisation method.

    Attributes
    ----------
    run_fn :
        The free function implementing the method.  Must take
        ``oracle, x_0`` and a set of keyword arguments.
    uses_approx : bool
        If ``True``, :func:`run_method` will attach ``approx_oracle`` and
        ``approx_hess_fn`` to the kwargs (typically setting
        ``is_approx=True`` via ``default_kw``).
    uses_wsm : bool
        If ``True``, the method consumes a problem's ``approx_hess_fn_wsm``
        rather than ``approx_hess_fn``.
    default_kw : dict
        Fixed keyword arguments injected on every call (e.g.
        ``is_approx=True`` to select a method's inexact branch).
    """

    run_fn: Callable[..., MethodResult]
    uses_approx: bool = False
    uses_wsm: bool = False
    default_kw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-method configurable hyperparameters that the runner threads through
# from the YAML ``methods:`` entries.  Names match the method kwargs.
# ---------------------------------------------------------------------------
_OPTIONAL_KWARGS: tuple[str, ...] = (
    # AGNS dials
    "momentum_offset",
    "restart_mode",
    "adaptive_search_max_iter",
    # GNS/AGNS shared
    "gamma_min",
    # First-order / quasi-Newton hyperparameters
    "alpha",
    "beta",
    "lr",
    "beta1",
    "beta2",
    "eps_num",
    "n_hutchinson_probes",
    "seed",
    "m",
    "pgtol",
    "factr",
    # Cubic / accelerated cubic / MS-ACN dials
    "sigma",
    "sigma_max",
    "probe_max",
    "monotone",
    "M_0",
    "M_min",
    "M_max",
    "H_0",
    "H_min",
    # Newton (Armijo)
    "c1",
    "rho_backtrack",
    "alpha_0",
    "max_backtracks",
    "tau_0",
    "tau_max",
    # Trust region
    "Delta_0",
    "Delta_max",
    "eta1",
    "eta2",
    "shrink_factor",
    "expand_factor",
    "cg_tol_rel",
    "cg_max_iter",
    # Linesearch / line-search
    "line_search",
    "L_0",
    "L_min",
    # WSM backend
    "invert_backend",
)


METHOD_REGISTRY: dict[str, MethodSpec] = {
    # -----------------------------------------------------------------------
    # GNS family
    # -----------------------------------------------------------------------
    "gns_exact": MethodSpec(
        run_fn=gns,
        default_kw={"is_approx": False},
    ),
    "gns_inexact": MethodSpec(
        run_fn=gns,
        uses_approx=True,
        default_kw={"is_approx": True},
    ),
    "gns_wsm": MethodSpec(
        run_fn=gns_wsm,
        uses_approx=True,
        uses_wsm=True,
        default_kw={"is_approx": True, "invert_backend": "wsm"},
    ),
    # -----------------------------------------------------------------------
    # AGNS family
    # -----------------------------------------------------------------------
    # ``agns_exact`` and ``agns_inexact`` both bind to the same ``agns``
    # update function; the registry default kwargs decide which branch
    # runs (exact ``H`` from the oracle vs Fisher / Gauss-Newton
    # approximation supplied by ``approx_hess_fn``).
    "agns_exact": MethodSpec(
        run_fn=agns,
        default_kw={"is_approx": False},
    ),
    "agns_inexact": MethodSpec(
        run_fn=agns,
        uses_approx=True,
        default_kw={"is_approx": True},
    ),
    "agns_wsm": MethodSpec(
        run_fn=agns_wsm,
        uses_approx=True,
        uses_wsm=True,
        default_kw={"is_approx": True, "invert_backend": "wsm"},
    ),
    # -----------------------------------------------------------------------
    # Newton-type baselines
    # -----------------------------------------------------------------------
    "newton": MethodSpec(
        run_fn=newton,
        default_kw={"c1": 1e-4, "rho_backtrack": 0.5, "alpha_0": 1.0},
    ),
    "super_newton": MethodSpec(
        run_fn=super_newton,
        default_kw={"alpha": 1.0},
    ),
    "cubic_newton": MethodSpec(run_fn=cubic_newton),
    "accelerated_cubic_newton": MethodSpec(
        run_fn=accelerated_cubic_newton,
        default_kw={"sigma": 1.0, "adaptive_search": True},
    ),
    "monteiro_svaiter_acn": MethodSpec(
        run_fn=monteiro_svaiter_acn,
        default_kw={"sigma_max": 0.9, "monotone": True},
    ),
    "trust_region": MethodSpec(
        run_fn=trust_region,
        default_kw={"Delta_0": 1.0, "eta1": 0.25, "eta2": 0.75},
    ),
    # -----------------------------------------------------------------------
    # First-order baselines
    # -----------------------------------------------------------------------
    "gradient": MethodSpec(
        run_fn=gradient,
        default_kw={"line_search": True},
    ),
    "fast_gradient": MethodSpec(
        run_fn=fast_gradient,
        default_kw={"line_search": True},
    ),
    "heavy_ball": MethodSpec(
        run_fn=heavy_ball,
        default_kw={"alpha": 1.0, "beta": 0.9, "line_search": True},
    ),
    "lbfgs": MethodSpec(
        run_fn=lbfgs,
        default_kw={"m": 10},
    ),
    # -----------------------------------------------------------------------
    # ML optimisers
    # -----------------------------------------------------------------------
    "adam": MethodSpec(
        run_fn=adam,
        default_kw={"lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps_num": 1e-8},
    ),
    "adahessian": MethodSpec(
        run_fn=adahessian,
        default_kw={
            "lr": 0.1,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps_num": 1e-4,
            "n_hutchinson_probes": 1,
            "seed": 0,
        },
    ),
}


# Methods that do not consume the GNS adaptive-search dial.
_DROP_ADAPTIVE_SEARCH: frozenset[Callable[..., Any]] = frozenset(
    {
        newton,
        heavy_ball,
        lbfgs,
        trust_region,
        monteiro_svaiter_acn,
        adam,
        adahessian,
        gradient,
        fast_gradient,
    }
)

# Methods that do not consume the GNS gamma_0 dial.
_DROP_GAMMA_0: frozenset[Callable[..., Any]] = frozenset(
    {
        newton,
        heavy_ball,
        lbfgs,
        trust_region,
        adam,
        adahessian,
        gradient,
        fast_gradient,
        super_newton,
        cubic_newton,
        accelerated_cubic_newton,
        monteiro_svaiter_acn,
    }
)


def run_method(
    spec: MethodSpec,
    oracle: Any,
    x_0: NDArray[np.float64],
    cfg: dict[str, Any],
    approx_oracle: Any,
    approx_hess_fn: Any,
) -> MethodResult:
    """Execute a method given its :class:`MethodSpec` and a generic ``cfg``.

    ``cfg`` is a flat dict that may carry any subset of the generic
    parameter names (``n_iters``, ``gamma_0``, ``eps``, ...) plus any of
    the optional method-specific dials listed in :data:`_OPTIONAL_KWARGS`.

    Parameters
    ----------
    spec : MethodSpec
        Registry entry for the method.
    oracle, x_0 :
        Forwarded to the method.
    cfg : dict
        Generic configuration; see above.
    approx_oracle, approx_hess_fn :
        Attached to the kwargs when ``spec.uses_approx`` is set.
    """
    kw: dict[str, Any] = {}
    kw.update(spec.default_kw)

    for opt in _OPTIONAL_KWARGS:
        if opt in cfg:
            kw[opt] = cfg[opt]

    # Generic dials.  Every method takes ``n_iters`` and ``eps``; only
    # GNS-family methods take ``gamma_0`` / ``adaptive_search``.
    kw["n_iters"] = cfg.get("n_iters", 1000)
    kw["eps"] = cfg.get("eps", 1e-8)

    if spec.run_fn not in _DROP_GAMMA_0:
        kw["gamma_0"] = cfg.get("gamma_0", 1.0)
    if spec.run_fn not in _DROP_ADAPTIVE_SEARCH:
        kw["adaptive_search"] = cfg.get("adaptive_search", True)

    if "B" in cfg:
        kw["B"] = cfg["B"]
    # ``cubic_newton`` solves the cubic subproblem matrix-free in B and
    # therefore does not take a Binv keyword.
    if "Binv" in cfg and spec.run_fn is not cubic_newton:
        kw["Binv"] = cfg["Binv"]
    if "f_star" in cfg:
        kw["f_star"] = cfg["f_star"]
    if cfg.get("grad_tol") is not None:
        kw["grad_tol"] = cfg["grad_tol"]
    if cfg.get("warnings"):
        kw["warnings"] = True

    kw["trace"] = True

    if spec.uses_approx and approx_oracle is not None and approx_hess_fn is not None:
        kw["approx_oracle"] = approx_oracle
        kw["approx_hess_fn"] = approx_hess_fn

    return spec.run_fn(oracle, x_0, **kw)
