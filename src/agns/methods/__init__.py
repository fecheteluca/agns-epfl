"""Optimisation methods.

Each method is a free function returning the canonical 3-tuple
``(x_final, status_string, history)``.  See :mod:`agns.methods.base` for
the shared types.  The mapping from registry keys (used in YAML configs)
to functions lives in :mod:`agns.pipeline.registry`.
"""

from __future__ import annotations

from agns.methods.accelerated_cubic_newton import accelerated_cubic_newton
from agns.methods.adahessian import adahessian
from agns.methods.adam import adam
from agns.methods.agns import agns
from agns.methods.agns_wsm import agns_wsm
from agns.methods.base import History, HistoryDict, MethodResult, OptimizationMethod, Status
from agns.methods.cubic_newton import cubic_newton, cubic_newton_step
from agns.methods.fast_gradient import fast_gradient
from agns.methods.gns import gns
from agns.methods.gns_wsm import gns_wsm
from agns.methods.gradient import gradient
from agns.methods.heavy_ball import heavy_ball
from agns.methods.lbfgs import lbfgs
from agns.methods.monteiro_svaiter import monteiro_svaiter_acn
from agns.methods.newton import newton
from agns.methods.super_newton import super_newton
from agns.methods.trust_region import trust_region

__all__ = [
    "History",
    "HistoryDict",
    "MethodResult",
    "OptimizationMethod",
    "Status",
    "accelerated_cubic_newton",
    "adahessian",
    "adam",
    "agns",
    "agns_wsm",
    "cubic_newton",
    "cubic_newton_step",
    "fast_gradient",
    "gns",
    "gns_wsm",
    "gradient",
    "heavy_ball",
    "lbfgs",
    "monteiro_svaiter_acn",
    "newton",
    "super_newton",
    "trust_region",
]
