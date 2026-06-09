"""Optimization methods (ported from epfml/grad-norm-smooth + novel AGNS)."""

from agns.methods.agns import agns
from agns.methods.agns_rank_one import agns_rank_one
from agns.methods.cubic import cubic_newton, cubic_newton_step
from agns.methods.fast_gradient import fast_gradient_method
from agns.methods.gns import grad_norm_smooth
from agns.methods.gns_rank_one import grad_norm_smooth_for_rank_one
from agns.methods.gradient import gradient_method
from agns.methods.normalized_gradient import grad_norm_smooth_gradient_method
from agns.methods.super_universal import super_newton

__all__ = [
    "agns",
    "agns_rank_one",
    "cubic_newton",
    "cubic_newton_step",
    "fast_gradient_method",
    "grad_norm_smooth",
    "grad_norm_smooth_for_rank_one",
    "gradient_method",
    "grad_norm_smooth_gradient_method",
    "super_newton",
]
