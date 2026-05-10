"""
Method registry.

Each entry in METHOD_REGISTRY maps a method key to a MethodSpec namedtuple
containing:
  - run_fn     : callable(oracle, x_0, **kw) -> (x_final, status, history)
  - param_map  : dict remapping generic param names to method-specific ones
                 (e.g. {"n_iters": "max_iter"} for gradient_method)
  - uses_approx: bool — whether the method takes approx_oracle / approx_hess_fn
  - uses_wsm   : bool — whether the method uses the WSM inversion backend
  - default_kw : method-specific keyword overrides applied at every call

The run() helper below translates the generic experiment configuration into
the concrete method call, hiding all per-method parameter name differences.
"""

from collections import namedtuple

from methods import (
    accelerated_grad_norm_smooth,
    accelerated_grad_norm_smooth_for_rank_one,
    cubic_newton,
    fast_gradient_method,
    grad_norm_smooth,
    grad_norm_smooth_for_rank_one,
    gradient_method,
    super_newton,
)

MethodSpec = namedtuple(
    "MethodSpec", ["run_fn", "param_map", "uses_approx", "uses_wsm", "default_kw"]
)

# Generic parameter set understood by every method entry:
#   n_iters, gamma_0, eps, adaptive_search, B, Binv, f_star, grad_tol
#   approx_oracle, approx_hess_fn  (only when uses_approx=True)

METHOD_REGISTRY = {
    # -----------------------------------------------------------------------
    # GNS — exact Hessian
    # -----------------------------------------------------------------------
    "gns_exact": MethodSpec(
        run_fn=grad_norm_smooth,
        param_map={},
        uses_approx=False,
        uses_wsm=False,
        default_kw={"is_approx": False},
    ),
    # -----------------------------------------------------------------------
    # GNS — inexact Hessian (full CHO solve)
    # -----------------------------------------------------------------------
    "gns_inexact": MethodSpec(
        run_fn=grad_norm_smooth,
        param_map={},
        uses_approx=True,
        uses_wsm=False,
        default_kw={"is_approx": True},
    ),
    # -----------------------------------------------------------------------
    # GNS rank-1 — inexact Hessian, Woodbury-Sherman-Morrison inversion
    # -----------------------------------------------------------------------
    "gns_wsm": MethodSpec(
        run_fn=grad_norm_smooth_for_rank_one,
        param_map={},
        uses_approx=True,
        uses_wsm=True,
        default_kw={"is_approx": True, "invert_backend": "wsm"},
    ),
    # -----------------------------------------------------------------------
    # AGNS — lookahead anchor (theoretically grounded default)
    # -----------------------------------------------------------------------
    "agns_lookahead": MethodSpec(
        run_fn=accelerated_grad_norm_smooth,
        param_map={},
        uses_approx=True,
        uses_wsm=False,
        default_kw={"is_approx": True, "hessian_anchor": "lookahead"},
    ),
    # -----------------------------------------------------------------------
    # AGNS — iterate anchor (ablation)
    # -----------------------------------------------------------------------
    "agns_iterate": MethodSpec(
        run_fn=accelerated_grad_norm_smooth,
        param_map={},
        uses_approx=True,
        uses_wsm=False,
        default_kw={"is_approx": True, "hessian_anchor": "iterate"},
    ),
    # -----------------------------------------------------------------------
    # AGNS exact — no Hessian approximation, lookahead anchor
    # -----------------------------------------------------------------------
    "agns_exact_lookahead": MethodSpec(
        run_fn=accelerated_grad_norm_smooth,
        param_map={},
        uses_approx=False,
        uses_wsm=False,
        default_kw={"is_approx": False, "hessian_anchor": "lookahead"},
    ),
    # -----------------------------------------------------------------------
    # AGNS rank-1 — WSM inversion, lookahead anchor
    # -----------------------------------------------------------------------
    "agns_wsm_lookahead": MethodSpec(
        run_fn=accelerated_grad_norm_smooth_for_rank_one,
        param_map={},
        uses_approx=True,
        uses_wsm=True,
        default_kw={
            "is_approx": True,
            "invert_backend": "wsm",
            "hessian_anchor": "lookahead",
        },
    ),
    # -----------------------------------------------------------------------
    # AGNS rank-1 — WSM inversion, iterate anchor (ablation)
    # -----------------------------------------------------------------------
    "agns_wsm_iterate": MethodSpec(
        run_fn=accelerated_grad_norm_smooth_for_rank_one,
        param_map={},
        uses_approx=True,
        uses_wsm=True,
        default_kw={
            "is_approx": True,
            "invert_backend": "wsm",
            "hessian_anchor": "iterate",
        },
    ),
    # -----------------------------------------------------------------------
    # Super-Universal Newton (cubic regularisation baseline)
    # -----------------------------------------------------------------------
    "super_newton": MethodSpec(
        run_fn=super_newton,
        param_map={"gamma_0": "H_0"},    # gamma_0 maps to H_0
        uses_approx=False,
        uses_wsm=False,
        default_kw={"alpha": 1.0},
    ),
    # -----------------------------------------------------------------------
    # Cubic Newton (ARC baseline)
    # -----------------------------------------------------------------------
    "cubic_newton": MethodSpec(
        run_fn=cubic_newton,
        param_map={"gamma_0": "H_0"},
        uses_approx=False,
        uses_wsm=False,
        default_kw={},
    ),
    # -----------------------------------------------------------------------
    # Gradient Method (first-order baseline)
    # -----------------------------------------------------------------------
    "gradient": MethodSpec(
        run_fn=gradient_method,
        param_map={"n_iters": "max_iter", "gamma_0": "L_0"},
        uses_approx=False,
        uses_wsm=False,
        default_kw={"line_search": True},
    ),
    # -----------------------------------------------------------------------
    # Fast (Nesterov-accelerated) Gradient Method
    # -----------------------------------------------------------------------
    "fast_gradient": MethodSpec(
        run_fn=fast_gradient_method,
        param_map={"n_iters": "max_iter", "gamma_0": "L_0"},
        uses_approx=False,
        uses_wsm=False,
        default_kw={"line_search": True},
    ),
}


def run_method(spec, oracle, x_0, cfg, approx_oracle, approx_hess_fn):
    """
    Execute a method given its MethodSpec and a standardised cfg dict.

    cfg keys (all optional, sensible defaults assumed):
        n_iters, gamma_0, eps, adaptive_search, B, Binv, f_star, grad_tol

    Returns (x_final, status, history) exactly as the underlying fn does.
    """
    # Build keyword arguments from generic cfg
    n_iters = cfg.get("n_iters", 1000)
    gamma_0 = cfg.get("gamma_0", 1.0)
    eps = cfg.get("eps", 1e-8)
    adaptive_search = cfg.get("adaptive_search", True)
    B = cfg.get("B")
    Binv = cfg.get("Binv")
    f_star = cfg.get("f_star")
    grad_tol = cfg.get("grad_tol")

    kw = {}

    # Apply default_kw from registry
    kw.update(spec.default_kw)

    # Generic → method-specific parameter name translation
    generic = {
        "n_iters": n_iters,
        "gamma_0": gamma_0,
        "eps": eps,
        "B": B,
        "Binv": Binv,
        "f_star": f_star,
        "adaptive_search": adaptive_search,
        "trace": True,
    }
    if grad_tol is not None:
        generic["grad_tol"] = grad_tol

    for key, val in generic.items():
        mapped_key = spec.param_map.get(key, key)
        kw[mapped_key] = val

    # gradient_method / fast_gradient_method use line_search instead of adaptive_search,
    # but they DO accept f_star, eps, grad_tol for early stopping.
    if spec.run_fn in (gradient_method, fast_gradient_method):
        kw.pop("adaptive_search", None)
        kw.pop("gamma_0", None)   # already remapped to L_0

    # cubic_newton does not accept grad_tol, B, Binv
    if spec.run_fn is cubic_newton:
        kw.pop("grad_tol", None)
        kw.pop("Binv", None)

    # Attach approx oracle if this method uses an inexact Hessian
    if spec.uses_approx and approx_oracle is not None and approx_hess_fn is not None:
        kw["approx_oracle"] = approx_oracle
        kw["approx_hess_fn"] = approx_hess_fn

    return spec.run_fn(oracle, x_0, **kw)
