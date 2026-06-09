"""Every registered method makes progress, and the descent methods never go uphill.

On a small, well-conditioned SPD quadratic (optimum ``f* = 0``) each method config is run
with the runner and required to drive the functional residual below tolerance. The methods
whose acceptance test guarantees decrease (GNS, gradient, cubic, normalized gradient) are
additionally required to be monotone. This is the validity gate for the whole method zoo;
``00_sanity.ipynb`` is the notebook-level counterpart over every problem family.
"""

from __future__ import annotations

import numpy as np
import pytest

from agns.runner import run_variant
from agns.utils.config import load_config
from agns.utils.paths import CONFIG_DIR
from tests.conftest import make_quadratic_problem

# A few methods are registered but ship no method config (they are solve-backend or
# baseline variants); fall back to a minimal inline variant for those.
_INLINE_VARIANTS = {
    "normalized_gradient": {
        "method": "normalized_gradient",
        "params": {"gamma_0": 1.0},
    },
}


def _variant(method_name):
    """Return the method config, or a minimal inline variant if no config file exists."""
    if (CONFIG_DIR / "methods" / f"{method_name}.yaml").exists():
        return load_config(f"methods/{method_name}.yaml")
    return _INLINE_VARIANTS[method_name]


# Newton-type methods solve the quadratic in a handful of steps.
SECOND_ORDER = [
    "gns",
    "super_universal",
    "cubic",
    "agns_exact",
    "agns_norestart",
    "agns_fvalue",
]
# First-order methods need a larger budget but still converge on a benign quadratic.
FIRST_ORDER = ["gradient", "fast_gradient", "normalized_gradient"]
# Methods whose acceptance test guarantees a non-increasing function value.
MONOTONE = ["gns", "cubic", "normalized_gradient", "gradient"]


def _residuals(history, f_star):
    return np.asarray(history["func"], dtype=float) - f_star


@pytest.mark.parametrize("method_name", SECOND_ORDER)
def test_second_order_reaches_tolerance(method_name):
    """Each Newton-type method drives ``f - f*`` below ``1e-7`` within a small budget."""
    problem = make_quadratic_problem(n=8, cond=10.0)
    variant = _variant(method_name)
    _, _, history = run_variant(variant, problem, n_iters=80, seed=0)
    assert _residuals(history, problem.f_star).min() < 1e-7


@pytest.mark.parametrize("method_name", FIRST_ORDER)
def test_first_order_reaches_tolerance(method_name):
    """Each first-order method drives ``f - f*`` below ``1e-6`` on a benign quadratic."""
    problem = make_quadratic_problem(n=8, cond=10.0)
    variant = _variant(method_name)
    _, _, history = run_variant(variant, problem, n_iters=8000, seed=0)
    assert _residuals(history, problem.f_star).min() < 1e-6


@pytest.mark.parametrize("method_name", MONOTONE)
def test_descent_methods_are_monotone(method_name):
    """Methods with a decrease-based acceptance test never increase ``f``."""
    problem = make_quadratic_problem(n=8, cond=10.0)
    variant = _variant(method_name)
    _, _, history = run_variant(variant, problem, n_iters=2000, seed=0)
    func = np.asarray(history["func"], dtype=float)
    increases = np.diff(func)
    tol = 1e-9 * max(1.0, float(np.abs(func[0])))
    assert increases.max() <= tol
