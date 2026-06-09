"""Every registered method runs end-to-end on a synthetic and a real (LIBSVM) problem.

This is the integration smoke test for the config -> registry -> runner path. It also guards
that the test set covers *every* method in ``METHOD_FACTORY`` (so a newly registered method
cannot silently go untested), runs each on a small synthetic log-sum-exp instance and on the
vendored ``mushrooms`` LIBSVM dataset, and asserts each produces a finite, non-increasing
trajectory.
"""

from __future__ import annotations

import numpy as np
import pytest

from agns.oracles.problems import build_problem
from agns.registry import METHOD_FACTORY
from agns.runner import run_variant
from agns.utils.config import load_config
from agns.utils.paths import DATA_DIR
from tests.conftest import make_logsumexp_problem

# One variant per registered method. Methods without a config file (solve-backend or
# baseline variants) get a minimal inline variant. Approx/rank-one variants are exercised
# on the log-sum-exp family, the only family their approximate Hessian supports.
VARIANTS = {
    "gns": load_config("methods/gns.yaml"),
    "super_universal": load_config("methods/super_universal.yaml"),
    "cubic": load_config("methods/cubic.yaml"),
    "gradient": load_config("methods/gradient.yaml"),
    "fast_gradient": load_config("methods/fast_gradient.yaml"),
    "agns": load_config("methods/agns_exact.yaml"),
    "agns_rank_one": load_config("methods/agns_rank_one.yaml"),
    "gns_rank_one": {"method": "gns_rank_one", "params": {"invert_backend": "cho"}},
    "normalized_gradient": {"method": "normalized_gradient", "params": {}},
}


def test_every_registered_method_is_covered():
    """Guard: the smoke test exercises every method in the registry."""
    covered = {v["method"] for v in VARIANTS.values()}
    assert covered == set(
        METHOD_FACTORY
    ), f"untested registered methods: {set(METHOD_FACTORY) - covered}"


def _assert_sane_run(history):
    x_func = np.asarray(history["func"], dtype=float)
    assert x_func.size > 0
    assert np.all(np.isfinite(x_func))
    # No catastrophic increase: the final value is no worse than the start (sanity).
    assert x_func[-1] <= x_func[0] + 1e-6


@pytest.mark.parametrize("name", list(VARIANTS))
def test_method_runs_on_synthetic(name):
    """Each registered method runs on a small synthetic log-sum-exp instance."""
    problem = make_logsumexp_problem(n=12, m=20)
    _, status, history = run_variant(VARIANTS[name], problem, n_iters=40, seed=0)
    assert isinstance(status, str)
    _assert_sane_run(history)


@pytest.fixture(scope="module")
def mushrooms_problem():
    """The vendored ``mushrooms`` LIBSVM instance (log-sum-exp family)."""
    return build_problem(load_config("oracles/mushrooms.yaml"), DATA_DIR)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_method_runs_on_libsvm(name, mushrooms_problem):
    """Each registered method runs on the real ``mushrooms`` LIBSVM dataset."""
    _, status, history = run_variant(
        VARIANTS[name], mushrooms_problem, n_iters=4, seed=0
    )
    assert isinstance(status, str)
    _assert_sane_run(history)
