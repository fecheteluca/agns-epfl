"""The hybrid seed protocol produces genuinely distinct, correctly-typed problems.

These tests guard the foundation of every statistical claim: because the methods are
deterministic, seeds must change the *problem* or every CI collapses to a point. We check
that ``instance`` seeds draw fresh data (distinct optima) with a fixed start, that ``start``
seeds hold the objective fixed (shared optimum) while moving the start, and that the
``vary`` mode is validated.
"""

from __future__ import annotations

import numpy as np
import pytest

from agns.sampling import problem_for_seed, spec_for_seed
from agns.utils.config import load_config
from agns.utils.paths import DATA_DIR


def test_spec_for_seed_instance_offsets_generation_seed():
    """``instance`` mode offsets the data seed and pins the start to be deterministic."""
    spec = {"family": "logsumexp", "vary": "instance", "seed": 100, "n": 5, "m": 7}
    out = spec_for_seed(spec, 3)
    assert out["seed"] == 103
    assert out["x0_random"] is False


def test_spec_for_seed_start_sets_random_start():
    """``start`` mode enables a per-seed random start and leaves the data seed alone."""
    spec = {"family": "rosenbrock", "vary": "start", "seed": 7}
    out = spec_for_seed(spec, 4)
    assert out["x0_random"] is True
    assert out["x0_seed"] == 4
    assert out["seed"] == 7  # objective data untouched


def test_spec_for_seed_rejects_unknown_vary():
    """An unknown ``vary`` mode is a configuration error."""
    with pytest.raises(ValueError, match="Unknown vary"):
        spec_for_seed({"family": "logsumexp", "vary": "bogus"}, 0)


def test_instance_seeds_give_distinct_optima():
    """Instance-sampled seeds draw different data, hence different ``f_star`` values."""
    spec = load_config("oracles/random_small.yaml")
    f_stars = [problem_for_seed(spec, s, DATA_DIR).f_star for s in range(4)]
    assert len({round(f, 9) for f in f_stars}) == 4


def test_instance_seeds_share_the_start():
    """Instance variation keeps the starting point fixed (spread is data-only)."""
    spec = load_config("oracles/random_small.yaml")
    starts = [problem_for_seed(spec, s, DATA_DIR).x_0 for s in range(3)]
    for s in starts[1:]:
        np.testing.assert_array_equal(s, starts[0])


def test_start_seeds_share_objective_but_move_start():
    """Start variation holds the objective (shared ``f_star``) and moves ``x_0``."""
    spec = load_config("oracles/conditioning_base.yaml")
    spec = {**spec, "cond": 50.0}
    problems = [problem_for_seed(spec, s, DATA_DIR) for s in range(3)]
    f_stars = [p.f_star for p in problems]
    np.testing.assert_allclose(f_stars, f_stars[0], rtol=1e-12)
    starts = [p.x_0 for p in problems]
    assert not np.allclose(starts[0], starts[1])
    assert not np.allclose(starts[1], starts[2])


def test_rosenbrock_start_variation_is_honored():
    """The Rosenbrock builder now draws a random start under the ``start`` protocol."""
    spec = load_config("oracles/rosenbrock.yaml")
    x0 = problem_for_seed(spec, 0, DATA_DIR).x_0
    x1 = problem_for_seed(spec, 1, DATA_DIR).x_0
    assert not np.allclose(x0, x1)
