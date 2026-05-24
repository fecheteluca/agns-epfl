"""Reproducible seed management.

Every experiment must be re-runnable from a single integer seed.
:func:`set_global_seed` seeds every relevant RNG (``random``, ``numpy``,
optionally ``torch``); :func:`with_seed` is a context manager that
seeds RNGs without permanently changing global state; :func:`make_rng`
returns a freshly-seeded ``numpy.random.Generator``.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator

import numpy as np

__all__ = ["make_rng", "set_global_seed", "with_seed"]


def set_global_seed(seed: int) -> None:
    """Seed the global RNGs: Python's ``random``, NumPy's legacy global RNG,
    and (when available) PyTorch's CPU + CUDA generators.

    Parameters
    ----------
    seed : int
        Non-negative integer used as the seed for every RNG.

    Notes
    -----
    ``np.random.default_rng()`` instances are *independent* of NumPy's
    legacy global RNG: this function does **not** influence freshly-
    constructed ``Generator`` objects.  Use :func:`make_rng` to obtain
    a deterministic ``Generator`` from a seed.

    PyTorch is imported lazily, so the core dependency graph stays free
    of torch.  If torch is unavailable, the torch step is skipped.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    try:  
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@contextlib.contextmanager
def with_seed(seed: int) -> Iterator[None]:
    """Seed every RNG temporarily, restoring previous state on exit.

    Examples
    --------
    >>> with with_seed(0):
    ...     _ = np.random.default_rng().standard_normal(3)
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        set_global_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def make_rng(seed: int) -> np.random.Generator:
    """Return a freshly-seeded :class:`numpy.random.Generator`."""
    return np.random.default_rng(seed)
