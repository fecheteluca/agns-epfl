"""Cross-cutting utilities.

Small, dependency-light helpers shared across the package:

* :mod:`agns.utils.timing` --- ``perf_counter``-based ``Timer`` used by
  every optimisation method to record wall-clock time.
* :mod:`agns.utils.seed` --- deterministic seeding of ``random`` and
  NumPy (and PyTorch when present), plus :func:`set_seed`.
"""

from __future__ import annotations

from agns.utils.seed import make_rng, set_global_seed, set_seed, with_seed
from agns.utils.timing import Timer, tic_toc

__all__ = [
    "Timer",
    "make_rng",
    "set_global_seed",
    "set_seed",
    "tic_toc",
    "with_seed",
]
