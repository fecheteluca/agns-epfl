"""Cross-cutting utilities.

Re-exports the small, dependency-light helpers shared across the
package.  :mod:`agns.utils.io` (atomic JSON / pickle writers) is used by
the benchmark runner and the aggregator, and
:class:`agns.utils.timing.Timer` by every optimisation method.  The
remaining helpers (:mod:`agns.utils.logging`, :mod:`agns.utils.memory`,
:mod:`agns.utils.seed`) are reusable building blocks for analysis
scripts and the test suite rather than wired into the pipeline itself.

* :mod:`agns.utils.io` --- atomic JSON / pickle write helpers with a
  NumPy-aware JSON encoder.
* :mod:`agns.utils.logging` --- ``get_logger`` factory (plain or JSON).
* :mod:`agns.utils.memory` --- peak-RSS tracking via :mod:`resource`.
* :mod:`agns.utils.seed` --- deterministic seeding of ``random``,
  NumPy, and (optionally) PyTorch.
* :mod:`agns.utils.timing` --- ``perf_counter``-based ``Timer``.
"""

from __future__ import annotations

from agns.utils.io import load_json, load_pickle, save_json, save_pickle
from agns.utils.logging import get_logger
from agns.utils.memory import MemorySnapshot, peak_memory_mb, track_memory
from agns.utils.seed import make_rng, set_global_seed, with_seed
from agns.utils.timing import Timer, tic_toc

__all__ = [
    "MemorySnapshot",
    "Timer",
    "get_logger",
    "load_json",
    "load_pickle",
    "make_rng",
    "peak_memory_mb",
    "save_json",
    "save_pickle",
    "set_global_seed",
    "tic_toc",
    "track_memory",
    "with_seed",
]
