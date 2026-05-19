"""Peak memory tracking helpers.

Optional helpers for downstream callers that want to record process-
level peak RSS alongside their results.  The current pipeline does not
auto-populate a memory field in ``summary.json``; these primitives are
exposed so a custom runner can pair an optimisation run with a
:func:`track_memory` context.

Implemented on top of :mod:`resource`, which is POSIX-only; on non-POSIX
platforms the helpers degrade to ``None``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

__all__ = ["MemorySnapshot", "peak_memory_mb", "track_memory"]

try:  # POSIX only: Linux, macOS
    import resource

    _HAS_RESOURCE = True
except ImportError:  # pragma: no cover -- exercised only on Windows
    _HAS_RESOURCE = False


def peak_memory_mb() -> float | None:
    """Return the process's peak resident set size in megabytes.

    Returns ``None`` on platforms without :mod:`resource` (e.g. Windows).
    Linux reports ``ru_maxrss`` in kilobytes, macOS in bytes; both are
    normalised to MB.
    """
    if not _HAS_RESOURCE:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Heuristic: if rss > 10 GB when interpreted as kilobytes, it's
    # almost certainly already in bytes (the macOS convention).
    if rss > 10 * 1024 * 1024:
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


@dataclass
class MemorySnapshot:
    """Memory snapshot taken at entry and exit of a context manager."""

    start_mb: float | None
    peak_mb: float | None

    @property
    def delta_mb(self) -> float | None:
        """Difference between peak and start, in MB.  ``None`` if unsupported."""
        if self.start_mb is None or self.peak_mb is None:
            return None
        return self.peak_mb - self.start_mb


@contextlib.contextmanager
def track_memory() -> Iterator[MemorySnapshot]:
    """Context manager that snapshots peak RSS at entry and exit."""
    snap = MemorySnapshot(start_mb=peak_memory_mb(), peak_mb=None)
    try:
        yield snap
    finally:
        snap.peak_mb = peak_memory_mb()
