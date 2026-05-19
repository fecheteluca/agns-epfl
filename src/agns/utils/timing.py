"""Wall-clock timing helpers.

Every optimisation method records ``time`` in its history trace via
:class:`Timer`, which wraps :func:`time.perf_counter` for monotonic
high-resolution measurements.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from time import perf_counter

__all__ = ["Timer", "tic_toc"]


@dataclass
class Timer:
    """Cumulative wall-clock timer based on :func:`time.perf_counter`.

    Construct a Timer, call :meth:`start` to begin accumulating, and
    :meth:`elapsed` at any point to read the wall-clock seconds since
    start.  Construction automatically calls :meth:`start`.
    """

    _t0: float = field(default_factory=perf_counter)

    def start(self) -> None:
        """(Re)start the timer."""
        self._t0 = perf_counter()

    def elapsed(self) -> float:
        """Seconds since the last call to :meth:`start` (or construction)."""
        return perf_counter() - self._t0


@contextlib.contextmanager
def tic_toc() -> Iterator[Timer]:
    """Context manager yielding a freshly-started :class:`Timer`.

    Examples
    --------
    >>> with tic_toc() as t:
    ...     pass
    >>> assert t.elapsed() >= 0.0
    """
    timer = Timer()
    timer.start()
    yield timer
