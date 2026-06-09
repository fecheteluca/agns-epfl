"""Wall-clock timing helper.

Mirrors the upstream pattern of stamping ``(datetime.now() - start).total_seconds()``
into each history record (``epfml/grad-norm-smooth``, Apache-2.0, commit 9f4ca00).
Timing is wall-clock and therefore excluded from the numeric equivalence tests.
"""

from __future__ import annotations

from datetime import datetime


class Timer:
    """Records elapsed wall-clock seconds since construction."""

    def __init__(self) -> None:
        self._start = datetime.now()

    def elapsed(self) -> float:
        """Return seconds elapsed since the timer was created."""
        return (datetime.now() - self._start).total_seconds()
