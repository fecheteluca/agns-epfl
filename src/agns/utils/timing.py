from __future__ import annotations

from datetime import datetime


class Timer:
    """Records elapsed wall-clock seconds since construction."""

    def __init__(self) -> None:
        self._start = datetime.now()

    def elapsed(self) -> float:
        """Return seconds elapsed since the timer was created."""
        return (datetime.now() - self._start).total_seconds()
