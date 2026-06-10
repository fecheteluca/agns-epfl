from __future__ import annotations

from collections import defaultdict
from typing import Any


def new_history() -> defaultdict[str, list[Any]]:
    """Create an empty trace container matching upstream (``defaultdict(list)``)."""
    return defaultdict(list)


def record_trace(history: defaultdict[str, list[Any]], **fields: Any) -> None:
    """Append one value per provided field to the history.

    Parameters
    ----------
    history:
        The trace container created by :func:`new_history`.
    **fields:
        ``key=value`` pairs; each ``value`` is appended to ``history[key]``.
    """
    for key, value in fields.items():
        history[key].append(value)
