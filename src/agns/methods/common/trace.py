"""History recording with the upstream schema.

Upstream (``epfml/grad-norm-smooth``, Apache-2.0, commit 9f4ca00) stores traces in a
``collections.defaultdict(list)`` and appends one value per key per iteration. We keep
the identical container and append semantics so ``utils/plotting.py`` (also ported)
works unchanged. Each method passes its own key set (e.g. ``gamma_k`` for GNS, ``H_k``
for Super-Universal, ``grad_sqr_norm``/``L`` for gradient methods).
"""

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
