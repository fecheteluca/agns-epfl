"""Safe serialisation helpers.

Atomic write-then-rename for pickle and JSON so an interrupted run never
leaves a half-written file on disk.  JSON serialisation transparently
handles NumPy scalars / arrays and :class:`pathlib.Path`.  Pickles use
protocol 4 for broad compatibility.
"""

from __future__ import annotations

import contextlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["load_json", "load_pickle", "save_json", "save_pickle"]

_PICKLE_PROTOCOL = 4


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands NumPy scalars, arrays, and Paths."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def _atomic_write(path: str | os.PathLike[str], data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (write-temp-then-rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, p)
    except Exception:
        with contextlib.suppress(OSError):  # pragma: no cover
            os.unlink(tmp)
        raise


def save_pickle(obj: Any, path: str | os.PathLike[str]) -> None:
    """Pickle ``obj`` to ``path`` atomically using protocol 4."""
    _atomic_write(path, pickle.dumps(obj, protocol=_PICKLE_PROTOCOL))


def load_pickle(path: str | os.PathLike[str]) -> Any:
    """Read a pickle from ``path``."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_json(obj: Any, path: str | os.PathLike[str], *, indent: int = 2) -> None:
    """Serialise ``obj`` to JSON at ``path`` atomically.

    Transparently encodes NumPy scalars, arrays, and :class:`pathlib.Path`.
    """
    encoded = json.dumps(obj, indent=indent, cls=_NumpyJSONEncoder).encode("utf-8")
    _atomic_write(path, encoded)


def load_json(path: str | os.PathLike[str]) -> Any:
    """Read a JSON document from ``path``."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
