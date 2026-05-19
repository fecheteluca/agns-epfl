"""Project logging configuration.

:func:`get_logger` returns a configured :class:`logging.Logger`.  Output
is plain text by default; pass ``json_output=True`` for one-line JSON
records suitable for the experiment harness.

The factory is idempotent: calling it more than once with the same name
returns the same logger without stacking handlers.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

__all__ = ["get_logger"]

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class _JsonFormatter(logging.Formatter):
    """Format log records as one-line JSON for structured ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


def get_logger(
    name: str = "agns",
    level: int = logging.INFO,
    *,
    json_output: bool = False,
) -> logging.Logger:
    """Return a configured logger.

    Idempotent: subsequent calls with the same ``name`` just return the
    existing logger.

    Parameters
    ----------
    name : str
        Logger name (dotted-path convention).
    level : int
        Minimum severity.  Default :data:`logging.INFO`.
    json_output : bool
        Emit one-line JSON records instead of plain text.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
