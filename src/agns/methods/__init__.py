"""Optimisation methods.

Each method is a free function returning the canonical 3-tuple
``(x_final, status_string, history)``.  See :mod:`agns.methods.base` for
the shared types.  The mapping from registry keys (used in YAML configs)
to functions lives in :mod:`agns.registry`.
"""

from __future__ import annotations

from agns.methods.agns import agns
from agns.methods.agns_wsm import agns_wsm
from agns.methods.base import History, HistoryDict, MethodResult, OptimizationMethod, Status
from agns.methods.gns import gns
from agns.methods.gns_wsm import gns_wsm

__all__ = [
    "History",
    "HistoryDict",
    "MethodResult",
    "OptimizationMethod",
    "Status",
    "agns",
    "agns_wsm",
    "gns",
    "gns_wsm",
]
