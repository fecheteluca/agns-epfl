"""LaTeX-table renderers consumed by :mod:`agns.cli.make_tables`.

One module per table type.  All four share the helpers in
:mod:`agns.tables._tex` so cell formatting and the booktabs scaffold
are identical across artefacts.
"""

from agns.tables import (
    noise_floor,
    per_campaign,
    real_world_summary,
    restart_density,
    speedup,
)

__all__ = [
    "noise_floor",
    "per_campaign",
    "real_world_summary",
    "restart_density",
    "speedup",
]
