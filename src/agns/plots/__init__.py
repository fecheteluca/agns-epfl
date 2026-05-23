"""Matplotlib figure renderers consumed by :mod:`agns.cli.make_plots`.

One module per plot type.  All four share the helpers in
:mod:`agns.plots._helpers` so figure setup, legend placement, atomic
saving, and per-method style resolution are identical across artefacts.
"""

from agns.plots import convergence, rate_loglog, scaling, sweep

__all__ = [
    "convergence",
    "rate_loglog",
    "scaling",
    "sweep",
]
