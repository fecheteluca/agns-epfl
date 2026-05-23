"""AGNS: gradient-regularised Newton optimisation methods.

This package implements a family of second-order optimisation methods (GNS,
AGNS) alongside a broad set of baselines (gradient, fast-gradient, Newton,
super-Newton, cubic Newton, accelerated cubic Newton, Monteiro-Svaiter ACN,
heavy-ball, L-BFGS, trust-region, Adam, AdaHessian), and the benchmark harness
that drives them.

Public entry points are documented in :mod:`agns.cli`.  See the top-level
``README.md`` for installation and usage.
"""

from __future__ import annotations

__version__: str = "0.1.0"

__all__ = ["__version__"]
