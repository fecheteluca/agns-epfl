"""AGNS: gradient-regularised Newton optimisation methods.

This package implements the gradient-regularised Newton step (GNS) of
Semenov-Jaggi-Doikov and its Nesterov-momentum + gradient-restart
acceleration (AGNS), together with the baselines used to study them
(gradient descent, Nesterov AGD, and the Super-Universal regularised
Newton method of Doikov-Mishchenko-Nesterov).

The package backs the experiments of an EPFL "Optimization for ML"
mini-project; see the top-level ``README.md`` for installation and how to
reproduce every figure and table.
"""

from __future__ import annotations

__version__: str = "0.1.0"

__all__ = ["__version__"]
