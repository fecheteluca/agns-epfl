"""Pipeline composition layer.

Glues YAML configs (:mod:`agns.pipeline.config`) and sweep expansion
(:mod:`agns.pipeline.sweeps`) to the method/problem registries
(:mod:`agns.pipeline.registry`).  These three modules are consumed by
:mod:`agns.cli.run_benchmark` and the experimentation helpers.
"""

from agns.pipeline.config import ConfigError, load_config
from agns.pipeline.registry import (
    METHOD_REGISTRY,
    PROBLEM_REGISTRY,
    MethodSpec,
    run_method,
)
from agns.pipeline.sweeps import expand_sweeps

__all__ = [
    "METHOD_REGISTRY",
    "PROBLEM_REGISTRY",
    "ConfigError",
    "MethodSpec",
    "expand_sweeps",
    "load_config",
    "run_method",
]
