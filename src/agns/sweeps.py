"""Expand ``sweep:`` directives in a campaign config.

A method entry in ``configs/.../<campaign>.yaml`` may carry a
``sweep:`` sub-block::

    methods:
      - include: [../_base/methods/agns_theory.yaml]
        n_iters: 200
        sweep:
          param: rho
          values: [0.5, 0.6, 0.7071, 0.8, 0.9]

:func:`expand_sweeps` unrolls this entry into one method clone per value:
each clone gets ``param`` set to one element of ``values`` and a unique
``key = <name>__<param>=<value>`` so the resulting pickles do not collide
on disk and the aggregator records them as distinct methods.  The
clone's ``label`` is similarly suffixed so the figure pipeline produces
a sweep-axis legend automatically.

Method entries without a ``sweep:`` block pass through unchanged: a
campaign config can mix swept and reference methods in the same list
(e.g. "AGNS-Theory at five rhos plus Super-Newton once").

The function is *idempotent*: calling it on an already-expanded config
is a no-op.
"""

from __future__ import annotations

import copy
from typing import Any

__all__ = ["expand_sweeps"]


def _format_value(v: Any) -> str:
    """Turn ``v`` into a filename-safe suffix.

    Numbers: scientific notation kept terse (``1e-3`` not ``0.001``).
    Strings: passed through verbatim (e.g. ``restart_mode=gradient``).
    Anything else: ``str(v)`` with non-alphanumerics replaced by ``_``.
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        if 1e-3 <= abs(v) < 1e4:
            return f"{v:g}".replace(".", "p")
        return f"{v:.3e}".replace("+", "").replace("-0", "-").replace(".", "p")
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return v
    return str(v).translate(str.maketrans(" ,()[]'\"", "________"))


def _clone_method(mcfg: dict[str, Any], param: str, value: Any) -> dict[str, Any]:
    out = copy.deepcopy(mcfg)
    out.pop("sweep", None)
    out[param] = value
    name = out.get("name", "method")
    suffix = f"{param}={_format_value(value)}"
    out["key"] = f"{name}__{suffix}"
    base_label = out.get("label", name)
    out["label"] = f"{base_label} [{param}={value}]"
    return out


def expand_sweeps(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``cfg`` with every method's ``sweep:`` unrolled.

    Method entries without a ``sweep:`` block are left untouched.

    Raises
    ------
    ValueError
        For malformed sweep blocks (missing ``param`` / ``values``,
        empty values, non-list values).
    """
    if "methods" not in cfg or not isinstance(cfg["methods"], list):
        return cfg

    new_cfg = copy.deepcopy(cfg)
    expanded: list[dict[str, Any]] = []
    for mcfg in new_cfg["methods"]:
        sw = mcfg.get("sweep")
        if sw is None:
            expanded.append(mcfg)
            continue
        if not isinstance(sw, dict):
            raise ValueError(f"sweep block must be a mapping, got {type(sw).__name__}")
        param = sw.get("param")
        values = sw.get("values")
        if not isinstance(param, str) or not param:
            raise ValueError(f"sweep.param must be a non-empty string, got {param!r}")
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep.values must be a non-empty list, got {values!r}")
        for v in values:
            expanded.append(_clone_method(mcfg, param, v))
    new_cfg["methods"] = expanded
    return new_cfg
