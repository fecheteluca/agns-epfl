"""YAML config loader with ``include:`` composition.

Leaf configs in :mod:`configs/` share large blocks of defaults: every
LogSumExp benchmark wants the same problem parameters, every method has
the same colour / linestyle / linewidth across figures, every adaptive-
search dial is the same.  Re-declaring these in every leaf is a recipe
for drift: one config will quietly disagree with another and readers
will not know why.

The loader implements deep-merge composition:

* A YAML node may declare ``include: [<path>, ...]``.  Each path is
  resolved *relative to the file containing the include*, the resulting
  document is loaded recursively (with its own includes resolved), and
  deep-merged *underneath* the current node.  The current node's keys
  take precedence over included values.

* Dicts merge deeply, key by key.

* Lists are replaced outright (the more specific list wins).  This is
  the only sane behaviour for benchmark plotting: if a leaf says
  ``methods: [A, B]`` it wants exactly those, not the union of every
  included method list.

* List elements that are themselves dicts may carry their own
  ``include:`` directive: that lets the ``methods:`` array reference
  ``_base/methods/<name>.yaml`` to pick up default colour / linestyle /
  iter budget while the leaf supplies only the overrides.

Cycle detection: includes are tracked via a visited-set; a cycle raises
:class:`ConfigError`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import yaml

__all__ = ["ConfigError", "load_config"]


class ConfigError(ValueError):
    """Raised on malformed config or include cycles."""


_INCLUDE_KEY = "include"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config from ``path``, resolving every ``include:``.

    Returns a deep-merged composition: each ``include:`` is loaded
    recursively and merged *underneath* the file that referenced it.

    Raises
    ------
    ConfigError
        If the file does not exist, parses as something other than a
        mapping, or sits inside an include cycle.
    """
    abs_path = Path(path).resolve()
    return _load_with_visited(abs_path, visited=set())


def _load_with_visited(path: Path, visited: set[Path]) -> dict[str, Any]:
    if path in visited:
        cycle = " -> ".join(str(p) for p in (*visited, path))
        raise ConfigError(f"include cycle detected: {cycle}")
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"top-level YAML node must be a mapping, got {type(raw).__name__}: {path}"
        )
    new_visited = visited | {path}
    return cast(
        "dict[str, Any]",
        _resolve_includes(raw, base_dir=path.parent, visited=new_visited),
    )


def _resolve_includes(
    node: Any,
    base_dir: Path,
    visited: set[Path],
) -> Any:
    """Walk ``node`` and expand every ``include:`` directive in place."""
    if isinstance(node, dict):
        includes = node.pop(_INCLUDE_KEY, None)
        walked = {k: _resolve_includes(v, base_dir, visited) for k, v in node.items()}
        if includes is None:
            return walked
        if not isinstance(includes, list):
            raise ConfigError(
                f"'include' must be a list of paths, got {type(includes).__name__}"
            )
        merged: dict[str, Any] = {}
        for inc in includes:
            inc_path = _resolve_path(inc, base_dir)
            inc_data = _load_with_visited(inc_path, visited)
            merged = _deep_merge(merged, inc_data)
        return _deep_merge(merged, walked)
    if isinstance(node, list):
        return [_resolve_includes(item, base_dir, visited) for item in node]
    return node


def _resolve_path(spec: Any, base_dir: Path) -> Path:
    if not isinstance(spec, str):
        raise ConfigError(
            f"include entries must be strings, got {type(spec).__name__}: {spec!r}"
        )
    p = Path(spec)
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``override`` merged on top of ``base``.

    Dicts merge deeply; every other type (lists included) is replaced
    outright by the override value.  Neither input is mutated.
    """
    out: dict[str, Any] = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out
