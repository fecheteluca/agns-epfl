from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agns.utils.paths import CONFIG_DIR


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins on scalar conflicts)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve(path: str | Path, base_dir: Path) -> Path:
    """Resolve a config path relative to ``base_dir`` then to ``CONFIG_DIR``."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    local = base_dir / candidate
    if local.exists():
        return local
    return CONFIG_DIR / candidate


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, applying ``extends:`` inheritance.

    Parameters
    ----------
    path:
        Path to the YAML file (absolute, or relative to ``config/``).

    Returns
    -------
    dict
        The fully merged configuration.
    """
    path = _resolve(path, CONFIG_DIR)
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw

    parent = load_config(_resolve(parent_ref, path.parent))
    return _deep_merge(parent, raw)
