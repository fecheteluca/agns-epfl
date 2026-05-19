"""Tests for :mod:`agns.config` (YAML loader with include composition)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agns.config import ConfigError, load_config


def _write(path: Path, content: str) -> None:
    path.write_text(content)


def test_load_basic(tmp_path: Path) -> None:
    p = tmp_path / "x.yaml"
    _write(p, "a: 1\nb:\n  c: 2\n")
    cfg = load_config(p)
    assert cfg == {"a": 1, "b": {"c": 2}}


def test_include_merges_underneath(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    leaf = tmp_path / "leaf.yaml"
    _write(base, "x: 1\ny: 2\n")
    _write(leaf, "include:\n  - base.yaml\ny: 99\nz: 3\n")
    cfg = load_config(leaf)
    # Override takes precedence on `y`, plus the new `z`.
    assert cfg == {"x": 1, "y": 99, "z": 3}


def test_deep_merge(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    leaf = tmp_path / "leaf.yaml"
    _write(base, "outer:\n  a: 1\n  b: 2\n")
    _write(leaf, "include: [base.yaml]\nouter:\n  b: 99\n  c: 3\n")
    cfg = load_config(leaf)
    assert cfg == {"outer": {"a": 1, "b": 99, "c": 3}}


def test_lists_replace_not_merge(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    leaf = tmp_path / "leaf.yaml"
    _write(base, "methods: [a, b]\n")
    _write(leaf, "include: [base.yaml]\nmethods: [c, d]\n")
    cfg = load_config(leaf)
    assert cfg["methods"] == ["c", "d"]


def test_methods_entry_can_include(tmp_path: Path) -> None:
    method = tmp_path / "method.yaml"
    leaf = tmp_path / "leaf.yaml"
    _write(method, "name: m\ncolor: red\nn_iters: 100\n")
    _write(leaf, "methods:\n  - include: [method.yaml]\n    n_iters: 200\n")
    cfg = load_config(leaf)
    assert cfg["methods"] == [{"name": "m", "color": "red", "n_iters": 200}]


def test_missing_file_raises(tmp_path: Path) -> None:
    leaf = tmp_path / "leaf.yaml"
    _write(leaf, "include: [missing.yaml]\n")
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(leaf)


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    leaf = tmp_path / "leaf.yaml"
    _write(leaf, "- 1\n- 2\n")
    with pytest.raises(ConfigError, match="top-level YAML node must be a mapping"):
        load_config(leaf)


def test_include_cycle_detected(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    _write(a, "include: [b.yaml]\n")
    _write(b, "include: [a.yaml]\n")
    with pytest.raises(ConfigError, match="include cycle"):
        load_config(a)


def test_include_must_be_list(tmp_path: Path) -> None:
    p = tmp_path / "x.yaml"
    _write(p, "include: not_a_list\n")
    with pytest.raises(ConfigError, match="must be a list"):
        load_config(p)


def test_include_entry_must_be_string(tmp_path: Path) -> None:
    p = tmp_path / "x.yaml"
    _write(p, "include:\n  - 42\n")
    with pytest.raises(ConfigError, match="must be strings"):
        load_config(p)


def test_empty_yaml_returns_empty_dict(tmp_path: Path) -> None:
    p = tmp_path / "x.yaml"
    _write(p, "")
    assert load_config(p) == {}


def test_repo_configs_all_loadable() -> None:
    """Every checked-in campaign config must load."""
    cfg_root = Path(__file__).resolve().parents[1] / "configs"
    for yaml_file in cfg_root.rglob("*.yaml"):
        cfg = load_config(yaml_file)
        assert isinstance(cfg, dict), yaml_file
