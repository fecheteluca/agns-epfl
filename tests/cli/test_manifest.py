"""Tests for :mod:`agns.cli.manifest`."""

from __future__ import annotations

import json
from pathlib import Path

from agns.cli.manifest import (
    _strip_wallclock,
    compute_manifest,
    load_manifest,
    sha256_canonical,
    write_manifest,
)


def test_strip_wallclock_recursive() -> None:
    obj = {
        "x": 1,
        "time": 0.5,
        "time_curve": {"x": [0], "median": [1]},
        "nested": {"time": 9, "ok": True},
        "list": [{"time": 9, "ok": 1}],
    }
    out = _strip_wallclock(obj)
    assert out == {
        "x": 1,
        "nested": {"ok": True},
        "list": [{"ok": 1}],
    }


def test_sha256_canonical_is_order_invariant(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"x": 1, "y": 2}))
    b.write_text(json.dumps({"y": 2, "x": 1}))
    assert sha256_canonical(a) == sha256_canonical(b)


def test_sha256_canonical_strips_time(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"x": 1, "time": 0.001}))
    b.write_text(json.dumps({"x": 1, "time": 999.0}))
    assert sha256_canonical(a) == sha256_canonical(b)


def test_compute_and_load_manifest_roundtrip(tmp_path: Path) -> None:
    agg = tmp_path / "agg"
    agg.mkdir()
    (agg / "alpha.json").write_text(json.dumps({"x": 1}))
    (agg / "beta.json").write_text(json.dumps({"x": 2}))

    manifest = compute_manifest(agg)
    assert set(manifest.keys()) == {"alpha", "beta"}

    manifest_path = tmp_path / "m.txt"
    write_manifest(manifest, manifest_path)
    parsed = load_manifest(manifest_path)
    assert parsed == manifest


def test_load_manifest_missing_returns_empty(tmp_path: Path) -> None:
    assert load_manifest(tmp_path / "nope") == {}
