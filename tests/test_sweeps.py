"""Tests for :mod:`agns.sweeps`."""

from __future__ import annotations

import pytest

from agns.sweeps import expand_sweeps


def test_no_methods_passthrough() -> None:
    assert expand_sweeps({}) == {}
    assert expand_sweeps({"common": {}}) == {"common": {}}


def test_no_sweep_passthrough() -> None:
    cfg = {"methods": [{"name": "m", "n_iters": 10}]}
    assert expand_sweeps(cfg)["methods"] == [{"name": "m", "n_iters": 10}]


def test_sweep_expands_values() -> None:
    cfg = {
        "methods": [
            {
                "name": "m",
                "label": "M",
                "sweep": {"param": "rho", "values": [0.5, 0.7]},
            }
        ]
    }
    out = expand_sweeps(cfg)
    assert len(out["methods"]) == 2
    a, b = out["methods"]
    assert a["name"] == "m" and a["rho"] == 0.5 and a["key"] == "m__rho=0p5"
    assert b["name"] == "m" and b["rho"] == 0.7 and b["key"] == "m__rho=0p7"
    assert a["label"] == "M [rho=0.5]"
    assert b["label"] == "M [rho=0.7]"


def test_sweep_mixed_with_reference() -> None:
    cfg = {
        "methods": [
            {"name": "m1", "sweep": {"param": "rho", "values": [0.5, 0.7]}},
            {"name": "m2"},
        ]
    }
    out = expand_sweeps(cfg)
    assert [m["name"] for m in out["methods"]] == ["m1", "m1", "m2"]


def test_sweep_with_string_values() -> None:
    cfg = {
        "methods": [
            {"name": "m", "sweep": {"param": "restart_mode", "values": ["gradient", "none"]}}
        ]
    }
    out = expand_sweeps(cfg)
    keys = [m["key"] for m in out["methods"]]
    assert keys == ["m__restart_mode=gradient", "m__restart_mode=none"]


def test_sweep_idempotent() -> None:
    cfg = {"methods": [{"name": "m", "sweep": {"param": "rho", "values": [0.5]}}]}
    once = expand_sweeps(cfg)
    twice = expand_sweeps(once)
    assert once == twice


def test_sweep_validates_param() -> None:
    with pytest.raises(ValueError):
        expand_sweeps({"methods": [{"name": "m", "sweep": {"param": "", "values": [1]}}]})
    with pytest.raises(ValueError):
        expand_sweeps({"methods": [{"name": "m", "sweep": {"values": [1]}}]})


def test_sweep_validates_values() -> None:
    with pytest.raises(ValueError):
        expand_sweeps({"methods": [{"name": "m", "sweep": {"param": "rho", "values": []}}]})
    with pytest.raises(ValueError):
        expand_sweeps({"methods": [{"name": "m", "sweep": {"param": "rho", "values": "not_list"}}]})
    with pytest.raises(ValueError):
        expand_sweeps({"methods": [{"name": "m", "sweep": "not_a_dict"}]})
