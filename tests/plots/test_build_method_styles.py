"""Direct tests for :func:`agns.plots._helpers.build_method_styles`.

The helper is the core piece that gives sweep clones distinct colour
shades and labels.  Coverage focuses on the invariants a paper-grade
ablation panel relies on:

* every method receives a style dict
* sweep clones share the base hue (HLS-perturbed) but never identical
* sweep-clone labels include ``[param=value]``
* non-sweep methods get their canonical legend abbreviation verbatim
"""

from __future__ import annotations

import colorsys
from pathlib import Path

from agns.plots._helpers import build_method_styles


def _hue_of(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return colorsys.rgb_to_hls(r, g, b)[0]


def _make_methods(keys: list[str]) -> dict[str, dict]:
    return {k: {"label": k} for k in keys}


class TestBuildMethodStyles:
    def test_returns_one_dict_per_input_method(self) -> None:
        methods = _make_methods(["gns_exact", "agns_inexact", "gradient"])
        out = build_method_styles(methods, Path("."))
        assert set(out) == set(methods)
        for v in out.values():
            assert {"color", "linestyle", "marker", "linewidth", "zorder", "label"} <= set(v)

    def test_non_sweep_methods_get_canonical_abbrev(self) -> None:
        methods = _make_methods(["gns_exact", "agns_inexact"])
        out = build_method_styles(methods, Path("."))
        assert out["gns_exact"]["label"] == "GNS"
        assert out["agns_inexact"]["label"] == "AGNS-WGN"

    def test_sweep_clones_share_hue(self) -> None:
        methods = _make_methods([
            "agns_inexact__gamma_0=0p01",
            "agns_inexact__gamma_0=0p1",
            "agns_inexact__gamma_0=1",
            "agns_inexact__gamma_0=10",
        ])
        out = build_method_styles(methods, Path("."))
        hues = {round(_hue_of(out[k]["color"]), 3) for k in methods}
        assert len(hues) == 1, f"sweep clones must share hue, got {hues}"

    def test_sweep_clones_get_distinct_colors(self) -> None:
        methods = _make_methods([
            "agns_inexact__gamma_0=0p01",
            "agns_inexact__gamma_0=1",
            "agns_inexact__gamma_0=100",
        ])
        out = build_method_styles(methods, Path("."))
        colors = {out[k]["color"] for k in methods}
        assert len(colors) == len(methods), f"colors must be distinct: {colors}"

    def test_sweep_clones_get_distinct_linestyles_within_first_five(self) -> None:
        methods = _make_methods([
            f"agns_inexact__gamma_0={i}" for i in (1, 2, 3, 4, 5)
        ])
        out = build_method_styles(methods, Path("."))
        styles = {out[k]["linestyle"] for k in methods}
        assert len(styles) == 5

    def test_sweep_clone_labels_include_param_and_value(self) -> None:
        methods = _make_methods(["agns_inexact__gamma_0=0p1"])
        out = build_method_styles(methods, Path("."))
        label = out["agns_inexact__gamma_0=0p1"]["label"]
        assert "AGNS-WGN" in label
        assert "gamma0" in label or "gamma_0" in label
        assert "0.1" in label

    def test_categorical_sweep_value_passes_through(self) -> None:
        methods = _make_methods(["agns_inexact__restart_mode=gradient"])
        out = build_method_styles(methods, Path("."))
        label = out["agns_inexact__restart_mode=gradient"]["label"]
        assert "AGNS-WGN" in label
        assert "gradient" in label

    def test_mixed_sweep_and_reference(self) -> None:
        # The user often runs sweep clones alongside a non-swept reference
        # (e.g.\ a plain GNS baseline next to the AGNS gamma_0 sweep).
        # Both flavours should resolve cleanly within one call.
        methods = _make_methods([
            "agns_inexact__gamma_0=0p1",
            "agns_inexact__gamma_0=1",
            "gns_exact",
        ])
        out = build_method_styles(methods, Path("."))
        assert "gamma0" in out["agns_inexact__gamma_0=0p1"]["label"] or "gamma_0" in out["agns_inexact__gamma_0=0p1"]["label"]
        assert out["gns_exact"]["label"] == "GNS"

    def test_independent_sweep_groups_are_independent(self) -> None:
        methods = _make_methods([
            "agns_inexact__gamma_0=0p1",
            "agns_inexact__gamma_0=1",
            "agns_inexact__restart_mode=gradient",
            "agns_inexact__restart_mode=none",
        ])
        out = build_method_styles(methods, Path("."))
        # Each (base, param) sweep group is independent: the colours
        # within each group span the same lightness range.
        gamma_colors = {
            out["agns_inexact__gamma_0=0p1"]["color"],
            out["agns_inexact__gamma_0=1"]["color"],
        }
        restart_colors = {
            out["agns_inexact__restart_mode=gradient"]["color"],
            out["agns_inexact__restart_mode=none"]["color"],
        }
        # Within each group, colors must differ.
        assert len(gamma_colors) == 2
        assert len(restart_colors) == 2
