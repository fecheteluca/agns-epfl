"""Shared helpers for the plot modules.

Figure setup, atomic PDF+PNG output, legend placement, per-method
style resolution, and the sweep-key parser used by the ablation
panels.  Every helper here is meant to be called from one of the
per-plot modules (:mod:`agns.plots.convergence` &c.); top-level
orchestration lives in :mod:`agns.cli.make_plots`.
"""

from __future__ import annotations

import colorsys
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agns.pipeline.config import ConfigError, load_config
from agns.plots._style import (
    LEGEND_ABBREV,
    PAPER_RC,
    RESID_FLOOR,
    STYLE_TABLE,
    UNKNOWN_STYLE,
)

__all__ = [
    "build_method_styles",
    "floor_log",
    "legend_below",
    "make_fig",
    "markevery",
    "parse_sweep_key",
    "resolve_style",
    "save_pair",
    "short_label",
    "style_for",
    "sweep_xscale",
]


def floor_log(arr: np.ndarray) -> np.ndarray:
    """Floor an array at :data:`RESID_FLOOR` so ``log(x)`` is safe."""
    return np.maximum(arr, RESID_FLOOR)


def short_label(label: str) -> str:
    """Strip the ``[param=value]`` sweep suffix from a label."""
    if " [" in label:
        return label.split(" [", 1)[0]
    return label


def markevery(n_points: int, target: int = 8) -> int | None:
    """Marker frequency that yields about ``target`` markers per curve."""
    if n_points <= target:
        return None
    return max(1, n_points // target)


def resolve_style(method_key: str) -> tuple[str, str, str, float, int]:
    """Look up the ``(color, linestyle, marker, linewidth, zorder)`` 5-tuple.

    Sweep clones (``agns_inexact__gamma_0=0p1``) resolve to the
    base-method style (``agns_inexact``).
    """
    base = method_key.split("__", 1)[0]
    return STYLE_TABLE.get(base, UNKNOWN_STYLE)


def style_for(method_key: str, rec: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolve the full style dict (visual style + legend label) for a method."""
    color, linestyle, marker, linewidth, zorder = resolve_style(method_key)
    base = method_key.split("__", 1)[0]
    abbr = LEGEND_ABBREV.get(base)
    if abbr is not None:
        label: str = abbr
    else:
        raw: str | None = rec.get("label")
        if raw is None:
            base_yaml = base_dir / f"{method_key}.yaml"
            if base_yaml.is_file():
                try:
                    cfg = load_config(base_yaml)
                    yaml_label = cfg.get("label")
                    if yaml_label is not None:
                        raw = str(yaml_label)
                except ConfigError:
                    pass
        label = short_label(raw) if raw else method_key
    return {
        "color": color,
        "linestyle": linestyle,
        "marker": marker,
        "linewidth": linewidth,
        "zorder": zorder,
        "label": label,
    }


def legend_below(ax: Any, *, ncol: int | None = None) -> None:
    """Place a compact legend below the axes.

    With the abbreviated method labels the legend fits within the plot
    width: up to five methods per row.  ``ncol`` is chosen automatically
    to balance the rows.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    n = len(handles)
    if ncol is None:
        rows = math.ceil(n / 5)
        ncol = math.ceil(n / rows)
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=ncol,
        borderaxespad=0.0,
        fontsize=11,
        columnspacing=1.0,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.3,
    )


def make_fig(rc_override: dict[str, Any] | None = None) -> tuple[Any, Any]:
    """Construct a single-axes figure using the project's rcParams."""
    plt.rcParams.update(PAPER_RC)
    if rc_override:
        plt.rcParams.update(rc_override)
    return plt.subplots()


def save_pair(fig: Any, out_dir: Path, name: str) -> None:
    """Write ``<name>.pdf`` and ``<name>.png`` with reproducible metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    # Suppress timestamps so a re-run on the same input produces a
    # byte-comparable PDF.
    metadata = {"CreationDate": None, "ModDate": None, "Producer": None}
    fig.savefig(pdf_path, metadata=metadata)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  [figure] {pdf_path}")


def parse_sweep_key(method_key: str) -> tuple[str, str, str] | None:
    """Split a sweep clone key into ``(base_method, param, value_str)``.

    Returns ``None`` for non-sweep keys (no ``__`` infix).
    """
    if "__" not in method_key:
        return None
    base, suffix = method_key.split("__", 1)
    if "=" not in suffix:
        return None
    param, value = suffix.split("=", 1)
    if not param or not value:
        return None
    return base, param, value


def sweep_xscale(values: list[float]) -> str:
    """Pick the x-axis scale for a numeric sweep.

    Log axis when the values are all positive and span a wide
    multiplicative range; linear axis otherwise.
    """
    if values and all(v > 0.0 for v in values) and max(values) / min(values) >= 50.0:
        return "log"
    return "linear"


# Linestyle cycle used to differentiate sweep clones within one group.
# Repeats after 5 entries -- the colour shade already separates them.
_SWEEP_LINESTYLE_CYCLE: tuple[str, ...] = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))


# Marker cycle for the same purpose.
_SWEEP_MARKER_CYCLE: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = (max(0, min(1, c)) for c in rgb)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _perturb_lightness(hex_color: str, t: float) -> str:
    """Shade ``hex_color`` toward white at ``t=0``, base at ``t=0.5``, dark at ``t=1``.

    Operates in HLS space so the hue is preserved.  The lightness span
    is constrained to ``[L_base - 0.18, L_base + 0.22]`` so the family
    identity stays visible (a darker AGNS shade is still recognisably
    AGNS).
    """
    r, g, b = _hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)  # noqa: E741
    l_max = min(l + 0.22, 0.86)
    l_min = max(l - 0.18, 0.18)
    l_new = (1.0 - t) * l_max + t * l_min
    return _rgb_to_hex(colorsys.hls_to_rgb(h, l_new, s))


def _sort_sweep_values(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Sort ``(method_key, raw_value)`` items numerically when possible."""

    def key(item: tuple[str, str]) -> tuple[int, float | str]:
        raw = item[1]
        try:
            return (0, float(raw.replace("p", ".")))
        except ValueError:
            return (1, raw)

    return sorted(items, key=key)


def _format_sweep_value(raw: str) -> str:
    """Pretty-print a sweep value for the legend (``0p01`` -> ``0.01``)."""
    try:
        return f"{float(raw.replace('p', '.')):g}"
    except ValueError:
        return raw


# Compact display names for swept hyperparameters, matching ``SWEEP_XLABEL``
# but written in plain text (no LaTeX math) so they read cleanly inside
# the legend bracket ``[param=value]``.
_SWEEP_PARAM_DISPLAY: dict[str, str] = {
    "gamma_0": "gamma0",
    "gamma0": "gamma0",
    "momentum_offset": "mom",
    "restart_mode": "restart",
    "inexact_rank": "rank",
}


def _format_sweep_param(param: str) -> str:
    return _SWEEP_PARAM_DISPLAY.get(param, param.replace("_", " "))


def build_method_styles(
    methods: dict[str, dict[str, Any]],
    base_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Return ``method_key -> style dict`` with sweep clones differentiated.

    Reference (non-sweep) methods get their base style from
    :func:`style_for` verbatim.  Sweep clones (``agns__gamma_0=0p1``)
    share the base method's hue but receive a distinct lightness shade,
    a distinct linestyle, and a label that includes ``[param=value]``
    so each curve is unambiguous in the legend.
    """
    # 1. Group sweep clones by (base, param) and sort numerically.
    sweep_groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for key in methods:
        parsed = parse_sweep_key(key)
        if parsed is None:
            continue
        base, param, raw_value = parsed
        sweep_groups.setdefault((base, param), []).append((key, raw_value))
    for k in sweep_groups:
        sweep_groups[k] = _sort_sweep_values(sweep_groups[k])

    out: dict[str, dict[str, Any]] = {}
    for method_key, rec in methods.items():
        parsed = parse_sweep_key(method_key)
        if parsed is None:
            out[method_key] = style_for(method_key, rec, base_dir)
            continue
        base, param, raw_value = parsed
        group = sweep_groups[(base, param)]
        i = next(idx for idx, (k, _) in enumerate(group) if k == method_key)
        n = len(group)
        base_color, _base_ls, _base_marker, base_lw, base_z = resolve_style(method_key)
        t = i / max(n - 1, 1)  # 0..1 across the swept range
        color = _perturb_lightness(base_color, t)
        linestyle = _SWEEP_LINESTYLE_CYCLE[i % len(_SWEEP_LINESTYLE_CYCLE)]
        marker = _SWEEP_MARKER_CYCLE[i % len(_SWEEP_MARKER_CYCLE)]
        label_base = LEGEND_ABBREV.get(base, base)
        label = f"{label_base} [{_format_sweep_param(param)}={_format_sweep_value(raw_value)}]"
        out[method_key] = {
            "color": color,
            "linestyle": linestyle,
            "marker": marker,
            "linewidth": base_lw,
            "zorder": base_z,
            "label": label,
        }
    return out
