"""Tests for the AGNS-vs-baseline noise-floor diagnostic renderer.

Mirrors the structural pattern of ``test_renderers.py``: synthetic
per-seed inputs with known properties, assert the flag column
behaves correctly across three scenarios:

* clear winner (CI strictly above 1.0, low p) -> no flag;
* clear tie (CI tight around 1.0, p == 1) -> flag fires;
* borderline (CI brackets 1.0 even with non-trivial point estimate) -> flag fires.

Also pins the renderer's skip behaviour (baseline absent, AGNS absent,
length mismatch, empty aggregated dir).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agns.tables import noise_floor


def _make_method_record(per_seed: list | None, n_seeds: int | None = None) -> dict:
    """Minimal aggregated-method record for the noise_floor renderer.

    Only ``iters_to_eps_per_seed`` is read; other fields are not
    needed for these tests.  When ``per_seed`` is ``None``, the
    field is omitted (mirrors a failure-only record or an older
    aggregated JSON predating the per-seed-iters schema addition).
    """
    rec: dict = {}
    if per_seed is not None:
        rec["iters_to_eps_per_seed"] = list(per_seed)
    if n_seeds is not None:
        rec["n_seeds_succeeded"] = n_seeds
    return rec


def _make_campaign(
    methods: dict[str, dict],
    *,
    campaign: str = "synthetic",
    eps: float = 1e-8,
    n_seeds: int = 10,
) -> dict:
    return {
        "campaign": campaign,
        "eps_target": eps,
        "n_seeds": n_seeds,
        "methods": methods,
    }


def _render_one(tmp_path: Path, doc: dict) -> str:
    agg_dir = tmp_path / "agg"
    agg_dir.mkdir()
    (agg_dir / f"{doc['campaign']}.json").write_text(json.dumps(doc))
    out = tmp_path / "out.tex"
    noise_floor.render(agg_dir, out)
    return out.read_text()


def _body(tex: str) -> str:
    """Extract the body rows of a booktabs table (between midrule/bottomrule)."""
    start = tex.index(r"\midrule")
    end = tex.index(r"\bottomrule")
    return tex[start:end]


# ---------------------------------------------------------------------------
# Flag column behaviour
# ---------------------------------------------------------------------------


class TestFlagColumn:
    def test_clear_winner_no_flag(self, tmp_path: Path) -> None:
        """AGNS at 5 iters vs GNS at 50 iters with 10 paired seeds.

        Point speedup = 10.0; CI strictly above 1.0; Wilcoxon p tiny.
        No flag expected.
        """
        doc = _make_campaign({
            "agns_inexact": _make_method_record([5] * 10),
            "gns_inexact": _make_method_record([50] * 10),
        })
        tex = _render_one(tmp_path, doc)
        body = _body(tex)
        # The body should contain a GNS row with speedup 10.00 and no flag.
        assert "GNS" in body
        assert "10.00" in body
        # Flag column: r"$\approx$" must not appear in the body.
        assert r"$\approx$" not in body

    def test_identical_inputs_flag_fires(self, tmp_path: Path) -> None:
        """AGNS and baseline with identical per-seed iters.

        Point speedup = 1.00; CI tight on 1.0; Wilcoxon short-circuits
        to p=1.0 on bit-identical inputs.  Flag fires.
        """
        doc = _make_campaign({
            "agns_inexact": _make_method_record([10] * 10),
            "gns_inexact": _make_method_record([10] * 10),
        })
        body = _body(_render_one(tmp_path, doc))
        assert "1.00" in body
        assert r"$\approx$" in body

    def test_borderline_ci_brackets_one_flag_fires(self, tmp_path: Path) -> None:
        """Overlapping random distributions: CI typically brackets 1.0."""
        # Two arrays with overlapping integer ranges; we expect at
        # least the GNS row to carry the flag.
        import numpy as np

        rng = np.random.default_rng(42)
        agns = rng.integers(8, 13, size=10).tolist()
        gns = rng.integers(8, 13, size=10).tolist()
        doc = _make_campaign({
            "agns_inexact": _make_method_record(agns),
            "gns_inexact": _make_method_record(gns),
        })
        body = _body(_render_one(tmp_path, doc))
        assert r"$\approx$" in body


# ---------------------------------------------------------------------------
# Row inclusion / skipping
# ---------------------------------------------------------------------------


class TestRowInclusion:
    def test_baseline_without_per_seed_field_skipped(self, tmp_path: Path) -> None:
        """A baseline whose record lacks per-seed iter data is skipped silently."""
        doc = _make_campaign({
            "agns_inexact": _make_method_record([5] * 10),
            "gns_inexact": _make_method_record(None),  # no iters_to_eps_per_seed
        })
        body = _body(_render_one(tmp_path, doc))
        # No usable rows -> renderer emits the documented placeholder row,
        # which is a single ``\\``-terminated line.  Counting ``\\`` in
        # the body gives the actual row count.
        assert body.count(r"\\") == 1
        assert "no campaigns" in body.lower()

    def test_no_agns_record_means_no_rows(self, tmp_path: Path) -> None:
        """Campaign without AGNS produces no rows for that campaign."""
        doc = _make_campaign({
            "gns_inexact": _make_method_record([5] * 10),
            "newton": _make_method_record([7] * 10),
        })
        body = _body(_render_one(tmp_path, doc))
        # Placeholder row appears: "(no campaigns with usable AGNS + baseline data)".
        assert "no campaigns" in body.lower()

    def test_length_mismatch_skipped(self, tmp_path: Path) -> None:
        """AGNS and baseline with different per-seed list lengths -> skipped."""
        doc = _make_campaign({
            "agns_inexact": _make_method_record([5] * 10),
            "gns_inexact": _make_method_record([10] * 5),  # mismatched length
        })
        body = _body(_render_one(tmp_path, doc))
        # As above: only the placeholder row appears.
        assert body.count(r"\\") == 1
        assert "no campaigns" in body.lower()

    def test_agns_exact_fallback_when_inexact_absent(self, tmp_path: Path) -> None:
        """Ridge campaigns only have agns_exact; the fallback chain picks it up."""
        doc = _make_campaign({
            "agns_exact": _make_method_record([5] * 10),
            "gns_exact": _make_method_record([50] * 10),
        })
        body = _body(_render_one(tmp_path, doc))
        # "AGNS variant" column should carry agns_exact (escaped).
        assert "agns" in body.lower()
        assert "GNS" in body
        assert "10.00" in body  # speedup 50/5

    def test_picard_acn_row_appears_when_present(self, tmp_path: Path) -> None:
        """Picard-ACN is one of the documented BASELINES; it gets a row."""
        doc = _make_campaign({
            "agns_inexact": _make_method_record([5] * 10),
            "picard_acn_2008": _make_method_record([20] * 10),
        })
        body = _body(_render_one(tmp_path, doc))
        assert "Picard-ACN" in body
        assert "4.00" in body  # 20/5


# ---------------------------------------------------------------------------
# Top-level renderer contract
# ---------------------------------------------------------------------------


def test_render_raises_on_empty_aggregated_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        noise_floor.render(empty, tmp_path / "out.tex")


def test_render_caption_explains_flag_and_resampling(tmp_path: Path) -> None:
    """Caption is self-decoding: explains $\\approx$ and the resampling design."""
    doc = _make_campaign({
        "agns_inexact": _make_method_record([5] * 10),
        "gns_inexact": _make_method_record([50] * 10),
    })
    tex = _render_one(tmp_path, doc)
    assert "approx" in tex.lower()
    assert "bootstrap" in tex.lower()
    assert "Wilcoxon" in tex


def test_baselines_list_includes_documented_methods() -> None:
    """The BASELINES exported list pins the headline baseline set.

    Used by downstream consumers to reason about which columns the
    noise-floor table covers.  This test guards against accidental
    removal of a documented baseline.
    """
    baseline_names = [name for name, _ in noise_floor.BASELINES]
    for required in ("GNS", "ACN", "L-BFGS", "Picard-ACN", "Newton"):
        assert required in baseline_names
