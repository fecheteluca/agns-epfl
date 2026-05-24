"""Tests for the reference-solution cache module.

Covers the cache-key hashing, round-trip persistence, lookup
precedence, and the manifest rebuild.  Aggregator integration with the
cache (declared > cached > fallback precedence + warning emission) is
tested separately in :mod:`tests.cli.test_aggregate_reference_cache`.
"""

from __future__ import annotations

from pathlib import Path

from agns.pipeline.reference_cache import (
    ReferenceEntry,
    canonical_params_hash,
    entry_path,
    load_entry,
    load_manifest,
    lookup,
    rebuild_manifest,
    save_entry,
)

# ---------------------------------------------------------------------------
# canonical_params_hash
# ---------------------------------------------------------------------------


def test_hash_excludes_seed() -> None:
    """The cache key is per-instance; ``seed`` is a separate axis."""
    h_a = canonical_params_hash({"m": 100, "n": 10, "reg": 1e-3, "seed": 0})
    h_b = canonical_params_hash({"m": 100, "n": 10, "reg": 1e-3, "seed": 99})
    assert h_a == h_b


def test_hash_is_key_order_invariant() -> None:
    h_a = canonical_params_hash({"a": 1, "b": 2})
    h_b = canonical_params_hash({"b": 2, "a": 1})
    assert h_a == h_b


def test_hash_changes_with_value() -> None:
    h_a = canonical_params_hash({"reg": 1e-3})
    h_b = canonical_params_hash({"reg": 1e-2})
    assert h_a != h_b


def test_hash_is_16_hex_chars() -> None:
    h = canonical_params_hash({"x": 1})
    assert len(h) == 16
    int(h, 16)  # hex-decodable


# ---------------------------------------------------------------------------
# entry path layout
# ---------------------------------------------------------------------------


def test_entry_path_layout(tmp_path: Path) -> None:
    params = {"reg": 1e-3, "max_samples": 4000}
    p = entry_path(tmp_path, "logistic_regression_libsvm", params, seed=2)
    assert p.parent == tmp_path / "logistic_regression_libsvm"
    h = canonical_params_hash(params)
    assert p.name == f"{h}_seed_2.json"


# ---------------------------------------------------------------------------
# save / load round trip
# ---------------------------------------------------------------------------


def _toy_entry(seed: int = 0, f_ref: float = 0.123) -> ReferenceEntry:
    return ReferenceEntry(
        problem_type="logistic_regression_synthetic",
        params={"m": 50, "n": 5, "reg": 1e-3},
        seed=seed,
        f_ref=f_ref,
        grad_norm_ref=1e-15,
        solver="newton",
        n_iters_spent=42,
        eps_target=1e-16,
        grad_tol_target=1e-14,
        status="success, 42 iters",
        wall_time_s=0.5,
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    entry = _toy_entry()
    written = save_entry(tmp_path, entry)
    assert written.is_file()
    loaded = load_entry(written)
    assert loaded.f_ref == entry.f_ref
    assert loaded.params == entry.params
    assert loaded.problem_type == entry.problem_type
    assert loaded.seed == entry.seed
    assert loaded.solver == entry.solver


def test_save_excludes_seed_from_persisted_params(tmp_path: Path) -> None:
    """Caller's params dict gets persisted verbatim, sans seed.

    Maintainer rule (see :mod:`agns.pipeline.reference_cache`): seed is
    a separate axis of the cache key.  The persisted ``params`` field
    therefore must not duplicate ``seed`` inside ``params``.
    """
    entry = ReferenceEntry(
        problem_type="logistic_regression_synthetic",
        params={"m": 50, "n": 5, "reg": 1e-3},  # caller already stripped seed
        seed=3,
        f_ref=0.0,
        grad_norm_ref=None,
        solver="newton",
        n_iters_spent=1,
        eps_target=1e-12,
        grad_tol_target=None,
        status="ok",
        wall_time_s=0.0,
    )
    save_entry(tmp_path, entry)
    loaded = load_entry(entry_path(tmp_path, entry.problem_type, entry.params, entry.seed))
    assert "seed" not in loaded.params


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------


def test_lookup_hit(tmp_path: Path) -> None:
    entry = _toy_entry(seed=2, f_ref=0.42)
    save_entry(tmp_path, entry)
    found = lookup(tmp_path, entry.problem_type, entry.params, entry.seed)
    assert found is not None
    assert found.f_ref == 0.42


def test_lookup_miss_returns_none(tmp_path: Path) -> None:
    found = lookup(tmp_path, "logistic_regression_synthetic", {"m": 1, "n": 1}, 0)
    assert found is None


def test_lookup_miss_when_seed_differs(tmp_path: Path) -> None:
    entry = _toy_entry(seed=0)
    save_entry(tmp_path, entry)
    found = lookup(tmp_path, entry.problem_type, entry.params, seed=1)
    assert found is None


# ---------------------------------------------------------------------------
# manifest rebuild
# ---------------------------------------------------------------------------


def test_rebuild_manifest_collects_all_entries(tmp_path: Path) -> None:
    save_entry(tmp_path, _toy_entry(seed=0, f_ref=0.1))
    save_entry(tmp_path, _toy_entry(seed=1, f_ref=0.2))
    save_entry(
        tmp_path,
        ReferenceEntry(
            problem_type="softmax_regression_synthetic",
            params={"m": 30, "n": 3, "K": 2, "reg": 1e-3},
            seed=0,
            f_ref=0.05,
            grad_norm_ref=None,
            solver="fast_gradient",
            n_iters_spent=200,
            eps_target=1e-12,
            grad_tol_target=None,
            status="ok",
            wall_time_s=0.0,
        ),
    )
    manifest_path = rebuild_manifest(tmp_path)
    assert manifest_path.is_file()
    rows = load_manifest(tmp_path)
    assert len(rows) == 3
    problem_types = {r["problem_type"] for r in rows}
    assert problem_types == {"logistic_regression_synthetic", "softmax_regression_synthetic"}


def test_load_manifest_returns_empty_when_absent(tmp_path: Path) -> None:
    assert load_manifest(tmp_path) == []
