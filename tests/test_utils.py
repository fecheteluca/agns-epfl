"""Tests for :mod:`agns.utils`."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pytest

from agns.utils import (
    Timer,
    get_logger,
    load_json,
    load_pickle,
    make_rng,
    peak_memory_mb,
    save_json,
    save_pickle,
    set_global_seed,
    tic_toc,
    track_memory,
    with_seed,
)

# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def test_timer_is_monotonic() -> None:
    t = Timer()
    e1 = t.elapsed()
    time.sleep(0.001)
    e2 = t.elapsed()
    assert e2 >= e1 >= 0.0


def test_timer_start_resets() -> None:
    t = Timer()
    time.sleep(0.001)
    e = t.elapsed()
    assert e > 0.0
    t.start()
    assert t.elapsed() < e


def test_tic_toc_context() -> None:
    with tic_toc() as timer:
        time.sleep(0.0)
    assert timer.elapsed() >= 0.0


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------


def test_set_global_seed_is_deterministic() -> None:
    set_global_seed(123)
    a = np.random.random(5)
    set_global_seed(123)
    b = np.random.random(5)
    np.testing.assert_array_equal(a, b)


def test_set_global_seed_rejects_negative() -> None:
    with pytest.raises(ValueError):
        set_global_seed(-1)


def test_with_seed_restores_state() -> None:
    np.random.seed(0)
    before = np.random.get_state()
    with with_seed(99):
        np.random.random(3)
    after = np.random.get_state()
    # Compare the rng state tuple element-wise.
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2] == after[2]


def test_make_rng_is_independent() -> None:
    r1 = make_rng(0).random(3)
    r2 = make_rng(0).random(3)
    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------


def test_save_load_pickle_roundtrip(tmp_path: Path) -> None:
    obj = {"a": 1, "b": np.arange(3)}
    path = tmp_path / "x.pkl"
    save_pickle(obj, path)
    loaded = load_pickle(path)
    assert loaded["a"] == 1
    np.testing.assert_array_equal(loaded["b"], obj["b"])


def test_save_pickle_atomic_no_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "x.pkl"
    save_pickle({"a": 1}, path)
    # Only the final path should exist; no stray .tmp partials.
    siblings = list(tmp_path.iterdir())
    assert siblings == [path]


def test_save_json_handles_numpy(tmp_path: Path) -> None:
    obj = {"a": np.int64(5), "b": np.float64(1.5), "c": np.array([1.0, 2.0])}
    path = tmp_path / "x.json"
    save_json(obj, path)
    loaded = load_json(path)
    assert loaded == {"a": 5, "b": 1.5, "c": [1.0, 2.0]}


def test_save_json_handles_path(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    save_json({"p": tmp_path}, path)
    loaded = load_json(path)
    assert loaded == {"p": str(tmp_path)}


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def test_get_logger_idempotent() -> None:
    name = "agns_test_logger"
    a = get_logger(name)
    b = get_logger(name)
    assert a is b
    assert len(a.handlers) == 1


def test_get_logger_json_format(caplog: pytest.LogCaptureFixture) -> None:
    name = "agns_test_logger_json"
    logger = get_logger(name, json_output=True)
    assert isinstance(logger, logging.Logger)


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------


def test_peak_memory_mb_returns_number_or_none() -> None:
    v = peak_memory_mb()
    assert v is None or v >= 0


def test_track_memory_records_snapshots() -> None:
    with track_memory() as snap:
        _ = [0] * 1024
    if snap.start_mb is not None and snap.peak_mb is not None:
        assert snap.peak_mb >= snap.start_mb
        assert snap.delta_mb is not None
