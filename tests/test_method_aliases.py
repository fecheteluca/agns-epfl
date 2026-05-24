"""Tests for the method-name deprecation alias machinery.

The registry exposes a small ``METHOD_ALIASES`` table that lets
legacy YAML configs (``name: monteiro_svaiter_acn``) keep working
for one release after a rename, while warning the operator to
migrate.  These tests pin the behavioural contract:

* canonical names resolve unchanged with no warning;
* deprecated names resolve to the canonical replacement and emit a
  ``DeprecationWarning`` exactly once per process;
* unknown names raise ``KeyError`` (preserves the old "unknown
  method" path in the runner).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pytest

from agns.cli import run_benchmark
from agns.pipeline.registry import (
    _WARNED_ALIASES,
    METHOD_ALIASES,
    METHOD_REGISTRY,
    resolve_method_name,
)


@pytest.fixture(autouse=True)
def _reset_warned_set() -> None:
    """Each test starts with a fresh ``_WARNED_ALIASES`` set.

    Otherwise the one-shot warning behaviour bleeds across tests:
    the first test that triggers an alias would warn, and every
    subsequent test would silently skip the warning.
    """
    _WARNED_ALIASES.clear()


# ---------------------------------------------------------------------------
# resolve_method_name
# ---------------------------------------------------------------------------


def test_canonical_name_returns_unchanged() -> None:
    """A canonical registry key resolves to itself with no warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = resolve_method_name("picard_acn_2008")
    assert result == "picard_acn_2008"
    assert not caught


def test_deprecated_name_resolves_to_canonical() -> None:
    """A deprecated alias resolves to the canonical replacement."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = resolve_method_name("monteiro_svaiter_acn")
    assert result == "picard_acn_2008"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_deprecated_warning_fires_only_once_per_process() -> None:
    """One-shot semantics so multi-seed runs don't spam the log."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_method_name("monteiro_svaiter_acn")
        resolve_method_name("monteiro_svaiter_acn")
        resolve_method_name("monteiro_svaiter_acn")
    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 1


def test_unknown_name_raises_keyerror() -> None:
    """Unknown names preserve the runner's existing "unknown method" path."""
    with pytest.raises(KeyError):
        resolve_method_name("absolutely_not_a_method")


def test_alias_target_is_canonical() -> None:
    """Every alias must point at a real registry key.

    Otherwise the deprecation path resolves to a name the runner
    cannot dispatch, and the user sees a confusing error instead
    of the deprecation guidance.
    """
    for alias, target in METHOD_ALIASES.items():
        assert target in METHOD_REGISTRY, (
            f"alias {alias!r} -> {target!r} but {target!r} is not in METHOD_REGISTRY"
        )


def test_warning_text_mentions_replacement_name() -> None:
    """The deprecation message tells the operator what to change to."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_method_name("monteiro_svaiter_acn")
    msg = str(caught[0].message)
    assert "monteiro_svaiter_acn" in msg
    assert "picard_acn_2008" in msg


# ---------------------------------------------------------------------------
# End-to-end: YAML with legacy method name still runs
# ---------------------------------------------------------------------------


def test_runner_accepts_legacy_method_name_in_yaml(tmp_path: Path) -> None:
    """A YAML config using the legacy name produces a history pickle.

    The pickle is written under the *legacy* key (because ``key``
    defaults to ``name`` and the runner preserves that for backwards
    compatibility with on-disk artefacts), but the underlying run
    uses the canonical method.  A DeprecationWarning fires.
    """
    cfg = {
        "problem": {
            "type": "logistic_regression_synthetic",
            "params": {"m": 30, "n": 3, "reg": 1.0e-2},
        },
        "common": {"gamma_0": 1.0, "eps": 1.0e-10, "adaptive_search": True},
        "methods": [
            # Legacy key still parses.
            {"name": "monteiro_svaiter_acn", "label": "Legacy-MS", "n_iters": 3},
        ],
    }
    save_dir = tmp_path / "seed_0"

    # Force the warning to be visible (project's pytest config sets
    # filterwarnings to "error" by default for unknown warnings, but
    # registers ignores for DeprecationWarning; capture explicitly).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_benchmark.run_experiment(
            cfg,
            argparse.Namespace(seed=0, methods=None, warnings=False),
            save_dir,
        )

    # The legacy method ran successfully -- pickle exists under the legacy key
    # (matches what the user's YAML asked for).
    assert (save_dir / "monteiro_svaiter_acn_history.pkl").is_file()
    summary = json.loads((save_dir / "summary.json").read_text())
    assert "monteiro_svaiter_acn" in summary["methods"]

    # The deprecation warning fired.
    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert any("monteiro_svaiter_acn" in str(w.message) for w in deprecation_warnings)


def test_runner_accepts_canonical_method_name_in_yaml(tmp_path: Path) -> None:
    """The new canonical key works without any warning."""
    cfg = {
        "problem": {
            "type": "logistic_regression_synthetic",
            "params": {"m": 30, "n": 3, "reg": 1.0e-2},
        },
        "common": {"gamma_0": 1.0, "eps": 1.0e-10, "adaptive_search": True},
        "methods": [
            {"name": "picard_acn_2008", "label": "Picard-ACN", "n_iters": 3},
        ],
    }
    save_dir = tmp_path / "seed_0"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_benchmark.run_experiment(
            cfg,
            argparse.Namespace(seed=0, methods=None, warnings=False),
            save_dir,
        )
    assert (save_dir / "picard_acn_2008_history.pkl").is_file()
    # No deprecation warnings for the canonical path.
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_legacy_and_canonical_produce_bit_identical_traces(tmp_path: Path) -> None:
    """The legacy alias dispatches to the same code path -- traces must match.

    Bit-stability is the load-bearing invariant of the project; the
    rename must not perturb numerics.  Run the same problem under
    both names and assert the function-value traces are byte-equal.
    """
    cfg_legacy = {
        "problem": {
            "type": "logistic_regression_synthetic",
            "params": {"m": 40, "n": 4, "reg": 1.0e-2},
        },
        "common": {"gamma_0": 1.0, "eps": 1.0e-10, "adaptive_search": True},
        "methods": [{"name": "monteiro_svaiter_acn", "label": "L", "n_iters": 5}],
    }
    cfg_new = dict(cfg_legacy)
    cfg_new["methods"] = [{"name": "picard_acn_2008", "label": "N", "n_iters": 5}]

    legacy_dir = tmp_path / "legacy"
    new_dir = tmp_path / "new"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        run_benchmark.run_experiment(
            cfg_legacy,
            argparse.Namespace(seed=0, methods=None, warnings=False),
            legacy_dir,
        )
    run_benchmark.run_experiment(
        cfg_new,
        argparse.Namespace(seed=0, methods=None, warnings=False),
        new_dir,
    )

    import pickle

    with open(legacy_dir / "monteiro_svaiter_acn_history.pkl", "rb") as fh:
        h_legacy = pickle.load(fh)
    with open(new_dir / "picard_acn_2008_history.pkl", "rb") as fh:
        h_new = pickle.load(fh)
    assert h_legacy["func"] == h_new["func"], "legacy alias diverged from canonical run"
