"""Cache of pre-computed reference ``f_star`` values keyed by problem instance.

The aggregator (:mod:`agns.cli.aggregate`) prefers a cached reference
``f_star`` over the fallback "per-seed minimum across methods" rule.
The fallback rule biases every residual toward whichever method was
the per-seed argmin: the winning method's residual collapses to zero
by construction, which is why a precomputed independent reference is
preferred whenever it exists.

Cache layout::

    results/reference_solutions/
    ├── manifest.json                                # flat list of all entries
    └── <problem_type>/
        └── <params_hash>_seed_<i>.json              # one per (instance, seed)

Cache key is ``(problem_type, sha256(canonical_params_json)[:16], seed)``,
where ``canonical_params_json`` is the problem ``params`` dict **excluding
seed** dumped with sorted keys.  Seed is a separate axis of the key
because the runner substitutes the global ``--seed`` into
``problem_params["seed"]`` (see ``agns.cli.run_benchmark.run_experiment``)
and the same underlying problem family produces a different data
instance per seed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_REFERENCE_DIR",
    "ReferenceEntry",
    "canonical_params_hash",
    "current_git_sha",
    "default_reference_dir",
    "entry_path",
    "load_entry",
    "load_manifest",
    "lookup",
    "rebuild_manifest",
    "save_entry",
]


def current_git_sha() -> str:
    """Return the abbreviated git SHA of the working tree, or ``""``.

    Used to stamp :class:`ReferenceEntry` records with provenance.
    Returns an empty string outside a git checkout or when ``git`` is
    unavailable -- the field is informational, not load-bearing.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.decode("utf-8", errors="ignore").strip()


#: Canonical default cache root.  Tests override with a tmp path.
DEFAULT_REFERENCE_DIR = Path("results/reference_solutions")


def default_reference_dir() -> Path:
    """Return the canonical reference-cache directory.

    Wrapper around :data:`DEFAULT_REFERENCE_DIR` for callers that want
    to import a function (e.g. tests that monkeypatch).
    """
    return DEFAULT_REFERENCE_DIR


@dataclass(frozen=True)
class ReferenceEntry:
    """One cached reference solution.

    Attributes mirror the on-disk JSON schema used by
    :func:`save_entry` / :func:`load_entry`.
    """

    problem_type: str
    params: dict[str, Any]
    seed: int
    f_ref: float
    grad_norm_ref: float | None
    solver: str
    n_iters_spent: int
    eps_target: float
    grad_tol_target: float | None
    status: str
    wall_time_s: float
    local_minimum_warning: bool = False
    code_commit: str = ""
    computed_at: str = ""
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_type": self.problem_type,
            "params": self.params,
            "seed": self.seed,
            "f_ref": self.f_ref,
            "grad_norm_ref": self.grad_norm_ref,
            "solver": self.solver,
            "n_iters_spent": self.n_iters_spent,
            "eps_target": self.eps_target,
            "grad_tol_target": self.grad_tol_target,
            "status": self.status,
            "wall_time_s": self.wall_time_s,
            "local_minimum_warning": self.local_minimum_warning,
            "code_commit": self.code_commit,
            "computed_at": self.computed_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ReferenceEntry:
        return cls(
            problem_type=str(raw["problem_type"]),
            params=dict(raw["params"]),
            seed=int(raw["seed"]),
            f_ref=float(raw["f_ref"]),
            grad_norm_ref=(
                float(raw["grad_norm_ref"]) if raw.get("grad_norm_ref") is not None else None
            ),
            solver=str(raw["solver"]),
            n_iters_spent=int(raw["n_iters_spent"]),
            eps_target=float(raw["eps_target"]),
            grad_tol_target=(
                float(raw["grad_tol_target"]) if raw.get("grad_tol_target") is not None else None
            ),
            status=str(raw.get("status", "")),
            wall_time_s=float(raw.get("wall_time_s", 0.0)),
            local_minimum_warning=bool(raw.get("local_minimum_warning", False)),
            code_commit=str(raw.get("code_commit", "")),
            computed_at=str(raw.get("computed_at", "")),
            schema_version=int(raw.get("schema_version", 1)),
        )


def canonical_params_hash(params: Mapping[str, Any]) -> str:
    """Return a stable 16-char SHA-256 prefix of ``params`` excluding ``seed``.

    The runner overwrites ``params["seed"]`` per multi-seed run; seed is
    therefore not part of the *instance* identity, only of the per-seed
    cache axis.  Excluding it here mirrors that contract.
    """
    filtered = {k: v for k, v in dict(params).items() if k != "seed"}
    canonical = json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def entry_path(root: Path, problem_type: str, params: Mapping[str, Any], seed: int) -> Path:
    """Resolve the on-disk JSON path for ``(problem_type, params, seed)``."""
    h = canonical_params_hash(params)
    return Path(root) / problem_type / f"{h}_seed_{seed}.json"


def save_entry(root: Path, entry: ReferenceEntry) -> Path:
    """Persist ``entry`` to its canonical path under ``root``; return the path.

    The directory layout is created on demand.  The manifest is **not**
    rewritten here; call :func:`rebuild_manifest` after a batch of
    writes (e.g. at the end of ``scripts/compute_reference_solutions.py``).
    """
    path = entry_path(Path(root), entry.problem_type, entry.params, entry.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(entry.to_dict(), indent=2).encode("utf-8")
    # Atomic write (simple variant; the project's utils/io.py atomic helper
    # would also work but keeping reference_cache.py free of cli imports).
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(encoded)
    tmp.replace(path)
    return path


def load_entry(path: Path) -> ReferenceEntry:
    """Load a single entry JSON; raise FileNotFoundError if missing."""
    with open(path, encoding="utf-8") as fh:
        return ReferenceEntry.from_dict(json.load(fh))


def load_manifest(root: Path) -> list[dict[str, Any]]:
    """Return the manifest as a list of dicts, or ``[]`` if not present."""
    manifest = Path(root) / "manifest.json"
    if not manifest.is_file():
        return []
    try:
        with open(manifest, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    return list(data) if isinstance(data, list) else []


def rebuild_manifest(root: Path) -> Path:
    """Re-scan ``root`` for entry JSONs and write ``manifest.json``.

    The manifest is a redundant index for fast existence checks; the
    entry JSONs are the source of truth.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for entry_json in sorted(root.glob("*/*_seed_*.json")):
        try:
            entry = load_entry(entry_json)
        except (OSError, KeyError, ValueError):
            continue
        rows.append(
            {
                "problem_type": entry.problem_type,
                "params_hash": canonical_params_hash(entry.params),
                "seed": entry.seed,
                "f_ref": entry.f_ref,
                "solver": entry.solver,
                "local_minimum_warning": entry.local_minimum_warning,
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2))
    return manifest


def lookup(
    root: Path,
    problem_type: str,
    params: Mapping[str, Any],
    seed: int,
) -> ReferenceEntry | None:
    """Return the cached entry for the (problem, seed) triple, or ``None``.

    Cheap: a single ``stat`` + ``open`` on the entry path.  Does not
    consult the manifest (which can drift stale relative to the entry
    files); the on-disk entry JSONs are the source of truth.
    """
    path = entry_path(Path(root), problem_type, params, seed)
    if not path.is_file():
        return None
    with contextlib.suppress(OSError, KeyError, ValueError):
        return load_entry(path)
    return None
