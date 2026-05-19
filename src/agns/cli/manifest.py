"""SHA-256 manifest of aggregated campaign JSONs.

Used in CI to detect silent drift in the experimental pipeline (e.g., a
non-deterministic BLAS path that bypassed thread pinning).  The manifest
records the canonical SHA-256 of every aggregated JSON under
``results/numerical/aggregated/``, with wall-clock-dependent fields
stripped before hashing.

Usage::

    # Regenerate the manifest from the current results.
    python -m agns.cli.manifest --write

    # Verify the on-disk artefacts against the committed manifest.
    python -m agns.cli.manifest --check

    # Show the diff without failing CI.
    python -m agns.cli.manifest --diff

Override the paths via ``--aggregated-dir`` / ``--manifest`` if you need
to compare alternate trees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

__all__ = ["compute_manifest", "load_manifest", "main", "sha256_canonical", "write_manifest"]

# Repo root is the parent of ``src/``; manifest.py lives at
# ``src/agns/cli/manifest.py`` so we go up four levels.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_AGG_DIR = _REPO_ROOT / "results" / "numerical" / "aggregated"
_DEFAULT_MANIFEST = _REPO_ROOT / "scripts" / "aggregated_checksums.txt"

HEADER = """# SHA-256 manifest of aggregated campaign JSONs.
#
# Regenerate with: python -m agns.cli.manifest --write
# Verify with:     python -m agns.cli.manifest --check
#
# Format: <campaign_name>  <sha256_hex>
#
"""


def _strip_wallclock(obj: object) -> object:
    """Recursively remove wall-clock-dependent fields from an aggregated JSON.

    ``time_curve`` and top-level ``time`` carry wall-clock measurements
    that vary across re-runs; everything else is deterministic under
    pinned BLAS threads.
    """
    if isinstance(obj, dict):
        return {k: _strip_wallclock(v) for k, v in obj.items() if k not in ("time_curve", "time")}
    if isinstance(obj, list):
        return [_strip_wallclock(v) for v in obj]
    return obj


def sha256_canonical(path: Path) -> str:
    """SHA-256 of the wall-clock-stripped, sorted-key canonical JSON at ``path``."""
    with open(path) as f:
        obj = json.load(f)
    stripped = _strip_wallclock(obj)
    canonical = json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def compute_manifest(agg_dir: Path) -> dict[str, str]:
    """SHA-256 of every aggregated JSON in ``agg_dir``, keyed by campaign."""
    return {p.stem: sha256_canonical(p) for p in sorted(agg_dir.glob("*.json"))}


def load_manifest(path: Path) -> dict[str, str]:
    """Parse a manifest file into a dict."""
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        result[parts[0]] = parts[1]
    return result


def write_manifest(manifest: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{name:<32s} {digest}" for name, digest in sorted(manifest.items()))
    path.write_text(HEADER + body + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="Regenerate the manifest.")
    g.add_argument("--check", action="store_true", help="Verify on-disk vs manifest.")
    g.add_argument("--diff", action="store_true", help="Show diff without exit code.")
    p.add_argument(
        "--aggregated-dir",
        default=str(_DEFAULT_AGG_DIR),
        help="Directory of aggregated JSONs (default: results/numerical/aggregated).",
    )
    p.add_argument(
        "--manifest",
        default=str(_DEFAULT_MANIFEST),
        help="Path to the manifest file (default: scripts/aggregated_checksums.txt).",
    )
    args = p.parse_args()

    agg_dir = Path(args.aggregated_dir).resolve()
    manifest_path = Path(args.manifest).resolve()

    if not agg_dir.is_dir():
        print(f"[manifest] aggregated dir not found: {agg_dir}", file=sys.stderr)
        return 1

    on_disk = compute_manifest(agg_dir)

    if args.write:
        write_manifest(on_disk, manifest_path)
        print(f"[manifest] wrote {len(on_disk)} entries to {manifest_path}")
        return 0

    committed = load_manifest(manifest_path)
    only_disk = set(on_disk) - set(committed)
    only_committed = set(committed) - set(on_disk)
    mismatched = [k for k in (set(on_disk) & set(committed)) if on_disk[k] != committed[k]]

    if not (only_disk or only_committed or mismatched):
        print(f"[manifest] OK: {len(on_disk)} campaigns match")
        return 0

    print(f"[manifest] DRIFT: on-disk has {len(on_disk)} entries; manifest has {len(committed)}")
    for c in sorted(only_disk):
        print(f"  on-disk only:  {c}  ({on_disk[c]})")
    for c in sorted(only_committed):
        print(f"  manifest only: {c}  (was {committed[c]})")
    for c in sorted(mismatched):
        print(f"  hash mismatch: {c}")
        print(f"    on-disk:   {on_disk[c]}")
        print(f"    manifest:  {committed[c]}")
    return 0 if args.diff else 1


if __name__ == "__main__":
    sys.exit(main())
