"""Host-fingerprint capture for cross-machine reproducibility.

The wall-clock speedup table embeds a one-line host fingerprint in its
caption so a reader can tell whether two runs of the same campaign
came from the same machine.  This module is the single source of
truth for what "host fingerprint" means.

Captured fields:

* ``cpu_model`` -- best-effort CPU model string (Linux: first
  ``model name`` from ``/proc/cpuinfo``; macOS/Windows: falls back to
  :func:`platform.processor`).
* ``platform`` -- :func:`platform.platform`.
* ``python_version`` -- ``X.Y.Z``.
* ``numpy_version`` / ``scipy_version`` -- as reported by the
  installed packages.
* ``blas_threads_env`` -- snapshot of the BLAS/OpenMP thread-count
  environment variables that the runner pins before any numpy import.
  Matches what was actually in effect during the run.

The capture is read-only: it inspects the current process and writes
nothing.  Repeated calls on the same machine return identical
fingerprints (modulo time-of-call fields, which are not captured).
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

__all__ = ["BLAS_ENV_VARS", "capture_host_info", "fmt_host_fingerprint"]


#: Environment variables read by the BLAS / OpenMP libraries.  The
#: runner pins these to ``"1"`` before any numpy import; the snapshot
#: lets the speedup table caption reproduce the exact thread state.
BLAS_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _cpu_model() -> str:
    """Best-effort CPU model name.

    On Linux, reads the first ``model name`` line from
    ``/proc/cpuinfo`` -- the most informative source.  On other
    platforms falls back to :func:`platform.processor`, which is
    sometimes empty (Windows / older macOS) -- empty values are
    surfaced as the literal string ``"unknown"`` so the caller can
    always print *something*.
    """
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(errors="ignore").splitlines():
                if line.startswith("model name"):
                    _, _, val = line.partition(":")
                    val = val.strip()
                    if val:
                        return val
        except OSError:  # pragma: no cover - defensive
            pass
    proc = platform.processor() or ""
    return proc.strip() or "unknown"


def _package_version(name: str) -> str:
    """Return ``<name>.__version__`` or ``"not installed"``."""
    try:
        mod = __import__(name)
    except ImportError:
        return "not installed"
    return str(getattr(mod, "__version__", "unknown"))


def capture_host_info() -> dict[str, Any]:
    """Snapshot the current host's reproducibility-relevant fingerprint.

    The returned dict is JSON-serialisable (no datetime, no Path) so
    it can be stamped directly into ``summary.json`` /
    ``aggregated/<campaign>.json``.
    """
    blas_env = {var: os.environ.get(var, "") for var in BLAS_ENV_VARS}
    return {
        "cpu_model": _cpu_model(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": _package_version("numpy"),
        "scipy_version": _package_version("scipy"),
        "blas_threads_env": blas_env,
    }


def fmt_host_fingerprint(host: dict[str, Any] | None) -> str:
    """Render a one-line caption fragment summarising the host.

    Returns an empty string for an empty / missing host record (so
    callers can append unconditionally without inserting a stray
    parenthesis).  When the BLAS env vars are all pinned to the
    same value (e.g. ``"1"`` for the canonical run), that value is
    summarised compactly; otherwise the variant ones are spelled out.
    """
    if not host:
        return ""
    cpu = str(host.get("cpu_model", "")).strip() or "unknown CPU"
    blas = host.get("blas_threads_env") or {}
    blas_vals = sorted(set(str(v) for v in blas.values() if v != ""))
    if len(blas_vals) == 1:
        blas_summary = f"BLAS threads = {blas_vals[0]}"
    elif blas_vals:
        # Hetereogeneous: list each var that is set.
        pieces = [f"{k}={v}" for k, v in blas.items() if v != ""]
        blas_summary = "BLAS threads: " + ", ".join(pieces)
    else:
        blas_summary = "BLAS threads unspecified"
    return f"{cpu}; {blas_summary}"
