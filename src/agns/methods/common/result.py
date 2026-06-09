"""Common return type for optimization methods.

Ported structure from ``epfml/grad-norm-smooth`` (Apache-2.0), commit 9f4ca00.
Upstream methods return the bare tuple ``(x_k, status, history)``; we keep the same
shape so trajectories and downstream plotting code stay byte-compatible.
"""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

# (final iterate, status string, history dict | None)
MethodResult = tuple[NDArray[Any], str, dict[str, Any] | None]
