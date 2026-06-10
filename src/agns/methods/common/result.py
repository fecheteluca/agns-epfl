from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

# (final iterate, status string, history dict | None)
MethodResult = tuple[NDArray[Any], str, dict[str, Any] | None]
