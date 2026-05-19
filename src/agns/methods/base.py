"""Common types shared by every optimisation method.

Every method in :mod:`agns.methods` is a free function returning the
canonical 3-tuple ``(x_final, status_string, history)``, where ``history``
is a plain :class:`collections.defaultdict` whose values are lists.
``HistoryDict`` is the type alias for this wire format; downstream
tooling (aggregator, figure pipeline) consumes it directly.

The :class:`History` dataclass is a typed-access convenience wrapper
provided for the analysis tooling.  It is **not** used by the
optimisation methods themselves --- they emit the raw ``defaultdict(list)``
so pickle output remains stable as the wrapper class evolves.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from agns.stopping import Status

__all__ = [
    "History",
    "HistoryDict",
    "MethodResult",
    "OptimizationMethod",
    "Status",
]


# Canonical wire format for per-iteration traces.
HistoryDict: TypeAlias = defaultdict[str, list[Any]]


@dataclass
class History:
    """Typed-access wrapper for analysis tooling.

    This class is **not** used by the optimisation methods themselves
    (they emit raw :data:`HistoryDict` instances).  Use
    :meth:`from_dict` to wrap a raw history dict loaded from disk.

    Canonical metric names (present when applicable) are:

    - ``func``: per-iteration function value
    - ``grad_norm``: dual norm of the gradient at the current iterate
    - ``gamma_k``: GNS-step adaptive search parameter
    - ``H_k``: cubic-Newton / super-Newton regulariser
    - ``M``: accelerated-cubic regulariser
    - ``L``: gradient / fast-gradient Lipschitz estimate
    - ``A_k``: Monteiro-Svaiter A-sequence
    - ``time``: wall-clock seconds since the run started
    - ``func_calls`` / ``grad_calls`` / ``hess_calls`` / ``hess_vec_calls``:
      oracle call counters
    - ``matrix_inverses``: dense linear-system solves
    - ``x_k``: per-iteration iterate (a full copy)
    """

    data: HistoryDict = field(default_factory=lambda: defaultdict(list))

    @classmethod
    def from_dict(cls, raw: HistoryDict | dict[str, list[Any]]) -> History:
        """Wrap a raw history dict (e.g. loaded from a pickle) as a History."""
        return cls(data=defaultdict(list, raw))

    def __getitem__(self, key: str) -> list[Any]:
        return self.data[key]

    def __setitem__(self, key: str, value: list[Any]) -> None:
        self.data[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self.data

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def keys(self) -> Any:
        return self.data.keys()

    def values(self) -> Any:
        return self.data.values()

    def items(self) -> Any:
        return self.data.items()

    def to_dict(self) -> dict[str, list[Any]]:
        """Plain-dict view of the wrapped history."""
        return dict(self.data)


# Canonical result tuple returned by every optimisation method.
MethodResult: TypeAlias = tuple[NDArray[np.float64], str, "HistoryDict | None"]


@runtime_checkable
class OptimizationMethod(Protocol):
    """Structural type for an optimisation method.

    Every method in :mod:`agns.methods` is a free function (not a class)
    matching this signature.  The protocol documents a single canonical
    interface to downstream tooling (the runner, the figure renderer).
    """

    def __call__(
        self,
        oracle: Any,
        x_0: NDArray[np.float64],
        **kwargs: Any,
    ) -> MethodResult: ...
