from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

VALID_RESTART_MODES: frozenset[str] = frozenset({"gradient", "function_value", "none"})


def validate_restart_mode(restart_mode: str) -> None:
    """Raise ``ValueError`` if ``restart_mode`` is not a supported scheme."""
    if restart_mode not in VALID_RESTART_MODES:
        raise ValueError(
            f"Unknown restart_mode {restart_mode!r}; "
            f"expected one of {sorted(VALID_RESTART_MODES)}."
        )


def should_restart(
    restart_mode: str,
    g_trial: NDArray[np.float64],
    x_trial: NDArray[np.float64],
    x_k: NDArray[np.float64],
    f_trial: float,
    f_k: float,
) -> bool:
    """Decide whether to reset the momentum sequence after a step.

    Parameters
    ----------
    restart_mode:
        One of :data:`VALID_RESTART_MODES`.
    g_trial, x_trial:
        Gradient and iterate produced by the accelerated step.
    x_k:
        The pre-step iterate (momentum anchor reference).
    f_trial, f_k:
        Trial and current function values.

    Returns
    -------
    bool
        ``True`` if the momentum should restart at this step.
    """
    if restart_mode == "gradient":
        return float(g_trial.dot(x_trial - x_k)) > 0.0
    if restart_mode == "function_value":
        return f_trial > f_k
    if restart_mode == "none":
        return False
    validate_restart_mode(restart_mode)
    return False  # unreachable; validate_restart_mode raises first
