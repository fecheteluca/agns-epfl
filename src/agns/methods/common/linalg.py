from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

_DEFAULT_JITTERS: tuple[float, ...] = (1e-12, 1e-10, 1e-8, 1e-6)


def safe_cho_solve(
    matrix: NDArray[np.float64],
    rhs: NDArray[np.float64],
    *,
    jitters: Sequence[float] = _DEFAULT_JITTERS,
) -> NDArray[np.float64]:
    """Solve ``matrix @ x = rhs`` via Cholesky, with jitter/lstsq fallback.

    Parameters
    ----------
    matrix:
        Symmetric (regularized) system matrix ``H + lam * B``.
    rhs:
        Right-hand side (upstream passes ``-g``).
    jitters:
        Increasing diagonal perturbations tried before the least-squares fallback.

    Returns
    -------
    numpy.ndarray
        The solution vector. Identical to the upstream ``cho_solve`` result whenever
        the unperturbed Cholesky factorization succeeds.
    """
    try:
        return scipy.linalg.cho_solve(scipy.linalg.cho_factor(matrix, lower=False), rhs)
    except (np.linalg.LinAlgError, ValueError):
        eye = np.eye(matrix.shape[0])
        for jitter in jitters:
            try:
                return scipy.linalg.cho_solve(
                    scipy.linalg.cho_factor(matrix + jitter * eye, lower=False), rhs
                )
            except (np.linalg.LinAlgError, ValueError):
                continue
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]
