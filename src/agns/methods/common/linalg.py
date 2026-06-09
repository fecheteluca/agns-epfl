"""Robust Cholesky solve for the regularized Newton system.

Upstream (``epfml/grad-norm-smooth``, Apache-2.0, commit 9f4ca00) solves the
regularized Newton step with::

    scipy.linalg.cho_solve(scipy.linalg.cho_factor(H + lam * B, lower=False), -g)

``safe_cho_solve`` wraps exactly that call. On a well-conditioned matrix the first
attempt succeeds and the result is *bit-identical* to upstream. Only when the factor
fails does it fall back to a small diagonal jitter and finally a least-squares solve,
so the refactored methods never abort mid-search on a transiently indefinite matrix.
"""

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
