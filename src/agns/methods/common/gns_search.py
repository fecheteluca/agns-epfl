"""The single shared GNS adaptive search loop.

Upstream ``epfml/grad-norm-smooth`` (Apache-2.0, commit 9f4ca00) **duplicates** this exact
inner loop inside ``grad_norm_smooth`` and ``grad_norm_smooth_for_rank_one``. We factor it
out **once** here and reuse it from ``gns``, ``gns_rank_one``, ``agns`` and ``agns_rank_one``.

The control flow reproduces upstream byte-for-byte on well-conditioned problems:

* ``lambda_k = ||g_base||_* / (gamma_k + eps)``;
* the regularized step is delegated to the caller's ``solve(lambda_k)`` (Cholesky or
  Woodbury), which on a well-conditioned system returns the same vector upstream would;
* GNS acceptance test ``f_base - f_trial >= ||g_trial||_*^2 / (8 lambda_k)``;
* on accept, ``gamma_k`` doubles with floor ``gamma_min``; on reject it halves (no floor);
* the exhaustion branch is gated on ``warnings`` exactly as upstream (when ``warnings`` is
  ``False`` the final ``i == max_iter`` iteration still performs a solve).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from agns.methods.common.norms import DualNormSqr
from agns.oracles.base import OracleCallsCounter

GnsSearchResult = tuple[
    NDArray[np.float64], float, NDArray[np.float64], float, float, int
]


def gns_adaptive_search(
    *,
    counter: OracleCallsCounter,
    base_x: NDArray[np.float64],
    f_base: float,
    g_base: NDArray[np.float64],
    g_base_norm: float,
    dual_norm_sqr: DualNormSqr,
    solve: Callable[[float], NDArray[np.float64]],
    gamma_k: float,
    gamma_min: float,
    adaptive_search: bool,
    max_iter: int,
    matrix_inverses: int,
    eps: float = 1e-8,
    warnings: bool = False,
    k: int = 0,
    keep_last_trial_on_exhaust: bool = True,
) -> GnsSearchResult:
    """Run the gradient-normalized adaptive search from ``base_x``.

    Parameters
    ----------
    counter:
        Oracle wrapper used to evaluate trial points (keeps call counts consistent).
    base_x, f_base, g_base, g_base_norm:
        The point the search steps from and its cached value/gradient/dual-norm.
    dual_norm_sqr:
        Squared dual-norm functional (from :func:`agns.methods.common.norms.build_dual_norm`).
    solve:
        Callback mapping ``lambda_k`` to the regularized step ``delta_x``.
    gamma_k, gamma_min:
        Current regularization parameter and its floor (applied on accept only).
    adaptive_search:
        If ``False``, take a single step without the acceptance loop.
    max_iter:
        Maximum inner iterations (upstream uses 40).
    matrix_inverses:
        Running solve count, threaded in and returned incremented.
    eps:
        Stabilizer in ``lambda_k = ||g_base||_* / (gamma_k + eps)``.
    warnings:
        If ``True``, print and break at the exhaustion iteration (upstream behavior).
    k:
        Outer iteration index (used only in the warning message).
    keep_last_trial_on_exhaust:
        If the loop exhausts without accepting, keep the last trial (``True``, upstream
        behavior) or discard it and return ``base_x`` unchanged (``False``).

    Returns
    -------
    tuple
        ``(x_trial, f_trial, g_trial, g_trial_norm_sqr, gamma_k, matrix_inverses)``.
    """
    x_trial: NDArray[np.float64] = base_x
    f_trial: Any = f_base
    g_trial: NDArray[np.float64] = g_base
    g_trial_norm_sqr: Any = g_base_norm**2
    stepped = False
    accepted = False

    for i in range(max_iter + 1):
        if i == max_iter and warnings:
            print(f"W: adaptive_iterations_exceeded, k = {k}", flush=True)
            break

        lambda_k = g_base_norm / (gamma_k + eps)
        delta_x = solve(lambda_k)
        matrix_inverses += 1

        x_trial = base_x + delta_x
        f_trial = counter.func(x_trial)
        g_trial = counter.grad(x_trial)
        g_trial_norm_sqr = dual_norm_sqr(g_trial)
        stepped = True

        if not adaptive_search:
            accepted = True
            break

        if f_base - f_trial >= g_trial_norm_sqr / (8.0 * lambda_k):
            gamma_k = max(gamma_k * 2.0, gamma_min)
            accepted = True
            break
        gamma_k *= 0.5

    if stepped and not accepted and not keep_last_trial_on_exhaust:
        # Exhausted without acceptance and the caller wants no move taken.
        x_trial = base_x
        f_trial = f_base
        g_trial = g_base
        g_trial_norm_sqr = g_base_norm**2

    return x_trial, f_trial, g_trial, g_trial_norm_sqr, gamma_k, matrix_inverses
