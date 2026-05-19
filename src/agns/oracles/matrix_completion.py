r"""Low-rank matrix completion (Burer-Monteiro factorisation).

Canonical objective: given a partially observed matrix ``M`` with
observations at indices ``Omega`` and target rank ``r``, factor
``M ~= U V^T`` with ``U \in R^{n_1 \times r}``, ``V \in R^{n_2 \times r}``,
and minimise

.. math::

    f(U, V) \;=\;
      \frac{1}{2|\Omega|} \sum_{(i, j) \in \Omega}
        \bigl( (U V^{\top})_{ij} - M_{ij} \bigr)^{2}
      \;+\; \frac{\lambda}{2}\,
        \bigl(\|U\|_{F}^{2} + \|V\|_{F}^{2}\bigr).

The objective is non-convex (the ``U V`` product is bilinear), so this
oracle is one of the project's non-convex stress tests alongside
Rosenbrock and Chebyshev.

The oracle operates on a *flattened* parameter vector
``w = [U.flatten(), V.flatten()]`` of size ``(n_1 + n_2) * r``, so it
fits the :class:`BaseSmoothOracle` interface.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from agns.oracles.base import BaseSmoothOracle


class MatrixCompletionOracle(BaseSmoothOracle):
    """Burer-Monteiro matrix completion (non-convex)."""

    def __init__(
        self,
        n1: int,
        n2: int,
        rank: int,
        obs_rows: NDArray[np.int64],
        obs_cols: NDArray[np.int64],
        obs_vals: NDArray[np.float64],
        reg: float = 1e-3,
    ) -> None:
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if reg < 0.0:
            raise ValueError(f"reg must be non-negative, got {reg}")
        self.n1, self.n2 = n1, n2
        self.r = rank
        self.obs_rows = np.asarray(obs_rows, dtype=np.int64).ravel()
        self.obs_cols = np.asarray(obs_cols, dtype=np.int64).ravel()
        self.obs_vals = np.asarray(obs_vals, dtype=float).ravel()
        if not (self.obs_rows.shape == self.obs_cols.shape == self.obs_vals.shape):
            raise ValueError("obs_rows, obs_cols, obs_vals must have the same shape")
        self.n_obs = self.obs_rows.shape[0]
        self.reg = reg

    def _unpack(self, w: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        U_size = self.n1 * self.r
        U = w[:U_size].reshape(self.n1, self.r)
        V = w[U_size:].reshape(self.n2, self.r)
        return U, V

    def _residuals(self, U: NDArray[np.float64], V: NDArray[np.float64]) -> NDArray[np.float64]:
        # res_k = (U V^T)_{i_k j_k} - M_{i_k j_k}
        pred = np.sum(U[self.obs_rows] * V[self.obs_cols], axis=1)
        return cast("NDArray[np.float64]", pred - self.obs_vals)

    def func(self, w: NDArray[np.float64]) -> float:
        U, V = self._unpack(w)
        r = self._residuals(U, V)
        data_term = 0.5 * float(r.dot(r)) / self.n_obs
        reg_term = 0.5 * self.reg * (float((U * U).sum()) + float((V * V).sum()))
        return float(data_term + reg_term)

    def grad(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        U, V = self._unpack(w)
        r = self._residuals(U, V)  # (n_obs,)

        # dL/dU_{i,:} = (1/n_obs) sum_{k: i_k = i} r_k * V_{j_k,:}
        gU = np.zeros((self.n1, self.r))
        np.add.at(gU, self.obs_rows, (r[:, None] * V[self.obs_cols]) / self.n_obs)
        gU += self.reg * U

        gV = np.zeros((self.n2, self.r))
        np.add.at(gV, self.obs_cols, (r[:, None] * U[self.obs_rows]) / self.n_obs)
        gV += self.reg * V

        return np.concatenate([gU.flatten(), gV.flatten()])

    def hess_vec(self, w: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Gauss-Newton + bilinear Hessian-vector product."""
        U, V = self._unpack(w)
        dU, dV = self._unpack(v)
        r = self._residuals(U, V)

        # First-order chain-rule terms (bilinear):
        # d/dU (UV^T - M)_{i j} = e_i V_{j,:}^T  -> action on dU: sum over obs
        # Gauss-Newton term:
        #   GN_U[i,:] = (1/n_obs) sum_k V_{j_k} (V_{j_k}^T dU_{i_k})    when i = i_k
        # Bilinear correction:
        #   B_U[i,:] = (1/n_obs) sum_{k: i_k = i} r_k * dV_{j_k}
        pred_diff = np.sum(
            dU[self.obs_rows] * V[self.obs_cols] + U[self.obs_rows] * dV[self.obs_cols],
            axis=1,
        )  # (n_obs,)

        # H v components:
        Hv_U = np.zeros((self.n1, self.r))
        np.add.at(
            Hv_U,
            self.obs_rows,
            (pred_diff[:, None] * V[self.obs_cols]) / self.n_obs,
        )
        np.add.at(
            Hv_U,
            self.obs_rows,
            (r[:, None] * dV[self.obs_cols]) / self.n_obs,
        )
        Hv_U += self.reg * dU

        Hv_V = np.zeros((self.n2, self.r))
        np.add.at(
            Hv_V,
            self.obs_cols,
            (pred_diff[:, None] * U[self.obs_rows]) / self.n_obs,
        )
        np.add.at(
            Hv_V,
            self.obs_cols,
            (r[:, None] * dU[self.obs_rows]) / self.n_obs,
        )
        Hv_V += self.reg * dV

        return np.concatenate([Hv_U.flatten(), Hv_V.flatten()])

    def hess(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        """Materialise the full Hessian via column-by-column ``hess_vec``.

        ``O((n1+n2)*r)`` matrix-free probes, each ``O(n_obs * r)`` time.
        Only practical for small Burer-Monteiro instances (rank-1
        Movielens prototypes); use :meth:`hess_vec` directly otherwise.
        """
        d = (self.n1 + self.n2) * self.r
        H = np.zeros((d, d))
        for k in range(d):
            ek = np.zeros(d)
            ek[k] = 1.0
            H[:, k] = self.hess_vec(w, ek)
        return cast("NDArray[np.float64]", 0.5 * (H + H.T))


def make_matrix_completion_synthetic(
    n1: int = 30,
    n2: int = 30,
    rank: int = 3,
    sampling_rate: float = 0.5,
    noise_std: float = 0.0,
    reg: float = 1e-3,
    seed: int = 0,
) -> dict[str, Any]:
    """Random low-rank matrix + uniform observation mask."""
    rng = np.random.default_rng(seed)
    U_true = rng.standard_normal((n1, rank))
    V_true = rng.standard_normal((n2, rank))
    M = U_true @ V_true.T
    if noise_std > 0:
        M = M + noise_std * rng.standard_normal((n1, n2))

    n_obs = round(sampling_rate * n1 * n2)
    all_idx = np.arange(n1 * n2)
    obs_idx = rng.choice(all_idx, size=n_obs, replace=False)
    obs_rows = obs_idx // n2
    obs_cols = obs_idx % n2
    obs_vals = M[obs_rows, obs_cols]

    oracle = MatrixCompletionOracle(n1, n2, rank, obs_rows, obs_cols, obs_vals, reg=reg)
    d = (n1 + n2) * rank
    x_0 = rng.standard_normal(d) * 0.1
    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": None,
        "B": None,
        "Binv": None,
        "approx_hess_fn": None,
        "approx_hess_fn_wsm": None,
        "meta": {
            "type": "matrix_completion_synthetic",
            "n1": n1,
            "n2": n2,
            "rank": rank,
            "sampling_rate": sampling_rate,
            "n_obs": n_obs,
            "reg": reg,
            "seed": seed,
        },
    }


__all__ = ["MatrixCompletionOracle", "make_matrix_completion_synthetic"]
