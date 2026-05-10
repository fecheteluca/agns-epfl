"""
Numerical correctness test for the Sherman-Morrison / Woodbury rank-1
inversion path used by `grad_norm_smooth_for_rank_one` and
`accelerated_grad_norm_smooth_for_rank_one`.

The system being solved is
    (lambda * B + alpha * g0 g0^T) * delta = -g
where alpha = (p-2) ‖u‖^(p-4) and g0 = J^T u.  The WSM path replaces the
explicit Cholesky on the dense matrix with two B^{-1} solves and a scalar
correction.  This test checks that both paths give the same delta to within
the conditioning floor of B.

Run:
    python tests/test_wsm_identity.py
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from oracles import NonlinearEquationsOracle


def _wsm_solve(g, lambda_k, B, Binv, alpha, g0):
    """Closed-form Sherman-Morrison inverse, mirrors src/methods.py."""
    inv_scale = 1.0 / lambda_k
    Pinv_gk = inv_scale * Binv.dot(-g)
    Pinv_g0 = inv_scale * Binv.dot(g0)
    denom = 1.0 + alpha * (g0.dot(Pinv_g0))
    return Pinv_gk - (alpha / denom) * Pinv_g0 * (g0.dot(Pinv_gk))


def test_wsm_matches_direct_solve():
    rng = np.random.default_rng(123)
    n, m = 30, 50
    A = rng.uniform(-1.0, 1.0, (n, m))
    b = rng.uniform(-1.0, 1.0, n)
    eps_reg = 1e-8
    B = A.T.dot(A) + eps_reg * np.eye(m)
    Binv = np.linalg.inv(B)

    for p in (2, 3, 4, 5):
        oracle = NonlinearEquationsOracle(p, A=A, b=b)
        x = rng.standard_normal(m)

        g = oracle.grad(x)
        g_norm = np.sqrt(Binv.dot(g).dot(g))
        gamma = 0.5
        lambda_k = g_norm / gamma

        u = oracle.func_u(x)
        norm_u = np.linalg.norm(u)
        J = oracle.jac_u(x)
        g0 = J.T.dot(u)
        alpha = (p - 2) * norm_u ** (p - 4) if p != 2 else 0.0

        H_approx = alpha * np.outer(g0, g0)
        delta_direct = np.linalg.solve(H_approx + lambda_k * B, -g)
        delta_wsm = _wsm_solve(g, lambda_k, B, Binv, alpha, g0)

        rel = np.linalg.norm(delta_direct - delta_wsm) / np.linalg.norm(delta_direct)
        assert rel < 1e-4, f"p={p}: WSM relerr {rel:.2e} too large"


if __name__ == "__main__":
    test_wsm_matches_direct_solve()
    print("WSM identity test passed.")
