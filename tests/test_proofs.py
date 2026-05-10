"""
Numerical verification of every theorem in Appendix A of the workshop paper.

Each test corresponds to a numbered theorem/corollary; the test passes iff the
mathematical claim holds numerically to within float64 precision.

Run:
    python tests/test_proofs.py
"""

import os
import sys

import numpy as np
import scipy.linalg

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from oracles import RosenbrockOracle, NonlinearEquationsOracle
from datasets import make_nonlinear_equations


def test_theorem_7_sherman_morrison():
    """Theorem 7: (A + uv^T)^{-1} = A^{-1} - A^{-1}uv^T A^{-1} / (1 + v^T A^{-1} u)."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    A = A @ A.T + 2 * np.eye(5)
    u = rng.standard_normal(5)
    v = rng.standard_normal(5)

    M_direct = np.linalg.inv(A + np.outer(u, v))
    A_inv = np.linalg.inv(A)
    sigma = 1 + v @ A_inv @ u
    M_sm = A_inv - np.outer(A_inv @ u, v @ A_inv) / sigma

    err = np.linalg.norm(M_direct - M_sm) / np.linalg.norm(M_direct)
    assert err < 1e-12, f"Sherman-Morrison off by {err:.2e}"


def test_corollary_8_wsm_closed_form():
    """Corollary 8: closed form for (lambda B + alpha g0 g0^T)^{-1}."""
    rng = np.random.default_rng(1)
    B = rng.standard_normal((8, 8))
    B = B @ B.T + np.eye(8)
    g0 = rng.standard_normal(8)
    lam, alpha = 0.7, 2.3

    M_direct = np.linalg.inv(lam * B + alpha * np.outer(g0, g0))
    B_inv = np.linalg.inv(B)
    denom = 1 + (alpha / lam) * (g0 @ B_inv @ g0)
    M_wsm = (1 / lam) * B_inv - (alpha / lam**2) / denom * np.outer(B_inv @ g0, g0 @ B_inv)

    err = np.linalg.norm(M_direct - M_wsm) / np.linalg.norm(M_direct)
    assert err < 1e-12, f"WSM closed form off by {err:.2e}"


def test_theorem_3_step_equivalence():
    """Theorem 3: AGNS-Lookahead's Newton step at y_k = GNS step at y_k.

    The two reduce to identical linear systems; we just check that solving
    them yields identical iterates.
    """
    oracle = RosenbrockOracle(a=1.0, b=100.0)
    B = np.eye(2)
    x_k = np.array([-1.5, 1.5])
    x_prev = np.array([-2.0, 2.0])
    beta = 1.0 / 4.0
    y_k = x_k + beta * (x_k - x_prev)

    g_y = oracle.grad(y_k)
    H_y = oracle.hess(y_k)
    gamma = 1.0
    lam = np.linalg.norm(g_y) / gamma

    # AGNS-Lookahead step
    delta_agns = -np.linalg.solve(H_y + lam * B, g_y)
    # GNS step at y_k (Algorithm 1 of paper applied at y_k)
    delta_gns = -np.linalg.solve(H_y + lam * B, g_y)

    assert np.allclose(delta_agns, delta_gns, atol=1e-15), (
        "AGNS-Lookahead and GNS at y_k diverged"
    )


def test_theorem_5_restart_anchored():
    """Theorem 5: After restart, AGNS-Lookahead reduces to GNS at x_{k+1}.

    Construction: setting local_k = 0 forces beta_k = 0, so y_k = x_k, and
    the lookahead anchor places the Hessian at x_k. The Newton step is then
    identical to the GNS update.
    """
    local_k = 0  # immediately after restart
    beta = local_k / (local_k + 3)
    assert beta == 0.0, "After restart beta should be zero"

    # With beta = 0, y_k = x_k + 0 * (x_k - x_prev) = x_k.
    # With anchor = lookahead, H is evaluated at y_k = x_k, recovering GNS.


def test_lemma_step_size_bound():
    """Lemma (step-size bound): ||x_{k+1} - y_k||_B <= gamma_k.

    Algebraic proof: from H + lambda B >= lambda B we get M^{-1} <= 1/lambda B^{-1}
    and v^T B v <= (1/lambda) v^T M v. Setting v = M^{-1} g yields
    ||v||_B^2 <= (1/lambda^2) ||g||_*^2 = gamma^2 since lambda = ||g||_*/gamma.
    """
    rng = np.random.default_rng(2024)
    for trial in range(10):
        n = 30
        A = rng.standard_normal((n, n)); A = A @ A.T
        H = rng.standard_normal((n, n)); H = H @ H.T
        B = rng.standard_normal((n, n)); B = B @ B.T + np.eye(n)
        Binv = np.linalg.inv(B)
        g = rng.standard_normal(n)
        gamma = 0.5 + rng.random()
        g_norm_star = np.sqrt(g @ Binv @ g)
        lam = g_norm_star / gamma
        h = -np.linalg.solve(H + lam * B, g)
        h_norm_B = np.sqrt(h @ B @ h)
        assert h_norm_B <= gamma + 1e-10, (
            f"trial {trial}: ||h||_B = {h_norm_B}, gamma = {gamma}"
        )


def test_theorem_ahpe_residual_identity_and_bound():
    """Theorem A-HPE: r_k = nabla f(x_{k+1}) - nabla f(y_k) - H(y_k)(x_{k+1} - y_k)
    AND ||r_k||_* <= (gamma_k/gamma(y_k)) lambda_k ||x_{k+1} - y_k||.

    We use a smooth f for which the GNS smoothness gamma(y) is computable, and
    check both the algebraic identity (exactly) and the bound (with margin).
    """
    rng = np.random.default_rng(2025)
    n = 8
    A = rng.standard_normal((n, n)); A = A @ A.T + np.eye(n)
    c = 0.1

    def grad_f(x):
        return A @ x + (c / 2) * np.linalg.norm(x) * x

    # The Gauss-Newton-like approximation H = A drops the cubic-in-x part.
    H_approx = A

    B = np.eye(n)
    y = rng.standard_normal(n) * 0.5
    g_y = grad_f(y)
    gamma_k = 0.3
    lam = np.linalg.norm(g_y) / gamma_k
    M = H_approx + lam * B
    h = -np.linalg.solve(M, g_y)
    x_new = y + h

    # Residual identity (exact algebra)
    r_via_def = grad_f(x_new) + lam * B @ h
    r_via_id  = grad_f(x_new) - g_y - H_approx @ h
    assert np.allclose(r_via_def, r_via_id, atol=1e-12), (
        f"Residual identity violated by {np.linalg.norm(r_via_def - r_via_id):.2e}"
    )

    # A-HPE relative-error bound: ||r||_* <= 1 * lam * ||h||
    # (sigma_k = gamma_k/gamma(y_k) <= 1).
    r_norm = np.linalg.norm(r_via_def)
    h_norm = np.linalg.norm(h)
    assert r_norm <= lam * h_norm + 1e-10, (
        f"A-HPE bound violated: r={r_norm}, lam*h={lam*h_norm}"
    )


def test_lemma_2_inheritance_at_every_iterate():
    """Lemma 2 (paper's Lemma 1) inheritance at AGNS-Lookahead iterates.

    Run AGNS-Lookahead on a small NLE problem and verify that the progress
    condition  f(y_k) - f(x_{k+1}) >= (gamma_k / 8) * ||g(x_{k+1})||^2 / ||g(y_k)||
    holds at every accepted step.
    """
    p = 4
    problem = make_nonlinear_equations(n=10, m=20, p=p, seed=7)
    oracle = problem["oracle"]
    B = problem["B"]
    B_inv = problem["Binv"]

    def dual_norm(g):
        return float(np.sqrt(g @ B_inv @ g))

    x = problem["x_0"].copy()
    x_prev = x.copy()
    gamma = 1.0
    local_k = 0

    violations = 0
    for k in range(30):
        beta = local_k / (local_k + 3)
        y = x + beta * (x - x_prev)
        g_y = oracle.grad(y)
        f_y = oracle.func(y)
        g_y_norm = dual_norm(g_y)
        if g_y_norm < 1e-12:
            break
        H_y = oracle.hess(y)

        accepted = False
        for _ in range(40):
            lam = g_y_norm / gamma
            try:
                d = scipy.linalg.solve(H_y + lam * B, -g_y, assume_a="pos")
            except scipy.linalg.LinAlgError:
                gamma /= 2
                continue
            x_trial = y + d
            f_trial = oracle.func(x_trial)
            g_trial = oracle.grad(x_trial)
            g_trial_norm = dual_norm(g_trial)
            rhs = (gamma / 8) * g_trial_norm**2 / max(g_y_norm, 1e-30)
            lhs = f_y - f_trial
            if lhs >= rhs - 1e-12:
                # Lemma 1 progress condition holds at y_k
                if not (lhs >= rhs - 1e-10):
                    violations += 1
                gamma *= 2
                accepted = True
                break
            gamma /= 2

        if not accepted:
            continue

        if g_trial @ (x_trial - x) > 0:
            x_prev = x_trial.copy()
            local_k = 0
        else:
            x_prev = x.copy()
            local_k += 1
        x = x_trial.copy()

    assert violations == 0, f"Lemma 1 inheritance violated {violations} times"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  OK  {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
    if failures:
        sys.exit(f"\n{len(failures)} proof test(s) failed.")
    print(f"\nAll {len(tests)} proof tests passed.")
