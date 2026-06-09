"""Frozen, self-contained reference implementation of AGNS (exact Hessian).

This module is the *golden* specification of the AGNS algorithm. It is deliberately
independent of ``agns.methods``: it re-implements the dual norm, the gradient-normalized
adaptive search, the robust Cholesky solve, and the O'Donoghue--Candes restart inline,
depending only on NumPy/SciPy. ``tests/test_agns_fidelity.py`` runs the production
``agns.methods.agns.agns`` against this reference and requires them to agree to ``1e-10``,
including identical restart events.

DO NOT refactor this file to call into the library, and do not "tidy" it when ``agns.py``
is refactored: its whole purpose is to drift-detect changes in the production method. It is
excluded from ruff/black in ``pyproject.toml`` for exactly this reason.

Behavior mirrors ``agns.methods.agns.agns`` at commit-of-record:
* momentum ``beta_k = local_k / (local_k + momentum_offset)``, ``y_k = x_k + beta_k (x_k - x_prev)``;
* GNS step from ``y_k``: ``lambda_k = ||g_y||_* / (gamma_k + eps)``, accept on
  ``f_y - f_trial >= ||g_trial||_*^2 / (8 lambda_k)``, ``gamma`` doubles (floored) on accept
  and halves on reject; the last trial is kept if the search exhausts;
* restart schemes "gradient" / "function_value" / "none".
"""

import numpy as np
import scipy.linalg

_JITTERS = (1e-12, 1e-10, 1e-8, 1e-6)


def _safe_cho_solve(matrix, rhs):
    """Cholesky solve with jitter/lstsq fallback (matches agns.methods.common.linalg)."""
    try:
        return scipy.linalg.cho_solve(scipy.linalg.cho_factor(matrix, lower=False), rhs)
    except (np.linalg.LinAlgError, ValueError):
        eye = np.eye(matrix.shape[0])
        for jitter in _JITTERS:
            try:
                return scipy.linalg.cho_solve(
                    scipy.linalg.cho_factor(matrix + jitter * eye, lower=False), rhs
                )
            except (np.linalg.LinAlgError, ValueError):
                continue
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def reference_agns(
    oracle,
    x_0,
    n_iters=1000,
    gamma_0=1.0,
    gamma_min=1e-9,
    adaptive_search=True,
    B=None,
    Binv=None,
    eps=1e-8,
    f_star=None,
    grad_tol=None,
    momentum_offset=3.0,
    restart_mode="gradient",
    adaptive_search_max_iter=40,
):
    """Reference AGNS run; returns ``(x_k, status, history)`` like the library method."""
    if restart_mode not in {"gradient", "function_value", "none"}:
        raise ValueError(f"Unknown restart_mode {restart_mode!r}")

    n = x_0.shape[0]
    if B is None:
        B_eff = np.eye(n)

        def dual_norm_sqr(v):
            return v.dot(v)

    else:
        B_eff = B
        Binv_eff = Binv if Binv is not None else np.linalg.inv(B)

        def dual_norm_sqr(v):
            return Binv_eff.dot(v).dot(v)

    # Count calls exactly as OracleCallsCounter does, but we only need grad_calls here.
    grad_calls = 0

    def f(v):
        return oracle.func(v)

    def g(v):
        nonlocal grad_calls
        grad_calls += 1
        return oracle.grad(v)

    x_k = np.copy(x_0)
    x_prev = np.copy(x_0)
    f_k = f(x_k)
    g_k = g(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    gamma_k = gamma_0
    local_k = 0
    n_restarts = 0
    restart_events = []

    history = {
        "func": [],
        "grad_norm": [],
        "gamma_k": [],
        "grad_calls": [],
        "matrix_inverses": [],
        "x_k": [],
        "local_k": [],
        "n_restarts_so_far": [],
    }
    matrix_inverses = 0
    status = ""

    for k in range(n_iters + 1):
        history["func"].append(f_k)
        history["grad_norm"].append(g_k_norm)
        history["gamma_k"].append(gamma_k)
        history["grad_calls"].append(grad_calls)
        history["matrix_inverses"].append(matrix_inverses)
        history["x_k"].append(x_k.copy())
        history["local_k"].append(local_k)
        history["n_restarts_so_far"].append(n_restarts)

        if (f_star is not None and f_k - f_star < eps) or (
            grad_tol is not None and g_k_norm < grad_tol
        ):
            status = f"success, {k} iters"
            break
        if k == n_iters:
            status = "iterations_exceeded"
            break

        beta_k = local_k / (local_k + momentum_offset)
        y_k = x_k + beta_k * (x_k - x_prev)

        g_y = g(y_k)
        g_y_norm = dual_norm_sqr(g_y) ** 0.5
        f_y = f(y_k)

        if g_y_norm < eps:
            if f_y <= f_k:
                x_prev = x_k.copy()
                x_k, f_k, g_k = y_k.copy(), f_y, g_y.copy()
                g_k_norm = g_y_norm
            else:
                x_prev = x_k.copy()
            local_k = 0
            continue

        Hess_k = oracle.hess(y_k)

        # ---- inline gradient-normalized adaptive search from y_k ----
        x_trial = y_k
        f_trial = f_y
        g_trial = g_y
        g_trial_norm_sqr = g_y_norm**2
        for i in range(adaptive_search_max_iter + 1):
            lambda_k = g_y_norm / (gamma_k + eps)
            delta_x = _safe_cho_solve(Hess_k + lambda_k * B_eff, -g_y)
            matrix_inverses += 1
            x_trial = y_k + delta_x
            f_trial = f(x_trial)
            g_trial = g(x_trial)
            g_trial_norm_sqr = dual_norm_sqr(g_trial)
            if not adaptive_search:
                break
            if f_y - f_trial >= g_trial_norm_sqr / (8.0 * lambda_k):
                gamma_k = max(gamma_k * 2.0, gamma_min)
                break
            gamma_k *= 0.5

        # ---- restart decision ----
        if restart_mode == "gradient":
            restart = float(g_trial.dot(x_trial - x_k)) > 0.0
        elif restart_mode == "function_value":
            restart = f_trial > f_k
        else:
            restart = False

        if restart:
            x_prev = x_trial.copy()
            local_k = 0
            n_restarts += 1
            restart_events.append(k)
        else:
            x_prev = x_k.copy()
            local_k += 1

        x_k = x_trial.copy()
        f_k = f_trial
        g_k = g_trial
        g_k_norm = g_trial_norm_sqr**0.5

        if not np.isfinite(f_k) or not np.isfinite(g_k_norm):
            status = "diverged_nonfinite"
            break

    history["n_restarts_total"] = [n_restarts]
    history["restart_events"] = restart_events
    history["restart_mode"] = [restart_mode]
    return x_k, status, history
