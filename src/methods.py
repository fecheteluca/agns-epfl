import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import scipy

from oracles import OracleCallsCounter


def grad_norm_smooth(
    oracle,
    x_0,
    n_iters=1000,
    gamma_0=1.0,
    adaptive_search=True,
    is_approx=False,
    approx_oracle=None,
    approx_hess_fn=None,
    trace=True,
    B=None,
    Binv=None,
    eps=1e-8,
    gamma_min=1e-9,
    f_star=None,
    grad_tol=None,
    warnings=False,
):
    """
    Run Gradient-Regularized Newton Method
    Algorithm 1 from the paper
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    # Dual norm initialization
    if B is None:
        B = np.eye(x_0.shape[0])
        dual_norm_sqr = lambda x: x.dot(x)
    else:
        if Binv is None:
            Binv = np.linalg.inv(B)
        dual_norm_sqr = lambda x: Binv.dot(x).dot(x)

    # Initialization
    x_k = np.copy(x_0)
    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    gamma_k = gamma_0

    history = defaultdict(list) if trace else None
    matrix_inverses = 0
    status = ""

    # Main optimization loop
    for k in range(n_iters + 1):
        if trace:
            history["func"].append(f_k)
            history["grad_norm"].append(g_k_norm)
            history["gamma_k"].append(gamma_k)
            history["grad_calls"].append(oracle.grad_calls)
            history["matrix_inverses"].append(matrix_inverses)
            history["time"].append((datetime.now() - start_timestamp).total_seconds())
            history["x_k"].append(x_k.copy())

        if (f_star is not None and f_k - f_star < eps) or (
            grad_tol is not None and g_k_norm < grad_tol
        ):
            status = f"success, {k} iters"
            break

        if k == n_iters:
            status = "iterations_exceeded"
            break

        # Choose Hessian
        if is_approx and approx_oracle is not None and callable(approx_hess_fn):
            Hess_k = approx_hess_fn(approx_oracle, x_k)
        else:
            Hess_k = oracle.hess(x_k)

        # Safe fallback if the inner loop exhausts its budget: stay at x_k.
        x_new = x_k.copy()
        f_new = f_k
        g_new = g_k
        g_new_norm_sqr = g_k_norm * g_k_norm

        # Adaptive search for gamma_k
        adaptive_search_max_iter = 40
        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter:
                if warnings:
                    print(f"W: adaptive_iterations_exceeded, k = {k}", flush=True)
                break

            lambda_k = g_k_norm / gamma_k
            try:
                delta_x = scipy.linalg.cho_solve(
                    scipy.linalg.cho_factor(Hess_k + lambda_k * B, lower=False), -g_k
                )
                matrix_inverses += 1
            except (np.linalg.LinAlgError, ValueError):
                if warnings:
                    print("W: linalg_error -> tightening gamma", flush=True)
                # Indefinite matrix: tighten regularization (smaller gamma → larger
                # lambda) and retry within the same inner loop. Mathematically this
                # is the same path as a rejected step.
                gamma_k = max(gamma_k * 0.5, gamma_min)
                continue

            x_trial = x_k + delta_x
            f_trial = oracle.func(x_trial)
            g_trial = oracle.grad(x_trial)
            g_trial_norm_sqr = dual_norm_sqr(g_trial)

            if not adaptive_search:
                x_new, f_new, g_new, g_new_norm_sqr = (
                    x_trial, f_trial, g_trial, g_trial_norm_sqr
                )
                break

            # Lemma 1 progress condition
            if f_k - f_trial >= g_trial_norm_sqr / (8.0 * lambda_k):
                x_new, f_new, g_new, g_new_norm_sqr = (
                    x_trial, f_trial, g_trial, g_trial_norm_sqr
                )
                gamma_k *= 2.0
                break
            else:
                gamma_k = max(gamma_k * 0.5, gamma_min)

        x_k = x_new
        f_k = f_new
        g_k = g_new
        g_k_norm = g_new_norm_sqr ** 0.5

    return x_k, status, history


def super_newton(
    oracle,
    x_0,
    n_iters=1000,
    H_0=1.0,
    alpha=1.0,
    adaptive_search=True,
    trace=True,
    B=None,
    Binv=None,
    eps=1e-8,
    H_min=1e-5,
    f_star=None,
    grad_tol=None,
    warnings=False,
):
    """
    Run Super-Universal Newton Method
    for 'n_iters' iterations, minimizing smooth function.

    'oracle' is an instance of BaseSmoothOracle representing the objective.
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    # Initialization of the dual norm
    if B is None:
        B = np.eye(x_0.shape[0])
        dual_norm_sqr = lambda x: x.dot(x)
    else:
        if Binv is None:
            Binv = np.linalg.inv(B)
        dual_norm_sqr = lambda x: Binv.dot(x).dot(x)

    # Initialization of the method
    x_k = np.copy(x_0)
    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    H_k = H_0

    history = defaultdict(list) if trace else None
    matrix_inverses = 0
    status = ""

    # Main loop
    for k in range(n_iters + 1):

        if trace:
            history["func"].append(f_k)
            history["grad_norm"].append(g_k_norm)
            history["H_k"].append(H_k)
            history["grad_calls"].append(oracle.grad_calls)
            history["matrix_inverses"].append(matrix_inverses)
            history["time"].append((datetime.now() - start_timestamp).total_seconds())
            history["x_k"].append(x_k.copy())

        if (f_star is not None and f_k - f_star < eps) or (
            grad_tol is not None and g_k_norm < grad_tol
        ):
            status = "success, %d iters" % k
            break

        if k == n_iters:
            status = "iterations_exceeded"
            break

        Hess_k = oracle.hess(x_k)

        # Safe fallback if the inner loop exits without an accepted step.
        x_new = x_k.copy()
        f_new = f_k
        g_new = g_k
        g_new_norm_sqr = g_k_norm * g_k_norm

        adaptive_search_max_iter = 40
        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter:
                if warnings:
                    print(("W: adaptive_iterations_exceeded, k = %d" % k), flush=True)
                break

            lambda_k = H_k * g_k_norm**alpha
            try:
                delta_x = scipy.linalg.cho_solve(
                    scipy.linalg.cho_factor(Hess_k + lambda_k * B, lower=False), -g_k
                )
                matrix_inverses += 1
            except (np.linalg.LinAlgError, ValueError) as e:
                if warnings:
                    print("W: linalg_error -> tightening H", flush=True)
                # Tighten regularization and retry within the inner loop.
                H_k *= 4
                continue

            x_trial = x_k + delta_x
            f_trial = oracle.func(x_trial)
            g_trial = oracle.grad(x_trial)
            g_trial_norm_sqr = dual_norm_sqr(g_trial)

            if not adaptive_search:
                x_new, f_new, g_new, g_new_norm_sqr = (
                    x_trial, f_trial, g_trial, g_trial_norm_sqr
                )
                break

            # Super-Universal acceptance condition
            if g_trial.dot(-delta_x) >= g_trial_norm_sqr / (4 * lambda_k):
                x_new, f_new, g_new, g_new_norm_sqr = (
                    x_trial, f_trial, g_trial, g_trial_norm_sqr
                )
                H_k = max(H_k * 0.25, H_min)
                break

            H_k *= 4

        x_k = x_new
        f_k = f_new
        g_k = g_new
        g_k_norm = g_new_norm_sqr ** 0.5

    return x_k, status, history


def cubic_newton_step(g, A, H, B=None, eps=1e-8):
    """
    Computes minimizer of the following function:
       f(x) = <g, x> + 1/2 * <Ax, x> + H/3 * ||x||^3,
    by finding the root of the equation:
       h(r) = r - ||(A + HrB)^{-1} g|| = 0.
    """
    n = g.shape[0]
    if B is None:
        B = np.eye(n)
        l2_norm_sqr = lambda x: x.dot(x)
    else:
        l2_norm_sqr = lambda x: B.dot(x).dot(x)

    def f(T, T_norm):
        return g.dot(T) + 0.5 * A.dot(T).dot(T) + H * T_norm**3 / 3.0

    def h(r, der=False):
        ArB_cho_factor = scipy.linalg.cho_factor(A + H * r * B, lower=False)
        T = scipy.linalg.cho_solve(ArB_cho_factor, -g)
        T_norm = l2_norm_sqr(T) ** 0.5
        h_r = r - T_norm
        if der:
            BT = B.dot(T)
            h_r_prime = 1 + H / T_norm * scipy.linalg.cho_solve(ArB_cho_factor, BT).dot(
                BT
            )
        else:
            h_r_prime = None
        return h_r, T_norm, T, h_r_prime

    try:
        max_r = 1.0
        max_iters = 50
        # Find max_r such that h(max_r) is nonnegative
        for i in range(max_iters):
            h_r, T_norm, T, _ = h(max_r)
            if h_r < -eps:
                max_r *= 2
            elif -eps <= h_r <= eps:
                return T, f(T, T_norm), "success"
            else:
                break

        # Univariate Newton's
        r = max_r
        for i in range(max_iters):
            h_r, T_norm, T, h_r_prime = h(r, der=True)
            if -eps <= h_r <= eps:
                return T, f(T, T_norm), "success"
            r -= h_r / h_r_prime

    except (np.linalg.LinAlgError, ValueError) as e:
        return np.zeros(n), 0.0, "linalg_error"

    return np.zeros(n), 0.0, "iterations_exceeded"


def cubic_newton(
    oracle,
    x_0,
    n_iters=1000,
    H_0=1.0,
    adaptive_search=True,
    trace=True,
    B=None,
    eps=1e-8,
    H_min=1e-5,
    f_star=None,
    warnings=False,
):
    """
    Run Cubic Newton Method
    for 'n_iters' iterations, minimizing smooth function.

    'oracle' is an instance of BaseSmoothOracle representing the objective.
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    # Initialization of the method
    x_k = np.copy(x_0)
    f_k = oracle.func(x_k)
    H_k = H_0

    history = defaultdict(list) if trace else None
    status = ""

    # Main loop
    for k in range(n_iters + 1):

        if trace:
            history["func"].append(f_k)
            history["grad_calls"].append(oracle.grad_calls)
            history["H_k"].append(H_k)
            history["time"].append((datetime.now() - start_timestamp).total_seconds())
            history["x_k"].append(x_k.copy())

        if f_star is not None and f_k - f_star < eps:
            status = "success, %d iters" % k
            break

        if k == n_iters:
            status = "iterations_exceeded"
            break

        g_k = oracle.grad(x_k)
        Hess_k = oracle.hess(x_k)

        # Safe fallback: if every cubic_newton_step fails or the model never accepts
        # within the budget, stay at x_k.
        x_new = x_k.copy()
        f_new = f_k

        adaptive_search_max_iter = 40
        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter:
                if warnings:
                    print("W: adaptive_iterations_exceeded", flush=True)
                break

            x_delta, model_value, message = cubic_newton_step(
                g_k, Hess_k, 0.5 * H_k, B, 1e-8
            )
            if message != "success":
                if warnings:
                    print("W: cubic_newton_step: %s" % message, end=" ", flush=True)
                # cannot trust x_delta; tighten regularization and retry
                H_k *= 2
                continue

            f_trial = oracle.func(x_k + x_delta)

            if not adaptive_search:
                x_new = x_k + x_delta
                f_new = f_trial
                break

            if f_trial <= f_k + model_value:
                x_new = x_k + x_delta
                f_new = f_trial
                H_k = max(H_k * 0.5, H_min)
                break

            H_k *= 2

        x_k = x_new
        f_k = f_new

    return x_k, status, history


def gradient_method(
    oracle,
    x_0,
    max_iter=1000,
    L_0=1.0,
    line_search=True,
    trace=True,
    B=None,
    Binv=None,
    L_min=1e-5,
    warnings=False,
    f_star=None,
    eps=1e-8,
    grad_tol=None,
):
    """
    Gradient method with backtracking line search.
    Stopping: f_k - f_star < eps, or ||grad||_* < grad_tol, or k == max_iter.
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    if B is None:
        l2_norm_sqr = lambda x: x.dot(x)
        dual_norm_sqr = lambda x: x.dot(x)
    else:
        l2_norm_sqr = lambda x: B.dot(x).dot(x)
        if Binv is None:
            Binv = np.linalg.inv(B)
        dual_norm_sqr = lambda x: Binv.dot(x).dot(x)
    precond = (lambda g: g) if Binv is None else (lambda g: Binv.dot(g))

    x_k = np.copy(x_0)
    func_k = oracle.func(x_k)
    grad_k = oracle.grad(x_k)
    grad_k_norm_sqr = dual_norm_sqr(grad_k)
    L_k = L_0

    history = defaultdict(list) if trace else None
    status = ""

    for k in range(max_iter + 1):
        if trace:
            history["func"].append(func_k)
            history["grad_sqr_norm"].append(grad_k_norm_sqr)
            history["L"].append(L_k)
            history["func_calls"].append(oracle.func_calls)
            history["grad_calls"].append(oracle.grad_calls)
            history["time"].append((datetime.now() - start_timestamp).total_seconds())
            history["x_k"].append(x_k.copy())

        if (f_star is not None and func_k - f_star < eps) or (
            grad_tol is not None and grad_k_norm_sqr ** 0.5 < grad_tol
        ):
            status = f"success, {k} iters"
            break

        if k == max_iter:
            status = "iterations_exceeded"
            break

        # Safe fallback: if line search fails entirely, stay at x_k.
        # We track 'accepted' so a divergent search (e.g. f(T)=NaN at every
        # probe) can't commit a poisoned iterate.
        accepted = False
        T = x_k.copy()
        func_T = func_k

        line_search_max_iter = 30
        for i in range(line_search_max_iter + 1):
            if i == line_search_max_iter:
                if warnings:
                    print("W: line_search_max_iter_reached", flush=True)
                break

            T_trial = x_k - precond(grad_k) / L_k
            func_trial = oracle.func(T_trial)

            if not np.isfinite(func_trial):
                # Step is too large, function diverged; tighten L and retry.
                L_k *= 2
                continue

            if not line_search:
                T = T_trial
                func_T = func_trial
                accepted = True
                break

            if func_trial <= func_k + grad_k.dot(T_trial - x_k) + 0.5 * L_k * l2_norm_sqr(
                T_trial - x_k
            ):
                T = T_trial
                func_T = func_trial
                L_k = max(L_k * 0.5, L_min)
                accepted = True
                break

            L_k *= 2

        if not accepted:
            status = "line_search_diverged"
            break

        x_k = T
        func_k = func_T
        grad_k = oracle.grad(x_k)
        grad_k_norm_sqr = dual_norm_sqr(grad_k)

    return x_k, status, history


def fast_gradient_method(
    oracle,
    x_0,
    max_iter=1000,
    L_0=1.0,
    line_search=True,
    trace=True,
    B=None,
    Binv=None,
    L_min=1e-5,
    warnings=False,
    f_star=None,
    eps=1e-8,
    grad_tol=None,
):
    """
    Nesterov's Fast Gradient Method (FISTA-style backtracking).
    Stopping: f_k - f_star < eps, or ||grad||_* < grad_tol, or k == max_iter.
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    if B is None:
        l2_norm_sqr = lambda x: x.dot(x)
        dual_norm_sqr = lambda x: x.dot(x)
        precond = lambda g: g
    else:
        l2_norm_sqr = lambda x: B.dot(x).dot(x)
        if Binv is None:
            Binv = np.linalg.inv(B)
        dual_norm_sqr = lambda x: Binv.dot(x).dot(x)
        precond = lambda g: Binv.dot(g)

    x_k = np.copy(x_0)
    v_k = np.copy(x_0)
    func_k = oracle.func(x_k)
    grad_k = oracle.grad(x_k)
    grad_k_norm_sqr = dual_norm_sqr(grad_k)
    L_k = L_0
    A_k = 0.0

    history = defaultdict(list) if trace else None
    status = ""

    for k in range(max_iter + 1):
        if trace:
            history["func"].append(func_k)
            history["grad_sqr_norm"].append(grad_k_norm_sqr)
            history["func_calls"].append(oracle.func_calls)
            history["grad_calls"].append(oracle.grad_calls)
            history["L"].append(L_k)
            history["time"].append((datetime.now() - start_timestamp).total_seconds())
            history["x_k"].append(x_k.copy())

        if (f_star is not None and func_k - f_star < eps) or (
            grad_tol is not None and grad_k_norm_sqr ** 0.5 < grad_tol
        ):
            status = f"success, {k} iters"
            break

        if k == max_iter:
            status = "iterations_exceeded"
            break

        # Safe fallback: keep previous iterate if line search fails entirely.
        # 'accepted' guards against committing a NaN/Inf trial when the search
        # diverges (e.g. high-p Rosenbrock with huge initial gradient norm).
        accepted = False
        T = x_k.copy()
        func_T = func_k
        a_k_new = 1.0 / L_k
        y_k = x_k.copy()
        grad_y_k = grad_k

        line_search_max_iter = 30
        for i in range(line_search_max_iter + 1):
            if i == line_search_max_iter:
                if warnings:
                    print("W: line_search_max_iter_reached", flush=True)
                break

            a_k_trial = (1 + (1 + 4 * A_k * L_k) ** 0.5) / (2 * L_k)
            y_k_trial = (a_k_trial * v_k + A_k * x_k) / (a_k_trial + A_k)
            grad_y_k_trial = oracle.grad(y_k_trial)
            func_y_k_trial = oracle.func(y_k_trial)

            T_trial = y_k_trial - precond(grad_y_k_trial) / L_k
            func_T_trial = oracle.func(T_trial)

            if not (np.isfinite(func_y_k_trial) and np.isfinite(func_T_trial)):
                L_k *= 2
                continue

            if not line_search:
                a_k_new = a_k_trial
                y_k = y_k_trial
                grad_y_k = grad_y_k_trial
                T = T_trial
                func_T = func_T_trial
                accepted = True
                break

            if func_T_trial <= func_y_k_trial + grad_y_k_trial.dot(T_trial - y_k_trial) \
                    + 0.5 * L_k * l2_norm_sqr(T_trial - y_k_trial):
                a_k_new = a_k_trial
                y_k = y_k_trial
                grad_y_k = grad_y_k_trial
                T = T_trial
                func_T = func_T_trial
                L_k = max(L_k * 0.5, L_min)
                accepted = True
                break

            L_k *= 2

        if not accepted:
            status = "line_search_diverged"
            break

        v_k = T + (A_k / a_k_new) * (T - x_k)
        A_k += a_k_new

        x_k = T
        func_k = func_T
        # Re-evaluate gradient at the new iterate so the trace and stopping
        # criterion use the correct value.
        grad_k = oracle.grad(x_k)
        grad_k_norm_sqr = dual_norm_sqr(grad_k)

    return x_k, status, history


def grad_norm_smooth_for_rank_one(
    oracle,
    x_0,
    n_iters=1000,
    gamma_0=1.0,
    adaptive_search=True,
    is_approx=False,
    approx_oracle=None,
    approx_hess_fn=None,
    trace=True,
    B=None,
    Binv=None,
    eps=1e-8,
    gamma_min=1e-9,
    f_star=None,
    grad_tol=None,
    warnings=False,
    invert_backend="cho",
):
    """
    Newton Method with Gradient-Normalized-Smoothness and faster invert for rank-1 approximation
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    if B is None:
        B = np.eye(x_0.shape[0])
        dual_norm_sqr = lambda x: x.dot(x)
        Binv = np.eye(x_0.shape[0])
    else:
        if Binv is None:
            Binv = np.linalg.inv(B)
        dual_norm_sqr = lambda x: Binv.dot(x).dot(x)

    x_k = np.copy(x_0)
    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    g_k_norm = np.sqrt(dual_norm_sqr(g_k))
    gamma_k = gamma_0

    history = defaultdict(list) if trace else None
    matrix_inverses = 0
    status = ""

    for k in range(n_iters + 1):
        if trace:
            history["func"].append(f_k)
            history["grad_norm"].append(g_k_norm)
            history["gamma_k"].append(gamma_k)
            history["grad_calls"].append(oracle.grad_calls)
            history["matrix_inverses"].append(matrix_inverses)
            history["time"].append((datetime.now() - start_timestamp).total_seconds())
            history["x_k"].append(x_k.copy())

        if (f_star is not None and f_k - f_star < eps) or (
            grad_tol is not None and g_k_norm < grad_tol
        ):
            status = f"success, {k} iters"
            break
        if k == n_iters:
            status = "iterations_exceeded"
            break

        # Only the 'cho' backend ever uses the dense Hess_k matrix; the
        # 'wsm' backend reads its rank-1 structure (alpha, g0) directly from
        # the approx_oracle, so building the full matrix is wasteful.
        if invert_backend == "cho":
            if is_approx and approx_oracle is not None and callable(approx_hess_fn):
                Hess_k = approx_hess_fn(approx_oracle, x_k)
            else:
                Hess_k = oracle.hess(x_k)
        else:
            Hess_k = None

        # Safe fallback if the inner loop exits without an accepted step.
        x_new = x_k.copy()
        f_new = f_k
        g_new = g_k
        g_new_norm_sqr = g_k_norm * g_k_norm

        adaptive_search_max_iter = 40
        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter:
                if warnings:
                    print(f"W: adaptive_iterations_exceeded, k = {k}", flush=True)
                break

            lambda_k = g_k_norm / gamma_k
            try:
                if invert_backend == "cho":
                    delta_x = scipy.linalg.cho_solve(
                        scipy.linalg.cho_factor(Hess_k + lambda_k * B, lower=False),
                        -g_k,
                    )
                    matrix_inverses += 1

                elif invert_backend == "wsm":
                    if not (is_approx and callable(approx_hess_fn)):
                        raise ValueError(
                            "wsm backend requires a rank-one approx_hess_fn"
                        )
                    u = approx_oracle.func_u(x_k)
                    norm_u = np.linalg.norm(u)
                    J = approx_oracle.jac_u(x_k)
                    if approx_oracle.p != 2:
                        alpha = (approx_oracle.p - 2) * norm_u ** (approx_oracle.p - 4)
                        g0 = J.T.dot(u)
                    else:
                        alpha = 0.0
                        g0 = np.zeros_like(g_k)
                    inv_scale = 1.0 / lambda_k
                    Pinv_gk = inv_scale * Binv.dot(-g_k)
                    Pinv_g0 = inv_scale * Binv.dot(g0)
                    denom = 1.0 + alpha * (g0.dot(Pinv_g0))
                    delta_x = Pinv_gk - (alpha / denom) * Pinv_g0 * (g0.dot(Pinv_gk))
                    matrix_inverses += 1

                else:
                    raise ValueError(f"Unknown invert_backend '{invert_backend}'")

            except (np.linalg.LinAlgError, ValueError) as e:
                if warnings:
                    print(f"W: linear solve error ({e}) -> tightening gamma", flush=True)
                gamma_k = max(gamma_k * 0.5, gamma_min)
                continue

            x_trial = x_k + delta_x
            f_trial = oracle.func(x_trial)
            g_trial = oracle.grad(x_trial)
            g_trial_norm_sqr = dual_norm_sqr(g_trial)

            if not adaptive_search:
                x_new, f_new, g_new, g_new_norm_sqr = (
                    x_trial, f_trial, g_trial, g_trial_norm_sqr
                )
                break

            if f_k - f_trial >= g_trial_norm_sqr / (8.0 * lambda_k):
                x_new, f_new, g_new, g_new_norm_sqr = (
                    x_trial, f_trial, g_trial, g_trial_norm_sqr
                )
                gamma_k *= 2.0
                break
            else:
                gamma_k = max(gamma_k * 0.5, gamma_min)

        x_k = x_new
        f_k = f_new
        g_k = g_new
        g_k_norm = g_new_norm_sqr ** 0.5

    return x_k, status, history


def accelerated_grad_norm_smooth(
    oracle,
    x_0,
    n_iters=1000,
    gamma_0=1.0,
    adaptive_search=True,
    is_approx=False,
    approx_oracle=None,
    approx_hess_fn=None,
    trace=True,
    B=None,
    Binv=None,
    eps=1e-8,
    gamma_min=1e-9,
    f_star=None,
    grad_tol=None,
    warnings=False,
    hessian_anchor="lookahead"
):
    """
    Accelerated Gradient-Regularized Newton Method (AGNS).

    THEORETICAL STATUS — IMPORTANT:
        AGNS is NOT proved convergent in Semenov, Jaggi, Doikov (2025).  The
        paper's Lemma 1 (eq. 8) only governs Algorithm 1 applied at the same
        point where the Hessian and gradient are evaluated.  Section 7 of the
        paper explicitly states that an accelerated variant of GNS is left
        as future work.

        AGNS-lookahead applies Algorithm 1 exactly at y_k, so Lemma 1 holds
        relative to f(y_k) — but f(y_k) need not lie below f(x_k) since the
        momentum step is purely heuristic descent.  AGNS-iterate has no
        Lemma 1 analogue (Hessian at x_k, gradient at y_k mixes anchors).

        We therefore ONLY claim AGNS as a strong empirical heuristic.  All
        formal complexity guarantees in this codebase belong to GNS, the
        Super-Universal Newton method, and Cubic Newton.

    Per-iteration structure (k >= 0):
      1. Extrapolate: y_k = x_k + beta_k * (x_k - x_{k-1}),
         beta_k = local_k / (local_k + 3).  local_k counts iterations since
         the last gradient restart and resets to 0 on restart, so beta_k = 0
         on the iteration immediately following a restart.
      2. Evaluate gradient at y_k.  Hessian anchor is selectable:
            'lookahead' (default): H at y_k.  Lemma 1 holds w.r.t. f(y_k).
            'iterate': H at x_k.  Ablation only — no formal grounding.
      3. Solve (H + lambda_k * B) * delta = -grad(y_k),  with
         lambda_k = ||grad(y_k)||_* / gamma_k.  Adaptive search on gamma_k
         enforces  f(y_k) - f(x_trial) >= g_trial_sq / (8 * lambda_k)
         (the Lemma-1 progress condition transplanted to y_k).
         Accept => gamma_k *= 2.  Reject => gamma_k = max(gamma_k/2, gamma_min).
      4. Gradient restart (O'Donoghue & Candes, 2015):
         if grad(x_trial) . (x_trial - x_k) > 0  the momentum was adversarial,
         so we set x_prev = x_trial and zero local_k (beta -> 0 next step).
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    # Dual norm initialization
    if B is None:
        B = np.eye(x_0.shape[0])
        dual_norm_sqr = lambda x: x.dot(x)
    else:
        if Binv is None:
            Binv = np.linalg.inv(B)
        dual_norm_sqr = lambda x: Binv.dot(x).dot(x)

    # Initialization
    x_k = np.copy(x_0)
    x_prev = np.copy(x_0)
    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    gamma_k = gamma_0
    # local_k counts steps since last restart; beta_k = local_k/(local_k+3)
    # This resets to 0 after a gradient restart so momentum is zeroed immediately.
    local_k = 0

    history = defaultdict(list) if trace else None
    matrix_inverses = 0
    status = ""

    # Main optimization loop
    for k in range(n_iters + 1):
        if trace:
            history["func"].append(f_k)
            history["grad_norm"].append(g_k_norm)
            history["gamma_k"].append(gamma_k)
            history["grad_calls"].append(oracle.grad_calls)
            history["matrix_inverses"].append(matrix_inverses)
            history["time"].append((datetime.now() - start_timestamp).total_seconds())
            history["x_k"].append(x_k.copy())

        if (f_star is not None and f_k - f_star < eps) or (
            grad_tol is not None and g_k_norm < grad_tol
        ):
            status = f"success, {k} iters"
            break

        if k == n_iters:
            status = "iterations_exceeded"
            break

        # 1. Momentum extrapolation using local counter (resets on restart)
        beta_k = local_k / (local_k + 3.0)
        y_k = x_k + beta_k * (x_k - x_prev)

        # 2. Oracle evaluation at lookahead point
        g_y = oracle.grad(y_k)
        g_y_norm = dual_norm_sqr(g_y) ** 0.5
        f_y = oracle.func(y_k)

        # Safe fallback in case the inner loop exits without a successful step
        # (e.g., Cholesky failure on the very first inner iteration).
        x_trial = y_k.copy()
        f_trial = f_y
        g_trial = g_y.copy()
        g_trial_norm_sqr = dual_norm_sqr(g_trial)

        # Guard: if gradient at lookahead is already zero, skip Newton step.
        if g_y_norm < eps:
            x_k = x_trial
            f_k = f_trial
            g_k = g_trial
            g_k_norm = g_trial_norm_sqr ** 0.5
            local_k += 1
            continue

        # 3. Choose Hessian anchor point
        anchor_point = x_k if hessian_anchor == "iterate" else y_k
        if is_approx and approx_oracle is not None and callable(approx_hess_fn):
            Hess_k = approx_hess_fn(approx_oracle, anchor_point)
        else:
            Hess_k = oracle.hess(anchor_point)

        # 4. Adaptive search for gamma_k
        # Ensures: f(y_k) - f(x_trial) >= (gamma_k/8) * ||g_trial||^2_* / ||g_y||_*
        # Equivalently:  f(y_k) - f(x_trial) >= g_trial_norm_sqr / (8 * lambda_k)
        # where lambda_k = g_y_norm / gamma_k.
        # This is exactly the progress condition of Lemma 1 (paper eq. 8) applied at y_k.
        adaptive_search_max_iter = 40
        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter:
                if warnings:
                    print(f"W: adaptive_iterations_exceeded, k = {k}", flush=True)
                break

            lambda_k = g_y_norm / gamma_k
            try:
                delta_y = scipy.linalg.cho_solve(
                    scipy.linalg.cho_factor(Hess_k + lambda_k * B, lower=False), -g_y
                )
                matrix_inverses += 1
            except (np.linalg.LinAlgError, ValueError):
                if warnings:
                    print("W: linalg_error -> tightening gamma", flush=True)
                gamma_k = max(gamma_k * 0.5, gamma_min)
                continue

            x_trial = y_k + delta_y
            f_trial = oracle.func(x_trial)
            g_trial = oracle.grad(x_trial)
            g_trial_norm_sqr = dual_norm_sqr(g_trial)

            if not adaptive_search:
                break

            if f_y - f_trial >= g_trial_norm_sqr / (8.0 * lambda_k):
                gamma_k *= 2.0
                break
            else:
                gamma_k = max(gamma_k * 0.5, gamma_min)

        # 5. Gradient restart (standard Nesterov restart criterion)
        # Restart if the gradient at x_trial is positively correlated with the full
        # step x_trial - x_k (gradient opposes the step direction → momentum was harmful).
        # On restart: zero local_k so beta resets to 0 at the next iteration.
        if g_trial.dot(x_trial - x_k) > 0:
            x_prev = x_trial.copy()
            local_k = 0
        else:
            x_prev = x_k.copy()
            local_k += 1

        # Update the point
        x_k = x_trial.copy()
        f_k = f_trial
        g_k = g_trial
        g_k_norm = g_trial_norm_sqr ** 0.5

    return x_k, status, history


def accelerated_grad_norm_smooth_for_rank_one(
    oracle,
    x_0,
    n_iters=1000,
    gamma_0=1.0,
    adaptive_search=True,
    is_approx=False,
    approx_oracle=None,
    approx_hess_fn=None,
    trace=True,
    B=None,
    Binv=None,
    eps=1e-8,
    gamma_min=1e-9,
    f_star=None,
    grad_tol=None,
    warnings=False,
    invert_backend="cho",
    hessian_anchor="lookahead"
):
    """
    AGNS with Sherman-Morrison rank-1 inversion.

    Same heuristic as `accelerated_grad_norm_smooth` — see that function's
    docstring for the (lack of) convergence guarantees.  This variant solves
        (lambda_k * B + alpha * g0 g0^T) * delta = -grad(y_k)
    in O(m^2) using a B^{-1} solve and a scalar correction (Woodbury formula
    [54] in the paper), instead of an O(m^3) Cholesky on the dense matrix.

    Backends:
      'cho'  — explicit Cholesky on (Hess + lambda*B); same as the standard
               AGNS variant but goes through the rank-1 control flow.
      'wsm'  — closed-form Sherman-Morrison.  Requires `approx_hess_fn` to
               return a rank-1 matrix of the form alpha * g0 g0^T (the Fisher
               term of f = (1/p)||u(x)||^p).
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    if B is None:
        B = np.eye(x_0.shape[0])
        dual_norm_sqr = lambda x: x.dot(x)
        Binv = np.eye(x_0.shape[0])
    else:
        if Binv is None:
            Binv = np.linalg.inv(B)
        dual_norm_sqr = lambda x: Binv.dot(x).dot(x)

    x_k = np.copy(x_0)
    x_prev = np.copy(x_0)
    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    g_k_norm = np.sqrt(dual_norm_sqr(g_k))
    gamma_k = gamma_0
    local_k = 0

    history = defaultdict(list) if trace else None
    matrix_inverses = 0
    status = ""

    for k in range(n_iters + 1):
        if trace:
            history["func"].append(f_k)
            history["grad_norm"].append(g_k_norm)
            history["gamma_k"].append(gamma_k)
            history["grad_calls"].append(oracle.grad_calls)
            history["matrix_inverses"].append(matrix_inverses)
            history["time"].append((datetime.now() - start_timestamp).total_seconds())
            history["x_k"].append(x_k.copy())

        if (f_star is not None and f_k - f_star < eps) or (grad_tol is not None and g_k_norm < grad_tol):
            status = f"success, {k} iters"
            break
        if k == n_iters:
            status = "iterations_exceeded"
            break

        # 1. Momentum Extrapolation using local_k (resets on restart)
        beta_k = local_k / (local_k + 3.0)
        y_k = x_k + beta_k * (x_k - x_prev)

        # 2. Evaluate gradient and function at lookahead
        g_y = oracle.grad(y_k)
        g_y_norm_sqr = dual_norm_sqr(g_y)
        g_y_norm = np.sqrt(g_y_norm_sqr)
        f_y = oracle.func(y_k)

        # Safe fallback if inner loop exits without a successful step
        x_trial = y_k.copy()
        f_trial = f_y
        g_trial = g_y.copy()
        g_trial_norm_sqr = g_y_norm_sqr

        # Guard: skip Newton step if gradient at lookahead is zero
        if g_y_norm < eps:
            x_k = x_trial
            f_k = f_trial
            g_k = g_trial
            g_k_norm = g_y_norm
            local_k += 1
            continue

        # 3. Choose Anchor Point
        anchor_point = x_k if hessian_anchor == "iterate" else y_k

        # The 'wsm' backend reads rank-1 structure (alpha, g0) from
        # approx_oracle directly, so building the full Hessian is wasteful.
        if invert_backend == "cho":
            if is_approx and approx_oracle is not None and callable(approx_hess_fn):
                Hess_k = approx_hess_fn(approx_oracle, anchor_point)
            else:
                Hess_k = oracle.hess(anchor_point)
        else:
            Hess_k = None

        # 4. Adaptive Line Search (Lemma 1 progress condition at y_k)
        adaptive_search_max_iter = 40
        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter:
                if warnings:
                    print(f"W: adaptive_iterations_exceeded, k = {k}", flush=True)
                break

            lambda_k = g_y_norm / gamma_k
            try:
                if invert_backend == "cho":
                    delta_y = scipy.linalg.cho_solve(
                        scipy.linalg.cho_factor(Hess_k + lambda_k * B, lower=False), -g_y
                    )
                    matrix_inverses += 1

                elif invert_backend == "wsm":
                    if not (is_approx and callable(approx_hess_fn)):
                        raise ValueError("wsm backend requires a rank-one approx_hess_fn")
                    u = approx_oracle.func_u(anchor_point)
                    norm_u = np.linalg.norm(u)
                    J = approx_oracle.jac_u(anchor_point)
                    if approx_oracle.p != 2:
                        alpha = (approx_oracle.p - 2) * norm_u ** (approx_oracle.p - 4)
                        g0 = J.T.dot(u)
                    else:
                        alpha = 0.0
                        g0 = np.zeros_like(g_y)
                    inv_scale = 1.0 / lambda_k
                    Pinv_gy = inv_scale * Binv.dot(-g_y)
                    Pinv_g0 = inv_scale * Binv.dot(g0)
                    denom = 1.0 + alpha * (g0.dot(Pinv_g0))
                    delta_y = Pinv_gy - (alpha / denom) * Pinv_g0 * (g0.dot(Pinv_gy))
                    matrix_inverses += 1
                else:
                    raise ValueError(f"Unknown invert_backend '{invert_backend}'")

            except (np.linalg.LinAlgError, ValueError) as e:
                if warnings:
                    print(f"W: linear solve error ({e}) -> tightening gamma", flush=True)
                gamma_k = max(gamma_k * 0.5, gamma_min)
                continue

            x_trial = y_k + delta_y
            f_trial = oracle.func(x_trial)
            g_trial = oracle.grad(x_trial)
            g_trial_norm_sqr = dual_norm_sqr(g_trial)

            if not adaptive_search:
                break

            if f_y - f_trial >= g_trial_norm_sqr / (8 * lambda_k):
                gamma_k *= 2.0
                break
            else:
                gamma_k = max(gamma_k * 0.5, gamma_min)

        # 5. Gradient Restart Check (standard Nesterov criterion)
        if g_trial.dot(x_trial - x_k) > 0:
            x_prev = x_trial.copy()
            local_k = 0
        else:
            x_prev = x_k.copy()
            local_k += 1

        x_k = x_trial.copy()
        f_k = f_trial
        g_k = g_trial
        g_k_norm = np.sqrt(g_trial_norm_sqr)

    return x_k, status, history