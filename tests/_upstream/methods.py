import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import scipy

from .oracles import OracleCallsCounter


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
    Alggorithm 1 from the paper
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

        # Adaptive search for gamma_k
        adaptive_search_max_iter = 40
        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter and warnings:
                print(f"W: adaptive_iterations_exceeded, k = {k}", flush=True)
                break

            lambda_k = g_k_norm / (gamma_k + eps)
            try:
                # Regularized Newton step
                delta_x = scipy.linalg.cho_solve(
                    scipy.linalg.cho_factor(Hess_k + lambda_k * B, lower=False), -g_k
                )
                matrix_inverses += 1
            except (np.linalg.LinAlgError, ValueError):
                if warnings:
                    print("W: linalg_error", flush=True)
                break

            f_new = oracle.func(x_k + delta_x)
            g_new = oracle.grad(x_k + delta_x)
            g_new_norm_sqr = dual_norm_sqr(g_new)

            if not adaptive_search:
                break

            # Check condition for gamma_k
            if f_k - f_new >= g_new_norm_sqr / (8 * lambda_k):
                gamma_k = max(gamma_k * 2.0, gamma_min)
                break
            else:
                gamma_k *= 0.5

        # Update the point
        x_k += delta_x
        f_k = f_new
        g_k = g_new
        g_k_norm = g_new_norm_sqr**0.5

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

        adaptive_search_max_iter = 40
        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter:
                if warnings:
                    print(("W: adaptive_iterations_exceeded, k = %d" % k), flush=True)
                break

            lambda_k = H_k * g_k_norm**alpha
            try:
                # Compute the regularized Newton step
                delta_x = scipy.linalg.cho_solve(
                    scipy.linalg.cho_factor(Hess_k + lambda_k * B, lower=False), -g_k
                )
                matrix_inverses += 1
            except (np.linalg.LinAlgError, ValueError) as e:
                if warnings:
                    print("W: linalg_error", flush=True)

            f_new = oracle.func(x_k + delta_x)
            g_new = oracle.grad(x_k + delta_x)
            g_new_norm_sqr = dual_norm_sqr(g_new)

            if not adaptive_search:
                break

            # Check condition for H_k
            if g_new.dot(-delta_x) >= g_new_norm_sqr / (4 * lambda_k):
                H_k *= 0.25
                H_k = max(H_k, H_min)
                break

            H_k *= 4

        # Update the point
        x_k += delta_x
        f_k = f_new
        g_k = g_new
        g_k_norm = g_new_norm_sqr**0.5

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

        adaptive_search_max_iter = 40
        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter:
                if warnings:
                    print("W: adaptive_iterations_exceeded", flush=True)
                break

            # Compute the Cubic Newton step
            x_delta, model_value, message = cubic_newton_step(
                g_k, Hess_k, 0.5 * H_k, B, 1e-8
            )
            if message != "success" and warnings:
                print("W: cubic_newton_step: %s" % message, end=" ", flush=True)
            f_new = oracle.func(x_k + x_delta)

            if not adaptive_search:
                break

            # Check condition for H_k
            if f_new <= f_k + model_value:
                H_k *= 0.5
                H_k = max(H_k, H_min)
                break

            H_k *= 2

        # Update the point
        x_k += x_delta
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
):
    """
    Run the Gradient Method for 'n_iters' iterations,
    minimizing smooth function.

    'oracle' is an instance of BaseSmoothOracle representing the objective.
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    # Initialization of the Euclidean metric
    if B is None:
        l2_norm_sqr = lambda x: x.dot(x)
        dual_norm_sqr = lambda x: x.dot(x)
        to_dual = lambda x: x
    else:
        l2_norm_sqr = lambda x: B.dot(x).dot(x)
        to_dual = lambda x: B.dot(x)
        if Binv is None:
            Binv = np.linalg.inv(B)
        dual_norm_sqr = lambda x: Binv.dot(x).dot(x)
    if Binv is None:
        precond = lambda g: g
    else:
        precond = lambda g: Binv.dot(g)

    # Initialization of the method
    x_k = np.copy(x_0)
    func_k = oracle.func(x_k)
    grad_k = oracle.grad(x_k)
    grad_k_norm_sqr = dual_norm_sqr(grad_k)
    L_k = L_0

    history = defaultdict(list) if trace else None
    status = ""

    # Main loop
    for k in range(max_iter + 1):

        if trace:
            history["func"].append(func_k)
            history["grad_sqr_norm"].append(grad_k_norm_sqr)
            history["L"].append(L_k)
            history["func_calls"].append(oracle.func_calls)
            history["grad_calls"].append(oracle.grad_calls)
            history["time"].append((datetime.now() - start_timestamp).total_seconds())
            history["x_k"].append(x_k.copy())

        if k == max_iter:
            status = "iterations_exceeded"
            break

        line_search_max_iter = 30
        for i in range(line_search_max_iter + 1):
            if i == line_search_max_iter:
                if warnings:
                    print("W: line_search_max_iter_reached", flush=True)
                break

            T = x_k - precond(grad_k) / L_k
            func_T = oracle.func(T)

            if not line_search:
                break

            if func_T <= func_k + grad_k.dot(T - x_k) + 0.5 * L_k * l2_norm_sqr(
                T - x_k
            ):
                L_k *= 0.5
                L_k = max(L_k, L_min)
                break

            L_k *= 2

        x_k = T
        func_k = func_T
        grad_k = oracle.grad(T)
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
):
    """
    Run the Fast Gradient Method for 'n_iters' iterations,
    minimizing smooth function.

    'oracle' is an instance of BaseSmoothOracle representing the objective.
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    # Initialization of the Euclidean metric
    if B is None:
        l2_norm_sqr = lambda x: x.dot(x)
        dual_norm_sqr = lambda x: x.dot(x)
        to_dual = lambda x: x
        precond = lambda g: g
    else:
        l2_norm_sqr = lambda x: B.dot(x).dot(x)
        to_dual = lambda x: B.dot(x)
        if Binv is None:
            Binv = np.linalg.inv(B)
        dual_norm_sqr = lambda x: Binv.dot(x).dot(x)
        precond = lambda g: Binv.dot(g)

    # Initialization of the method
    x_k = np.copy(x_0)
    v_k = np.copy(x_0)
    func_k = oracle.func(x_k)
    grad_k = oracle.grad(x_k)
    grad_k_norm_sqr = dual_norm_sqr(grad_k)
    L_k = L_0
    A_k = 0.0

    history = defaultdict(list) if trace else None
    status = ""

    # Main loop
    for k in range(max_iter + 1):

        if trace:
            history["func"].append(func_k)
            history["grad_sqr_norm"].append(grad_k_norm_sqr)
            history["func_calls"].append(oracle.func_calls)
            history["grad_calls"].append(oracle.grad_calls)
            history["L"].append(L_k)
            history["time"].append((datetime.now() - start_timestamp).total_seconds())
            history["x_k"].append(x_k.copy())

        if k == max_iter:
            status = "iterations_exceeded"
            break

        line_search_max_iter = 30
        for i in range(line_search_max_iter + 1):
            if i == line_search_max_iter:
                if warnings:
                    print("W: line_search_max_iter_reached", flush=True)
                break

            a_k_new = (1 + (1 + 4 * A_k * L_k) ** 0.5) / (2 * L_k)
            y_k = (a_k_new * v_k + A_k * x_k) / (a_k_new + A_k)
            grad_y_k = oracle.grad(y_k)

            T = y_k - precond(grad_y_k) / L_k
            func_T = oracle.func(T)

            if not line_search:
                break

            func_y_k = oracle.func(y_k)
            if func_T <= func_y_k + grad_y_k.dot(T - y_k) + 0.5 * L_k * l2_norm_sqr(
                T - y_k
            ):
                L_k *= 0.5
                L_k = max(L_k, L_min)
                break

            L_k = 2 * L_k

        grad_T = oracle.grad(T)

        v_k = T + A_k / a_k_new * (T - x_k)
        A_k += a_k_new

        x_k = T
        func_k = func_T
        grad_k_norm_sqr = dual_norm_sqr(grad_k)

    return x_k, status, history


def grad_norm_smooth_gradient_method(
    oracle,
    x_0,
    n_iters=1000,
    gamma_0=1.0,
    adaptive_search=True,
    trace=True,
    is_precond=False,
    precond_oracle=None,
    precond_fn=None,
    eps=1e-8,
    B=None,
    Binv=None,
    gamma_min=1e-9,
    f_star=None,
    grad_tol=None,
    warnings=False,
):
    """
    Run Normalized Gradient Method
    with an adaptive search procedure
    as in Algorithm 1from the paper.
    """
    oracle = OracleCallsCounter(oracle)
    start_timestamp = datetime.now()

    # Initialization of the Euclidean metric
    if B is None:
        dual_norm_sqr = lambda x: x.dot(x)
    else:
        if Binv is None:
            Binv = np.linalg.inv(B)
        dual_norm_sqr = lambda x: Binv.dot(x).dot(x)
    if Binv is None:
        precond = lambda g: g
    else:
        precond = lambda g: Binv.dot(g)

    # Initialization of the method
    x_k = np.copy(x_0)
    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    g_k_norm = dual_norm_sqr(g_k) ** 0.5
    gamma_k = gamma_0

    history = defaultdict(list) if trace else None
    status = ""

    # Main loop
    for k in range(n_iters + 1):
        if trace:
            history["func"].append(f_k)
            history["grad_norm"].append(g_k_norm)
            history["gamma_k"].append(gamma_k)
            history["func_calls"].append(oracle.func_calls)
            history["grad_calls"].append(oracle.grad_calls)
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

        adaptive_search_max_iter = 40
        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter:
                if warnings:
                    print("W: adaptive_search_max_iter_exceeded", flush=True)
                break

            lambda_k = g_k_norm / (gamma_k + eps)

            if is_precond and precond_oracle is not None and callable(precond_fn):
                P_k = precond_fn(precond_oracle, x_k) + eps * np.eye(x_k.shape[0])
                P_k_inv = np.linalg.inv(P_k)
                precond = lambda g: P_k_inv.dot(g)

            T = x_k - precond(g_k) / lambda_k
            f_new = oracle.func(T)
            g_new = oracle.grad(T)
            g_new_norm_sqr = dual_norm_sqr(g_new)

            if not adaptive_search:
                break

            if f_k - f_new >= g_new_norm_sqr / (8 * lambda_k):
                gamma_k = max(gamma_k * 2.0, gamma_min)
                break
            else:
                gamma_k *= 0.5

        x_k = T
        f_k = f_new
        g_k = g_new
        g_k_norm = g_new_norm_sqr**0.5

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

        if (f_star is not None and f_k - f_star < eps) or (
            grad_tol is not None and g_k_norm < grad_tol
        ):
            status = f"success, {k} iters"
            break
        if k == n_iters:
            status = "iterations_exceeded"
            break

        if is_approx and approx_oracle is not None and callable(approx_hess_fn):
            Hess_k = approx_hess_fn(approx_oracle, x_k)
        else:
            Hess_k = oracle.hess(x_k)

        adaptive_search_max_iter = 40
        for i in range(adaptive_search_max_iter + 1):
            if i == adaptive_search_max_iter and warnings:
                print(f"W: adaptive_iterations_exceeded, k = {k}", flush=True)
                break

            lambda_k = g_k_norm / (gamma_k + eps)
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
                    print(f"W: linear solve error ({e})", flush=True)
                break

            x_trial = x_k + delta_x
            f_new = oracle.func(x_trial)
            g_new = oracle.grad(x_trial)
            g_new_norm_sqr = dual_norm_sqr(g_new)

            if not adaptive_search:
                break

            if f_k - f_new >= g_new_norm_sqr / (8 * lambda_k):
                gamma_k = max(gamma_k * 2.0, gamma_min)
                break
            else:
                gamma_k *= 0.5

        x_k = x_trial
        f_k = f_new
        g_k = g_new
        g_k_norm = np.sqrt(g_new_norm_sqr)

    return x_k, status, history
