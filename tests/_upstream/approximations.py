import numpy as np

from .oracles import *


def approx_hess_fn_logsumexp(approx_oracle, x_k):
    """
    Weighted Gauss-Newton approximation:
    H = (A^T (diag(pi)) A) / mu
    """
    approx_oracle._update_a_and_pi(x_k)
    pi = approx_oracle.pi
    return approx_oracle.matmat_ATsA(pi) / approx_oracle.mu


def approx_hess_fn_fisher_term(oracle, x):
    """
    Approximate Hessian of f(x)=1/p||u(x)||^p by the Fisher term:
      (p-2)||u||^{p-4}(J^T u)(J^T u)^T
    """
    u = oracle.func_u(x)
    norm_u = np.linalg.norm(u)
    J = oracle.jac_u(x)
    H_approx = np.zeros((x.shape[0], x.shape[0]))
    if oracle.p != 2:
        g = J.T.dot(u)
        H_approx += (oracle.p - 2) * norm_u ** (oracle.p - 4) * np.outer(g, g)
    return H_approx


def approx_hess_nonlinear_equations(oracle, x):
    """
    Approximate Hessian of f(x)=1/p||u(x)||^p as out theory suggests:
    H = (||u||^{p-2} J^T J) + (p-2)(||u||^{p-4} (J^T u)(J^T u)^T)
    where J is the Jacobian of u(x).
    """
    u, J = oracle._compute_u_and_jac(x)
    norm_u = np.linalg.norm(u)
    p = oracle.p
    if norm_u == 0:
        return (norm_u ** (p - 2)) * (J.T.dot(J))
    H = (norm_u ** (p - 2)) * (J.T.dot(J))
    if p != 2:
        Ju = J.T.dot(u)
        outer_term = np.outer(Ju, Ju)
        H += (p - 2) * (norm_u ** (p - 4)) * outer_term

    return H


def approx_hess_fn_chebyshev(oracle, x):
    """
    Approximate Hessian of f(x)=1/p||u(x)||^p as out theory suggests:
    H = (||u||^{p-2} J^T J) + (p-2)(||u||^{p-4} (J^T u)(J^T u)^T)
    where J is the Jacobian of u(x).
    """
    u = oracle.func_u(x)
    norm_u = np.linalg.norm(u)
    J = oracle.jac_u(x)
    if norm_u == 0:
        if oracle.p == 2:
            return J.T.dot(J)
        else:
            return np.zeros((oracle.n, oracle.n), dtype=float)

    p = oracle.p
    H_approx = (norm_u ** (p - 2)) * (J.T.dot(J))
    if p != 2:
        g_vec = J.T.dot(u)
        H_approx += (p - 2) * (norm_u ** (p - 4)) * np.outer(g_vec, g_vec)

    return H_approx
