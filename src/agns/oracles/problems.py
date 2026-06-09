"""Optimization-problem builders for every oracle family.

A :class:`Problem` bundles an oracle with the starting point, metric and known optimum the
methods need. :func:`build_problem` dispatches on the config ``family`` and returns a
``Problem``, so experiments treat the convex softmax problem and the non-convex stress
problems uniformly. All non-LogSumExp families have an analytic optimum ``f* = 0``, which
makes their functional-residual plots exact.

Families
--------
``logsumexp``            softmax / LogSumExp (random or LIBSVM), optimum shifted to the origin
``rosenbrock``           the classic non-convex banana valley, ``f* = 0`` at ``(a, a^2)``
``nonlinear_equations``  ``(1/p) ||A x - b||^p`` with invertible ``A``, ``f* = 0`` at ``A^{-1} b``
``chebyshev``            non-convex Chebyshev least squares, ``f* = 0`` at ``x = 1``
``polytope``             ``sum_i (a_i x - b_i)_+^p`` feasibility, ``f* = 0`` on the polytope
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_svmlight_file

from agns.oracles.chebyshev import ChebyshevOracle
from agns.oracles.logsumexp import create_log_sum_exp_zero_oracle
from agns.oracles.nonlinear_equations import NonlinearEquationsOracle
from agns.oracles.polytope import PolytopeFeasibility
from agns.oracles.rosenbrock import RosenbrockOracle
from agns.oracles.separable import SeparableCurvatureOracle

Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]


@dataclass
class Problem:
    """A fully specified optimization instance."""

    name: str
    label: str
    oracle: Any
    x_0: Vector
    f_star: float
    eps: float
    B: Matrix | None = None
    Binv: Matrix | None = None
    n: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers


def _metric(
    D: Matrix, use_metric: bool, ridge: float
) -> tuple[Matrix | None, Matrix | None]:
    """Data metric ``B = D^T D (+ ridge I)`` and its inverse, or ``(None, None)``."""
    if not use_metric:
        return None, None
    B = D.T.dot(D)
    if ridge > 0.0:
        B = B + ridge * np.eye(B.shape[0])
    return B, np.linalg.inv(B)


def _start(
    n: int,
    *,
    x0_value: float,
    x0_random: bool,
    x0_seed: int,
    x0_jitter: float | None = None,
) -> Vector:
    """Constant, seeded-random, or seeded-jittered starting point.

    When ``x0_jitter`` is given, the start is the constant ``x0_value`` plus Gaussian noise of
    that scale -- a small per-seed perturbation that stays in the good basin (used for
    multi-basin objectives like Chebyshev, where a full random start lands in a wrong basin).
    Otherwise ``x0_random`` selects a full random draw scaled by ``x0_value`` (fine for the
    convex log-sum-exp family) or a constant start.
    """
    if x0_jitter is not None and x0_random:
        rng = np.random.RandomState(x0_seed)
        return np.full(n, x0_value, dtype=float) + rng.standard_normal(n) * x0_jitter
    if x0_random:
        return np.random.RandomState(x0_seed).standard_normal(n) * x0_value
    return np.full(n, x0_value, dtype=float)


def _conditioned_design(n: int, m: int, cond: float, seed: int) -> Matrix:
    """Random ``(n, m)`` design whose ``A A^T`` has condition number ``cond``."""
    rng = np.random.RandomState(seed)
    u, _ = np.linalg.qr(rng.standard_normal((n, n)))
    v, _ = np.linalg.qr(rng.standard_normal((m, m)))
    s = np.geomspace(1.0, np.sqrt(cond), n)
    sigma = np.zeros((n, m))
    sigma[:n, :n] = np.diag(s)
    return u @ sigma @ v.T


# ----------------------------------------------------------------------- logsumexp


def _build_logsumexp(spec: dict[str, Any], data_dir: Path) -> Problem:
    kind = spec.get("kind", "random")
    mu = float(spec.get("mu", 1.0))
    eps = float(spec.get("eps", 1e-8))
    use_metric = bool(spec.get("use_metric", True))
    label = spec.get("label", spec.get("name", "logsumexp"))

    if kind == "random":
        n, m = int(spec["n"]), int(spec["m"])
        seed = int(spec.get("seed", 3124))
        cond = float(spec.get("cond", 0.0))
        if cond and cond > 1.0:
            A = _conditioned_design(n, m, cond, seed)
        else:
            A = np.random.RandomState(seed).rand(n, m) * 2 - 1
        b = np.random.RandomState(seed + 1).rand(m) * 2 - 1
        D = A.T  # (m, n): m linear pieces, n variables
        ridge = float(spec.get("ridge", 0.0))
        name = f"random_n{n}_m{m}_mu{mu:g}"
    elif kind == "libsvm":
        path = data_dir / spec["dataset"]
        X, y = load_svmlight_file(str(path))
        D = np.asarray(X.todense(), dtype=float)
        y = np.asarray(y, dtype=float)
        max_samples = spec.get("max_samples")
        if max_samples is not None and D.shape[0] > int(max_samples):
            idx = np.random.RandomState(int(spec.get("seed", 0))).choice(
                D.shape[0], size=int(max_samples), replace=False
            )
            D, y = D[idx], y[idx]
        if bool(spec.get("normalize", False)):
            norms = np.linalg.norm(D, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            D = D / norms
        b = y
        ridge = float(spec.get("ridge", 1e-6))
        name = spec.get("name", path.stem)
    else:
        raise ValueError(f"Unknown logsumexp kind {kind!r}")

    m_rows, n = D.shape
    oracle = create_log_sum_exp_zero_oracle(D, b, mu)
    f_star = float(oracle.func(np.zeros(n)))
    x_0 = _start(
        n,
        x0_value=float(spec.get("x0_value", 1.0)),
        x0_random=bool(spec.get("x0_random", False)),
        x0_seed=int(spec.get("x0_seed", 0)),
    )
    B, Binv = _metric(D, use_metric, ridge)
    return Problem(
        name=name,
        label=label,
        oracle=oracle,
        x_0=x_0,
        f_star=f_star,
        eps=eps,
        B=B,
        Binv=Binv,
        n=n,
        meta={"family": "logsumexp", "mu": mu, "m": m_rows, "cond": spec.get("cond")},
    )


# --------------------------------------------------------------- non-convex / other


def _build_rosenbrock(spec: dict[str, Any], _data_dir: Path) -> Problem:
    a = float(spec.get("a", 1.0))
    b = float(spec.get("b", 100.0))
    base_x0 = np.asarray(spec.get("x0", [-1.2, 1.0]), dtype=float)
    if bool(spec.get("x0_random", False)):
        # Random start near the classic hard corner (the "start" seed protocol).
        rng = np.random.RandomState(int(spec.get("x0_seed", 0)))
        jitter = float(spec.get("x0_jitter", 0.5))
        x_0 = base_x0 + rng.standard_normal(2) * jitter
    else:
        x_0 = base_x0
    return Problem(
        name="rosenbrock",
        label=spec.get("label", "Rosenbrock"),
        oracle=RosenbrockOracle(a=a, b=b),
        x_0=x_0,
        f_star=0.0,
        eps=float(spec.get("eps", 1e-8)),
        n=2,
        meta={"family": "rosenbrock", "a": a, "b": b},
    )


def _build_nonlinear_equations(spec: dict[str, Any], _data_dir: Path) -> Problem:
    n = int(spec.get("n", 8))
    p = int(spec.get("p", 4))
    seed = int(spec.get("seed", 7))
    rng = np.random.RandomState(seed)
    A = rng.standard_normal((n, n)) + n * np.eye(n)  # well-conditioned, invertible
    b = rng.standard_normal(n)
    oracle = NonlinearEquationsOracle(p=p, A=A, b=b)
    x_0 = _start(
        n,
        x0_value=float(spec.get("x0_value", 1.0)),
        x0_random=bool(spec.get("x0_random", True)),
        x0_seed=int(spec.get("x0_seed", 0)),
    )
    return Problem(
        name=f"nleq_n{n}_p{p}",
        label=spec.get("label", f"Nonlinear eq. ($p={p}$)"),
        oracle=oracle,
        x_0=x_0,
        f_star=0.0,
        eps=float(spec.get("eps", 1e-8)),
        n=n,
        meta={"family": "nonlinear_equations", "p": p},
    )


def _build_chebyshev(spec: dict[str, Any], _data_dir: Path) -> Problem:
    n = int(spec.get("n", 8))
    p = int(spec.get("p", 4))
    jitter = spec.get("x0_jitter")
    x_0 = _start(
        n,
        x0_value=float(spec.get("x0_value", 0.0)),
        x0_random=bool(spec.get("x0_random", False)),
        x0_seed=int(spec.get("x0_seed", 0)),
        x0_jitter=float(jitter) if jitter is not None else None,
    )
    return Problem(
        name=f"chebyshev_n{n}_p{p}",
        label=spec.get("label", f"Chebyshev ($p={p}$)"),
        oracle=ChebyshevOracle(n, p),
        x_0=x_0,
        f_star=0.0,
        eps=float(spec.get("eps", 1e-8)),
        n=n,
        meta={"family": "chebyshev", "p": p},
    )


def _build_polytope(spec: dict[str, Any], _data_dir: Path) -> Problem:
    n = int(spec.get("n", 50))
    m = int(spec.get("m", 100))
    p = int(spec.get("p", 4))
    seed = int(spec.get("seed", 0))
    rng = np.random.RandomState(seed)
    A = rng.standard_normal((m, n))
    # Make the polytope feasible: pick x_feas and set b so that A x_feas <= b, giving an
    # interior point with f = 0 (otherwise a random A x <= b is almost surely infeasible
    # and the true minimum is positive, breaking the analytic f* = 0).
    x_feas = rng.standard_normal(n)
    b = A.dot(x_feas) + np.abs(rng.standard_normal(m))
    x_0 = _start(
        n,
        x0_value=float(spec.get("x0_value", 1.0)),
        x0_random=bool(spec.get("x0_random", True)),
        x0_seed=int(spec.get("x0_seed", 0)),
    )
    return Problem(
        name=f"polytope_n{n}_m{m}_p{p}",
        label=spec.get("label", f"Polytope ($p={p}$)"),
        oracle=PolytopeFeasibility(A, b, p),
        x_0=x_0,
        f_star=0.0,
        eps=float(spec.get("eps", 1e-8)),
        n=n,
        meta={"family": "polytope", "p": p},
    )


def _build_curvature_variation(spec: dict[str, Any], _data_dir: Path) -> Problem:
    n = int(spec.get("n", 50))
    cond = float(spec.get("cond", 100.0))
    kappa = float(spec.get("kappa", 1.0))
    # Geometric eigenvalue spread fixes the condition number at the optimum (= cond),
    # independent of kappa; kappa controls curvature variation along the path.
    c = np.geomspace(1.0, cond, n)
    oracle = SeparableCurvatureOracle(c=c, kappa=kappa)
    x_0 = _start(
        n,
        x0_value=float(spec.get("x0_value", 1.0)),
        x0_random=bool(spec.get("x0_random", False)),
        x0_seed=int(spec.get("x0_seed", 0)),
    )
    return Problem(
        name=f"curvature_n{n}_cond{cond:g}_kappa{kappa:g}",
        label=spec.get("label", rf"Curvature ($\kappa={kappa:g}$)"),
        oracle=oracle,
        x_0=x_0,
        f_star=0.0,
        eps=float(spec.get("eps", 1e-8)),
        n=n,
        meta={"family": "curvature_variation", "cond": cond, "kappa": kappa},
    )


_BUILDERS = {
    "logsumexp": _build_logsumexp,
    "rosenbrock": _build_rosenbrock,
    "nonlinear_equations": _build_nonlinear_equations,
    "chebyshev": _build_chebyshev,
    "polytope": _build_polytope,
    "curvature_variation": _build_curvature_variation,
}


def build_problem(spec: dict[str, Any], data_dir: str | Path) -> Problem:
    """Build a :class:`Problem` from a config dict, dispatching on ``family``."""
    family = spec.get("family", "logsumexp")
    try:
        builder = _BUILDERS[family]
    except KeyError as exc:
        raise ValueError(
            f"Unknown problem family {family!r}; available: {sorted(_BUILDERS)}"
        ) from exc
    return builder(spec, Path(data_dir))
