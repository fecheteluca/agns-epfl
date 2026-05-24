"""Problem oracles and factories.

Re-exports every concrete oracle class and ``make_<problem>`` factory so
callers can write ``from agns.oracles import NonlinearEquationsOracle``
without knowing the per-file layout.  The :data:`PROBLEM_REGISTRY` dict
maps the ``problem.type`` string used in YAML configs to a factory
callable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agns.oracles.base import BaseSmoothOracle, OracleCallsCounter
from agns.oracles.chebyshev import ChebyshevOracle, make_chebyshev
from agns.oracles.logistic_regression import (
    LogisticRegressionOracle,
    make_logistic_regression_libsvm,
    make_logistic_regression_synthetic,
)
from agns.oracles.logsumexp import (
    LogSumExpOracle,
    make_logsumexp_pair_oracle,
    make_logsumexp_real,
    make_logsumexp_synthetic,
    make_logsumexp_zero_oracle,
)
from agns.oracles.matrix_completion import (
    MatrixCompletionOracle,
    make_matrix_completion_synthetic,
)
from agns.oracles.nn_torch import (
    TorchOracle,
    make_mlp_classifier_synthetic,
)
from agns.oracles.nonlinear_equations import (
    NonlinearEquationsOracle,
    make_nonlinear_equations,
)
from agns.oracles.phase_retrieval import (
    PhaseRetrievalOracle,
    make_phase_retrieval_synthetic,
)
from agns.oracles.ridge_regression import (
    RidgeRegressionOracle,
    make_ridge_regression_libsvm,
    make_ridge_regression_synthetic,
)
from agns.oracles.rosenbrock import (
    NonlinearEquationsRosenbrockOracle,
    RosenbrockOracle,
    make_rosenbrock_2d,
    make_rosenbrock_nle,
)
from agns.oracles.smoothed_lasso import (
    SmoothedLassoOracle,
    make_smoothed_lasso_synthetic,
)
from agns.oracles.soft_svm import (
    SoftSVMOracle,
    make_soft_svm_libsvm,
    make_soft_svm_synthetic,
)
from agns.oracles.softmax_regression import (
    SoftmaxRegressionOracle,
    make_softmax_regression_synthetic,
)

#: Problem-type string -> factory callable.  Consumed by the runner.
PROBLEM_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "logsumexp_synthetic": make_logsumexp_synthetic,
    "logsumexp_real": make_logsumexp_real,
    "nonlinear_equations": make_nonlinear_equations,
    "chebyshev": make_chebyshev,
    "rosenbrock_nle": make_rosenbrock_nle,
    "rosenbrock_2d": make_rosenbrock_2d,
    "logistic_regression_synthetic": make_logistic_regression_synthetic,
    "logistic_regression_libsvm": make_logistic_regression_libsvm,
    "ridge_regression_synthetic": make_ridge_regression_synthetic,
    "ridge_regression_libsvm": make_ridge_regression_libsvm,
    "soft_svm_synthetic": make_soft_svm_synthetic,
    "soft_svm_libsvm": make_soft_svm_libsvm,
    "smoothed_lasso_synthetic": make_smoothed_lasso_synthetic,
    "softmax_regression_synthetic": make_softmax_regression_synthetic,
    "matrix_completion_synthetic": make_matrix_completion_synthetic,
    "phase_retrieval_synthetic": make_phase_retrieval_synthetic,
    "mlp_classifier_synthetic": make_mlp_classifier_synthetic,
}

__all__ = [
    "PROBLEM_REGISTRY",
    "BaseSmoothOracle",
    "ChebyshevOracle",
    "LogSumExpOracle",
    "LogisticRegressionOracle",
    "MatrixCompletionOracle",
    "NonlinearEquationsOracle",
    "NonlinearEquationsRosenbrockOracle",
    "OracleCallsCounter",
    "PhaseRetrievalOracle",
    "RidgeRegressionOracle",
    "RosenbrockOracle",
    "SmoothedLassoOracle",
    "SoftSVMOracle",
    "SoftmaxRegressionOracle",
    "TorchOracle",
    "make_chebyshev",
    "make_logistic_regression_libsvm",
    "make_logistic_regression_synthetic",
    "make_logsumexp_pair_oracle",
    "make_logsumexp_real",
    "make_logsumexp_synthetic",
    "make_logsumexp_zero_oracle",
    "make_matrix_completion_synthetic",
    "make_mlp_classifier_synthetic",
    "make_nonlinear_equations",
    "make_phase_retrieval_synthetic",
    "make_ridge_regression_libsvm",
    "make_ridge_regression_synthetic",
    "make_rosenbrock_2d",
    "make_rosenbrock_nle",
    "make_smoothed_lasso_synthetic",
    "make_soft_svm_libsvm",
    "make_soft_svm_synthetic",
    "make_softmax_regression_synthetic",
]
