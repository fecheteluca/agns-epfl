r"""PyTorch oracle wrapper for neural-network training.

A :class:`TorchOracle` wraps any ``torch.nn.Module`` so it satisfies the
:class:`agns.oracles.base.BaseSmoothOracle` interface that every method
in the registry consumes.  The wrapper exposes:

* :meth:`func` -- training loss at parameter vector ``w``,
* :meth:`grad` -- flattened gradient via autograd,
* :meth:`hess_vec` -- Hessian-vector product via double backward,
* :meth:`hess` -- full Hessian, materialised column-by-column from
  :meth:`hess_vec` (only feasible for tiny networks; meant for unit
  tests and the smoke benchmark).

All numerical traffic at the interface boundary is ``numpy.float64``
to match the convention used by every other oracle in the registry.
The wrapped module is cast to ``torch.float64`` so the autograd
computation does not silently downcast the optimisation iterates.

Larger architectures (CIFAR ResNet, transformer blocks) need
:meth:`hess_vec` only --- materialising the full Hessian is out of
scope.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from agns.oracles.base import BaseSmoothOracle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_params(module: nn.Module) -> torch.Tensor:
    """Concatenate every trainable parameter of ``module`` into a 1-D tensor."""
    return torch.cat([p.detach().reshape(-1) for p in module.parameters() if p.requires_grad])


def _set_params_from_flat(module: nn.Module, w: torch.Tensor) -> None:
    """Copy the entries of ``w`` (1-D tensor) back into the module's parameters,
    in the same order :func:`_flatten_params` produced them.

    Operates in-place; preserves the parameter ``requires_grad`` flags.
    """
    offset = 0
    with torch.no_grad():
        for p in module.parameters():
            if not p.requires_grad:
                continue
            numel = p.numel()
            p.copy_(w[offset : offset + numel].view_as(p))
            offset += numel
    if offset != w.numel():
        raise ValueError(f"flat-vector size {w.numel()} != total param count {offset}")


def _flatten_grads(module: nn.Module) -> torch.Tensor:
    """Read ``param.grad`` (after a backward pass) into a single 1-D tensor.

    Parameters that received no gradient contribute zeros, matching the
    convention used by ``torch.optim``.
    """
    pieces = []
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            pieces.append(torch.zeros_like(p).reshape(-1))
        else:
            pieces.append(p.grad.detach().reshape(-1))
    return torch.cat(pieces)


# ---------------------------------------------------------------------------
# TorchOracle
# ---------------------------------------------------------------------------


class TorchOracle(BaseSmoothOracle):
    r"""Wrap a ``torch.nn.Module`` as an AGNS-compatible oracle.

    Parameters
    ----------
    model
        The network.  Will be cast to ``torch.float64`` and moved to
        ``device`` in place.
    X, y
        Training inputs and targets.  Either tensors or numpy arrays;
        they are cast to ``float64`` (or ``int64`` for integer labels)
        and moved to ``device`` on construction.
    loss_fn
        Callable ``loss_fn(preds, y_batch) -> scalar tensor``.  E.g.
        ``torch.nn.CrossEntropyLoss(reduction='mean')`` or
        ``torch.nn.MSELoss(reduction='mean')``.
    reg
        L2 regularisation strength.  Added as ``0.5 * reg * ||w||^2``
        on top of the data-fit loss.
    device
        ``'cpu'`` (default) or ``'cuda'``.  All tensors live on this
        device; numpy inputs are converted at the boundary.
    """

    def __init__(
        self,
        model: nn.Module,
        X: Any,
        y: Any,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        reg: float = 0.0,
        device: str = "cpu",
    ) -> None:
        if reg < 0.0:
            raise ValueError(f"reg must be non-negative, got {reg}")
        self.device = torch.device(device)
        # Cast the model to float64 so the algorithms see the same
        # numerical precision as every other oracle in the package.
        self.model = model.to(dtype=torch.float64, device=self.device)
        self.loss_fn = loss_fn
        self.reg = float(reg)

        # Inputs / targets at boundary -> torch tensors on ``device``.
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(np.asarray(X, dtype=np.float64))
        self.X: torch.Tensor = X.to(dtype=torch.float64, device=self.device)

        if isinstance(y, np.ndarray):
            # Heuristic: integer dtype -> classification target;
            # float dtype -> regression target.
            if np.issubdtype(y.dtype, np.integer):
                y = torch.from_numpy(np.asarray(y, dtype=np.int64))
            else:
                y = torch.from_numpy(np.asarray(y, dtype=np.float64))
        if y.dtype.is_floating_point:
            y = y.to(dtype=torch.float64, device=self.device)
        else:
            y = y.to(device=self.device)
        self.y: torch.Tensor = y

        # Cache parameter-list reference and total dimension for the
        # flatten/unflatten round-trip.
        self._params: list[torch.nn.Parameter] = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        self.n: int = int(sum(p.numel() for p in self._params))

    # ------------------------------------------------------------------
    # Helpers operating on the wrapped torch module
    # ------------------------------------------------------------------
    def _load_w(self, w: NDArray[np.float64]) -> torch.Tensor:
        """Copy ``w`` (numpy) into the module's parameters and return the
        same vector as a torch tensor on the model's device."""
        w_arr = np.asarray(w, dtype=np.float64).ravel()
        if w_arr.shape[0] != self.n:
            raise ValueError(f"w has {w_arr.shape[0]} entries; expected {self.n}")
        w_t = torch.from_numpy(w_arr).to(device=self.device)
        _set_params_from_flat(self.model, w_t)
        return w_t

    def _data_loss(self) -> torch.Tensor:
        """Forward pass: return scalar data-fit loss tensor (no reg)."""
        preds = self.model(self.X)
        return self.loss_fn(preds, self.y)

    # ------------------------------------------------------------------
    # BaseSmoothOracle interface
    # ------------------------------------------------------------------
    def func(self, w: NDArray[np.float64]) -> float:
        w_t = self._load_w(w)
        with torch.no_grad():
            loss = self._data_loss()
            reg_term = 0.5 * self.reg * float(w_t.dot(w_t).item())
        return float(loss.item()) + reg_term

    def grad(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        w_t = self._load_w(w)
        self.model.zero_grad(set_to_none=True)
        loss = self._data_loss()
        loss.backward()
        g = _flatten_grads(self.model).detach().cpu().numpy().astype(np.float64)
        if self.reg > 0.0:
            g = g + self.reg * w_t.detach().cpu().numpy().astype(np.float64)
        return np.asarray(g, dtype=np.float64)

    def hess_vec(self, w: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Hessian-vector product via double backward.

        Builds ``grad(L)`` with ``create_graph=True`` so a second
        ``autograd.grad`` against ``(grad . v)`` yields the HVP.  The
        L2 regulariser contributes a constant ``reg * v`` correction.
        """
        self._load_w(w)
        v_arr = np.asarray(v, dtype=np.float64).ravel()
        if v_arr.shape[0] != self.n:
            raise ValueError(f"v has {v_arr.shape[0]} entries; expected {self.n}")
        v_t = torch.from_numpy(v_arr).to(device=self.device)

        # First-order grad with create_graph so we can backprop through it.
        loss = self._data_loss()
        grads = torch.autograd.grad(loss, self._params, create_graph=True)
        flat_grad = torch.cat([gp.reshape(-1) for gp in grads])
        gv = flat_grad.dot(v_t)
        hvp = torch.autograd.grad(gv, self._params, retain_graph=False)
        flat_hvp = torch.cat([hp.detach().reshape(-1) for hp in hvp])
        hv = flat_hvp.cpu().numpy().astype(np.float64)
        if self.reg > 0.0:
            hv = hv + self.reg * v_arr
        return np.asarray(hv, dtype=np.float64)

    def hess(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        """Materialise the full ``n x n`` Hessian via column-by-column HVP.

        ``O(n)`` double-backward calls; only practical for ``n`` up to a
        few thousand parameters.  Used by the unit tests and by Newton-
        style methods that explicitly want the dense matrix.
        """
        H = np.zeros((self.n, self.n), dtype=np.float64)
        ek = np.zeros(self.n, dtype=np.float64)
        for k in range(self.n):
            ek[k] = 1.0
            H[:, k] = self.hess_vec(w, ek)
            ek[k] = 0.0
        # Symmetrise to absorb numerical asymmetry from non-deterministic
        # GPU reductions; on CPU this is essentially a no-op.
        return 0.5 * (H + H.T)


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


def _build_mlp(
    n_features: int,
    hidden_dims: tuple[int, ...],
    n_classes: int,
    activation: str = "tanh",
) -> nn.Module:
    """Tanh-MLP for synthetic classification benchmarks.

    Tanh is preferred over ReLU because the second-order methods rely
    on a well-defined Hessian, and ReLU's derivative is discontinuous.
    Tanh is smooth and gives a well-defined Hessian everywhere.
    """
    act_cls = {"tanh": torch.nn.Tanh, "sigmoid": torch.nn.Sigmoid, "relu": torch.nn.ReLU}
    if activation not in act_cls:
        raise ValueError(f"unknown activation {activation}; expected one of {list(act_cls)}")
    layers: list[Any] = []
    in_dim = n_features
    for h in hidden_dims:
        layers.append(torch.nn.Linear(in_dim, h))
        layers.append(act_cls[activation]())
        in_dim = h
    layers.append(torch.nn.Linear(in_dim, n_classes))
    return cast("nn.Module", torch.nn.Sequential(*layers))


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_mlp_classifier_synthetic(
    n_samples: int = 200,
    n_features: int = 6,
    n_classes: int = 3,
    hidden_dims: tuple[int, ...] = (8,),
    reg: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Tiny MLP on synthetic Gaussian features for smoke test.

    Targets come from a random ``n_classes``-centroid rule plus mild
    label noise, giving a well-conditioned but non-trivial decision
    boundary.  The default architecture has ``n_features * 8 + 8 *
    n_classes + biases`` parameters -- small enough that the full
    Hessian can be materialised (``hess`` is O(n) HVPs).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    X = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    # Random ``n_classes`` cluster centroids -> assign labels by nearest centroid.
    centroids = rng.standard_normal((n_classes, n_features)).astype(np.float64)
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    y = dists.argmin(axis=1).astype(np.int64)
    # 10% label noise to keep the loss strictly positive.
    flip = rng.random(n_samples) < 0.1
    y[flip] = (y[flip] + rng.integers(1, n_classes, size=flip.sum())) % n_classes

    model = _build_mlp(n_features, hidden_dims, n_classes, activation="tanh")
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    oracle = TorchOracle(model, X, y, loss_fn, reg=reg, device=device)
    x_0 = _flatten_params(oracle.model).detach().cpu().numpy().astype(np.float64)

    return {
        "oracle": oracle,
        "x_0": x_0,
        "f_star": None,  # non-convex, unknown
        "B": None,
        "Binv": None,
        "approx_hess_fn": None,
        "approx_hess_fn_wsm": None,
        "meta": {
            "type": "mlp_classifier_synthetic",
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "hidden_dims": list(hidden_dims),
            "reg": reg,
            "seed": seed,
            "n_params": oracle.n,
        },
    }


__all__ = [
    "TorchOracle",
    "make_mlp_classifier_synthetic",
]
