# Gradient-Regularized Newton Methods with Approximate Hessians and Acceleration

> Empirical study of the **Gradient-Normalized Smoothness (GNS)** framework of
> Semenov, Jaggi & Doikov (2025), and a novel accelerated variant called
> **AGNS** that combines GNS with Nesterov momentum and a gradient-restart
> heuristic.

**Author.** Luca-Andrei Fechete &middot; `luca.fechete123@gmail.com`

**Reference paper.** *Gradient-Normalized Smoothness for Optimization with
Approximate Hessians*, Semenov, Jaggi & Doikov, arXiv:2506.13710 (2025) —
included as [`docs/2506.13710v1.pdf`](docs/2506.13710v1.pdf).

---

## Abstract

This repository implements, audits, and benchmarks the GNS family of
gradient-regularized Newton methods with approximate Hessians, on six
problem families spanning convex (LogSumExp / softmax) and non-convex
(nonlinear equations, Chebyshev polynomial chains, Rosenbrock and its
NLE-formulation) settings. We further propose **AGNS**, an accelerated
variant that applies Nesterov-style extrapolation to GNS with O'Donoghue–
Candès gradient restart and an optional Sherman-Morrison (Woodbury)
solver for rank-1 Fisher approximations. Across 25 configurations and 157
method runs, AGNS-Lookahead reduces iteration counts by **1.5–2.5x** over
GNS-Inexact on hard non-convex problems while never increasing them on
the convex / well-conditioned ones, and the WSM solver yields a
**30–100x wall-clock speedup** on the nonlinear-equations benchmark.

---

## 1. Background

For a smooth (possibly non-convex) objective `f : R^n -> R`, the GNS update is

```text
x_{k+1} = x_k - (H(x_k) + (||grad f(x_k)||_* / gamma_k) B)^{-1} grad f(x_k),
```

where `H(x_k)` is an arbitrary symmetric PSD approximation of the
Hessian, `gamma_k > 0` is a step size selected by an adaptive search
enforcing the **Lemma 1 progress condition**

```text
f(x_k) - f(x_{k+1})  >=  (gamma_k / 8) * ||grad f(x_{k+1})||_*^2 / ||grad f(x_k)||_*,
```

and `B` is a positive-definite metric defining `||h|| = <Bh, h>^{1/2}`.
GNS recovers state-of-the-art rates in every standard problem class
(Hölder Hessian, Hölder third derivative, quasi-self-concordant,
generalized self-concordant, `(L0, L1)`-smooth) without knowing which
class is active, while remaining robust to Hessian inexactness.

The paper explicitly leaves the **acceleration of GNS as future work**.
Closing this gap **empirically** is the main contribution of this codebase.

## 2. Methods

| Registry key | Algorithm | Reference |
| --- | --- | --- |
| `gns_exact` | GNS, `H = grad^2 f` | Semenov–Jaggi–Doikov 2025, Alg. 1 |
| `gns_inexact` | GNS with PSD approx (Fisher / Weighted GN) | Semenov–Jaggi–Doikov 2025, Sec. 5 |
| `gns_wsm` | GNS with rank-1 Fisher + Sherman–Morrison solve | This work |
| `agns_lookahead` | **AGNS** (ours) — Nesterov momentum on GNS, Hessian at `y_k` | This work |
| `agns_iterate` | **AGNS** (ours) — Hessian at `x_k` (ablation) | This work |
| `agns_wsm_lookahead` | AGNS-Lookahead with WSM solve | This work |
| `agns_wsm_iterate` | AGNS-Iterate with WSM solve (ablation) | This work |
| `agns_exact_lookahead` | AGNS with the exact Hessian | This work |
| `super_newton` | Super-Universal Newton method | Doikov–Mishchenko–Nesterov 2024 |
| `cubic_newton` | Adaptive cubic regularisation | Nesterov–Polyak 2006; Cartis–Gould–Toint 2011 |
| `gradient` | Gradient method, backtracking line search | Nesterov 2018 |
| `fast_gradient` | Nesterov accelerated gradient method (FISTA-style) | Beck & Teboulle 2009 |

The GNS adaptive search and the AGNS gradient-restart criterion are
implemented from scratch in pure NumPy / SciPy. The WSM identity used by
the rank-1 backend is

```text
(lambda B + alpha g0 g0^T)^{-1}
    = (1/lambda) B^{-1}
      - (alpha / lambda^2) / (1 + (alpha / lambda) g0^T B^{-1} g0)
        * B^{-1} g0 (B^{-1} g0)^T,
```

which reduces the per-iteration cost from `O(m^3)` (Cholesky factorisation)
to `O(m^2)` (two `B^{-1}` solves).

## 3. Repository layout

```text
optml_project/
|-- README.md                  this file
|-- requirements.txt           minimal pinned deps
|-- main.py                    YAML-driven experiment runner
|-- src/
|   |-- oracles.py             BaseSmoothOracle and 7 concrete oracles
|   |-- methods.py             10 optimization algorithms (GNS, AGNS, ...)
|   |-- approximations.py      Hessian approximations
|   |-- datasets.py            Problem-instance factory + metric construction
|   |-- models.py              Method registry + generic dispatch
|   `-- utils.py               Plotting helpers
|-- tests/
|   |-- test_oracles.py        12 finite-difference oracle correctness tests
|   |-- test_wsm_identity.py   Sherman-Morrison vs dense Cholesky agreement
|   `-- test_methods_smoke.py  Convergence + monotone-descent regressions
|-- config/                    11 YAML experiment configurations
|-- scripts/
|   |-- run_all.sh             Full benchmark suite (~25 min)
|   |-- run_logsumexp.sh       LSE incl. real LIBSVM datasets and seed sweep
|   |-- run_nonlinear_equations.sh
|   |-- run_chebyshev.sh
|   |-- run_rosenbrock.sh
|   `-- download_data.sh       Pulls a9a, mushrooms, w8a from LIBSVM
|-- data/                      (populated by download_data.sh)
|-- results/                   Per-config: summary.json, *_history.pkl, plots/
|-- notebooks/examples.ipynb   Walk-through notebook
|-- docs/2506.13710v1.pdf      Reference paper
`-- paper/                     Workshop write-up (LaTeX + figures + bib)
```

## 4. Installation

```bash
conda create -n ml_opt python=3.10 pip -y
conda activate ml_opt
pip install -r requirements.txt
```

`requirements.txt` is now minimal: `numpy`, `scipy`, `scikit-learn`,
`pyyaml`, `matplotlib`. None of the heavyweight ML deps (torch, optuna,
tensorboard, accelerate) are needed for any of the benchmarks.

## 5. Reproducing the experiments

### One command, full sweep

```bash
bash scripts/download_data.sh    # one-time: pulls a9a, mushrooms, w8a
bash scripts/run_all.sh          # ~25 min; runs 25 configs end-to-end
```

This produces 25 result directories under `results/` containing, per
configuration, `summary.json`, one `*_history.pkl` per method, and PNG
plots in `plots/{iterations,time,grad_calls,matrix_inverses}.png`.

### Single experiment

```bash
python main.py --config config/rosenbrock_nle.yaml
python main.py --config config/nonlinear_equations_wsm.yaml --seed 42
python main.py --config config/chebyshev.yaml --methods gns_exact agns_lookahead
```

### Sweeps

| Sweep axis | Script | Configurations |
| --- | --- | --- |
| LSE seeds | `scripts/run_logsumexp.sh` | seeds 1–5 of `logsumexp_synthetic_sweep.yaml` |
| Chebyshev dimension | `scripts/run_chebyshev.sh` | `n in {200, 500, 1000, 2000}` |
| NLE power | `scripts/run_nonlinear_equations.sh` | `p in {2, 3, 4, 5, 6}` |
| Rosenbrock-NLE power | `scripts/run_rosenbrock.sh` | `p in {2, 3, 4, 5, 6, 8}` |

### Tests

```bash
python tests/test_oracles.py        # 12 finite-difference oracle tests
python tests/test_wsm_identity.py   # Sherman-Morrison vs dense Cholesky
python tests/test_methods_smoke.py  # convergence smoke + monotone descent
python tests/test_proofs.py         # numerical verification of Appendix A theorems
```

All 22 tests run in under 1 second. The proof tests cross-check every
theorem in the workshop-paper appendix (`paper/main.tex`, Appendix A)
against numerical experiments: Sherman-Morrison identity (Theorem 7),
its specialised WSM closed form (Corollary 8), Theorem 3
(AGNS-Lookahead step equals a GNS step at $y_k$), Theorem 5
(restart-anchored progress), and Lemma 2 inheritance (the paper's
Lemma 1 progress condition holds at every accepted AGNS iterate).

## 6. Headline results

The complete results live under `results/`. The figures used in the
accompanying workshop paper are in [`paper/figs/`](paper/figs/). A few
numbers from the full sweep (target accuracy `eps = 1e-8`):

| Problem | GNS-Exact | AGNS-Lookahead (ours) | Speedup |
| --- | ---: | ---: | ---: |
| Rosenbrock-NLE p=5 | 56 iters | **32** iters | **1.75x** |
| Rosenbrock-NLE p=8 | 59 iters | **36** iters | **1.64x** |
| Chebyshev n=1000, p=4 | 31 iters | **20** iters | **1.55x** |
| Chebyshev n=2000, p=4 | 34 iters | **19** iters | **1.79x** |
| Nonlinear equations p=4 | 27 iters | **15** iters | **1.80x** |

Wall-clock impact of the rank-1 (WSM) solver on the nonlinear-equations
benchmark (`n=100, m=200, p=4`):

| Method | Wall-clock to `eps = 1e-8` |
| --- | ---: |
| GNS-Exact (Cholesky) | 0.110 s |
| GNS-Fisher (WSM, ours) | 0.003 s |
| AGNS-WSM-Lookahead (ours) | 0.003 s |

On Rosenbrock-NLE at high `p`, the gradient methods cleanly report
`line_search_diverged` (the function grows polynomially of degree `p`;
no power-of-two step from `x_0 = (-2, 2)` yields a finite probe value),
while every Newton-style method converges. This is a meaningful
experimental finding that earlier versions of the code masked with NaN
values; the `accepted`-flag fix in [`src/methods.py`](src/methods.py)
makes it visible.

## 7. Code architecture

The dispatcher pattern in [`src/models.py`](src/models.py) is what makes
the YAML pipeline composable. Every method is registered as

```python
"agns_lookahead": MethodSpec(
    run_fn=accelerated_grad_norm_smooth,
    param_map={},
    uses_approx=True,
    uses_wsm=False,
    default_kw={"is_approx": True, "hessian_anchor": "lookahead"},
),
```

and is then invoked through a generic
`run_method(spec, oracle, x_0, cfg, approx_oracle, approx_hess_fn)` that
translates a single canonical parameter set (`n_iters`, `gamma_0`,
`eps`, ...) into whatever each algorithm expects (`gamma_0` ↔ `H_0` ↔
`L_0`, `n_iters` ↔ `max_iter`, etc.). Adding a new algorithm is one
entry in the registry and one `run_method` call.

Every history dict shares the same key set (`func`, `time`,
`grad_calls`, `matrix_inverses`, `x_k`), so plotting and summary code
never branches on method type.

## 8. Limitations and honest reporting

- **AGNS has no convergence proof.** The paper's Lemma 1 covers GNS but
  not its momentum-augmented version; we reproduce the Lemma-1 progress
  condition relative to `f(y_k)`, which together with the
  O'Donoghue–Candès gradient restart suffices for empirical descent but
  is not a guarantee. Docstrings flag this clearly.
- **AGNS is not a free lunch.** On problems where GNS-Exact already
  achieves quadratic local convergence in a handful of iterations
  (LogSumExp on a9a, mushrooms), AGNS does *not* reduce iteration count
  — the win comes on hard, polynomially-flavoured non-convex problems.
  Both regimes are reported in the workshop paper.
- **AGNS-Iterate (Hessian at `x_k`, gradient at `y_k`) is included as
  an ablation only.** It mixes anchors in a way that has no Lemma-1
  analogue; we recommend the Lookahead variant.
- **Plain gradient descent diverges at `p >= 6`** on Rosenbrock-NLE for
  the chosen `L_0 = 1` (the descent condition cannot bracket a finite
  trial). This is reported cleanly as `line_search_diverged` rather
  than via NaN propagation.

## 9. Citation

If you use this codebase, please cite the underlying paper:

```bibtex
@article{semenov2025gradient,
  title   = {Gradient-Normalized Smoothness for Optimization with Approximate Hessians},
  author  = {Semenov, Andrei and Jaggi, Martin and Doikov, Nikita},
  journal = {arXiv preprint arXiv:2506.13710},
  year    = {2025}
}
```

## 10. License and acknowledgements

The reference paper is © its authors and reproduced under arXiv's
distribution licence in `docs/`. The LIBSVM datasets used in
`config/logsumexp_real_*.yaml` are © Chih-Chung Chang & Chih-Jen Lin
(see [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)).
The contents of this repository are released for research and
educational use.
