# AGNS — Accelerated Gradient-Normalized Smooth Newton

Experiments code for an *Optimization for ML* mini-project. It re-architects the official **Gradient-Normalized Smoothness (GNS)** implementation and adds **AGNS**: a Nesterov-momentum + adaptive-restart acceleration of the gradient-regularized Newton step. 

Built on [`epfml/grad-norm-smooth`](https://github.com/epfml/grad-norm-smooth) (Apache-2.0).

> Semenov, Jaggi, Doikov. *Gradient-Normalized Smoothness for Optimization with Approximate
> Hessians.* ICLR 2026. arXiv:2506.13710

## Setup

```bash
pip install -e ".[dev,notebooks]"
bash scripts/download_data.sh        # downloads a9a/w8a (mushrooms is vendored)
```

## Experiments

Each notebook reads as a self-contained argument for one research question — motivation, experiment, figures, quantitative reading, conclusion — and writes its figures/table to `results/`:

| Notebook | Question | Evidence |
|---|---|---|
| `00_sanity.ipynb` | every method decreases `f` | convergence smoke test |
| `01_rq1_phenomenon.ipynb` | does explicit momentum accelerate GNS? | softmax + non-convex families + conditioning sweep |
| `02_rq2_restart.ipynb` | how much does the restart matter? | ablation + oscillation metrics + Rosenbrock stress + momentum sweep |
| `03_rq3_vs_super_universal.ipynb` | explicit vs implicit acceleration | head-to-head + Dolan–Moré performance profile + real-data table |
| `04_rq4_inexact.ipynb` | does acceleration survive an approximate Hessian? | Weighted Gauss-Newton + other approximation families |

Run a notebook top-to-bottom . Set `QUICK = True` in the first cell for a fast, small-problem pass. All hyperparameters live in `config/`. Because the methods do a variable number of linear solves per iteration, convergence is plotted against **matrix inverses** (and gradient calls) as well as iterations, with median +
inter-quartile bands over **20 seeds**; aggregate comparisons use Dolan–Moré performance profiles and bootstrap CIs / paired Wilcoxon / TOST equivalence statistics. Each seed draws a *fresh problem* (a new random instance, or a new start for parametric/real problems — see each oracle's `vary` key in `config/oracles/`).

All figures and regenerate from a clean notebook run.