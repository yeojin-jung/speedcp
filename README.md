# 📘 SpeedCP
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the reference implementation and experiment code for
**SpeedCP: Fast Kernel-based Conditional Conformal Prediction**, accepted at
ICML.

SpeedCP is a kernel-based conditional conformal prediction method with fast
path-tracing algorithms for hyperparameter selection and test-time cutoff
computation. The repository includes the core implementation, baselines used in the paper, experiment entrypoints, and notebooks for result analysis.

---
## ✨ Key Features
- **ML Evaluation**: Provides rigorous, finite-sample conditional coverage guarantees for prediction sets, enabling uncertainty quantification for individual predictions of any ML model.
- **Fast Path-Tracing Optimization**: Dramatically accelerates the calibration process by deriving piecewise linear updates for parameters. This allows the solver to trace the entire solution path across different hyperparameter values efficiently. 
- **Simultaneous Hyperparameter Tuning**: Efficiently searches the joint space of the kernel bandwidth ($\gamma$) and the smoothness regularizer ($\lambda$). This ensures the selection of the tightest possible cutoffs for prediction sets without the cost of exhaustive grid searches.
- **Dual-Path Speedup**:
  - $\lambda$-path: Computes the full regularization path to find the optimal smoothness control.
  - $S$-path: Derives the piecewise linear solution of parameters relative to the score $S$ to calculate final prediction intervals.
- **Scalability via Latent Embeddings**: Integrated with low-rank latent embeddings to maintain conditional validity even in high-dimensional feature spaces.
- **Proven Performance**: Achieves a 40-fold speedup and improves interval efficiency (length) by 30% compared to existing RKHS-based conformal frameworks.


## 📦 Installation

The core SpeedCP solver requires `numpy`, `scipy`, `scikit-learn`, and
`cvxopt`. Additional experiment dependencies are listed in `requirements.txt`.
```bash
git clone https://github.com/yeojin-jung/speedcp.git
cd speedcp
pip install -r requirements.txt
pip install -e .
```
After installation, import SpeedCP with:

```python
from speedcp import SpeedCP
```

## 📂 Repository Structure

- `speedcp/`: Core SpeedCP package and path-tracing implementation.
- `experiments/`: Reproducibility scripts for synthetic, molecule graphs, arXiv, MRI, hyperparameter, and elbow experiments.
- `notebooks/`: Analysis notebooks for figures and result summaries.

## 🧠 SpeedCP Usage
SpeedCP uses a fast kernel-based pipeline that searches for optimal kernel smoothness hyperparameters ($\gamma, \lambda$) before computing the conformal cutoffs for test points.

### 1. Initialize the model

The `SpeedCP` object handles the path-following solvers and hyperparameter tuning.

```python
from speedcp import SpeedCP
import numpy as np

model = SpeedCP(
    alpha=0.1,              # Quantile level
    gamma_grid=np.logspace(0, 2, 30), # Search grid for kernel bandwidth
    use_cv=True,            # Enable cross-validation for gamma/lambda
    randomize=True,         # Enables local guarantees via randomization
    start_side='left',      # Direction for the S-path solver
    ridge=1e-4              # L2 regularizer for QP subproblems
)
```

| Parameter   | Description                                                      |
|-------------|------------------------------------------------------------------|
| `max_steps` | Maximum steps for the path solvers.                              |
| `eps/tol`   | Numerical tolerances for the solver and initialization.          |
| `thres`     | Threshold for early stopping in the $\lambda$-path.             |
| `gamma`     | If provided, skips grid search and uses this specific bandwidth. |

### 2. Fit and predict
The `.fit()` method executes the 3-step pipeline: (1) Grid search over $\gamma$ where for each $\gamma$, we run the $\lambda$-path solver  (2) Optimal hyperparameter pair selection with validation score (3) Final cutoff computation using the $S$-path.

Inputs to `fit`:

- `X_cal`: calibration features used by the kernel, shape `(n_cal, p)`.
- `Phi_cal`: calibration linear/basis features for conditional constraints,
  shape `(n_cal, d)`.
- `scores_cal`: calibration nonconformity scores, shape `(n_cal,)`.
- `X_test`: test features used by the kernel, shape `(n_test, p)`.
- `Phi_test`: test linear/basis features, shape `(n_test, d)`.
- `random_state`: random seed for the optional calibration/validation split.

```bash
cutoffs, _ = model.fit(
    X_cal, 
    Phi_calib, 
    scores_calib.ravel(),
    X_test,  
    Phi_test, 
    random_state=42,
)

# Calculate coverage results
covers = (scores_test <= cutoffs).astype(int)
print(f"Empirical Conditional Coverage: {covers.mean():.3f}")
```

The first return value, `cutoffs`, has shape `(n_test,)`. The second return
value stores the initial S-path value for each test point and is mainly useful
for diagnostics.

Useful fitted attributes:

- `model.gamma`: selected RBF bandwidth.
- `model.lam`: selected regularization parameter.
- `model.time_tune`: time spent selecting `gamma` and `lambda`.
- `model.elbow_sizes`: lambda-path elbow sizes for diagnostics.
- `model.s_path_elbow_sizes`: S-path elbow sizes for diagnostics.

Common options:

- Set `gamma=<float>` to skip bandwidth search.
- Set `gamma_grid=np.logspace(...)` to control the bandwidth grid.
- Set `use_cv=True` to select bandwidth by cross-validation.
- Set `use_split=True` to reserve half of calibration data for tuning.
- Set `randomize=False` for deterministic non-randomized cutoffs.

## 🧪 Experiments and Baselines
The synthetic experiments compare SpeedCP against several state-of-the-art baselines in the conformal prediction literature:

- **SplitCP**: Standard split conformal prediction.

- **CondConf**: Conformal prediction with conditional guarantees [[Gibbs et al., 2024]](https://github.com/jjcherian/conditional-conformal).

- **PCP**: Posterior Conformal Prediction [[Zhang and Candes, 2024]](https://github.com/yaozhang24/pcp).

- **RLCP**: Conformal prediction with local weights [[Hore and Barber, 2024]](https://github.com/rohanhore/RLCP).

CondConf and PCP/RLCP are not vendored in this repository. To reproduce the
baseline runs, install those authors' repositories separately and make their
modules importable as `conditionalconformal` and `PCP`.

Example setup:

```bash
git clone https://github.com/jjcherian/conditional-conformal.git ../conditional-conformal
git clone https://github.com/yaozhang24/pcp.git ../pcp

export PYTHONPATH="$PWD:../conditional-conformal:../pcp:$PYTHONPATH"
```

If those external repositories provide installable packages in your environment,
an editable install is also fine:

```bash
pip install -e ../conditional-conformal
pip install -e ../pcp
```

Baseline scripts import:

- `from conditionalconformal import CondConf`
- `from PCP.utils import PCP, RLCP`

### Running the Benchmark
Run scripts from the repository root with the module form so imports resolve
consistently:

```bash
python -m experiments.synthetic.run_mixture_predictors \
  --experiment predictor \
  --predictor rf \
  --n 1000 \
  --ntrials 10 \
  --outdir results/synthetic/mixture_predictors
```

Other experiment entrypoints:

- `python -m experiments.synthetic.run_mixture_original`
- `python -m experiments.synthetic.run_hyperparameter_sweep`
- `python -m experiments.synthetic.run_elbow_diagnostics`
- `python -m experiments.molecular.run_esol`
- `python -m experiments.molecular.run_qm7b`
- `python -m experiments.molecular.run_qm9`
- `python -m experiments.real_data.run_arxiv`
- `python -m experiments.real_data.run_mri`

Outputs are written under `results/` by default. Local datasets are expected
under `data/`, for example `data/arxiv/` and `data/mri/`.

Some experiments require external datasets or heavier optional dependencies in
`requirements.txt`, including PyTorch Geometric and RDKit for molecular
experiments.

## 📜 Citation

If you use this code, please cite:
```bibtex
@inproceedings{liu2026speedcp,
  title = {SpeedCP: Fast Kernel-based Conditional Conformal Prediction},
  author = {Liu, Yating and Jung, Yeo Jin and Wu, Zixuan and Jeong, Sowon and Donnat, Claire},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year = {2026}
}
```
## 🔗 Links
- 💻 GitHub: https://github.com/yeojin-jung/speedcp
- 📑 Paper: https://arxiv.org/abs/2509.24100
