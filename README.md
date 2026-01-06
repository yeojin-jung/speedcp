# 📘 SpeedCP
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SpeedCP is a Python package for efficient, kernel-based conditional conformal prediction. It provides fast algorithms to construct prediction sets with rigorous conditional coverage guarantees, as introduced in our paper: **SpeedCP: Fast Kernel-based Conditional Conformal Prediction** [[link]](https://arxiv.org/abs/2509.24100).

This repository provides:

- A cleaned and installable **Python package** `speedcp`
- Code for **synthetic experiments** and a tutorial on downstream analysis.

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

Install the dependencies and the package in editable mode:
```bash
git clone https://github.com/yeojin-jung/speedcp.git
cd speedcp
pip install -r requirements.txt
pip install -e .
```

## 📂 Repository Structure

- `speedcp/`: Core Python library containing the SpeedCP implementation.

- `codes/`: Scripts for running large-scale synthetic experiments.

## 🧠 Basic Usage
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

```bash
# W: Features (could be raw, or low-dimensional projections)
# Phi: Additional covariates
# res: Non-conformity scores

cutoffs, _ = model.fit(
    W_calib, 
    Phi_calib, 
    res_calib.ravel(),
    W_test,  
    Phi_test, 
    seed=42
)

# Calculate coverage results
covers = (res_test <= cutoffs).astype(int)
print(f"Empirical Conditional Coverage: {covers.mean():.3f}")
```

## 🧪 Experiments and Baselines
The synthetic experiments compare SpeedCP against several state-of-the-art baselines in the conformal prediction literature:

- **SplitCP**: Standard split conformal prediction.

- **CondConf**: Conformal prediction with conditional guarantees [[Gibbs et al., 2024]](https://github.com/jjcherian/conditional-conformal).

- **PCP**: Posterior Conformal Prediction [[Zhang and Candes, 2024]](https://github.com/yaozhang24/pcp).

- **RLCP**: Conformal prediction with local weights [[Hore and Barber, 2024]](https://github.com/rohanhore/RLCP).

### Running the Benchmark
You can run the synthetic experiments using the CLI:
```bash
python3 codes/synthetic_data.py \
        --predictor rf \
        --n 1000 \
        --ntrials 10 \
        --outdir "../results/mixture"
```
## 📜 Citation

If you use this code, please cite:
```bash
@inproceedings{jung2025speedcp,
  title={SpeedCP: Fast Kernel-based Conditional Conformal Prediction},
  author={Jung, Yeo Jin and Liu, Yating and Wu, Zixuan and Jeong, Sowon and Donnat, Claire},
  year={2025}
}
```
## 🔗 Links
- 💻 GitHub: https://github.com/yeojin-jung/speedcp
- 📑 Paper: https://arxiv.org/abs/2509.24100
