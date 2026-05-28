# Experiments

Run scripts from the repository root with `python -m` so package imports resolve
consistently.

Generated artifacts are written under `results/` by default. Local datasets are
expected under `data/`.

## Layout

- `common/`: Shared experiment helpers.
- `synthetic/`: Synthetic mixture, hyperparameter, and path diagnostics.
- `molecular/`: MoleculeNet/QM experiments.
- `real_data/`: arXiv and MRI experiments.

## Entry Points

| Module | Purpose | Notes |
| --- | --- | --- |
| `experiments.synthetic.run_mixture_predictors` | Main synthetic mixture benchmark with configurable predictor and latent representation settings. | Requires external CondConf and PCP/RLCP modules for baseline runs. |
| `experiments.synthetic.run_mixture_original` | Original synthetic mixture benchmark. | Kept for reproducibility of the first experiment script. |
| `experiments.synthetic.run_hyperparameter_sweep` | Hyperparameter sensitivity experiments. | Writes mixture sweep outputs. |
| `experiments.synthetic.run_elbow_diagnostics` | Path elbow-size diagnostics. | Writes diagnostic figures/tables. |
| `experiments.molecular.run_esol` | MoleculeNet ESOL experiment. | Requires PyTorch Geometric/RDKit and external baselines. |
| `experiments.molecular.run_qm7b` | QM7b molecular property experiment. | Requires PyTorch Geometric/RDKit and external baselines. |
| `experiments.molecular.run_qm9` | QM9 molecular property experiment. | Requires PyTorch Geometric/RDKit and external baselines. |
| `experiments.real_data.run_arxiv` | arXiv embedding experiment. | Expects local arXiv feature/label files. |
| `experiments.real_data.run_mri` | MRI feature-score experiment. | Expects prepared MRI tensors/data. |

## Data And Output Paths

| Experiment | Input path | Output path |
| --- | --- | --- |
| Synthetic mixture predictors | Simulated in script | `results/synthetic/mixture_predictors` |
| Synthetic mixture original | Simulated in script | `results/synthetic/mixture_original` |
| Hyperparameter sweep | Simulated in script | `results/synthetic/hyperparameter_sweep` |
| Elbow diagnostics | Simulated in script | `results/synthetic/elbow_diagnostics` |
| ESOL | Downloaded/cached by PyTorch Geometric under `data/MoleculeNet` | `results/molecular/esol` |
| QM7b | Downloaded/cached by PyTorch Geometric under `data/QM7b` | `results/molecular/qm7b` |
| QM9 | Downloaded/cached by PyTorch Geometric under `data/QM9` | `results/molecular/qm9` |
| arXiv | `data/arxiv/X_arxiv.csv`, `data/arxiv/y_arxiv.csv`, `data/arxiv/W_arxiv.csv` | `results/real_data/arxiv` |
| MRI | `data/mri/*.pt`, `data/mri/mri_model.pth` | `results/real_data/mri` |

## External Baselines

CondConf and PCP/RLCP are external repositories and are not included here.
Install them separately before running scripts that compare against those
baselines. The scripts expect:

- `from conditionalconformal import CondConf`
- `from PCP.utils import PCP, RLCP`

Example:

```bash
python -m experiments.synthetic.run_mixture_predictors \
  --experiment predictor \
  --predictor rf \
  --n 1000 \
  --ntrials 10 \
  --outdir results/synthetic/mixture_predictors
```

Large generated outputs, local datasets, and model artifacts should be written
outside version control, for example under `results/`.
