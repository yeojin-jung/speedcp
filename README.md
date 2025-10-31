# SpeedCP

[![PyPI version](https://badge.fury.io/py/speedcp.svg)](https://pypi.org/project/speedcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SpeedCP is a Python package for our paper  **SpeedCP: Kernel-based conditional conformal prediction** [link](https://www.arxiv.org/abs/2509.24100).  
It provides efficient algorithms for constructing prediction sets with **conditional coverage guarantees**.

---

## 🚀 Installation

Once released on PyPI:

```bash
pip install -r requirements.txt
pip install speedcp
```
Or from source:
```bash
pip install -r requirements.txt
git clone https://github.com/yeojin-jung/speedcp.git
cd speedcp
pip install -e .
```

## 📖 Usage
```bash
import speedcp
model = speedcp.SpeedCP(alpha=0.1)
cutoffs, _ = model.fit(X_cal, Phi_cal, S_cal, X_test, Phi_test)
covers = (S_test <= cutoffs).astype(int)
```

## 📂 Repository Structure
- speedcp/ – core implementation
- experiments/ – scripts & notebooks to reproduce results from the paper
- requirements.txt – exact dependencies for experiments

## 📜 Citation

If you use this code, please cite:
```bash
@inproceedings{your2025speedcp,
  title={SpeedCP: Fast Kernel-based Conditional Conformal Prediction},
  author={Yeo Jin Jung, Yating Liu, Zixuan Wu, Sowon Jeong, Claire Donnat},
  year={2025}
}
```
## 🔗 Links
- 📦 PyPI: 
- 💻 GitHub: github.com/yeojin-jung/speedcp
- 📑 Paper: https://www.arxiv.org/abs/2509.24100
