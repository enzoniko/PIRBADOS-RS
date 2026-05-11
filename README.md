# PIRBADOS-RS

This repository contains the code and resources for the IECON paper:
**"Physics-Informed Residual-Based Anomaly Detection and Open-Set Recognition System: A Case Study on Ring Bearings"**.

The project is organized around a three-stage workflow:

1. Direct PINN residual generation (`direct_analysis/`)
2. Residual-based anomaly detection (`anomaly_detection/`)
3. Siamese/open-set analysis (`siamese_analysis/`)

It also includes an embedded C++ benchmark (`embedded_analysis/`) and shared data/model utilities (`Data/`, `Models/`, `utils/`, `common/`).

## Repository Layout

```
PIRBADOS-RS/
├── direct_analysis/            # Stage 1: Direct PINN residual workflow
├── anomaly_detection/          # Stage 2: EVT and Isolation Forest detectors
├── siamese_analysis/           # Stage 3: Siamese/open-set workflow
├── embedded_analysis/          # C++ embedded benchmark
├── Data/                       # Data loading and preprocessing utilities
├── Models/                     # PINN model definitions
├── common/                     # Shared visualization/helpers
├── utils/                      # Additional utilities
├── get_direct_pinn_residuals.py
├── test_direct_pinn.py
└── main_IECON.tex
```

## Installation

This repository currently does not ship a `requirements.txt`, `pyproject.toml`, or Poetry lockfile.
Install dependencies directly in your environment (venv/conda) before running the pipelines.

Example setup (pip + venv):

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install torch numpy scipy scikit-learn matplotlib pandas seaborn h5py tqdm pywavelets umap-learn fastdtw dtw-python torchsummary ssqueezepy

# Optional: advanced feature extraction in some modules
pip install tsfresh
```

## Quick Start

### Stage 1: Direct PINN Residuals

Primary code paths are under `direct_analysis/` and helper scripts at repository root.

- Direct analysis module documentation: [`direct_analysis/README.md`](./direct_analysis/README.md)
- Legacy helper scripts: `get_direct_pinn_residuals.py`, `test_direct_pinn.py`

To execute the direct analysis entry point currently present in the repository:

```bash
python -m direct_analysis.cli
```

### Stage 2: Anomaly Detection (EVT / Isolation Forest)

Run anomaly detection from residual files:

```bash
python -m anomaly_detection.cli \
  --residuals path/to/residuals.pth \
  --source direct \
  --methods evt \
  --output anomaly_detection_results.json
```

Module documentation: [`anomaly_detection/README.md`](./anomaly_detection/README.md)

### Stage 3: Siamese Open-Set Analysis

The Siamese module is in `siamese_analysis/`.

```bash
python siamese_analysis/cli.py --help
```

Module documentation: [`siamese_analysis/README.md`](./siamese_analysis/README.md)

### Embedded Benchmark (C++)

```bash
cd embedded_analysis
make
make run
```

Module documentation: [`embedded_analysis/README.md`](./embedded_analysis/README.md)

## Data Files

Most training/evaluation scripts use `.pth` files under `Data/` (for example `X_normal.pth`, `Y_normal.pth` and fault variants).

Residual pipelines generally produce dictionary-based `.pth` artifacts keyed by data type/fault label.

## Paper

DOI: https://doi.org/10.1109/IECON58223.2025.11221472

## License

MIT License. See `LICENSE`.
