# Hybrid PINNs for Anomaly Detection in Ring Bearings

This repository contains the code and resources for the IECON paper: **"Physics-Informed Residual-Based Anomaly Detection and Open-Set Recognition System: A Case Study on Ring Bearings"**.

## Overview

This work presents a three-stage diagnostic framework for bearing fault detection and classification:

1. **Physics-Based Residual (PBR) Generation** - Using Direct PINNs to generate residuals that capture deviations from expected physical behavior
2. **EVT Anomaly Detection** - Using Extreme Value Theory to detect anomalies from the residual distributions
3. **Siamese Neural Network (SNN) Open-Set Recognition** - Using metric learning to classify fault types and enable open-set recognition of unknown faults

Additionally, an **Edge Computing** implementation provides a C++ benchmark for embedded deployment on resource-constrained platforms.

## Repository Structure

```
josafat/
├── direct_analysis/          # Stage 1: PBR Residual Generation
├── anomaly_detection/        # Stage 2: EVT Anomaly Detection
├── siamese_analysis/         # Stage 3: SNN Open-Set Recognition
├── embedded_analysis/        # Edge Computing Implementation
├── Data/                     # Data loading utilities
├── Models/                   # PINN model definitions
├── utils/                    # Utility functions
├── common/                   # Shared utilities
├── diagrams/                 # System diagrams
├── main_IECON.tex            # Paper manuscript
├── get_direct_pinn_residuals.py    # Script to generate PBR residuals
└── test_direct_pinn.py             # Evaluation script for Direct PINN
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hybrid_pinns.git
cd hybrid_pinns/josafat

# Install dependencies (using poetry)
poetry install

# Or using pip
pip install -r requirements.txt
```

## Quick Start

### Stage 1: Generate Physics-Based Residuals

The Direct PINN analysis module generates residuals that capture deviations from expected physical behavior:

```bash
# Run the complete Direct PINN pipeline
python -m direct_analysis.cli run --output-dir results/direct_analysis

# Or use the provided script
python get_direct_pinn_residuals.py --output-dir results/residuals
```

**Output**: Residuals file (`direct_pinn_residuals.pth`) containing predictions and physics-based residuals for all datasets.

For more details, see [`direct_analysis/README.md`](./direct_analysis/README.md).

### Stage 2: Anomaly Detection with EVT

The anomaly detection module uses Extreme Value Theory to detect anomalies from residual distributions:

```bash
# Run anomaly detection on generated residuals
python -m anomaly_detection.cli \
    --residuals results/residuals/direct_pinn_residuals.pth \
    --source direct \
    --output anomaly_detection_results.json \
    --methods evt
```

**Output**: Anomaly detection metrics (precision, recall, F1) and threshold parameters.

For more details, see [`anomaly_detection/README.md`](./anomaly_detection/README.md).

### Stage 3: Open-Set Recognition with Siamese Neural Network

The Siamese network module performs open-set recognition and fault classification:

```bash
# Run the complete Siamese analysis pipeline
python siamese_analysis/cli.py \
    --residuals results/residuals/direct_pinn_residuals.pth \
    --source direct \
    --output-dir results/siamese \
    --num-triplets 10000 \
    --evaluate-baselines
```

**Output**: 
- Trained Siamese model (`siamese_model.pt`)
- Classification reports for all hierarchy levels
- t-SNE/UMAP visualizations
- Baseline comparison results

For more details, see [`siamese_analysis/README.md`](./siamese_analysis/README.md).

### Edge Computing Benchmark

The embedded analysis module provides a C++ implementation for benchmarking on embedded platforms:

```bash
cd embedded_analysis

# Build release version
make

# Run benchmark
make run
```

**Output**: Average execution time and standard deviation for the complete detection pipeline.

For more details, see [`embedded_analysis/README.md`](./embedded_analysis/README.md).

## Complete Pipeline Example

To run the complete three-stage diagnostic pipeline:

```bash
# 1. Generate residuals
python -m direct_analysis.cli run --output-dir results/stage1

# 2. Detect anomalies
python -m anomaly_detection.cli \
    --residuals results/stage1/models/direct_pinn_residuals.pth \
    --source direct \
    --output results/stage2/anomaly_results.json

# 3. Open-set recognition
python siamese_analysis/cli.py \
    --residuals results/stage1/models/direct_pinn_residuals.pth \
    --source direct \
    --output-dir results/stage3 \
    --evaluate-baselines
```

## Evaluation Scripts

- `test_direct_pinn.py` - Evaluate Direct PINN model performance
- `test_direct_pinn.py` can be used to validate the trained model on test datasets

## Data Format

The pipeline expects data in `.pth` (PyTorch) format with the following structure:

```python
{
    'normal': {
        'X': tensor([...]),  # Input features
        'y': tensor([...])   # Target outputs
    },
    'fault_type_1': {...},
    'fault_type_2': {...},
    ...
}
```

## Paper

DOI: https://doi.org/10.1109/IECON58223.2025.11221472

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on the repository or contact the authors.
