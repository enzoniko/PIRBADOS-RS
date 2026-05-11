# IECON Paper Repository Restructuring Plan

This plan outlines the exact steps to clean and reorganize the repository so it exclusively contains the code and resources relevant to the IECON paper: *"Physics-Informed Residual-Based Anomaly Detection and Open-Set Recognition System: A Case Study on Ring Bearings"*.

## 1. Preserved Components (What Stays)

*   **Residual Generation (PBR)**: 
    *   `direct_analysis/` directory (Confirmed as the correct module).
    *   `get_direct_pinn_residuals.py` (Script to generate PBR residuals).
    *   `test_direct_pinn.py` (Evaluation script for the direct PINN).
*   **EVT Anomaly Detection**: 
    *   `anomaly_detection/` directory (The modularized version of the anomaly detection stage).
*   **Siamese Neural Network (SNN)**: 
    *   `siamese_analysis_v3/` directory. *(Reasoning: After analyzing the codebase, `v3` is the only version that fully implements the advanced feature extraction (Time-domain, Fourier, Wavelet db4), hierarchical evaluation, and PCA + KNN baselines explicitly described in the paper's methodology. We will rename this to `siamese_analysis/`.)*
*   **Edge Computing**:
    *   `embedded_analysis/` directory.
*   **Core Data/Utils**:
    *   `Data/`, `Models/`, `utils/`, `common/`.
*   **Documentation**:
    *   `main_IECON.tex`.

## 2. Directories to Safely Discard (What Goes)

*   **Legacy Training Methods**: `training_scripts/`, `backup_training_scripts/` (contains the complex 8 adaptive loss schemes not used in this paper).
*   **Unused Methodologies**: `hybrid_analysis/`, `data_driven_analysis/`.
*   **Legacy Siamese Versions**: `siamese_analysis/` (old), `siamese_analysis_v0/`, `siamese_analysis_v1/`, `siamese_analysis_v2/`.
*   **Results & Checkpoints**: `Saved Models/` and all `siamese_results_*` directories (as no pre-trained models or results need to be kept).
*   **Unrelated Projects**: `PAF/`.
*   **Python Caches**: `__pycache__` and `*.egg-info`.

## 3. Specific Files to Safely Discard (What Goes)

*   **Hybrid/RNN Scripts**: `get_hybrid_rnn_pinn_residuals.py`, `test_rnn_pinn.py`, `plot_hybrid_rnn_pinn_metrics.py`.
*   **Root Legacy Anomaly/Clustering Scripts**: `anomaly_detection_methods*.py`, `test_anomaly_detection_methods_v*.py`, `evaluate_residuals_clustering*.py`.
*   **One-off Analysis Scripts**: `TSSA.py`, `osr_umap_metrics_automation.py`, `analyze_overnight_results.py`, `overnight_siamese_analysis.py`, `run_complete_analysis.py`, `generate_all_plots.py`, etc.
*   **Redundant Root LaTeX Fragments**: `residual_analysis_all_models.tex`, `normal_residual_statistics_table.tex`, `residual_analysis_brdr.tex`, etc. (Everything except `main_IECON.tex`).
*   **Heavy Cache Files**: `dtw_distance_matrix.npy`, `dtw_distance_matrix_saved.npy`.

## 4. Reorganization & Documentation Steps

1.  **Execute the Purge**: Delete all the directories and files listed in sections 2 and 3.
2.  **Rename**: Move the contents of `siamese_analysis_v3/` into `siamese_analysis/` (replacing the deleted old version).
3.  **Generate `README.md`**: Write a completely fresh `README.md` tailored strictly to the three-stage diagnostic framework (PBR Residuals $\rightarrow$ EVT Anomaly Detection $\rightarrow$ SNN Open-Set Recognition), mapping out how to run the isolated pipeline.

