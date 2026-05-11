"""
Comprehensive visualization script for the MaFaulDa dataset.

This script provides various visualization tools to analyze and compare
normal vs faulty behavior in the rotor-bearing system data.

Data Structure:
- X data: (samples, timesteps, 9 features)
  - Features 0-3: Displacements (x_2, y_2, x_3, y_3)
  - Features 4-7: Velocities (x_2_dot, y_2_dot, x_3_dot, y_3_dot)
  - Feature 8: Angular velocity (omega)
- Y data: (samples, timesteps, 4 features)
  - Features 0-3: Accelerations (x_2_ddot, y_2_ddot, x_3_ddot, y_3_ddot)

Bearing locations:
- Underhang bearing: x_2, y_2 (radial, tangential)
- Overhang bearing: x_3, y_3 (radial, tangential)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr
import os
from typing import List, Tuple, Optional, Dict
import argparse
from pathlib import Path

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MaFaulDaVisualizer:
    """Comprehensive visualizer for MaFaulDa dataset."""
    
    def __init__(self, data_dir: str = "Data"):
        """
        Initialize the visualizer.
        
        Args:
            data_dir: Directory containing the .pth files
        """
        self.data_dir = Path(data_dir)
        self.available_datasets = self._get_available_datasets()
        
        # Feature names for better labeling
        self.x_feature_names = [
            'Underhang Radial Displacement', 'Underhang Tangential Displacement',
            'Overhang Radial Displacement', 'Overhang Tangential Displacement',
            'Underhang Radial Velocity', 'Underhang Tangential Velocity',
            'Overhang Radial Velocity', 'Overhang Tangential Velocity',
            'Angular Velocity'
        ]
        
        self.y_feature_names = [
            'Underhang Radial Acceleration', 'Underhang Tangential Acceleration',
            'Overhang Radial Acceleration', 'Overhang Tangential Acceleration'
        ]
        
        # Bearing names for clarity
        self.bearing_names = ['Underhang', 'Overhang']
        
    def _get_available_datasets(self) -> List[str]:
        """Get list of available datasets from .pth files."""
        datasets = []
        for file in self.data_dir.glob("X_*.pth"):
            dataset_name = file.stem[2:]  # Remove "X_" prefix
            if (self.data_dir / f"Y_{dataset_name}.pth").exists():
                datasets.append(dataset_name)
        return sorted(datasets)
    
    def load_dataset(self, dataset_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load X and Y data for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Tuple of (X_data, Y_data) tensors
        """
        x_path = self.data_dir / f"X_{dataset_name}.pth"
        y_path = self.data_dir / f"Y_{dataset_name}.pth"
        
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found")
        
        X = torch.load(x_path)
        Y = torch.load(y_path)
        
        print(f"Loaded {dataset_name}:")
        print(f"  X shape: {X.shape}")
        print(f"  Y shape: {Y.shape}")
        
        return X, Y
    
    def plot_time_series_comparison(self, normal_data: str, faulty_data: str, 
                                   sample_idx: int = 0, time_window: Optional[Tuple[int, int]] = None,
                                   features: Optional[List[int]] = None):
        """
        Compare time series between normal and faulty conditions.
        
        Args:
            normal_data: Name of normal dataset
            faulty_data: Name of faulty dataset
            sample_idx: Sample index to plot
            time_window: (start, end) time window to plot
            features: List of feature indices to plot (None for all)
        """
        X_norm, Y_norm = self.load_dataset(normal_data)
        X_fault, Y_fault = self.load_dataset(faulty_data)
        
        # Convert to numpy for plotting
        X_norm_np = X_norm[sample_idx].numpy()
        Y_norm_np = Y_norm[sample_idx].numpy()
        X_fault_np = X_fault[sample_idx].numpy()
        Y_fault_np = Y_fault[sample_idx].numpy()
        
        if time_window:
            start, end = time_window
            X_norm_np = X_norm_np[start:end]
            Y_norm_np = Y_norm_np[start:end]
            X_fault_np = X_fault_np[start:end]
            Y_fault_np = Y_fault_np[start:end]
        
        # Plot X features (displacements, velocities, omega)
        if features is None:
            features = list(range(X_norm_np.shape[1]))
        
        fig, axes = plt.subplots(len(features), 2, figsize=(15, 3*len(features)))
        if len(features) == 1:
            axes = axes.reshape(1, -1)
        
        time_axis = np.arange(X_norm_np.shape[0])
        
        for i, feat_idx in enumerate(features):
            # Normal condition
            axes[i, 0].plot(time_axis, X_norm_np[:, feat_idx], 'b-', label='Normal', linewidth=1)
            axes[i, 0].set_title(f'{self.x_feature_names[feat_idx]} - Normal')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Faulty condition
            axes[i, 1].plot(time_axis, X_fault_np[:, feat_idx], 'r-', label='Faulty', linewidth=1)
            axes[i, 1].set_title(f'{self.x_feature_names[feat_idx]} - Faulty')
            axes[i, 1].set_ylabel('Amplitude')
            axes[i, 1].grid(True, alpha=0.3)
        
        axes[-1, 0].set_xlabel('Time Steps')
        axes[-1, 1].set_xlabel('Time Steps')
        
        plt.tight_layout()
        plt.suptitle(f'Time Series Comparison: {normal_data} vs {faulty_data}', y=1.02)
        plt.show()
        
        # Plot Y features (accelerations)
        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
        axes = axes.flatten()
        
        for i in range(4):
            axes[i].plot(time_axis, Y_norm_np[:, i], 'b-', label='Normal', linewidth=1)
            axes[i].plot(time_axis, Y_fault_np[:, i], 'r-', label='Faulty', linewidth=1)
            axes[i].set_title(f'{self.y_feature_names[i]}')
            axes[i].set_ylabel('Acceleration')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Time Steps')
        axes[3].set_xlabel('Time Steps')
        
        plt.tight_layout()
        plt.suptitle(f'Acceleration Comparison: {normal_data} vs {faulty_data}', y=1.02)
        plt.show()
    
    def plot_phase_space(self, normal_data: str, faulty_data: str, 
                        sample_idx: int = 0, time_window: Optional[Tuple[int, int]] = None):
        """
        Create phase space plots comparing normal vs faulty conditions.
        
        Args:
            normal_data: Name of normal dataset
            faulty_data: Name of faulty dataset
            sample_idx: Sample index to plot
            time_window: (start, end) time window to plot
        """
        X_norm, _ = self.load_dataset(normal_data)
        X_fault, _ = self.load_dataset(faulty_data)
        
        # Convert to numpy
        X_norm_np = X_norm[sample_idx].numpy()
        X_fault_np = X_fault[sample_idx].numpy()
        
        if time_window:
            start, end = time_window
            X_norm_np = X_norm_np[start:end]
            X_fault_np = X_fault_np[start:end]
        
        # Create phase space plots for each bearing
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Underhang bearing: displacement vs velocity
        axes[0, 0].scatter(X_norm_np[:, 0], X_norm_np[:, 4], c='blue', alpha=0.6, s=1, label='Normal')
        axes[0, 0].scatter(X_fault_np[:, 0], X_fault_np[:, 4], c='red', alpha=0.6, s=1, label='Faulty')
        axes[0, 0].set_xlabel('Underhang Radial Displacement')
        axes[0, 0].set_ylabel('Underhang Radial Velocity')
        axes[0, 0].set_title('Underhang Radial Phase Space')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Underhang bearing: tangential
        axes[0, 1].scatter(X_norm_np[:, 1], X_norm_np[:, 5], c='blue', alpha=0.6, s=1, label='Normal')
        axes[0, 1].scatter(X_fault_np[:, 1], X_fault_np[:, 5], c='red', alpha=0.6, s=1, label='Faulty')
        axes[0, 1].set_xlabel('Underhang Tangential Displacement')
        axes[0, 1].set_ylabel('Underhang Tangential Velocity')
        axes[0, 1].set_title('Underhang Tangential Phase Space')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overhang bearing: radial
        axes[1, 0].scatter(X_norm_np[:, 2], X_norm_np[:, 6], c='blue', alpha=0.6, s=1, label='Normal')
        axes[1, 0].scatter(X_fault_np[:, 2], X_fault_np[:, 6], c='red', alpha=0.6, s=1, label='Faulty')
        axes[1, 0].set_xlabel('Overhang Radial Displacement')
        axes[1, 0].set_ylabel('Overhang Radial Velocity')
        axes[1, 0].set_title('Overhang Radial Phase Space')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Overhang bearing: tangential
        axes[1, 1].scatter(X_norm_np[:, 3], X_norm_np[:, 7], c='blue', alpha=0.6, s=1, label='Normal')
        axes[1, 1].scatter(X_fault_np[:, 3], X_fault_np[:, 7], c='red', alpha=0.6, s=1, label='Faulty')
        axes[1, 1].set_xlabel('Overhang Tangential Displacement')
        axes[1, 1].set_ylabel('Overhang Tangential Velocity')
        axes[1, 1].set_title('Overhang Tangential Phase Space')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'Phase Space Comparison: {normal_data} vs {faulty_data}', y=1.02)
        plt.show()
    
    def plot_frequency_spectrum(self, normal_data: str, faulty_data: str,
                               sample_idx: int = 0, features: Optional[List[int]] = None):
        """
        Compare frequency spectrums between normal and faulty conditions.
        
        Args:
            normal_data: Name of normal dataset
            faulty_data: Name of faulty dataset
            sample_idx: Sample index to plot
            features: List of feature indices to plot (None for all)
        """
        X_norm, Y_norm = self.load_dataset(normal_data)
        X_fault, Y_fault = self.load_dataset(faulty_data)
        
        # Convert to numpy
        X_norm_np = X_norm[sample_idx].numpy()
        Y_norm_np = Y_norm[sample_idx].numpy()
        X_fault_np = X_fault[sample_idx].numpy()
        Y_fault_np = Y_fault[sample_idx].numpy()
        
        # Combine X and Y for comprehensive analysis
        all_norm = np.concatenate([X_norm_np, Y_norm_np], axis=1)
        all_fault = np.concatenate([X_fault_np, Y_fault_np], axis=1)
        
        if features is None:
            features = list(range(all_norm.shape[1]))
        
        # Calculate sampling frequency (assuming 1 Hz for now)
        fs = 1.0  # Hz
        n_fft = min(2048, all_norm.shape[0])
        
        fig, axes = plt.subplots(len(features), 2, figsize=(15, 3*len(features)))
        if len(features) == 1:
            axes = axes.reshape(1, -1)
        
        all_feature_names = self.x_feature_names + self.y_feature_names
        
        for i, feat_idx in enumerate(features):
            # Calculate FFT
            f_norm, psd_norm = signal.welch(all_norm[:, feat_idx], fs=fs, nperseg=n_fft//4)
            f_fault, psd_fault = signal.welch(all_fault[:, feat_idx], fs=fs, nperseg=n_fft//4)
            
            # Normal condition
            axes[i, 0].semilogy(f_norm, psd_norm, 'b-', label='Normal', linewidth=1)
            axes[i, 0].set_title(f'{all_feature_names[feat_idx]} - Normal')
            axes[i, 0].set_ylabel('Power Spectral Density')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].legend()
            
            # Faulty condition
            axes[i, 1].semilogy(f_fault, psd_fault, 'r-', label='Faulty', linewidth=1)
            axes[i, 1].set_title(f'{all_feature_names[feat_idx]} - Faulty')
            axes[i, 1].set_ylabel('Power Spectral Density')
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].legend()
        
        axes[-1, 0].set_xlabel('Frequency (Hz)')
        axes[-1, 1].set_xlabel('Frequency (Hz)')
        
        plt.tight_layout()
        plt.suptitle(f'Frequency Spectrum Comparison: {normal_data} vs {faulty_data}', y=1.02)
        plt.show()
    
    def plot_statistical_comparison(self, normal_data: str, faulty_data: str,
                                   sample_idx: int = 0):
        """
        Compare statistical properties between normal and faulty conditions.
        
        Args:
            normal_data: Name of normal dataset
            faulty_data: Name of faulty dataset
            sample_idx: Sample index to analyze
        """
        X_norm, Y_norm = self.load_dataset(normal_data)
        X_fault, Y_fault = self.load_dataset(faulty_data)
        
        # Convert to numpy
        X_norm_np = X_norm[sample_idx].numpy()
        Y_norm_np = Y_norm[sample_idx].numpy()
        X_fault_np = X_fault[sample_idx].numpy()
        Y_fault_np = Y_fault[sample_idx].numpy()
        
        # Combine all features
        all_norm = np.concatenate([X_norm_np, Y_norm_np], axis=1)
        all_fault = np.concatenate([X_fault_np, Y_fault_np], axis=1)
        
        all_feature_names = self.x_feature_names + self.y_feature_names
        
        # Calculate statistics
        stats_norm = {
            'mean': np.mean(all_norm, axis=0),
            'std': np.std(all_norm, axis=0),
            'rms': np.sqrt(np.mean(all_norm**2, axis=0)),
            'peak': np.max(np.abs(all_norm), axis=0),
            'kurtosis': self._calculate_kurtosis(all_norm),
            'skewness': self._calculate_skewness(all_norm)
        }
        
        stats_fault = {
            'mean': np.mean(all_fault, axis=0),
            'std': np.std(all_fault, axis=0),
            'rms': np.sqrt(np.mean(all_fault**2, axis=0)),
            'peak': np.max(np.abs(all_fault), axis=0),
            'kurtosis': self._calculate_kurtosis(all_fault),
            'skewness': self._calculate_skewness(all_fault)
        }
        
        # Plot statistical comparisons
        stat_names = ['Mean', 'Std Dev', 'RMS', 'Peak', 'Kurtosis', 'Skewness']
        stat_keys = ['mean', 'std', 'rms', 'peak', 'kurtosis', 'skewness']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (stat_name, stat_key) in enumerate(zip(stat_names, stat_keys)):
            x_pos = np.arange(len(all_feature_names))
            width = 0.35
            
            axes[i].bar(x_pos - width/2, stats_norm[stat_key], width, label='Normal', alpha=0.8)
            axes[i].bar(x_pos + width/2, stats_fault[stat_key], width, label='Faulty', alpha=0.8)
            
            axes[i].set_xlabel('Features')
            axes[i].set_ylabel(stat_name)
            axes[i].set_title(f'{stat_name} Comparison')
            axes[i].legend()
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'Statistical Comparison: {normal_data} vs {faulty_data}', y=1.02)
        plt.show()
        
        # Print summary statistics
        print(f"\nStatistical Summary - {normal_data} vs {faulty_data}")
        print("=" * 60)
        for i, name in enumerate(all_feature_names):
            print(f"\n{name}:")
            for stat_name, stat_key in zip(stat_names, stat_keys):
                norm_val = stats_norm[stat_key][i]
                fault_val = stats_fault[stat_key][i]
                change = ((fault_val - norm_val) / norm_val) * 100
                print(f"  {stat_name}: {norm_val:.4f} → {fault_val:.4f} ({change:+.2f}%)")
    
    def _calculate_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Calculate kurtosis for each feature."""
        n = data.shape[0]
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        kurtosis = np.mean(((data - mean) / std) ** 4, axis=0) - 3
        return kurtosis
    
    def _calculate_skewness(self, data: np.ndarray) -> np.ndarray:
        """Calculate skewness for each feature."""
        n = data.shape[0]
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        skewness = np.mean(((data - mean) / std) ** 3, axis=0)
        return skewness
    
    def plot_correlation_matrix(self, normal_data: str, faulty_data: str,
                              sample_idx: int = 0):
        """
        Compare correlation matrices between normal and faulty conditions.
        
        Args:
            normal_data: Name of normal dataset
            faulty_data: Name of faulty dataset
            sample_idx: Sample index to analyze
        """
        X_norm, Y_norm = self.load_dataset(normal_data)
        X_fault, Y_fault = self.load_dataset(faulty_data)
        
        # Convert to numpy
        X_norm_np = X_norm[sample_idx].numpy()
        Y_norm_np = Y_norm[sample_idx].numpy()
        X_fault_np = X_fault[sample_idx].numpy()
        Y_fault_np = Y_fault[sample_idx].numpy()
        
        # Combine all features
        all_norm = np.concatenate([X_norm_np, Y_norm_np], axis=1)
        all_fault = np.concatenate([X_fault_np, Y_fault_np], axis=1)
        
        all_feature_names = self.x_feature_names + self.y_feature_names
        
        # Calculate correlation matrices
        corr_norm = np.corrcoef(all_norm.T)
        corr_fault = np.corrcoef(all_fault.T)
        
        # Plot correlation matrices
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Normal condition
        im1 = axes[0].imshow(corr_norm, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_title(f'Correlation Matrix - {normal_data}')
        axes[0].set_xticks(range(len(all_feature_names)))
        axes[0].set_yticks(range(len(all_feature_names)))
        axes[0].set_xticklabels(all_feature_names, rotation=45, ha='right')
        axes[0].set_yticklabels(all_feature_names)
        plt.colorbar(im1, ax=axes[0])
        
        # Faulty condition
        im2 = axes[1].imshow(corr_fault, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_title(f'Correlation Matrix - {faulty_data}')
        axes[1].set_xticks(range(len(all_feature_names)))
        axes[1].set_yticks(range(len(all_feature_names)))
        axes[1].set_xticklabels(all_feature_names, rotation=45, ha='right')
        axes[1].set_yticklabels(all_feature_names)
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
        
        # Plot difference matrix
        diff_matrix = corr_fault - corr_norm
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        im = ax.imshow(diff_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax.set_title(f'Correlation Difference: {faulty_data} - {normal_data}')
        ax.set_xticks(range(len(all_feature_names)))
        ax.set_yticks(range(len(all_feature_names)))
        ax.set_xticklabels(all_feature_names, rotation=45, ha='right')
        ax.set_yticklabels(all_feature_names)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()
    
    def plot_3d_trajectory(self, normal_data: str, faulty_data: str,
                           sample_idx: int = 0, time_window: Optional[Tuple[int, int]] = None):
        """
        Create 3D trajectory plots comparing normal vs faulty conditions.
        
        Args:
            normal_data: Name of normal dataset
            faulty_data: Name of faulty dataset
            sample_idx: Sample index to plot
            time_window: (start, end) time window to plot
        """
        X_norm, _ = self.load_dataset(normal_data)
        X_fault, _ = self.load_dataset(faulty_data)
        
        # Convert to numpy
        X_norm_np = X_norm[sample_idx].numpy()
        X_fault_np = X_fault[sample_idx].numpy()
        
        if time_window:
            start, end = time_window
            X_norm_np = X_norm_np[start:end]
            X_fault_np = X_fault_np[start:end]
        
        fig = plt.figure(figsize=(15, 10))
        
        # Underhang bearing 3D trajectory
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(X_norm_np[:, 0], X_norm_np[:, 1], X_norm_np[:, 4], 'b-', alpha=0.7, label='Normal')
        ax1.plot(X_fault_np[:, 0], X_fault_np[:, 1], X_fault_np[:, 4], 'r-', alpha=0.7, label='Faulty')
        ax1.set_xlabel('Radial Displacement')
        ax1.set_ylabel('Tangential Displacement')
        ax1.set_zlabel('Radial Velocity')
        ax1.set_title('Underhang Bearing 3D Trajectory')
        ax1.legend()
        
        # Overhang bearing 3D trajectory
        ax2 = fig.add_subplot(222, projection='3d')
        ax2.plot(X_norm_np[:, 2], X_norm_np[:, 3], X_norm_np[:, 6], 'b-', alpha=0.7, label='Normal')
        ax2.plot(X_fault_np[:, 2], X_fault_np[:, 3], X_fault_np[:, 6], 'r-', alpha=0.7, label='Faulty')
        ax2.set_xlabel('Radial Displacement')
        ax2.set_ylabel('Tangential Displacement')
        ax2.set_zlabel('Radial Velocity')
        ax2.set_title('Overhang Bearing 3D Trajectory')
        ax2.legend()
        
        # Combined trajectory (displacement space)
        ax3 = fig.add_subplot(223, projection='3d')
        ax3.plot(X_norm_np[:, 0], X_norm_np[:, 2], X_norm_np[:, 1], 'b-', alpha=0.7, label='Normal')
        ax3.plot(X_fault_np[:, 0], X_fault_np[:, 2], X_fault_np[:, 1], 'r-', alpha=0.7, label='Faulty')
        ax3.set_xlabel('Underhang Radial')
        ax3.set_ylabel('Overhang Radial')
        ax3.set_zlabel('Underhang Tangential')
        ax3.set_title('Combined Displacement Trajectory')
        ax3.legend()
        
        # Velocity space trajectory
        ax4 = fig.add_subplot(224, projection='3d')
        ax4.plot(X_norm_np[:, 4], X_norm_np[:, 6], X_norm_np[:, 5], 'b-', alpha=0.7, label='Normal')
        ax4.plot(X_fault_np[:, 4], X_fault_np[:, 6], X_fault_np[:, 5], 'r-', alpha=0.7, label='Faulty')
        ax4.set_xlabel('Underhang Radial Vel')
        ax4.set_ylabel('Overhang Radial Vel')
        ax4.set_zlabel('Underhang Tangential Vel')
        ax4.set_title('Combined Velocity Trajectory')
        ax4.legend()
        
        plt.tight_layout()
        plt.suptitle(f'3D Trajectory Comparison: {normal_data} vs {faulty_data}', y=1.02)
        plt.show()
    
    def comprehensive_analysis(self, normal_data: str, faulty_data: str,
                             sample_idx: int = 0, time_window: Optional[Tuple[int, int]] = None):
        """
        Perform comprehensive analysis comparing normal vs faulty conditions.
        
        Args:
            normal_data: Name of normal dataset
            faulty_data: Name of faulty dataset
            sample_idx: Sample index to analyze
            time_window: (start, end) time window to analyze
        """
        print(f"Comprehensive Analysis: {normal_data} vs {faulty_data}")
        print("=" * 60)
        
        # Load data
        X_norm, Y_norm = self.load_dataset(normal_data)
        X_fault, Y_fault = self.load_dataset(faulty_data)
        
        # Convert to numpy
        X_norm_np = X_norm[sample_idx].numpy()
        Y_norm_np = Y_norm[sample_idx].numpy()
        X_fault_np = X_fault[sample_idx].numpy()
        Y_fault_np = Y_fault[sample_idx].numpy()
        
        if time_window:
            start, end = time_window
            X_norm_np = X_norm_np[start:end]
            Y_norm_np = Y_norm_np[start:end]
            X_fault_np = X_fault_np[start:end]
            Y_fault_np = Y_fault_np[start:end]
        
        # Calculate key metrics
        all_norm = np.concatenate([X_norm_np, Y_norm_np], axis=1)
        all_fault = np.concatenate([X_fault_np, Y_fault_np], axis=1)
        
        all_feature_names = self.x_feature_names + self.y_feature_names
        
        print(f"\nSample {sample_idx} Analysis:")
        print(f"Time window: {time_window if time_window else 'Full signal'}")
        print(f"Signal length: {all_norm.shape[0]} points")
        
        # Energy analysis
        energy_norm = np.sum(all_norm**2, axis=0)
        energy_fault = np.sum(all_fault**2, axis=0)
        energy_change = ((energy_fault - energy_norm) / energy_norm) * 100
        
        print(f"\nEnergy Analysis:")
        for i, name in enumerate(all_feature_names):
            print(f"  {name}: {energy_norm[i]:.4f} → {energy_fault[i]:.4f} ({energy_change[i]:+.2f}%)")
        
        # Peak analysis
        peak_norm = np.max(np.abs(all_norm), axis=0)
        peak_fault = np.max(np.abs(all_fault), axis=0)
        peak_change = ((peak_fault - peak_norm) / peak_norm) * 100
        
        print(f"\nPeak Analysis:")
        for i, name in enumerate(all_feature_names):
            print(f"  {name}: {peak_norm[i]:.4f} → {peak_fault[i]:.4f} ({peak_change[i]:+.2f}%)")
        
        # Frequency domain analysis
        fs = 1.0  # Hz
        n_fft = min(2048, all_norm.shape[0])
        
        print(f"\nFrequency Domain Analysis:")
        for i, name in enumerate(all_feature_names):
            f_norm, psd_norm = signal.welch(all_norm[:, i], fs=fs, nperseg=n_fft//4)
            f_fault, psd_fault = signal.welch(all_fault[:, i], fs=fs, nperseg=n_fft//4)
            
            # Find dominant frequency
            peak_freq_norm = f_norm[np.argmax(psd_norm)]
            peak_freq_fault = f_fault[np.argmax(psd_fault)]
            
            print(f"  {name}:")
            print(f"    Dominant freq (normal): {peak_freq_norm:.3f} Hz")
            print(f"    Dominant freq (faulty): {peak_freq_fault:.3f} Hz")
            print(f"    Frequency shift: {peak_freq_fault - peak_freq_norm:+.3f} Hz")
        
        # Correlation analysis
        corr_norm = np.corrcoef(all_norm.T)
        corr_fault = np.corrcoef(all_fault.T)
        
        print(f"\nCorrelation Analysis:")
        print(f"  Average correlation (normal): {np.mean(np.abs(corr_norm)):.4f}")
        print(f"  Average correlation (faulty): {np.mean(np.abs(corr_fault)):.4f}")
        print(f"  Correlation change: {np.mean(np.abs(corr_fault)) - np.mean(np.abs(corr_norm)):+.4f}")
        
        # Generate all plots
        print(f"\nGenerating visualizations...")
        self.plot_time_series_comparison(normal_data, faulty_data, sample_idx, time_window)
        self.plot_phase_space(normal_data, faulty_data, sample_idx, time_window)
        self.plot_frequency_spectrum(normal_data, faulty_data, sample_idx)
        self.plot_statistical_comparison(normal_data, faulty_data, sample_idx)
        self.plot_correlation_matrix(normal_data, faulty_data, sample_idx)
        self.plot_3d_trajectory(normal_data, faulty_data, sample_idx, time_window)
        
        print(f"\nAnalysis complete!")


def main():
    """Main function to run the visualizer."""
    parser = argparse.ArgumentParser(description='MaFaulDa Dataset Visualizer')
    parser.add_argument('--normal', type=str, default='normal', 
                       help='Normal condition dataset name')
    parser.add_argument('--faulty', type=str, required=True,
                       help='Faulty condition dataset name')
    parser.add_argument('--sample', type=int, default=0,
                       help='Sample index to analyze')
    parser.add_argument('--start_time', type=int, default=None,
                       help='Start time for analysis window')
    parser.add_argument('--end_time', type=int, default=None,
                       help='End time for analysis window')
    parser.add_argument('--plot_type', type=str, default='all',
                       choices=['time_series', 'phase_space', 'frequency', 
                               'statistics', 'correlation', 'trajectory', 'all'],
                       help='Type of plot to generate')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = MaFaulDaVisualizer()
    
    # Print available datasets
    print("Available datasets:")
    for dataset in visualizer.available_datasets:
        print(f"  - {dataset}")
    
    # Set time window if provided
    time_window = None
    if args.start_time is not None and args.end_time is not None:
        time_window = (args.start_time, args.end_time)
    
    # Run analysis based on plot type
    if args.plot_type == 'all':
        visualizer.comprehensive_analysis(args.normal, args.faulty, args.sample, time_window)
    elif args.plot_type == 'time_series':
        visualizer.plot_time_series_comparison(args.normal, args.faulty, args.sample, time_window)
    elif args.plot_type == 'phase_space':
        visualizer.plot_phase_space(args.normal, args.faulty, args.sample, time_window)
    elif args.plot_type == 'frequency':
        visualizer.plot_frequency_spectrum(args.normal, args.faulty, args.sample)
    elif args.plot_type == 'statistics':
        visualizer.plot_statistical_comparison(args.normal, args.faulty, args.sample)
    elif args.plot_type == 'correlation':
        visualizer.plot_correlation_matrix(args.normal, args.faulty, args.sample)
    elif args.plot_type == 'trajectory':
        visualizer.plot_3d_trajectory(args.normal, args.faulty, args.sample, time_window)


if __name__ == "__main__":
    main() 