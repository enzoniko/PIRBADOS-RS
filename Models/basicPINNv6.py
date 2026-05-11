import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import argparse

"""
Physics-Informed Neural Network (PINN) for system identification of a dynamical system.

This version supports:
1. Using normalized data in residual computation to improve training stability
2. Multiple scaling methods (min-max and standard scaling)
3. Command line arguments to select normalization options:
   - --normalized: Use normalized data in residuals instead of denormalizing
   - --scaling: Choose between 'min_max' or 'standard' scaling

The motivation for using normalized data in residuals:
Since our equations are linear, we can apply scaling to the data before sending them 
to the equations, and they would still be recoverable. This improves convergence during
training and reduces problems with coefficients and exploding gradients caused by the 
ill-posed nature of the PINNs loss landscape.
"""

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from direct_analysis.data_utils import prepare_data, normalize_data

class PINN(nn.Module):

    def __init__(self):
        super(PINN, self).__init__()

        # The PINN will receive an input containing 10 values:
        # x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, omega, t (the velocities and displacements at time t)

        # The PINN will output 4 values:
        # x2_ddot, y2_ddot, x3_ddot, y3_ddot (the accelerations at time t)

        # There will be a network for estimating unmeasured parameters (NNforUnmeasured)
        # This network can have any architecture, but it will receive 15 inputs (the inputs + some learnable parameters) and output 4 values

        # There will be a network for estimating the accelerations (NNforAccelerations)
        # This network can have any architecture, but it will receive 9 inputs and output 4 values

        # Besides the neural networks, the PINN will have learnable parameters:
        # M1, M2, M3, D1, D2, D3, K1, K2, E1 (masses, damping coefficients, spring constants, E1)
        # Which are invariant parameters of the system
        
        # Define the neural network architecture for estimating the sum of the 
        # unknown unmmeasured parameters values:
        # fA: [M_{1}\ddot{x_1} + D_{1}\dot{x_1} + K_{1}x_1 + K_{2}x_1 - F_{xs}]
        # fB: [M_{1}\ddot{y_1} + D_{1}\dot{y_1} + K_{1}y_1 + K_{2}y_1 - F_{ys}]
        # fC: [F_{x}(x_3, y_3, \dot{x_3}, \dot{y_3}) - \frac{K_2}{K_1}F_{x}(x_2, y_2, \dot{x_2}, \dot{y_2})]
        # fD: [F_{y}(x_3, y_3, \dot{x_3}, \dot{y_3}) - \frac{K_2}{K_1}F_{y}(x_2, y_2, \dot{x_2}, \dot{y_2})]

        # Use Xavier initialization for better stability
        self.NNforUnmeasured = nn.Sequential(
            nn.Linear(15, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4)
        )
        
        # Apply Xavier initialization
        for layer in self.NNforUnmeasured:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Define the neural network architecture for estimating the accelerations:
        # x2_ddot, x3_ddot, y2_ddot, y3_ddot
        self.NNforAccelerations = nn.Sequential(
            nn.Linear(10, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4)
        )
        
        # Apply Xavier initialization
        for layer in self.NNforAccelerations:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Initialize learnable parameters with more realistic values
        # For masses, springs, etc., start with positive values in the expected range
        self.M1 = nn.Parameter(torch.tensor(10.0))  # Start with higher positive values
        self.M2 = nn.Parameter(torch.tensor(10.0))
        self.M3 = nn.Parameter(torch.tensor(11.0))
        self.D1 = nn.Parameter(torch.tensor(10.0))
        self.D2 = nn.Parameter(torch.tensor(10.0))
        self.D3 = nn.Parameter(torch.tensor(10.0))
        self.K1 = nn.Parameter(torch.tensor(10.0))  # Start with higher values for spring constants
        self.K2 = nn.Parameter(torch.tensor(10.0))
        self.E1 = nn.Parameter(torch.tensor(10.0))

        # Define the gravity constant
        self.g = torch.tensor(9.81)

    def forward(self, x):
        # Enforce constraints on physical parameters to prevent division by zero
        # and ensure physically meaningful values
        M1 = torch.clamp(self.M1, min=0.1)    
        M2 = torch.clamp(self.M2, min=0.1)
        M3 = torch.clamp(self.M3, min=0.1)
        D1 = torch.clamp(self.D1, min=0.0)
        D2 = torch.clamp(self.D2, min=0.0)
        D3 = torch.clamp(self.D3, min=0.0)
        K1 = torch.clamp(self.K1, min=1.0)  # Ensure K1 is sufficiently away from zero
        K2 = torch.clamp(self.K2, min=0.1)
        E1 = torch.clamp(self.E1, min=0.0)

        # Assuming x has shape (batch_size, num_features)
        batch_size = x.size(0)

        # Stack all the parameters along the feature dimension and expand them to match the batch size
        params = torch.cat([
            M1.unsqueeze(0), 
            D1.unsqueeze(0), 
            K1.unsqueeze(0), 
            K2.unsqueeze(0), 
            E1.unsqueeze(0), 
        ], dim=0)

        params = params.expand(batch_size, -1)  # Expand to match the batch size

        # Concatenate x with the expanded parameters
        input_for_unmeasured = torch.cat((x, params), dim=1)

        # Estimate the sum of the unknown unmeasured parameters
        fA, fB, fC, fD = torch.split(self.NNforUnmeasured(input_for_unmeasured), 1, dim=1)

        # Set A, B, C, D as attributes of the model to be used in the residuals
        self.fA = fA
        self.fB = fB
        self.fC = fC
        self.fD = fD

        # Estimate the accelerations
        x2_ddot, y2_ddot, x3_ddot, y3_ddot = torch.split(self.NNforAccelerations(x), 1, dim=1)

        # Return the accelerations as a tensor
        output = torch.cat((x2_ddot, y2_ddot, x3_ddot, y3_ddot), dim=1)
        return output
    
    def compute_residuals(self, x, pred, use_normalized=False, scaling_params=None, scaling_method='min_max'):
        """
        Compute the physics-based residuals for the model
        
        Parameters:
        - x: input data (could be normalized or denormalized)
        - pred: predicted accelerations (could be normalized or denormalized)
        - use_normalized: if True, x and pred are in normalized space and equations need to be scaled
        - scaling_params: tuple of (max_x, min_x, max_y, min_y) for denormalization if needed
        - scaling_method: 'min_max' or 'standard' scaling
        
        Returns:
        - residual1, residual2, residual3, residual4: physics equation residuals
        - resMass1_loss, resMass2_loss: mass constraint residuals
        """
        # Use clamped parameters for residual calculations
        M1 = torch.clamp(self.M1, min=0.1)    
        M2 = torch.clamp(self.M2, min=0.1)
        M3 = torch.clamp(self.M3, min=0.1)
        D1 = torch.clamp(self.D1, min=0.0)
        D2 = torch.clamp(self.D2, min=0.0)
        D3 = torch.clamp(self.D3, min=0.0)
        K1 = torch.clamp(self.K1, min=1.0)  # Critical to prevent division by zero
        K2 = torch.clamp(self.K2, min=0.1)
        E1 = torch.clamp(self.E1, min=0.0)
        g = self.g

        # Get predicted accelerations
        x2_ddot, y2_ddot, x3_ddot, y3_ddot = torch.split(pred, 1, dim=1)

        # Get the input features
        x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, omega, t = torch.split(x, 1, dim=1)
        
        # If using normalized values, we need scaling factors to adjust the physical equations
        if use_normalized and scaling_params is not None:
            max_x, min_x, max_y, min_y = scaling_params

            if scaling_method == 'min_max':
                # For min-max scaling, we need factors to properly scale terms in the equations
                
                # Define ranges for each feature
                x_dot_range = max_x[0] - min_x[0]  # Range for velocities in x
                y_dot_range = max_x[1] - min_x[1]  # Range for velocities in y
                x_range = max_x[4] - min_x[4]      # Range for positions in x
                y_range = max_x[5] - min_x[5]      # Range for positions in y
                omega_range = max_x[8] - min_x[8]  # Range for omega
                t_range = max_x[9] - min_x[9]      # Range for time
                accel_range = max_y[0] - min_y[0]  # Range for accelerations
                
                # Offset values (min values for each feature)
                x_dot_offset = min_x[0]
                y_dot_offset = min_x[1]
                x_offset = min_x[4]
                y_offset = min_x[5]
                omega_offset = min_x[8]
                t_offset = min_x[9]
                accel_offset = min_y[0]
                
                # Scale the different terms in the physical equations
                # When a normalized term is denormalized: 
                # denorm_x = norm_x * x_range + x_offset
                
                # For the product of terms, we need to adjust by the range
                # Example: For K*x terms:
                # K*x_denorm = K*(norm_x * x_range + x_offset)
                # So in normalized space: K*norm_x * x_range + K*x_offset
                
                # Safely compute K2/K1 ratio (K1 is guaranteed to be at least 1.0)
                K2_K1_ratio = K2 / K1
                
                # Scale mass times acceleration
                # M*a_denorm = M*(norm_a * accel_range + accel_offset)
                # When we work in normalized space, we need M*norm_a * accel_range + M*accel_offset
                
                # Scale damping coefficient times velocity 
                # D*v_denorm = D*(norm_v * v_range + v_offset)
                # In normalized space: D*norm_v * v_range + D*v_offset
                
                # Scale spring constant times position
                # K*x_denorm = K*(norm_x * x_range + x_offset)
                # In normalized space: K*norm_x * x_range + K*x_offset
                
                # Scale omega^2*E1*cos(omega*t)
                # omega_denorm = norm_omega * omega_range + omega_offset
                # t_denorm = norm_t * t_range + t_offset
                # In normalized space: need to be careful with nonlinear terms like sine/cosine
                
                # Therefore, the residuals are (with scaling factors):
                residual1 = (K1*x2 * x_range + K1*x_offset) + (K2*x3 * x_range + K2*x_offset) + \
                           (M1*((omega * omega_range + omega_offset)**2)*(E1)*torch.cos((omega * omega_range + omega_offset)*(t * t_range + t_offset))) - \
                           self.fA
                           
                residual2 = (K1*y2 * y_range + K1*y_offset) + (K2*y3 * y_range + K2*y_offset) - \
                           (M1*g) + \
                           (M1*((omega * omega_range + omega_offset)**2)*(E1)*torch.sin((omega * omega_range + omega_offset)*(t * t_range + t_offset))) - \
                           self.fB
                           
                residual3 = (M3*x3_ddot * accel_range + M3*accel_offset) + \
                           (D3*x3_dot * x_dot_range + D3*x_dot_offset) + \
                           (K2*x3 * x_range + K2*x_offset) - \
                           (K2_K1_ratio*M2*x2_ddot * accel_range + K2_K1_ratio*M2*accel_offset) - \
                           (K2_K1_ratio*D2*x2_dot * x_dot_range + K2_K1_ratio*D2*x_dot_offset) - \
                           (K2*x2 * x_range + K2*x_offset) - \
                           self.fC
                           
                residual4 = (M3*y3_ddot * accel_range + M3*accel_offset) + \
                           (D3*y3_dot * y_dot_range + D3*y_dot_offset) + \
                           (K2*y3 * y_range + K2*y_offset) - \
                           (K2_K1_ratio*M2*y2_ddot * accel_range + K2_K1_ratio*M2*accel_offset) - \
                           (K2_K1_ratio*D2*y2_dot * y_dot_range + K2_K1_ratio*D2*y_dot_offset) - \
                           (K2*y2 * y_range + K2*y_offset) - \
                           (K2_K1_ratio*M2*g) + (M3*g) - \
                           self.fD
                           
            elif scaling_method == 'standard':
                # For standard scaling (z-score normalization)
                # The normalized value is (x - mean) / std
                # x = normalized_x * std + mean
                
                # We would need to have the means and stds for all features
                # This implementation assumes min_x and max_x contain means and stds respectively
                # for standard scaling mode
                
                means_x = min_x  # In standard scaling mode, min_x should contain means
                stds_x = max_x   # In standard scaling mode, max_x should contain stds
                means_y = min_y  # In standard scaling mode, min_y should contain means
                stds_y = max_y   # In standard scaling mode, max_y should contain stds
                
                # Extract means and stds for each feature
                x_dot_mean, y_dot_mean = means_x[0], means_x[1]
                x_mean, y_mean = means_x[4], means_x[5]
                omega_mean, t_mean = means_x[8], means_x[9]
                accel_mean = means_y[0]
                
                x_dot_std, y_dot_std = stds_x[0], stds_x[1]
                x_std, y_std = stds_x[4], stds_x[5]
                omega_std, t_std = stds_x[8], stds_x[9]
                accel_std = stds_y[0]
                
                # Safely compute K2/K1 ratio (K1 is guaranteed to be at least 1.0)
                K2_K1_ratio = K2 / K1
                
                # Apply the standard scaling adjustments
                residual1 = (K1*(x2 * x_std + x_mean)) + (K2*(x3 * x_std + x_mean)) + \
                           (M1*((omega * omega_std + omega_mean)**2)*(E1)*torch.cos((omega * omega_std + omega_mean)*(t * t_std + t_mean))) - \
                           self.fA
                           
                residual2 = (K1*(y2 * y_std + y_mean)) + (K2*(y3 * y_std + y_mean)) - \
                           (M1*g) + \
                           (M1*((omega * omega_std + omega_mean)**2)*(E1)*torch.sin((omega * omega_std + omega_mean)*(t * t_std + t_mean))) - \
                           self.fB
                           
                residual3 = (M3*(x3_ddot * accel_std + accel_mean)) + \
                           (D3*(x3_dot * x_dot_std + x_dot_mean)) + \
                           (K2*(x3 * x_std + x_mean)) - \
                           (K2_K1_ratio*M2*(x2_ddot * accel_std + accel_mean)) - \
                           (K2_K1_ratio*D2*(x2_dot * x_dot_std + x_dot_mean)) - \
                           (K2*(x2 * x_std + x_mean)) - \
                           self.fC
                           
                residual4 = (M3*(y3_ddot * accel_std + accel_mean)) + \
                           (D3*(y3_dot * y_dot_std + y_dot_mean)) + \
                           (K2*(y3 * y_std + y_mean)) - \
                           (K2_K1_ratio*M2*(y2_ddot * accel_std + accel_mean)) - \
                           (K2_K1_ratio*D2*(y2_dot * y_dot_std + y_dot_mean)) - \
                           (K2*(y2 * y_std + y_mean)) - \
                           (K2_K1_ratio*M2*g) + (M3*g) - \
                           self.fD
            
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}")
            
        else:
            # Original computation with denormalized data
            # Safely compute K2/K1 ratio (K1 is guaranteed to be at least 1.0)
            K2_K1_ratio = K2 / K1
            
            # Therefore, the residuals are:
            residual1 = K1*x2 + K2*x3 + M1*omega**2*E1*torch.cos(omega*t) - self.fA
            residual2 = K1*y2 + K2*y3 - M1*self.g + M1*omega**2*E1*torch.sin(omega*t) - self.fB
            residual3 = M3*x3_ddot + D3*x3_dot + K2*x3 - K2_K1_ratio*M2*x2_ddot - K2_K1_ratio*D2*x2_dot - K2*x2 - self.fC
            residual4 = M3*y3_ddot + D3*y3_dot + K2*y3 - K2_K1_ratio*M2*y2_ddot - K2_K1_ratio*D2*y2_dot - K2*y2 - K2_K1_ratio*M2*self.g + M3*self.g - self.fD

        # Extra residuals for mass remain the same regardless of normalization
        resMass1_loss = M1 + M2 + M3 - 22.0  # The total mass of the system is 22 kg
        resMass2_loss = M2 - M3  # The masses of the overhang and underhang are equal

        return residual1, residual2, residual3, residual4, resMass1_loss, resMass2_loss

def custom_loss(model, x, y_true, max_x_values, min_x_values, max_y_values, min_y_values, use_normalized_residuals=False, scaling_method='min_max'):
    """
    Compute the loss for the PINN model
    
    Parameters:
    - model: the PINN model
    - x: input data (normalized)
    - y_true: true output data (normalized)
    - max_x_values, min_x_values, max_y_values, min_y_values: normalization parameters
    - use_normalized_residuals: if True, compute residuals using normalized data
    - scaling_method: 'min_max' or 'standard' scaling
    """
    # Get the predicted accelerations
    y_pred = model(x)

    # If not using normalized residuals, we need to denormalize for data loss calculation
    if not use_normalized_residuals:
        # Denormalize the data for data loss calculation
        y_pred_denorm = y_pred * (max_y_values - min_y_values) + min_y_values
        y_true_denorm = y_true * (max_y_values - min_y_values) + min_y_values
        
        # Compute the data loss
        data_loss = torch.sqrt(nn.MSELoss()(y_pred_denorm, y_true_denorm))
        
        # Denormalize x for residual computation
        x_for_residuals = x * (max_x_values - min_x_values) + min_x_values
        y_for_residuals = y_pred_denorm
    else:
        # Use normalized data for data loss calculation
        data_loss = torch.sqrt(nn.MSELoss()(y_pred, y_true))
        
        # Use normalized data for residual computation
        x_for_residuals = x
        y_for_residuals = y_pred

    # Compute residuals
    residual1, residual2, residual3, residual4, resMass1_loss, resMass2_loss = model.compute_residuals(
        x_for_residuals, y_for_residuals, 
        use_normalized=use_normalized_residuals, 
        scaling_params=(max_x_values, min_x_values, max_y_values, min_y_values), 
        scaling_method=scaling_method
    )

    # Compute individual losses for each residual with protection against NaN/inf
    # Handle NaNs and clip extreme values
    residual1 = torch.nan_to_num(residual1, nan=0.0, posinf=1e10, neginf=-1e10)
    residual2 = torch.nan_to_num(residual2, nan=0.0, posinf=1e10, neginf=-1e10)
    residual3 = torch.nan_to_num(residual3, nan=0.0, posinf=1e10, neginf=-1e10)
    residual4 = torch.nan_to_num(residual4, nan=0.0, posinf=1e10, neginf=-1e10)
    resMass1_loss = torch.nan_to_num(resMass1_loss, nan=0.0, posinf=1e10, neginf=-1e10)
    resMass2_loss = torch.nan_to_num(resMass2_loss, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Clip residuals to avoid extreme values
    residual1 = torch.clamp(residual1, -1e8, 1e8)
    residual2 = torch.clamp(residual2, -1e8, 1e8)
    residual3 = torch.clamp(residual3, -1e8, 1e8)
    residual4 = torch.clamp(residual4, -1e8, 1e8)
    resMass1_loss = torch.clamp(resMass1_loss, -1e8, 1e8)
    resMass2_loss = torch.clamp(resMass2_loss, -1e8, 1e8)
    
    # Compute individual RMSE losses for each residual
    res1_loss = torch.sqrt(torch.mean(residual1**2))
    res2_loss = torch.sqrt(torch.mean(residual2**2))
    res3_loss = torch.sqrt(torch.mean(residual3**2))
    res4_loss = torch.sqrt(torch.mean(residual4**2))
    resMass1_loss = resMass1_loss**2
    resMass2_loss = resMass2_loss**2
    
    # Protect against NaN in losses
    data_loss = torch.nan_to_num(data_loss, nan=1e4, posinf=1e4, neginf=1e4)
    res1_loss = torch.nan_to_num(res1_loss, nan=1e4, posinf=1e4, neginf=1e4)
    res2_loss = torch.nan_to_num(res2_loss, nan=1e4, posinf=1e4, neginf=1e4)
    res3_loss = torch.nan_to_num(res3_loss, nan=1e4, posinf=1e4, neginf=1e4)
    res4_loss = torch.nan_to_num(res4_loss, nan=1e4, posinf=1e4, neginf=1e4)
    resMass1_loss = torch.nan_to_num(resMass1_loss, nan=1e4, posinf=1e4, neginf=1e4)
    resMass2_loss = torch.nan_to_num(resMass2_loss, nan=1e4, posinf=1e4, neginf=1e4)

    # Print loss components
    print(f"Data: {data_loss.detach():.4f}, res1: {res1_loss.detach():.4f}, res2: {res2_loss.detach():.4f}, res3: {res3_loss.detach():.4f}, res4: {res4_loss.detach():.4f}, resMass1: {resMass1_loss.detach():.4f}, resMass2: {resMass2_loss.detach():.4f}", end='\r')
    
    return data_loss, res1_loss, res2_loss, res3_loss, res4_loss, resMass1_loss, resMass2_loss


def pinn_training_with_algorithm(use_normalized_residuals=False, scaling_method='min_max'):
    """
    Train the PINN model with the adaptive weighting algorithm
    
    Parameters:
    - use_normalized_residuals: if True, compute residuals using normalized data
    - scaling_method: 'min_max' or 'standard' scaling
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory for saving model
    output_dir = f'results/pinn_test_{"normalized" if use_normalized_residuals else "denormalized"}_{scaling_method}'
    models_dir = os.path.join(output_dir, "models")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Training with {'normalized' if use_normalized_residuals else 'denormalized'} residuals using {scaling_method} scaling")
    
    # Training configuration parameters
    max_samples = 5000        # Maximum number of samples to use for training
    batch_size = 64          # Batch size for training
    n_epochs = 1000           # Number of epochs (reduced from 10000)
    
    print("Loading and preparing data...")
    # Load normal data
    X = torch.load('Data/X_normal.pth')[:, :249998, :]
    y = torch.load('Data/Y_normal.pth')[:, :249998, :]
    
    print(f"Original data shapes: X={X.shape}, y={y.shape}")
    
    # Add time feature to X
    t = torch.arange(0, 2e-5 * X.size(1), 2e-5).view(1, -1, 1).expand(X.size(0), -1, -1)[:, :249998, :]
    X = torch.cat((X, t), dim=2)
    
    print(f"After adding time feature: X={X.shape}, y={y.shape}")
    
    # Reshape the tensors to 2D
    X = X.reshape(-1, X.size(-1))
    y = y.reshape(-1, y.size(-1))
    
    print(f"After reshaping: X={X.shape}, y={y.shape}")
    
    # Prepare data for training using the direct_analysis prepare_data function
    data_dict = prepare_data(X, y, batch_size=batch_size, normalize=True, 
                            test_size=0.1, val_size=0.2, max_samples=max_samples)
    
    # Instantiate the model and move to device
    model = PINN().to(device)
    print("\nInitial model parameters:")
    for name, param in model.named_parameters():
        if not name.startswith('NNforUnmeasured') and not name.startswith('NNforAccelerations'):
            print(f"{name}: {param.item():.6f}")

    # Get data loaders from the data_dict
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    test_loader = data_dict['test_loader']
    
    # Get normalization parameters
    X_min = data_dict['X_min'].to(device)
    X_max = data_dict['X_max'].to(device)
    y_min = data_dict['y_min'].to(device)
    y_max = data_dict['y_max'].to(device)

    # Hyperparameters
    alpha = 0.95           # Increased from 0.9 for smoother weight updates
    f = 100                # Reduced update frequency for more stability
    eta = 1e-2             # Reduced learning rate for stability (from 1e-3)
    grad_clip_value = 1.0  # Add gradient clipping

    # Individual Lagrangian multipliers for each loss component
    lambda_data = torch.tensor(0.3, requires_grad=False).to(device)
    lambda_res1 = torch.tensor(0.1, requires_grad=False).to(device)
    lambda_res2 = torch.tensor(0.1, requires_grad=False).to(device)
    lambda_res3 = torch.tensor(0.1, requires_grad=False).to(device)
    lambda_res4 = torch.tensor(0.1, requires_grad=False).to(device)
    lambda_mass1 = torch.tensor(0.1, requires_grad=False).to(device)
    lambda_mass2 = torch.tensor(0.1, requires_grad=False).to(device)
    
    # For tracking best model
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10000          # Early stopping patience
    
    # Lists for tracking metrics over time
    epochs_list = []
    
    # Physical parameters
    M1_history = []
    M2_history = []
    M3_history = []
    D1_history = []
    D2_history = []
    D3_history = []
    K1_history = []
    K2_history = []
    E1_history = []
    
    # Lagrangian multipliers
    lambda_data_history = []
    lambda_res1_history = []
    lambda_res2_history = []
    lambda_res3_history = []
    lambda_res4_history = []
    lambda_mass1_history = []
    lambda_mass2_history = []
    
    # Loss components
    train_total_loss_history = []
    train_data_loss_history = []
    train_res1_loss_history = []
    train_res2_loss_history = []
    train_res3_loss_history = []
    train_res4_loss_history = []
    train_mass1_loss_history = []
    train_mass2_loss_history = []
    
    val_total_loss_history = []
    val_data_loss_history = []
    val_res1_loss_history = []
    val_res2_loss_history = []
    val_res3_loss_history = []
    val_res4_loss_history = []
    val_mass1_loss_history = []
    val_mass2_loss_history = []
    
    print("\nStarting training loop...")
    # Training loop
    step = 0
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_res1_loss = 0.0
        epoch_res2_loss = 0.0
        epoch_res3_loss = 0.0
        epoch_res4_loss = 0.0
        epoch_mass1_loss = 0.0
        epoch_mass2_loss = 0.0
        num_batches = 0
        
        for x_batch, y_batch in train_loader:
            step += 1
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward pass and prediction
            y_pred = model(x_batch)
            
            # Calculate losses - now we get individual residual losses
            data_loss, res1_loss, res2_loss, res3_loss, res4_loss, resMass1_loss, resMass2_loss = custom_loss(
                model, x_batch, y_batch, 
                X_max, X_min, y_max, y_min,
                use_normalized_residuals=use_normalized_residuals,
                scaling_method=scaling_method
            )
           
            # Total loss with individual weighting for each residual
            total_loss = (lambda_data * data_loss + 
                          lambda_res1 * res1_loss + 
                          lambda_res2 * res2_loss + 
                          lambda_res3 * res3_loss + 
                          lambda_res4 * res4_loss + 
                          lambda_mass1 * resMass1_loss + 
                          lambda_mass2 * resMass2_loss)

            # Report if total_loss is not finite (but don't replace it)
            if not torch.isfinite(total_loss):
                print(f"\nWarning: Loss is not finite at step {step}. Loss values: data={data_loss.item():.4f}, res1={res1_loss.item():.4f}, res2={res2_loss.item():.4f}, res3={res3_loss.item():.4f}, res4={res4_loss.item():.4f}, mass1={resMass1_loss.item():.4f}, mass2={resMass2_loss.item():.4f}")
                
            # Periodically update global weights
            if step % f == 0:
                # Use try-except to catch any gradient computation errors
                try:
                    grads = torch.autograd.grad(total_loss, model.parameters(), retain_graph=True, create_graph=True)
                    
                    # Apply gradient clipping
                    grads = [torch.clamp(g, -grad_clip_value, grad_clip_value) if g is not None else g for g in grads]
                    
                    # Function to compute the norm of gradients, safely handling None values
                    norm = lambda g: torch.sqrt(sum(torch.sum(p**2) for p in g if p is not None) + 1e-12)

                    # Approximate loss gradients for each component
                    g_data = torch.autograd.grad(data_loss, model.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
                    g_res1 = torch.autograd.grad(res1_loss, model.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
                    g_res2 = torch.autograd.grad(res2_loss, model.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
                    g_res3 = torch.autograd.grad(res3_loss, model.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
                    g_res4 = torch.autograd.grad(res4_loss, model.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
                    g_mass1 = torch.autograd.grad(resMass1_loss, model.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
                    g_mass2 = torch.autograd.grad(resMass2_loss, model.parameters(), retain_graph=True, create_graph=True, allow_unused=True)

                    # Calculate gradient norms
                    norm_data = norm(g_data)
                    norm_res1 = norm(g_res1)
                    norm_res2 = norm(g_res2)
                    norm_res3 = norm(g_res3)
                    norm_res4 = norm(g_res4)
                    norm_mass1 = norm(g_mass1)
                    norm_mass2 = norm(g_mass2)

                    # Apply adaptive weighting only if all norms are finite
                    norms_ok = (torch.isfinite(norm_data) and torch.isfinite(norm_res1) and 
                                torch.isfinite(norm_res2) and torch.isfinite(norm_res3) and 
                                torch.isfinite(norm_res4) and torch.isfinite(norm_mass1) and 
                                torch.isfinite(norm_mass2))
                    
                    if norms_ok:
                        sum_norms = norm_data + norm_res1 + norm_res2 + norm_res3 + norm_res4 + norm_mass1 + norm_mass2
                        lambda_data_new = norm_data / sum_norms
                        lambda_res1_new = norm_res1 / sum_norms
                        lambda_res2_new = norm_res2 / sum_norms
                        lambda_res3_new = norm_res3 / sum_norms
                        lambda_res4_new = norm_res4 / sum_norms
                        lambda_mass1_new = norm_mass1 / sum_norms
                        lambda_mass2_new = norm_mass2 / sum_norms

                        # Update the Lagrangian multipliers with smoothing
                        lambda_data = alpha * lambda_data + (1 - alpha) * lambda_data_new.detach()
                        lambda_res1 = alpha * lambda_res1 + (1 - alpha) * lambda_res1_new.detach()
                        lambda_res2 = alpha * lambda_res2 + (1 - alpha) * lambda_res2_new.detach()
                        lambda_res3 = alpha * lambda_res3 + (1 - alpha) * lambda_res3_new.detach()
                        lambda_res4 = alpha * lambda_res4 + (1 - alpha) * lambda_res4_new.detach()
                        lambda_mass1 = alpha * lambda_mass1 + (1 - alpha) * lambda_mass1_new.detach()
                        lambda_mass2 = alpha * lambda_mass2 + (1 - alpha) * lambda_mass2_new.detach()

                    # Manual parameter update with gradient clipping
                    with torch.no_grad():
                        for param, grad in zip(model.parameters(), grads):
                            if grad is not None:
                                param -= eta * grad
                                
                except Exception as e:
                    print(f"\nError during gradient computation: {e}")
                    continue
            
            # Only add to epoch_loss if loss is finite
            if torch.isfinite(total_loss):
                epoch_loss += total_loss.item()
                epoch_data_loss += data_loss.item()
                epoch_res1_loss += res1_loss.item()
                epoch_res2_loss += res2_loss.item()
                epoch_res3_loss += res3_loss.item()
                epoch_res4_loss += res4_loss.item()
                epoch_mass1_loss += resMass1_loss.item()
                epoch_mass2_loss += resMass2_loss.item()
                num_batches += 1
            
            # Print progress
            if step % 10 == 0:
                lambdas_str = f"λd={lambda_data:.2f}, λr1={lambda_res1:.2f}, λr2={lambda_res2:.2f}, λr3={lambda_res3:.2f}, λr4={lambda_res4:.2f}, λm1={lambda_mass1:.2f}, λm2={lambda_mass2:.2f}"
                print(f"Epoch {epoch+1}/{n_epochs}, Step {step}, Loss: {total_loss.item():.4f}, {lambdas_str}")
                
                # Print current parameter values every 100 steps
                if step % 100 == 0:
                    print("\nCurrent parameters:")
                    for name, param in model.named_parameters():
                        if not name.startswith('NNforUnmeasured') and not name.startswith('NNforAccelerations'):
                            print(f"{name}: {param.item():.6f}")
        
        # Calculate average epoch loss
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            avg_data_loss = epoch_data_loss / num_batches
            avg_res1_loss = epoch_res1_loss / num_batches
            avg_res2_loss = epoch_res2_loss / num_batches
            avg_res3_loss = epoch_res3_loss / num_batches
            avg_res4_loss = epoch_res4_loss / num_batches
            avg_mass1_loss = epoch_mass1_loss / num_batches
            avg_mass2_loss = epoch_mass2_loss / num_batches
        else:
            avg_epoch_loss = float('nan')
            avg_data_loss = float('nan')
            avg_res1_loss = float('nan')
            avg_res2_loss = float('nan')
            avg_res3_loss = float('nan')
            avg_res4_loss = float('nan')
            avg_mass1_loss = float('nan')
            avg_mass2_loss = float('nan')
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_data_loss_sum = 0.0
        val_res1_loss_sum = 0.0
        val_res2_loss_sum = 0.0
        val_res3_loss_sum = 0.0
        val_res4_loss_sum = 0.0
        val_mass1_loss_sum = 0.0
        val_mass2_loss_sum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                
                # Forward pass for validation
                y_val_pred = model(x_val)
                
                # Calculate validation losses
                val_data_loss, val_res1_loss, val_res2_loss, val_res3_loss, val_res4_loss, val_mass1_loss, val_mass2_loss = custom_loss(
                    model, x_val, y_val, 
                    X_max, X_min, y_max, y_min,
                    use_normalized_residuals=use_normalized_residuals,
                    scaling_method=scaling_method
                )
                
                # Total validation loss
                val_total_loss = (lambda_data * val_data_loss + 
                                 lambda_res1 * val_res1_loss + 
                                 lambda_res2 * val_res2_loss + 
                                 lambda_res3 * val_res3_loss + 
                                 lambda_res4 * val_res4_loss + 
                                 lambda_mass1 * val_mass1_loss + 
                                 lambda_mass2 * val_mass2_loss)
                
                # Handle infinite loss in validation
                if torch.isfinite(val_total_loss):
                    val_loss += val_total_loss.item()
                    val_data_loss_sum += val_data_loss.item()
                    val_res1_loss_sum += val_res1_loss.item()
                    val_res2_loss_sum += val_res2_loss.item()
                    val_res3_loss_sum += val_res3_loss.item()
                    val_res4_loss_sum += val_res4_loss.item()
                    val_mass1_loss_sum += val_mass1_loss.item()
                    val_mass2_loss_sum += val_mass2_loss.item()
                    val_batches += 1
        
        # Calculate average validation loss (avoid division by zero)
        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
            avg_val_data_loss = val_data_loss_sum / val_batches
            avg_val_res1_loss = val_res1_loss_sum / val_batches
            avg_val_res2_loss = val_res2_loss_sum / val_batches
            avg_val_res3_loss = val_res3_loss_sum / val_batches
            avg_val_res4_loss = val_res4_loss_sum / val_batches
            avg_val_mass1_loss = val_mass1_loss_sum / val_batches
            avg_val_mass2_loss = val_mass2_loss_sum / val_batches
        else:
            avg_val_loss = float('nan')
            avg_val_data_loss = float('nan')
            avg_val_res1_loss = float('nan')
            avg_val_res2_loss = float('nan')
            avg_val_res3_loss = float('nan')
            avg_val_res4_loss = float('nan')
            avg_val_mass1_loss = float('nan')
            avg_val_mass2_loss = float('nan')
        
        # Save metrics for this epoch
        epochs_list.append(epoch + 1)
        
        # Save parameter values
        M1_history.append(model.M1.item())
        M2_history.append(model.M2.item())
        M3_history.append(model.M3.item())
        D1_history.append(model.D1.item())
        D2_history.append(model.D2.item())
        D3_history.append(model.D3.item())
        K1_history.append(model.K1.item())
        K2_history.append(model.K2.item())
        E1_history.append(model.E1.item())
        
        # Save Lagrangian multipliers
        lambda_data_history.append(lambda_data.item())
        lambda_res1_history.append(lambda_res1.item())
        lambda_res2_history.append(lambda_res2.item())
        lambda_res3_history.append(lambda_res3.item())
        lambda_res4_history.append(lambda_res4.item())
        lambda_mass1_history.append(lambda_mass1.item())
        lambda_mass2_history.append(lambda_mass2.item())
        
        # Save loss components - training
        train_total_loss_history.append(avg_epoch_loss)
        train_data_loss_history.append(avg_data_loss)
        train_res1_loss_history.append(avg_res1_loss)
        train_res2_loss_history.append(avg_res2_loss)
        train_res3_loss_history.append(avg_res3_loss)
        train_res4_loss_history.append(avg_res4_loss)
        train_mass1_loss_history.append(avg_mass1_loss)
        train_mass2_loss_history.append(avg_mass2_loss)
        
        # Save loss components - validation
        val_total_loss_history.append(avg_val_loss)
        val_data_loss_history.append(avg_val_data_loss)
        val_res1_loss_history.append(avg_val_res1_loss)
        val_res2_loss_history.append(avg_val_res2_loss)
        val_res3_loss_history.append(avg_val_res3_loss)
        val_res4_loss_history.append(avg_val_res4_loss)
        val_mass1_loss_history.append(avg_val_mass1_loss)
        val_mass2_loss_history.append(avg_val_mass2_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{n_epochs} completed:")
        print(f"  Training Loss: {avg_epoch_loss:.6f}")
        print(f"  Validation Loss: {avg_val_loss:.6f}")
        print(f"  λd={lambda_data:.2f}, λr1={lambda_res1:.2f}, λr2={lambda_res2:.2f}, λr3={lambda_res3:.2f}, λr4={lambda_res4:.2f}, λm1={lambda_mass1:.2f}, λm2={lambda_mass2:.2f}")
        
        # Early stopping (only if validation loss is finite)
        if torch.isfinite(torch.tensor(avg_val_loss)) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(models_dir, 'best_model.pth'))
            print(f"  New best model saved!")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs.")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
                
        # Save checkpoint every 10 epochs
        if epoch % 10 == 9:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss,
                'lambda_data': lambda_data,
                'lambda_res1': lambda_res1,
                'lambda_res2': lambda_res2,
                'lambda_res3': lambda_res3,
                'lambda_res4': lambda_res4,
                'lambda_mass1': lambda_mass1,
                'lambda_mass2': lambda_mass2,
            }, os.path.join(models_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"  Checkpoint saved at epoch {epoch+1}")
    
    # Print final parameter values
    print("\nTraining complete!")
    print("Final parameter values:")
    for name, param in model.named_parameters():
        if not name.startswith('NNforUnmeasured') and not name.startswith('NNforAccelerations'):
            print(f"{name}: {param.item():.6f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(models_dir, 'final_model.pth'))
    print(f"Final model saved to {os.path.join(models_dir, 'final_model.pth')}")
    
    # Save all metrics as numpy arrays
    metrics_data = {
        'epochs': np.array(epochs_list),
        # Parameters
        'M1': np.array(M1_history),
        'M2': np.array(M2_history),
        'M3': np.array(M3_history),
        'D1': np.array(D1_history),
        'D2': np.array(D2_history),
        'D3': np.array(D3_history),
        'K1': np.array(K1_history),
        'K2': np.array(K2_history),
        'E1': np.array(E1_history),
        # Lagrangians
        'lambda_data': np.array(lambda_data_history),
        'lambda_res1': np.array(lambda_res1_history),
        'lambda_res2': np.array(lambda_res2_history),
        'lambda_res3': np.array(lambda_res3_history),
        'lambda_res4': np.array(lambda_res4_history),
        'lambda_mass1': np.array(lambda_mass1_history),
        'lambda_mass2': np.array(lambda_mass2_history),
        # Training losses
        'train_total_loss': np.array(train_total_loss_history),
        'train_data_loss': np.array(train_data_loss_history),
        'train_res1_loss': np.array(train_res1_loss_history),
        'train_res2_loss': np.array(train_res2_loss_history),
        'train_res3_loss': np.array(train_res3_loss_history),
        'train_res4_loss': np.array(train_res4_loss_history),
        'train_mass1_loss': np.array(train_mass1_loss_history),
        'train_mass2_loss': np.array(train_mass2_loss_history),
        # Validation losses
        'val_total_loss': np.array(val_total_loss_history),
        'val_data_loss': np.array(val_data_loss_history),
        'val_res1_loss': np.array(val_res1_loss_history),
        'val_res2_loss': np.array(val_res2_loss_history),
        'val_res3_loss': np.array(val_res3_loss_history),
        'val_res4_loss': np.array(val_res4_loss_history),
        'val_mass1_loss': np.array(val_mass1_loss_history),
        'val_mass2_loss': np.array(val_mass2_loss_history),
    }
    
    np.savez(os.path.join(models_dir, 'training_metrics.npz'), **metrics_data)
    
    # Generate and save plots
    print("\nGenerating plots...")
    
    # 1. Plot parameter evolution
    plt.figure(figsize=(20, 20))
    
    # Plot masses
    plt.subplot(4, 3, 1)
    plt.plot(epochs_list, M1_history, 'r-', label='M1')
    plt.plot(epochs_list, M2_history, 'g-', label='M2')
    plt.plot(epochs_list, M3_history, 'b-', label='M3')
    plt.xlabel('Epoch')
    plt.ylabel('Mass Value')
    plt.title('Evolution of Mass Parameters')
    plt.legend()
    plt.grid(True)
    
    # Plot damping coefficients
    plt.subplot(4, 3, 2)
    plt.plot(epochs_list, D1_history, 'r-', label='D1')
    plt.plot(epochs_list, D2_history, 'g-', label='D2')
    plt.plot(epochs_list, D3_history, 'b-', label='D3')
    plt.xlabel('Epoch')
    plt.ylabel('Damping Value')
    plt.title('Evolution of Damping Parameters')
    plt.legend()
    plt.grid(True)
    
    # Plot spring constants
    plt.subplot(4, 3, 3)
    plt.plot(epochs_list, K1_history, 'r-', label='K1')
    plt.plot(epochs_list, K2_history, 'g-', label='K2')
    plt.xlabel('Epoch')
    plt.ylabel('Spring Constant Value')
    plt.title('Evolution of Spring Constants')
    plt.legend()
    plt.grid(True)
    
    # Plot E1
    plt.subplot(4, 3, 4)
    plt.plot(epochs_list, E1_history, 'm-', label='E1')
    plt.xlabel('Epoch')
    plt.ylabel('E1 Value')
    plt.title('Evolution of E1 Parameter')
    plt.legend()
    plt.grid(True)
    
    # 2. Plot all Lagrangian multipliers
    plt.subplot(4, 3, 5)
    plt.plot(epochs_list, lambda_data_history, 'k-', label='λ_data')
    plt.plot(epochs_list, lambda_res1_history, 'r-', label='λ_res1')
    plt.plot(epochs_list, lambda_res2_history, 'g-', label='λ_res2')
    plt.plot(epochs_list, lambda_res3_history, 'b-', label='λ_res3')
    plt.plot(epochs_list, lambda_res4_history, 'c-', label='λ_res4')
    plt.plot(epochs_list, lambda_mass1_history, 'm-', label='λ_mass1')
    plt.plot(epochs_list, lambda_mass2_history, 'y-', label='λ_mass2')
    plt.xlabel('Epoch')
    plt.ylabel('Lagrangian Value')
    plt.title('Evolution of All Lagrangian Multipliers')
    plt.legend()
    plt.grid(True)
    
    # 3. Plot training loss components - data and regularization
    plt.subplot(4, 3, 6)
    plt.plot(epochs_list, train_data_loss_history, 'b-', label='Data Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Data Loss')
    plt.legend()
    plt.grid(True)
    
    # 4. Plot training loss components - physics residuals
    plt.subplot(4, 3, 7)
    plt.plot(epochs_list, train_res1_loss_history, 'r-', label='Res1 Loss')
    plt.plot(epochs_list, train_res2_loss_history, 'g-', label='Res2 Loss')
    plt.plot(epochs_list, train_res3_loss_history, 'b-', label='Res3 Loss')
    plt.plot(epochs_list, train_res4_loss_history, 'c-', label='Res4 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Training: Physics Residual Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use log scale for better visualization
    
    # 5. Plot training loss components - mass residuals
    plt.subplot(4, 3, 8)
    plt.plot(epochs_list, train_mass1_loss_history, 'm-', label='Mass1 Loss')
    plt.plot(epochs_list, train_mass2_loss_history, 'y-', label='Mass2 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Training: Mass Constraint Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use log scale for better visualization
    
    # 6. Plot validation loss components - data and regularization
    plt.subplot(4, 3, 9)
    plt.plot(epochs_list, val_data_loss_history, 'b-', label='Data Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Validation: Data Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use log scale for better visualization
    
    # 7. Plot validation loss components - physics residuals
    plt.subplot(4, 3, 10)
    plt.plot(epochs_list, val_res1_loss_history, 'r-', label='Res1 Loss')
    plt.plot(epochs_list, val_res2_loss_history, 'g-', label='Res2 Loss')
    plt.plot(epochs_list, val_res3_loss_history, 'b-', label='Res3 Loss')
    plt.plot(epochs_list, val_res4_loss_history, 'c-', label='Res4 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Validation: Physics Residual Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use log scale for better visualization
    
    # 8. Plot validation loss components - mass residuals
    plt.subplot(4, 3, 11)
    plt.plot(epochs_list, val_mass1_loss_history, 'm-', label='Mass1 Loss')
    plt.plot(epochs_list, val_mass2_loss_history, 'y-', label='Mass2 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Validation: Mass Constraint Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use log scale for better visualization
    
    # 9. Compare train vs validation total loss
    plt.subplot(4, 3, 12)
    plt.plot(epochs_list, train_total_loss_history, 'b-', label='Training Loss')
    plt.plot(epochs_list, val_total_loss_history, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Training vs Validation Total Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use log scale for better visualization
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_evolution.png'), dpi=300)
    plt.close()
    
    # Create additional plot for parameters only (with better scale)
    plt.figure(figsize=(15, 10))
    
    # Plot all parameters together
    plt.subplot(1, 1, 1)
    plt.plot(epochs_list, M1_history, 'r-', label='M1')
    plt.plot(epochs_list, M2_history, 'g-', label='M2')
    plt.plot(epochs_list, M3_history, 'b-', label='M3')
    plt.plot(epochs_list, D1_history, 'r--', label='D1')
    plt.plot(epochs_list, D2_history, 'g--', label='D2')
    plt.plot(epochs_list, D3_history, 'b--', label='D3')
    plt.plot(epochs_list, K1_history, 'r-.', label='K1')
    plt.plot(epochs_list, K2_history, 'g-.', label='K2')
    plt.plot(epochs_list, E1_history, 'm-', label='E1')
    plt.xlabel('Epoch')
    plt.ylabel('Parameter Value')
    plt.title('Evolution of All Physical Parameters')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'parameters_evolution.png'), dpi=300)
    plt.close()
    
    # Create additional plot just for the Lagrangian multipliers
    plt.figure(figsize=(15, 8))
    
    plt.plot(epochs_list, lambda_data_history, 'k-', linewidth=2, label='λ_data')
    plt.plot(epochs_list, lambda_res1_history, 'r-', label='λ_res1')
    plt.plot(epochs_list, lambda_res2_history, 'g-', label='λ_res2')
    plt.plot(epochs_list, lambda_res3_history, 'b-', label='λ_res3')
    plt.plot(epochs_list, lambda_res4_history, 'c-', label='λ_res4')
    plt.plot(epochs_list, lambda_mass1_history, 'm-', label='λ_mass1')
    plt.plot(epochs_list, lambda_mass2_history, 'y-', label='λ_mass2')
    plt.xlabel('Epoch')
    plt.ylabel('Lagrangian Value')
    plt.title('Evolution of Lagrangian Multipliers')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'lagrangian_evolution.png'), dpi=300)
    plt.close()
    
    print(f"Plots saved to {plots_dir}")
    
    return model, metrics_data

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Train PINN with different normalization options')
    parser.add_argument('--normalized', action='store_true', 
                        help='Use normalized data in residual computation')
    parser.add_argument('--scaling', type=str, choices=['min_max', 'standard'], default='min_max',
                        help='Scaling method to use (min_max or standard)')
    
    args = parser.parse_args()
    
    # Run training with the specified options
    pinn_training_with_algorithm(
        use_normalized_residuals=args.normalized,
        scaling_method=args.scaling
    )