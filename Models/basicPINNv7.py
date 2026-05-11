import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from direct_analysis.data_utils import prepare_data, normalize_data

class ConfigurableMLP(nn.Module):
    """
    A configurable multi-layer perceptron neural network.
    
    Parameters:
    - input_dim: Input dimension
    - hidden_layers: List of integers defining the width of each hidden layer
    - output_dim: Output dimension
    - activation: Activation function (default: 'tanh')
    - dropout_rate: Dropout rate for regularization (default: 0.0)
    - init_method: Weight initialization method (default: 'xavier_normal')
    """
    def __init__(self, input_dim, hidden_layers, output_dim, 
                 activation='tanh', dropout_rate=0.0, init_method='xavier_normal'):
        super(ConfigurableMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        
        layers = []
        
        # Input layer
        current_dim = input_dim
        
        # Hidden layers
        for h_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, h_dim))
            
            # Activation function
            if activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation.lower() == 'elu':
                layers.append(nn.ELU())
            elif activation.lower() == 'selu':
                layers.append(nn.SELU())
            elif activation.lower() == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
            
            # Add dropout if specified
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            current_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights(init_method)
        
    def _initialize_weights(self, method):
        """Initialize the weights using the specified method."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif method == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif method == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='tanh')
                elif method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='tanh')
                elif method == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                elif method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=0.1)
                elif method == 'uniform':
                    nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                
                # Initialize bias
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x)

class ConfigurablePINN(nn.Module):
    """
    A configurable Physics-Informed Neural Network (PINN) model.
    
    Parameters:
    - unmeasured_net_config: Configuration for the unmeasured parameters network
        - hidden_layers: List of integers for hidden layer widths
        - activation: Activation function to use
        - dropout_rate: Dropout rate
        - init_method: Weight initialization method
    - acceleration_net_config: Configuration for the acceleration network
        - hidden_layers: List of integers for hidden layer widths
        - activation: Activation function to use
        - dropout_rate: Dropout rate
        - init_method: Weight initialization method
    - param_init_config: Configuration for physical parameter initialization
        - method: Initialization method ('fixed', 'uniform', 'normal')
        - values: Dictionary of initial values or distribution parameters
    - enable_mass_constraints: Whether to include mass constraints (default: True for real data compatibility)
    """
    def __init__(self, unmeasured_net_config=None, acceleration_net_config=None, param_init_config=None, enable_mass_constraints=True):
        super(ConfigurablePINN, self).__init__()
        
        # Store mass constraint flag
        self.enable_mass_constraints = enable_mass_constraints
        
        # Default configurations if not provided
        if unmeasured_net_config is None:
            unmeasured_net_config = {
                'hidden_layers': [64, 64],
                'activation': 'tanh',
                'dropout_rate': 0.0,
                'init_method': 'xavier_normal'
            }
            
        if acceleration_net_config is None:
            acceleration_net_config = {
                'hidden_layers': [64, 64],
                'activation': 'tanh',
                'dropout_rate': 0.0,
                'init_method': 'xavier_normal'
            }
            
        if param_init_config is None:
            param_init_config = {
                'method': 'fixed',
                'values': {
                    'M1': 50.0, 'M2': 3.5, 'M3': 3.5,
                    'D1': 3000.0, 'D2': 3000.0, 'D3': 3000.0,
                    'K1': 3.4635e6, 'K2': 3.8127e6, 'E1': 5.0e-6
                }
            }

        # Network for estimating unmeasured parameters (15 inputs, 4 outputs)
        self.NNforUnmeasured = ConfigurableMLP(
            input_dim=15,  # 10 features + 5 parameters
            hidden_layers=unmeasured_net_config['hidden_layers'],
            output_dim=4,  # 4 unmeasured parameters
            activation=unmeasured_net_config['activation'],
            dropout_rate=unmeasured_net_config['dropout_rate'],
            init_method=unmeasured_net_config['init_method']
        )
        
        # Network for estimating accelerations (10 inputs, 4 outputs)
        self.NNforAccelerations = ConfigurableMLP(
            input_dim=10,  # 10 features
            hidden_layers=acceleration_net_config['hidden_layers'],
            output_dim=4,  # 4 accelerations
            activation=acceleration_net_config['activation'],
            dropout_rate=acceleration_net_config['dropout_rate'],
            init_method=acceleration_net_config['init_method']
        )
        
        # Initialize learnable physical parameters based on configuration
        self._initialize_physical_parameters(param_init_config)
        
        # Register gravity constant as a buffer so it follows the model's device
        self.register_buffer('g', torch.tensor(9.81))
        
    def _initialize_physical_parameters(self, config):
        """
        Initialize physical parameters based on configuration.
        
        Parameters:
        - config: Dictionary with 'method' and 'values' keys
        """
        method = config['method']
        values = config['values']
        
        if method == 'fixed':
            # Initialize with fixed values
            self.M1 = nn.Parameter(torch.tensor(float(values.get('M1', 10.0))))
            self.M2 = nn.Parameter(torch.tensor(float(values.get('M2', 10.0))))
            self.M3 = nn.Parameter(torch.tensor(float(values.get('M3', 11.0))))
            self.D1 = nn.Parameter(torch.tensor(float(values.get('D1', 10.0))))
            self.D2 = nn.Parameter(torch.tensor(float(values.get('D2', 10.0))))
            self.D3 = nn.Parameter(torch.tensor(float(values.get('D3', 10.0))))
            self.K1 = nn.Parameter(torch.tensor(float(values.get('K1', 10.0))))
            self.K2 = nn.Parameter(torch.tensor(float(values.get('K2', 10.0))))
            self.E1 = nn.Parameter(torch.tensor(float(values.get('E1', 10.0))))
        
        elif method == 'uniform':
            # Initialize with uniform distribution
            M1_range = values.get('M1', (5.0, 15.0))
            M2_range = values.get('M2', (5.0, 15.0))
            M3_range = values.get('M3', (5.0, 15.0))
            D1_range = values.get('D1', (5.0, 15.0))
            D2_range = values.get('D2', (5.0, 15.0))
            D3_range = values.get('D3', (5.0, 15.0))
            K1_range = values.get('K1', (5.0, 15.0))
            K2_range = values.get('K2', (5.0, 15.0))
            E1_range = values.get('E1', (5.0, 15.0))
            
            self.M1 = nn.Parameter(torch.FloatTensor(1).uniform_(*M1_range))
            self.M2 = nn.Parameter(torch.FloatTensor(1).uniform_(*M2_range))
            self.M3 = nn.Parameter(torch.FloatTensor(1).uniform_(*M3_range))
            self.D1 = nn.Parameter(torch.FloatTensor(1).uniform_(*D1_range))
            self.D2 = nn.Parameter(torch.FloatTensor(1).uniform_(*D2_range))
            self.D3 = nn.Parameter(torch.FloatTensor(1).uniform_(*D3_range))
            self.K1 = nn.Parameter(torch.FloatTensor(1).uniform_(*K1_range))
            self.K2 = nn.Parameter(torch.FloatTensor(1).uniform_(*K2_range))
            self.E1 = nn.Parameter(torch.FloatTensor(1).uniform_(*E1_range))
        
        elif method == 'normal':
            # Initialize with normal distribution
            M1_params = values.get('M1', (10.0, 1.0))
            M2_params = values.get('M2', (10.0, 1.0))
            M3_params = values.get('M3', (11.0, 1.0))
            D1_params = values.get('D1', (10.0, 1.0))
            D2_params = values.get('D2', (10.0, 1.0))
            D3_params = values.get('D3', (10.0, 1.0))
            K1_params = values.get('K1', (10.0, 1.0))
            K2_params = values.get('K2', (10.0, 1.0))
            E1_params = values.get('E1', (10.0, 1.0))
            
            self.M1 = nn.Parameter(torch.FloatTensor(1).normal_(*M1_params))
            self.M2 = nn.Parameter(torch.FloatTensor(1).normal_(*M2_params))
            self.M3 = nn.Parameter(torch.FloatTensor(1).normal_(*M3_params))
            self.D1 = nn.Parameter(torch.FloatTensor(1).normal_(*D1_params))
            self.D2 = nn.Parameter(torch.FloatTensor(1).normal_(*D2_params))
            self.D3 = nn.Parameter(torch.FloatTensor(1).normal_(*D3_params))
            self.K1 = nn.Parameter(torch.FloatTensor(1).normal_(*K1_params))
            self.K2 = nn.Parameter(torch.FloatTensor(1).normal_(*K2_params))
            self.E1 = nn.Parameter(torch.FloatTensor(1).normal_(*E1_params))
        
        else:
            raise ValueError(f"Unsupported parameter initialization method: {method}")

    def forward(self, x):
        # Enforce constraints on physical parameters
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

        # Stack parameters
        # Ensure parameters are properly shaped for concatenation
        # First reshape to [1, 1] if needed, then expand to batch size
        M1_reshaped = M1.view(1, 1)
        D1_reshaped = D1.view(1, 1)
        K1_reshaped = K1.view(1, 1)
        K2_reshaped = K2.view(1, 1)
        E1_reshaped = E1.view(1, 1)
        
        # Concatenate along dimension 1
        params = torch.cat([
            M1_reshaped,
            D1_reshaped,
            K1_reshaped,
            K2_reshaped,
            E1_reshaped
        ], dim=1)
        
        # Expand to match batch size
        params = params.expand(batch_size, -1)

        # Concatenate x with the expanded parameters
        input_for_unmeasured = torch.cat((x, params), dim=1)

        # Estimate the sum of the unknown unmeasured parameters
        fA, fB, fC, fD = torch.split(self.NNforUnmeasured(input_for_unmeasured), 1, dim=1)

        # Set these as attributes to be used in compute_residuals
        self.fA = fA
        self.fB = fB
        self.fC = fC
        self.fD = fD

        # Estimate the accelerations
        x2_ddot, y2_ddot, x3_ddot, y3_ddot = torch.split(self.NNforAccelerations(x), 1, dim=1)

        # Return the accelerations
        return torch.cat((x2_ddot, y2_ddot, x3_ddot, y3_ddot), dim=1)
    
    def compute_residuals(self, x, pred, X_max=None, X_min=None, y_max=None, y_min=None):
        """
        Compute the physics-based residuals for the system.
        
        Parameters:
        - x: Input tensor (batch_size, features) - NORMALIZED
        - pred: Predicted accelerations (batch_size, 4) - NORMALIZED
        - X_max, X_min: Input normalization parameters (not used for physics)
        - y_max, y_min: Output normalization parameters (not used for physics)
        
        Returns:
        - Tuple of residual tensors
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

        # Get predicted accelerations (normalized)
        x2_ddot, y2_ddot, x3_ddot, y3_ddot = torch.split(pred, 1, dim=1)

        # Get the input features (all normalized to [0,1])
        x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, omega, t = torch.split(x, 1, dim=1)
        
        # CRITICAL FIX: Denormalize omega and t for correct trigonometric physics
        # Original omega range: ~77-386 rad/s, t range: 0-5 seconds
        omega = omega * (386.0 - 77.0) + 77.0  # Denormalize omega to rad/s
        t = t * 5.0  # Denormalize t to seconds (t_min = 0)
        
        # Safely compute K2/K1 ratio (K1 is guaranteed to be at least 1.0)
        K2_K1_ratio = K2 / K1
        
        # Calculate the residuals using normalized values for positions/velocities/accelerations
        # but real-world values for omega and t (which are needed for correct physics)
        residual1 = K1*x2 + K2*x3 + M1*omega**2*E1*torch.cos(omega*t) - self.fA
        residual2 = K1*y2 + K2*y3 - M1*self.g + M1*omega**2*E1*torch.sin(omega*t) - self.fB
        residual3 = M3*x3_ddot + D3*x3_dot + K2*x3 - K2_K1_ratio*M2*x2_ddot - K2_K1_ratio*D2*x2_dot - K2*x2 - self.fC
        residual4 = M3*y3_ddot + D3*y3_dot + K2*y3 - K2_K1_ratio*M2*y2_ddot - K2_K1_ratio*D2*y2_dot - K2*y2 - K2_K1_ratio*M2*self.g + M3*self.g - self.fD

        # SANITIZE RESIDUALS: Replace NaN with 0.0 and clamp extreme values
        residual1 = torch.nan_to_num(residual1, nan=0.0, posinf=1e8, neginf=-1e8)
        residual2 = torch.nan_to_num(residual2, nan=0.0, posinf=1e8, neginf=-1e8)
        residual3 = torch.nan_to_num(residual3, nan=0.0, posinf=1e8, neginf=-1e8)
        residual4 = torch.nan_to_num(residual4, nan=0.0, posinf=1e8, neginf=-1e8)
        
        # Clamp residuals to prevent extreme values
        residual1 = torch.clamp(residual1, -1e8, 1e8)
        residual2 = torch.clamp(residual2, -1e8, 1e8)
        residual3 = torch.clamp(residual3, -1e8, 1e8)
        residual4 = torch.clamp(residual4, -1e8, 1e8)

        # Conditionally include mass constraints based on flag
        if self.enable_mass_constraints:
            # Extra residuals for mass constraints (only for real data)
            residualMass1 = M1 + M2 + M3 - 22.0  # The total mass of the system is 22 kg
            residualMass2 = M2 - M3  # The masses of the overhang and underhang are equal
            
            # Sanitize mass residuals
            residualMass1 = torch.nan_to_num(residualMass1, nan=0.0, posinf=1e8, neginf=-1e8)
            residualMass2 = torch.nan_to_num(residualMass2, nan=0.0, posinf=1e8, neginf=-1e8)
            residualMass1 = torch.clamp(residualMass1, -1e8, 1e8)
            residualMass2 = torch.clamp(residualMass2, -1e8, 1e8)
            
            return residual1, residual2, residual3, residual4, residualMass1, residualMass2
        else:
            # For synthetic data, return zeros for mass constraints to maintain compatibility
            # Use zeros_like with a tensor that requires gradients to ensure grad_fn exists
            residualMass1 = torch.zeros_like(residual1)
            residualMass2 = torch.zeros_like(residual2)
            return residual1, residual2, residual3, residual4, residualMass1, residualMass2

def get_default_pinn_config():
    """Return the default configuration for a ConfigurablePINN model."""
    return {
        'unmeasured_net_config': {
            'hidden_layers': [64, 64],
            'activation': 'tanh',
            'dropout_rate': 0.0,
            'init_method': 'xavier_normal'
        },
        'acceleration_net_config': {
            'hidden_layers': [64, 64],
            'activation': 'tanh',
            'dropout_rate': 0.0,
            'init_method': 'xavier_normal'
        },
        'param_init_config': {
            'method': 'fixed',
            'values': {
                'M1': 50.0, 'M2': 3.5, 'M3': 3.5,
                'D1': 3000.0, 'D2': 3000.0, 'D3': 3000.0,
                'K1': 3.4635e6, 'K2': 3.8127e6, 'E1': 5.0e-6
            }
        },
        'enable_mass_constraints': True  # Default to True for backward compatibility
    }

def get_synthetic_pinn_config():
    """Return the configuration for a ConfigurablePINN model optimized for synthetic data."""
    return {
        'unmeasured_net_config': {
            'hidden_layers': [64, 64],
            'activation': 'tanh',
            'dropout_rate': 0.0,
            'init_method': 'xavier_normal'
        },
        'acceleration_net_config': {
            'hidden_layers': [64, 64],
            'activation': 'tanh',
            'dropout_rate': 0.0,
            'init_method': 'xavier_normal'
        },
        'param_init_config': {
            'method': 'fixed',
            'values': {
                'M1': 50.0, 'M2': 3.5, 'M3': 3.5,
                'D1': 3000.0, 'D2': 3000.0, 'D3': 3000.0,
                'K1': 3.4635e6, 'K2': 3.8127e6, 'E1': 5.0e-6
            }
        },
        'enable_mass_constraints': False  # Disable mass constraints for synthetic data
    }

def adaptive_custom_loss(model, x, y_true, X_max=None, X_min=None, y_max=None, y_min=None):
    """
    Custom loss function that returns individual loss components for adaptive weighting.
    Computes physics residuals using normalized values for proper scaling.
    
    Parameters:
    - model: The PINN model being trained
    - x: Input data tensor (already normalized)
    - y_true: Target output tensor (already normalized)
    - X_max, X_min: Input normalization parameters (not used for physics)
    - y_max, y_min: Output normalization parameters (not used for physics)
    
    Returns:
    - Tuple of individual loss components
    """
    try:
        # Get model predictions (model expects normalized inputs)
        y_pred = model(x)
        
        # Compute the data loss using RMSE on normalized values
        data_loss = torch.sqrt(nn.MSELoss()(y_pred, y_true) + 1e-8)
        
        # Compute physics-based residuals using normalized values
        # Pass normalization parameters for compatibility (not used in compute_residuals)
        residuals = model.compute_residuals(x, y_pred, X_max, X_min, y_max, y_min)
        
        # Extract and handle individual residuals
        residual1, residual2, residual3, residual4, residualMass1, residualMass2 = residuals
        
        # Handle NaNs and clip extreme values
        residual1 = torch.nan_to_num(residual1, nan=0.0, posinf=1e10, neginf=-1e10)
        residual2 = torch.nan_to_num(residual2, nan=0.0, posinf=1e10, neginf=-1e10)
        residual3 = torch.nan_to_num(residual3, nan=0.0, posinf=1e10, neginf=-1e10)
        residual4 = torch.nan_to_num(residual4, nan=0.0, posinf=1e10, neginf=-1e10)
        residualMass1 = torch.nan_to_num(residualMass1, nan=0.0, posinf=1e10, neginf=-1e10)
        residualMass2 = torch.nan_to_num(residualMass2, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Clip residuals to avoid extreme values
        residual1 = torch.clamp(residual1, -1e8, 1e8)
        residual2 = torch.clamp(residual2, -1e8, 1e8)
        residual3 = torch.clamp(residual3, -1e8, 1e8)
        residual4 = torch.clamp(residual4, -1e8, 1e8)
        residualMass1 = torch.clamp(residualMass1, -1e8, 1e8)
        residualMass2 = torch.clamp(residualMass2, -1e8, 1e8)
        
        # Compute individual RMSE losses for each residual
        res1_loss = torch.sqrt(torch.mean(residual1**2) + 1e-8)
        res2_loss = torch.sqrt(torch.mean(residual2**2) + 1e-8)
        res3_loss = torch.sqrt(torch.mean(residual3**2) + 1e-8)
        res4_loss = torch.sqrt(torch.mean(residual4**2) + 1e-8)
        
        # CRITICAL FIX: Conditionally handle mass constraint losses
        # For synthetic data (enable_mass_constraints=False), mass residuals are zeros
        # We should not include them in the loss computation to avoid constant losses
        if model.enable_mass_constraints:
            # For real data: compute actual mass constraint losses
            resMass1_loss = torch.sqrt(torch.mean(residualMass1**2) + 1e-8)
            resMass2_loss = torch.sqrt(torch.mean(residualMass2**2) + 1e-8)
        else:
            # For synthetic data: use zeros to maintain compatibility with training scripts
            # but these won't affect the actual loss computation
            resMass1_loss = torch.tensor(0.0, device=x.device, requires_grad=False)
            resMass2_loss = torch.tensor(0.0, device=x.device, requires_grad=False)
        
        # Protect against NaN in losses
        data_loss = torch.nan_to_num(data_loss, nan=1e4, posinf=1e4, neginf=1e4)
        res1_loss = torch.nan_to_num(res1_loss, nan=1e4, posinf=1e4, neginf=1e4)
        res2_loss = torch.nan_to_num(res2_loss, nan=1e4, posinf=1e4, neginf=1e4)
        res3_loss = torch.nan_to_num(res3_loss, nan=1e4, posinf=1e4, neginf=1e4)
        res4_loss = torch.nan_to_num(res4_loss, nan=1e4, posinf=1e4, neginf=1e4)
        resMass1_loss = torch.nan_to_num(resMass1_loss, nan=1e4, posinf=1e4, neginf=1e4)
        resMass2_loss = torch.nan_to_num(resMass2_loss, nan=1e4, posinf=1e4, neginf=1e4)
        
        return data_loss, res1_loss, res2_loss, res3_loss, res4_loss, resMass1_loss, resMass2_loss
    
    except Exception as e:
        # Fallback values in case of error
        device = x.device
        print(f"Error in adaptive_custom_loss: {e}")
        return (
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device)
        ) 