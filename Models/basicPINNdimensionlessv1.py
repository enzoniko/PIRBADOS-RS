import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
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
    A configurable Physics-Informed Neural Network (PINN) model with dimensionless formulation.
    
    This implementation uses on-the-fly dimensionless transformation as described in the theoretical framework.
    The neural networks operate on normalized data [0,1], which is then denormalized and converted to 
    dimensionless form using current learned parameters before physics calculations.
    
    Parameters:
    - unmeasured_net_config: Configuration for the unmeasured parameters network
    - acceleration_net_config: Configuration for the acceleration network  
    - param_init_config: Configuration for physical parameter initialization
    - enable_mass_constraints: Whether to include mass constraints (default: True)
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
                    'K1': 3.4635e6, 'K2': 3.8127e6, 'E1': 5.0e-6,
                    'c': 1e-4  # Characteristic bearing clearance
                }
            }

        # Network for estimating unmeasured parameters (10 inputs, 2 outputs)
        # Inputs: 10 dimensionless features (same as acceleration network)
        # Outputs: 2 unmodeled force terms (fC, fD) from theoretical framework
        self.NNforUnmeasured = ConfigurableMLP(
            input_dim=10,  # 10 dimensionless features
            hidden_layers=unmeasured_net_config['hidden_layers'],
            output_dim=2,  # 2 unmeasured force parameters (fC, fD)
            activation=unmeasured_net_config['activation'],
            dropout_rate=unmeasured_net_config['dropout_rate'],
            init_method=unmeasured_net_config['init_method']
        )
        
        # Network for estimating accelerations (10 inputs, 4 outputs)
        # Inputs: 10 dimensionless features
        # Outputs: 4 dimensionless accelerations
        self.NNforAccelerations = ConfigurableMLP(
            input_dim=10,  # 10 dimensionless features
            hidden_layers=acceleration_net_config['hidden_layers'],
            output_dim=4,  # 4 dimensionless accelerations
            activation=acceleration_net_config['activation'],
            dropout_rate=acceleration_net_config['dropout_rate'],
            init_method=acceleration_net_config['init_method']
        )
        
        # Initialize learnable physical parameters with reparameterization for positivity
        self._initialize_physical_parameters(param_init_config)
        
        # Register gravity constant as a buffer so it follows the model's device
        self.register_buffer('g', torch.tensor(9.81, dtype=torch.float64))
        
        # Convert model to double precision
        self.double()
        
    def _initialize_physical_parameters(self, config):
        """
        Initialize physical parameters using reparameterization for positivity constraints.
        Parameters are stored as raw values and converted to positive via exp() or softplus().
        """
        method = config['method']
        values = config['values']
        
        if method == 'fixed':
            # Initialize raw parameters (will be mapped to positive values)
            self.M1_raw = nn.Parameter(torch.tensor(math.log(float(values.get('M1', 10.0))), dtype=torch.float64))
            self.M2_raw = nn.Parameter(torch.tensor(math.log(float(values.get('M2', 10.0))), dtype=torch.float64))
            self.M3_raw = nn.Parameter(torch.tensor(math.log(float(values.get('M3', 11.0))), dtype=torch.float64))
            self.D1_raw = nn.Parameter(torch.tensor(math.log(float(values.get('D1', 10.0))), dtype=torch.float64))
            self.D2_raw = nn.Parameter(torch.tensor(math.log(float(values.get('D2', 10.0))), dtype=torch.float64))
            self.D3_raw = nn.Parameter(torch.tensor(math.log(float(values.get('D3', 10.0))), dtype=torch.float64))
            self.K1_raw = nn.Parameter(torch.tensor(math.log(float(values.get('K1', 10.0))), dtype=torch.float64))
            self.K2_raw = nn.Parameter(torch.tensor(math.log(float(values.get('K2', 10.0))), dtype=torch.float64))
            self.E1_raw = nn.Parameter(torch.tensor(math.log(float(values.get('E1', 10.0))), dtype=torch.float64))
            self.c_raw = nn.Parameter(torch.tensor(math.log(float(values.get('c', 1e-4))), dtype=torch.float64))
        
        elif method == 'uniform':
            # Initialize with uniform distribution (in log space for positivity)
            ranges = {k: v if isinstance(v, tuple) else (v*0.5, v*1.5) for k, v in values.items()}
            self.M1_raw = nn.Parameter(torch.tensor(math.log(np.random.uniform(*ranges.get('M1', (5.0, 15.0)))), dtype=torch.float64))
            self.M2_raw = nn.Parameter(torch.tensor(math.log(np.random.uniform(*ranges.get('M2', (5.0, 15.0)))), dtype=torch.float64))
            self.M3_raw = nn.Parameter(torch.tensor(math.log(np.random.uniform(*ranges.get('M3', (5.0, 15.0)))), dtype=torch.float64))
            self.D1_raw = nn.Parameter(torch.tensor(math.log(np.random.uniform(*ranges.get('D1', (5.0, 15.0)))), dtype=torch.float64))
            self.D2_raw = nn.Parameter(torch.tensor(math.log(np.random.uniform(*ranges.get('D2', (5.0, 15.0)))), dtype=torch.float64))
            self.D3_raw = nn.Parameter(torch.tensor(math.log(np.random.uniform(*ranges.get('D3', (5.0, 15.0)))), dtype=torch.float64))
            self.K1_raw = nn.Parameter(torch.tensor(math.log(np.random.uniform(*ranges.get('K1', (5.0, 15.0)))), dtype=torch.float64))
            self.K2_raw = nn.Parameter(torch.tensor(math.log(np.random.uniform(*ranges.get('K2', (5.0, 15.0)))), dtype=torch.float64))
            self.E1_raw = nn.Parameter(torch.tensor(math.log(np.random.uniform(*ranges.get('E1', (5.0, 15.0)))), dtype=torch.float64))
            self.c_raw = nn.Parameter(torch.tensor(math.log(np.random.uniform(*ranges.get('c', (1e-5, 1e-3)))), dtype=torch.float64))
        
        elif method == 'normal':
            # Initialize with normal distribution (in log space for positivity)
            params = {k: v if isinstance(v, tuple) else (v, v*0.1) for k, v in values.items()}
            self.M1_raw = nn.Parameter(torch.tensor(math.log(abs(np.random.normal(*params.get('M1', (10.0, 1.0))))), dtype=torch.float64))
            self.M2_raw = nn.Parameter(torch.tensor(math.log(abs(np.random.normal(*params.get('M2', (10.0, 1.0))))), dtype=torch.float64))
            self.M3_raw = nn.Parameter(torch.tensor(math.log(abs(np.random.normal(*params.get('M3', (11.0, 1.0))))), dtype=torch.float64))
            self.D1_raw = nn.Parameter(torch.tensor(math.log(abs(np.random.normal(*params.get('D1', (10.0, 1.0))))), dtype=torch.float64))
            self.D2_raw = nn.Parameter(torch.tensor(math.log(abs(np.random.normal(*params.get('D2', (10.0, 1.0))))), dtype=torch.float64))
            self.D3_raw = nn.Parameter(torch.tensor(math.log(abs(np.random.normal(*params.get('D3', (10.0, 1.0))))), dtype=torch.float64))
            self.K1_raw = nn.Parameter(torch.tensor(math.log(abs(np.random.normal(*params.get('K1', (10.0, 1.0))))), dtype=torch.float64))
            self.K2_raw = nn.Parameter(torch.tensor(math.log(abs(np.random.normal(*params.get('K2', (10.0, 1.0))))), dtype=torch.float64))
            self.E1_raw = nn.Parameter(torch.tensor(math.log(abs(np.random.normal(*params.get('E1', (10.0, 1.0))))), dtype=torch.float64))
            self.c_raw = nn.Parameter(torch.tensor(math.log(abs(np.random.normal(*params.get('c', (1e-4, 1e-5))))), dtype=torch.float64))
        
        else:
            raise ValueError(f"Unsupported parameter initialization method: {method}")

    # Property methods for accessing positive parameters (for compatibility)
    @property
    def M1(self):
        return torch.exp(self.M1_raw)
    
    @property 
    def M2(self):
        return torch.exp(self.M2_raw)
        
    @property
    def M3(self):
        return torch.exp(self.M3_raw)
        
    @property
    def D1(self):
        return torch.exp(self.D1_raw)
        
    @property
    def D2(self):
        return torch.exp(self.D2_raw)
        
    @property
    def D3(self):
        return torch.exp(self.D3_raw)
        
    @property
    def K1(self):
        return torch.exp(self.K1_raw)
        
    @property
    def K2(self):
        return torch.exp(self.K2_raw)
        
    @property
    def E1(self):
        return torch.exp(self.E1_raw)
        
    @property
    def c(self):
        return torch.exp(self.c_raw)

    def _denormalize_data(self, x_norm, X_max, X_min):
        """
        Denormalize input data from [0,1] range to physical units.
        
        Parameters:
        - x_norm: Normalized input tensor [0,1] range
        - X_max, X_min: Normalization parameters
        
        Returns:
        - Denormalized data in physical units
        """
        if X_max is None or X_min is None:
            return x_norm  # Return as-is if no normalization params
            
        X_max = X_max.double()
        X_min = X_min.double()
        
        # Handle 1D normalization parameters
        if X_min.dim() == 1:
            X_min = X_min.unsqueeze(0)  # Shape: (1, features)
            X_max = X_max.unsqueeze(0)
            
        X_range = X_max - X_min + 1e-12
        return x_norm * X_range + X_min

    def _convert_to_dimensionless(self, x_phys):
        """
        Convert physical quantities to dimensionless using current learned parameters.
        
        Parameters:
        - x_phys: Physical quantities [x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, omega, t]
        
        Returns:
        - Dimensionless quantities
        """
        x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, omega, t = torch.split(x_phys, 1, dim=1)
        
        # Get current parameters
        c = self.c
        
        # Apply dimensionless transformations from theoretical framework
        x2_dimless = x2 / c
        y2_dimless = y2 / c  
        x3_dimless = x3 / c
        y3_dimless = y3 / c
        x2_dot_dimless = x2_dot / (c * omega)
        y2_dot_dimless = y2_dot / (c * omega)
        x3_dot_dimless = x3_dot / (c * omega) 
        y3_dot_dimless = y3_dot / (c * omega)
        
        # Omega and time remain as-is for the networks (they need these values)
        return torch.cat([x2_dot_dimless, y2_dot_dimless, x3_dot_dimless, y3_dot_dimless,
                          x2_dimless, y2_dimless, x3_dimless, y3_dimless, omega, t], dim=1)

    def forward(self, x):
        """
        Forward pass with corrected and simplified architecture.
        
        Both neural networks now operate on the same 10-feature normalized input vector 'x',
        which represents the dimensionless state of the system after proper scaling.
        
        Parameters:
        - x: Normalized input tensor [0,1] range with shape (batch_size, 10)
        
        Returns:
        - Normalized accelerations for data loss computation
        """
        # Ensure input is double precision
        x = x.double()

        # Both networks now operate on the same 10-feature normalized input vector 'x'.
        # This vector represents the dimensionless state of the system.
        
        # Estimate the unmeasured force parameters (fC, fD) from theoretical framework
        unmeasured_forces = self.NNforUnmeasured(x)
        self.fC, self.fD = torch.split(unmeasured_forces, 1, dim=1)

        # Estimate the accelerations from the same normalized input
        accelerations_norm = self.NNforAccelerations(x)

        # Return normalized accelerations (for data loss)
        return accelerations_norm
    
    def compute_residuals(self, x, pred, X_max=None, X_min=None, y_max=None, y_min=None):
        """
        Compute the dimensionless physics-based residuals for the system.
        
        This method:
        1. Denormalizes inputs and outputs to physical units
        2. Converts to dimensionless using current parameters  
        3. Applies dimensionless physics equations from theoretical framework
        
        Parameters:
        - x: Input tensor (batch_size, features) - NORMALIZED [0,1]
        - pred: Predicted accelerations (batch_size, 4) - NORMALIZED [0,1]
        - X_max, X_min: Input normalization parameters
        - y_max, y_min: Output normalization parameters
        
        Returns:
        - Dictionary containing residuals and intermediate dimensionless quantities
        """
        # Ensure double precision
        x = x.double()
        pred = pred.double()
        
        # Step 1: Denormalize inputs to physical units
        x_phys = self._denormalize_data(x, X_max, X_min)
        
        # Step 2: Denormalize predictions to physical accelerations
        if y_max is not None and y_min is not None:
            y_max = y_max.double()
            y_min = y_min.double()
            
            if y_min.dim() == 1:
                y_min = y_min.unsqueeze(0)
                y_max = y_max.unsqueeze(0)
                
            y_range = y_max - y_min + 1e-12
            pred_phys = pred * y_range + y_min
        else:
            pred_phys = pred  # Fallback if no normalization params
            
        # Step 3: Extract physical quantities
        x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, omega, t = torch.split(x_phys, 1, dim=1)
        x2_ddot, y2_ddot, x3_ddot, y3_ddot = torch.split(pred_phys, 1, dim=1)
        
        # Step 4: Convert to dimensionless using current parameters
        c = self.c
        M1, M2, M3 = self.M1, self.M2, self.M3
        D1, D2, D3 = self.D1, self.D2, self.D3
        K1, K2 = self.K1, self.K2
        
        # Dimensionless kinematic states
        x2_dimless = x2 / c
        y2_dimless = y2 / c
        x3_dimless = x3 / c  
        y3_dimless = y3 / c
        x2_dot_dimless = x2_dot / (c * omega)
        y2_dot_dimless = y2_dot / (c * omega)
        x3_dot_dimless = x3_dot / (c * omega)
        y3_dot_dimless = y3_dot / (c * omega)
        x2_ddot_dimless = x2_ddot / (c * omega**2)
        y2_ddot_dimless = y2_ddot / (c * omega**2)
        x3_ddot_dimless = x3_ddot / (c * omega**2)
        y3_ddot_dimless = y3_ddot / (c * omega**2)
        
        # Step 5: Apply dimensionless physics equations from theoretical framework
        # Based on equations (21) and (22) from the paper
        
        # Dimensionless parameters
        d1 = D1 / (M1 * omega)
        d2 = D2 / (M2 * omega)  
        d3 = D3 / (M3 * omega)
        k1_over_m1 = K1 / (M1 * omega**2)
        k1_over_m2 = K1 / (M2 * omega**2)
        k1_over_m3 = K1 / (M3 * omega**2)
        k2_over_m2 = K2 / (M2 * omega**2)
        k2_over_m3 = K2 / (M3 * omega**2)
        
        # Physics residuals in dimensionless form (from theoretical framework)
        # e_phy_C: Equation (21)
        residual_C = (x3_ddot_dimless + d3 * x3_dot_dimless + k2_over_m3 * (x3_dimless - x2_dimless) 
                     - (K2 * M2) / (K1 * M3) * (x2_ddot_dimless + d2 * x2_dot_dimless)) - self.fC
        
        # e_phy_D: Equation (22)  
        residual_D = (y3_ddot_dimless + d3 * y3_dot_dimless + k2_over_m3 * (y3_dimless - y2_dimless)
                     - (K2 * M2) / (K1 * M3) * (y2_ddot_dimless + d2 * y2_dot_dimless)
                     + (K2 * M2 * self.g) / (K1 * M3 * c * omega**2) 
                     - self.g / (c * omega**2)) - self.fD
        
        # For compatibility with existing training scripts, we need 6 residuals
        # Use the same residuals for the first 4 components
        residual1 = residual_C
        residual2 = residual_D  
        residual3 = residual_C  # These could be different physics equations if needed
        residual4 = residual_D
        
        # Mass constraints (dimensional, as in original)
        if self.enable_mass_constraints:
            residualMass1 = M1 + M2 + M3 - 22.0  # Total mass constraint
            residualMass2 = M2 - M3                # Equal bearing masses
        else:
            # For synthetic data compatibility
            residualMass1 = torch.zeros_like(residual1)
            residualMass2 = torch.zeros_like(residual2)
            
        # Return dictionary with detailed information for debugging
        results = {
            'residuals': (residual1, residual2, residual3, residual4, residualMass1, residualMass2),
            'dimensionless_kinematics': {
                'x2_dimless': x2_dimless, 'y2_dimless': y2_dimless,
                'x3_dimless': x3_dimless, 'y3_dimless': y3_dimless,
                'x2_dot_dimless': x2_dot_dimless, 'y2_dot_dimless': y2_dot_dimless,
                'x3_dot_dimless': x3_dot_dimless, 'y3_dot_dimless': y3_dot_dimless,
                'x2_ddot_dimless': x2_ddot_dimless, 'y2_ddot_dimless': y2_ddot_dimless,
                'x3_ddot_dimless': x3_ddot_dimless, 'y3_ddot_dimless': y3_ddot_dimless
            },
            'dimensionless_params': {
                'd1': d1, 'd2': d2, 'd3': d3, 
                'k1_over_m1': k1_over_m1, 'k1_over_m2': k1_over_m2, 'k1_over_m3': k1_over_m3,
                'k2_over_m2': k2_over_m2, 'k2_over_m3': k2_over_m3
            }
        }
        return results

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
                'K1': 3.4635e6, 'K2': 3.8127e6, 'E1': 5.0e-6,
                'c': 1e-4
            }
        },
        'enable_mass_constraints': True
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
                'K1': 3.4635e6, 'K2': 3.8127e6, 'E1': 5.0e-6,
                'c': 1e-4
            }
        },
        'enable_mass_constraints': False  # Disable for synthetic data
    }

def adaptive_custom_loss(model, x, y_true, X_max=None, X_min=None, y_max=None, y_min=None, debug=False):
    """
    Custom loss function for dimensionless PINN that handles proper scaling.
    
    This function:
    1. Computes data loss using normalized predictions and targets
    2. Computes physics residuals using denormalized quantities in dimensionless form
    3. Returns the same 7-component structure as the original implementation
    
    Parameters:
    - model: The dimensionless PINN model
    - x: Input data tensor (normalized [0,1])
    - y_true: Target output tensor (normalized [0,1])
    - X_max, X_min: Input normalization parameters
    - y_max, y_min: Output normalization parameters
    - debug: Whether to print debug information
    
    Returns:
    - Tuple of individual loss components (7 components for compatibility)
    """
    try:
        # Ensure double precision
        x = x.double()
        y_true = y_true.double()
        
        # Get model predictions (normalized accelerations)
        y_pred = model(x)
        
        # Compute data loss using normalized values (RMSE)
        data_loss = torch.sqrt(torch.mean((y_pred - y_true)**2) + 1e-12)
        
        # Compute physics residuals using denormalized quantities
        results_dict = model.compute_residuals(x, y_pred, X_max, X_min, y_max, y_min)
        
        # Extract residuals from the returned dictionary
        residuals = results_dict['residuals']
        residual1, residual2, residual3, residual4, residualMass1, residualMass2 = residuals
        
        # Handle NaNs (no clipping for dimensionless residuals)
        residual1 = torch.nan_to_num(residual1, nan=0.0)
        residual2 = torch.nan_to_num(residual2, nan=0.0)
        residual3 = torch.nan_to_num(residual3, nan=0.0)
        residual4 = torch.nan_to_num(residual4, nan=0.0)
        residualMass1 = torch.nan_to_num(residualMass1, nan=0.0)
        residualMass2 = torch.nan_to_num(residualMass2, nan=0.0)
        
        # Compute individual RMSE losses for each residual
        res1_loss = torch.sqrt(torch.mean(residual1**2) + 1e-12)
        res2_loss = torch.sqrt(torch.mean(residual2**2) + 1e-12)
        res3_loss = torch.sqrt(torch.mean(residual3**2) + 1e-12)
        res4_loss = torch.sqrt(torch.mean(residual4**2) + 1e-12)
        
        # Handle mass constraints based on model settings
        if model.enable_mass_constraints:
            resMass1_loss = torch.sqrt(torch.mean(residualMass1**2) + 1e-12)
            resMass2_loss = torch.sqrt(torch.mean(residualMass2**2) + 1e-12)
        else:
            # For synthetic data: zeros with gradients for compatibility
            resMass1_loss = torch.tensor(0.0, device=x.device, dtype=torch.float64, requires_grad=True)
            resMass2_loss = torch.tensor(0.0, device=x.device, dtype=torch.float64, requires_grad=True)
        
        if debug:
            print(f"Data loss: {data_loss.item():.6e}")
            print(f"Physics losses: {res1_loss.item():.6e}, {res2_loss.item():.6e}, {res3_loss.item():.6e}, {res4_loss.item():.6e}")
            print(f"Mass losses: {resMass1_loss.item():.6e}, {resMass2_loss.item():.6e}")
            print(f"Current clearance c: {model.c.item():.6e}")
        
        return data_loss, res1_loss, res2_loss, res3_loss, res4_loss, resMass1_loss, resMass2_loss
    
    except Exception as e:
        # Fallback values in case of error
        device = x.device
        print(f"Error in adaptive_custom_loss: {e}")
        return (
            torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True),
            torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True),
            torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True),
            torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True),
            torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True),
            torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True),
            torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True)
        )