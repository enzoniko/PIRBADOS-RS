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

# ========================================
# DENORMALIZATION CONTROL FLAG
# ========================================
# This flag controls how input/output data is processed in compute_residuals():
#
# USE_PROPER_DENORMALIZATION = True (RECOMMENDED):
#   - Converts normalized [0,1] inputs back to physical units
#   - Applies proper dimensionless scaling (x/c, t*omega, etc.)
#   - Uses correct physics-based dimensionless parameters
#   - This is the CORRECT dimensionless PINN formulation
#
# USE_PROPER_DENORMALIZATION = False (EXPERIMENTAL):
#   - Treats normalized [0,1] inputs as if they were physical values
#   - Uses simplified constant dimensionless parameters
#   - FOR TESTING ONLY - not physically meaningful
#   - Useful for debugging normalization issues
#
# To test: Change the flag below, save, and re-import the module
# ========================================
USE_PROPER_DENORMALIZATION = True
# ========================================

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
    Hybrid Dimensionless Physics-Informed Neural Network (PINN) v2.
    
    This version implements a comprehensive approach with three neural networks:
    1. NNforUnmeasured: Predicts unmodeled bearing forces (fC, fD)
    2. NNforAccelerations: Predicts bearing accelerations  
    3. NNforRotorKinematics: Predicts unmeasured rotor kinematics
    
    Uses four physics residuals based on the complete equations of motion:
    - Residual 1: Rotor X-direction (with cos unbalance term)
    - Residual 2: Rotor Y-direction (with sin unbalance term)  
    - Residual 3: Bearing 2 X-direction
    - Residual 4: Bearing 2 Y-direction
    
    Parameters:
    - unmeasured_net_config: Configuration for the unmeasured forces network
    - acceleration_net_config: Configuration for the acceleration network
    - rotor_net_config: Configuration for the rotor kinematics network
    - param_init_config: Configuration for physical parameter initialization
    - enable_mass_constraints: Whether to include mass constraints (default: True)
    """
    def __init__(self, unmeasured_net_config=None, acceleration_net_config=None, 
                 rotor_net_config=None, param_init_config=None, enable_mass_constraints=True):
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
            
        if rotor_net_config is None:
            rotor_net_config = {
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

        # Network 1: Predicts unmodeled bearing forces (fC, fD)
        # Inputs: 10 dimensionless features
        # Outputs: 2 unmodeled force terms (fC, fD)
        self.NNforUnmeasured = ConfigurableMLP(
            input_dim=10,  # 10 dimensionless features
            hidden_layers=unmeasured_net_config['hidden_layers'],
            output_dim=2,  # 2 unmeasured force parameters (fC, fD)
            activation=unmeasured_net_config['activation'],
            dropout_rate=unmeasured_net_config['dropout_rate'],
            init_method=unmeasured_net_config['init_method']
        )
        
        # Network 2: Predicts bearing accelerations
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
        
        # Network 3: Predicts unmeasured rotor kinematics (NEW in v2)
        # Inputs: 10 dimensionless features (from bearings)
        # Outputs: 6 dimensionless rotor states (x1, y1, x1_dot, y1_dot, x1_ddot, y1_ddot)
        self.NNforRotorKinematics = ConfigurableMLP(
            input_dim=10,  # 10 dimensionless features
            hidden_layers=rotor_net_config['hidden_layers'],
            output_dim=6,  # 6 rotor kinematic states
            activation=rotor_net_config['activation'],
            dropout_rate=rotor_net_config['dropout_rate'],
            init_method=rotor_net_config['init_method']
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
        Parameters are stored as raw values and converted to positive via exp().
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

    def forward(self, x):
        """
        Forward pass for the hybrid architecture with three neural networks.
        
        All three networks operate on the same 10-feature normalized input vector 'x',
        which represents the dimensionless state of the system.
        
        Parameters:
        - x: Normalized input tensor [0,1] range with shape (batch_size, 10)
        
        Returns:
        - Normalized accelerations for data loss computation
        """
        # Ensure input is double precision
        x = x.double()

        # Network 1: Predict unmodeled bearing forces (fC, fD)
        unmeasured_forces = self.NNforUnmeasured(x)
        self.fC, self.fD = torch.split(unmeasured_forces, 1, dim=1)

        # Network 2: Predict unmeasured rotor kinematics (NEW in v2)
        rotor_kinematics = self.NNforRotorKinematics(x)
        # Store predictions for use in compute_residuals
        (self.x1_dimless, self.y1_dimless,
         self.x1_dot_dimless, self.y1_dot_dimless,
         self.x1_ddot_dimless, self.y1_ddot_dimless) = torch.split(rotor_kinematics, 1, dim=1)

        # Network 3: Predict bearing accelerations (main output for data loss)
        accelerations_norm = self.NNforAccelerations(x)

        # Return normalized accelerations (for data loss)
        return accelerations_norm
    
    def compute_residuals(self, x, pred, X_max=None, X_min=None, y_max=None, y_min=None):
        """
        Compute four comprehensive physics-based residuals using the hybrid approach.
        
        This method implements the complete equations of motion:
        1. Rotor X-direction (with cos unbalance term)
        2. Rotor Y-direction (with sin unbalance term)  
        3. Bearing 2 X-direction
        4. Bearing 2 Y-direction
        
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
        
        if USE_PROPER_DENORMALIZATION:
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
                
            # Debug info only on first call
            if not hasattr(self, '_denorm_debug_printed'):
                print(f"[INFO] Using proper denormalization - physical units mode")
                self._denorm_debug_printed = True
        else:
            # EXPERIMENTAL: Skip denormalization (use normalized values as if they were physical)
            # WARNING: This is NOT the correct dimensionless formulation!
            x_phys = x
            pred_phys = pred
            # Debug info only on first call
            if not hasattr(self, '_exp_debug_printed'):
                print(f"[WARNING] EXPERIMENTAL MODE: Skipping denormalization - normalized values treated as physical")
                self._exp_debug_printed = True
            
        # Step 3: Extract physical quantities
        x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, omega, t = torch.split(x_phys, 1, dim=1)
        x2_ddot, y2_ddot, x3_ddot, y3_ddot = torch.split(pred_phys, 1, dim=1)
        
        # Step 4: Get dimensional parameters
        M1, M2, M3 = self.M1, self.M2, self.M3
        D1, D2, D3 = self.D1, self.D2, self.D3
        K1, K2 = self.K1, self.K2
        E1 = self.E1
        c = self.c
        m1_total = M1  # Assuming Mf (seal mass) is negligible for MAFAULDA
        
        # Step 5: Convert measured bearing states to dimensionless form
        if USE_PROPER_DENORMALIZATION:
            # Proper dimensionless conversion (physical units → dimensionless)
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
        else:
            # EXPERIMENTAL: Treat normalized values as already dimensionless
            # Since inputs are [0,1], we can use them directly as dimensionless quantities
            x2_dimless = x2
            y2_dimless = y2
            x3_dimless = x3
            y3_dimless = y3
            x2_dot_dimless = x2_dot
            y2_dot_dimless = y2_dot
            x3_dot_dimless = x3_dot
            y3_dot_dimless = y3_dot
            x2_ddot_dimless = x2_ddot
            y2_ddot_dimless = y2_ddot
            x3_ddot_dimless = x3_ddot
            y3_ddot_dimless = y3_ddot
            # Minimal debug output
        
        # Step 6: Define dimensionless parameters for physics equations
        if USE_PROPER_DENORMALIZATION:
            # Proper dimensionless parameter calculation
            d1_dimless = D1 / (m1_total * omega)
            d2_dimless = D2 / (M2 * omega)
            d3_dimless = D3 / (M3 * omega)
            k1_m1_dimless = K1 / (m1_total * omega**2)
            k2_m1_dimless = K2 / (m1_total * omega**2)
            k1_m2_dimless = K1 / (M2 * omega**2)
            k2_m3_dimless = K2 / (M3 * omega**2)
            e1_dimless = E1 / c
            g_bar_dimless = self.g / (c * omega**2)
        else:
            # EXPERIMENTAL: Use simplified dimensionless parameters
            # Since we're treating normalized values as dimensionless, 
            # we need to scale the parameters appropriately
            omega_normalized = omega  # omega is already in normalized range [0,1]
            d1_dimless = torch.ones_like(omega) * 0.1  # Small damping
            d2_dimless = torch.ones_like(omega) * 0.1
            d3_dimless = torch.ones_like(omega) * 0.1
            k1_m1_dimless = torch.ones_like(omega) * 1.0  # Moderate stiffness
            k2_m1_dimless = torch.ones_like(omega) * 1.0
            k1_m2_dimless = torch.ones_like(omega) * 1.0
            k2_m3_dimless = torch.ones_like(omega) * 1.0
            e1_dimless = torch.ones_like(omega) * 0.01  # Small unbalance
            g_bar_dimless = torch.ones_like(omega) * 0.01  # Small gravity effect
            # Using simplified parameters for experimental mode
        
        # Step 7: Compute the four physics residuals
        if USE_PROPER_DENORMALIZATION:
            # Use proper omega*t for unbalance terms
            omega_t = omega * t
        else:
            # EXPERIMENTAL: Use normalized time values
            # Since omega and t are normalized [0,1], their product gives a phase-like quantity
            omega_t = omega * t * 10.0  # Scale up to get reasonable phase variation
        
        # RESIDUAL 1: Rotor X-direction equation (with cos unbalance term)
        residual1 = (self.x1_ddot_dimless + d1_dimless * self.x1_dot_dimless
                     + k1_m1_dimless * (self.x1_dimless - x2_dimless)
                     + k2_m1_dimless * (self.x1_dimless - x3_dimless)
                     - e1_dimless * torch.cos(omega_t))

        # RESIDUAL 2: Rotor Y-direction equation (with sin unbalance term)
        residual2 = (self.y1_ddot_dimless + d1_dimless * self.y1_dot_dimless
                     + k1_m1_dimless * (self.y1_dimless - y2_dimless)
                     + k2_m1_dimless * (self.y1_dimless - y3_dimless)
                     - e1_dimless * torch.sin(omega_t)
                     + g_bar_dimless)

        # RESIDUAL 3: Bearing 2 X-direction equation
        residual3 = (x2_ddot_dimless + d2_dimless * x2_dot_dimless
                     + k1_m2_dimless * (x2_dimless - self.x1_dimless)
                     - self.fC)  # fC learns the dimensionless bearing force

        # RESIDUAL 4: Bearing 2 Y-direction equation  
        residual4 = (y2_ddot_dimless + d2_dimless * y2_dot_dimless
                     + k1_m2_dimless * (y2_dimless - self.y1_dimless)
                     - self.fD   # fD learns the dimensionless bearing force
                     + g_bar_dimless)
        
        # Mass constraints (dimensional, as in original)
        if self.enable_mass_constraints:
            residualMass1 = M1 + M2 + M3 - 22.0  # Total mass constraint
            residualMass2 = M2 - M3                # Equal bearing masses
        else:
            # For synthetic data compatibility
            residualMass1 = torch.zeros_like(residual1)
            residualMass2 = torch.zeros_like(residual2)
            
        # Optional debug summary (only when verbose debugging is needed)
        # mode_str = "PROPER DENORMALIZATION" if USE_PROPER_DENORMALIZATION else "EXPERIMENTAL (NO DENORM)"
        # print(f"[DEBUG] Residual magnitudes: R1={residual1.abs().mean().item():.3e}, R2={residual2.abs().mean().item():.3e}, R3={residual3.abs().mean().item():.3e}, R4={residual4.abs().mean().item():.3e}")
        
        # Return dictionary with detailed information for debugging
        results = {
            'residuals': (residual1, residual2, residual3, residual4, residualMass1, residualMass2),
            'dimensionless_kinematics': {
                # Measured bearing states
                'x2_dimless': x2_dimless, 'y2_dimless': y2_dimless,
                'x3_dimless': x3_dimless, 'y3_dimless': y3_dimless,
                'x2_dot_dimless': x2_dot_dimless, 'y2_dot_dimless': y2_dot_dimless,
                'x3_dot_dimless': x3_dot_dimless, 'y3_dot_dimless': y3_dot_dimless,
                'x2_ddot_dimless': x2_ddot_dimless, 'y2_ddot_dimless': y2_ddot_dimless,
                'x3_ddot_dimless': x3_ddot_dimless, 'y3_ddot_dimless': y3_ddot_dimless,
                # Predicted rotor states (NEW in v2)
                'x1_dimless': self.x1_dimless, 'y1_dimless': self.y1_dimless,
                'x1_dot_dimless': self.x1_dot_dimless, 'y1_dot_dimless': self.y1_dot_dimless,
                'x1_ddot_dimless': self.x1_ddot_dimless, 'y1_ddot_dimless': self.y1_ddot_dimless
            },
            'dimensionless_params': {
                'd1': d1_dimless, 'd2': d2_dimless, 'd3': d3_dimless,
                'k1_m1': k1_m1_dimless, 'k2_m1': k2_m1_dimless,
                'k1_m2': k1_m2_dimless, 'k2_m3': k2_m3_dimless,
                'e1': e1_dimless, 'g_bar': g_bar_dimless
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
        'rotor_net_config': {
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
        'rotor_net_config': {
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
    Custom loss function for hybrid dimensionless PINN v2.
    
    This function computes loss components for the comprehensive four-residual approach:
    1. Data loss using normalized predictions and targets
    2. Four physics residuals: rotor (2) + bearing (2) equations
    3. Mass constraint losses
    
    Parameters:
    - model: The hybrid dimensionless PINN model
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
        
        # Compute physics residuals using the comprehensive four-residual approach
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
            print(f"Rotor states range: x1=[{model.x1_dimless.min().item():.3f}, {model.x1_dimless.max().item():.3f}], y1=[{model.y1_dimless.min().item():.3f}, {model.y1_dimless.max().item():.3f}]")
        
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