import torch 
import torch.nn as nn

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

        self.NNforUnmeasured = nn.Sequential(
            nn.Linear(15, 64), # 15 inputs (x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, Omega, t, M1, D1, K1, K2, E1)
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4) # 4 outputs (fA, fB, fC, fD)
        )

        # Define the neural network architecture for estimating the accelerations:
        # x2_ddot, x3_ddot, y2_ddot, y3_ddot
        self.NNforAccelerations = nn.Sequential(
            nn.Linear(10, 64), # 8 inputs (x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, omega, t)
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4) # 4 outputs (x2_ddot, y2_ddot, x3_ddot, y3_ddot)
        )

        # Define learnable parameters (masses, damping coefficients, spring constants, E1)
        self.M1 = nn.Parameter(torch.tensor(1.0))
        self.M2 = nn.Parameter(torch.tensor(1.0))
        self.M3 = nn.Parameter(torch.tensor(1.0))
        self.D1 = nn.Parameter(torch.tensor(1.0))
        self.D2 = nn.Parameter(torch.tensor(1.0))
        self.D3 = nn.Parameter(torch.tensor(1.0))
        self.K1 = nn.Parameter(torch.tensor(1.0))
        self.K2 = nn.Parameter(torch.tensor(1.0))
        self.E1 = nn.Parameter(torch.tensor(1.0))

        # Define the gravity constant
        self.g = torch.tensor(9.81)

        # Define the Lagrange Parameters of the loss function
        self.lagrange1 = nn.Parameter(torch.tensor(1.0))
        self.lagrange2 = nn.Parameter(torch.tensor(1.0))
        self.lagrange3 = nn.Parameter(torch.tensor(1.0))
        self.lagrange4 = nn.Parameter(torch.tensor(1.0))
        self.lagrange5 = nn.Parameter(torch.tensor(1.0))
        self.lagrange6 = nn.Parameter(torch.tensor(1.0))
        self.lagrange7 = nn.Parameter(torch.tensor(1.0))
        self.lagrange8 = nn.Parameter(torch.tensor(1.0))
        self.lagrange9 = nn.Parameter(torch.tensor(1.0))
        self.lagrange10 = nn.Parameter(torch.tensor(1.0))
        self.lagrange11 = nn.Parameter(torch.tensor(1.0))
        self.lagrange12 = nn.Parameter(torch.tensor(1.0))
        self.lagrange13 = nn.Parameter(torch.tensor(1.0))
        self.lagrange14 = nn.Parameter(torch.tensor(1.0))


    def forward(self, x):

        # Use clamp to ensure positive values for masses, damping coefficients, spring constants, E1
        """ M1 = torch.clamp(self.M1, 1e-8)    
        M2 = torch.clamp(self.M2, 1e-8)
        M3 = torch.clamp(self.M3, 1e-8)
        D1 = torch.clamp(self.D1, 1e-8)
        D2 = torch.clamp(self.D2, 1e-8)
        D3 = torch.clamp(self.D3, 1e-8)
        K1 = torch.clamp(self.K1, 1e-8)
        K2 = torch.clamp(self.K2, 1e-8)
        E1 = torch.clamp(self.E1, 1e-8) """

        M1 = self.M1    
        M2 = self.M2
        M3 = self.M3
        D1 = self.D1
        D2 = self.D2
        D3 = self.D3
        K1 = self.K1
        K2 = self.K2
        E1 = self.E1


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
    
    def compute_residuals(self, x, pred):
        
        """ M1 = torch.clamp(self.M1, 1e-8)    
        M2 = torch.clamp(self.M2, 1e-8)
        M3 = torch.clamp(self.M3, 1e-8)
        D1 = torch.clamp(self.D1, 1e-8)
        D2 = torch.clamp(self.D2, 1e-8)
        D3 = torch.clamp(self.D3, 1e-8)
        K1 = torch.clamp(self.K1, 1e-8)
        K2 = torch.clamp(self.K2, 1e-8)
        E1 = torch.clamp(self.E1, 1e-8) """
        M1 = self.M1    
        M2 = self.M2
        M3 = self.M3
        D1 = self.D1
        D2 = self.D2
        D3 = self.D3
        K1 = self.K1
        K2 = self.K2
        E1 = self.E1

        # Get predicted accelerations
        x2_ddot, y2_ddot, x3_ddot, y3_ddot = torch.split(pred, 1, dim=1)

        # Get the input features
        x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, omega, t = torch.split(x, 1, dim=1)
        
        # Compute residuals based on system of equations
        # Eq1: [K_{1}x_2 + K_{2}x_3 + M_{1}\Omega^2E_1\cos{(\Omega t)}]
        # Eq2: [K_{1}y_2 + K_{2}y_3 - M_{1}g + M_{1}\Omega^2E_1\sin{(\Omega t)}]
        # Eq3: [M_{3}\ddot{x_3} + D_{3}\dot{x_3} + K_{2}x_3 - \frac{K_2}{K_1}M_{2}\ddot{x_2} - \frac{K_2}{K_1}D_{2}\dot{x_2} - K_{2}x_2]
        # Eq4: [M_{3}\ddot{y_3} + D_{3}\dot{y_3} + K_{2}y_3 - \frac{K_2}{K_1}M_{2}\ddot{y_2} - \frac{K_2}{K_1}D_{2}\dot{y_2} - K_{2}y_2 - \frac{K_2}{K_1}M_{2}g + M_{3}g]

        # The system of equations is completely caracterized by:
        # Eq1 = fA
        # Eq2 = fB
        # Eq3 = fC
        # Eq4 = fD

        # Therefore, the residuals are:
        residual1 = K1*x2 + K2*x3 + self.M1*omega**2*E1*torch.cos(omega*t) - self.fA
        residual2 = K1*y2 + K2*y3 - self.M1*self.g + self.M1*omega**2*E1*torch.sin(omega*t) - self.fB
        residual3 = self.M3*x3_ddot + D3*x3_dot + K2*x3 - (K2/K1)*x2_ddot - (K2/K1)*x2_dot - K2*x2 - self.fC
        residual4 = self.M3*y3_ddot + D3*y3_dot + K2*y3 - (K2/K1)*y2_ddot - (K2/K1)*y2_dot - K2*y2 - (K2/K1)*self.M2*self.g + self.M3*self.g - self.fD

        # Extra residuals for mass:
        residualMass1 = self.M1 + self.M2 + self.M3 - 22.0 # The total mass of the system is 22 kg
        residualMass2 = self.M2 - self.M3 # The masses of the overhang and underhang are equal

        return residual1, residual2, residual3, residual4, residualMass1, residualMass2

def custom_loss(model, x, y_true, max_x_values, min_x_values, max_y_values, min_y_values, alpha, beta):

    # Get the predicted accelerations
    y_pred = model(x)

    # COMMENT THIS BLOCK TO USE UNNORMALIZED DATA
    # Denormalize the data
    y_pred = y_pred * (max_y_values - min_y_values) + min_y_values
    y_true = y_true * (max_y_values - min_y_values) + min_y_values

    # Compute the data loss
    data_loss = torch.sqrt(nn.MSELoss()(y_pred, y_true))
    #data_loss = nn.L1Loss()(y_pred, y_true)

    x = x * (max_x_values - min_x_values) + min_x_values

    # Compute residuals
    residual1, residual2, residual3, residual4, residualMass1, residualMass2 = model.compute_residuals(x, y_pred)

    # Compute the physics-informed loss
    #physics_loss = torch.mean(torch.stack([torch.mean(residual**2) for residual in [residual1, residual2, residual3, residual4]]))
    #physics_loss = torch.mean(torch.stack([torch.mean(torch.abs(residual)) for residual in [residual1, residual2, residual3, residual4]]))
    physics_loss = torch.sqrt(torch.mean(torch.stack([torch.mean(residual**2) for residual in [residual1, residual2, residual3, residual4, residualMass1, residualMass2]])))
    #physics_loss = torch.sqrt(torch.mean(torch.stack([torch.mean(residual**2) for residual in [residualMass1, residualMass2]])))
    
    # Define a loss term to penalize negative values of the learnable parameters
    total_loss = [torch.mean(residual1)*model.lagrange1, torch.mean(residual2)*model.lagrange2, torch.mean(residual3)*model.lagrange3, torch.mean(residual4)*model.lagrange4, residualMass1*model.lagrange5, residualMass2*model.lagrange6, model.M1*model.lagrange7, model.M2*model.lagrange8, model.M3*model.lagrange9, model.D1*model.lagrange10, model.D2*model.lagrange11, model.D3*model.lagrange12, data_loss*model.lagrange13, physics_loss*model.lagrange14]
    
    total_loss = torch.sqrt(torch.sum(torch.stack(total_loss)**2))
    # Save the data loss, physical loss, residuals, model parameters, and lagrange coefficients over time for analysis in lagrange_coefficients.csv
    with open('lagrange_coefficients.csv', 'a') as f:
        f.write(f'{data_loss.cpu().item()},{physics_loss.cpu().item()},{torch.mean(residual1).cpu().item()},{torch.mean(residual2).cpu().item()},{torch.mean(residual3).cpu().item()},{torch.mean(residual4).cpu().item()},{residualMass1.cpu().item()},{residualMass2.cpu().item()},{model.M1.cpu().item()},{model.M2.cpu().item()},{model.M3.cpu().item()},{model.D1.cpu().item()},{model.D2.cpu().item()},{model.D3.cpu().item()},{model.K1.cpu().item()},{model.K2.cpu().item()},{model.E1.cpu().item()},{model.lagrange1.cpu().item()},{model.lagrange2.cpu().item()},{model.lagrange3.cpu().item()},{model.lagrange4.cpu().item()},{model.lagrange5.cpu().item()},{model.lagrange6.cpu().item()},{model.lagrange7.cpu().item()},{model.lagrange8.cpu().item()},{model.lagrange9.cpu().item()},{model.lagrange10.cpu().item()},{model.lagrange11.cpu().item()},{model.lagrange12.cpu().item()},{model.lagrange13.cpu().item()},{model.lagrange14.cpu().item()}\n')

    # TODO: Use lagrange coefficients for other parameters??

    return total_loss

from torch.optim import Optimizer

class CustomGradientDescent(Optimizer):
    def __init__(self, params, lr=1e-3, lr_lagrange=1e-3):
        """
        Custom Gradient Descent optimizer that applies gradient descent 
        for regular parameters and maximizes Lagrange coefficients.
        
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            lr: Learning rate for the regular parameters.
            lr_lagrange: Learning rate for the Lagrange coefficients.
        """
        # Different learning rates for main parameters and Lagrange coefficients
        defaults = dict(lr=lr, lr_lagrange=lr_lagrange)
        super(CustomGradientDescent, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        # Check for NaNs or Infs in the parameters
        i = 1
        for param in self.param_groups[0]['params']:
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f'NaNs or Infs found in regular parameters {i}')
                return loss
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f'NaNs or Infs found in regular gradients {i}')
                return loss
            i += 1
        
        j = 1
        for param in self.param_groups[1]['params']:
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f'NaNs or Infs found in lagrange parameters {j}')
                return loss
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f'NaNs or Infs found in lagrange gradients {j}')
                return loss
            
            j += 1
        
    

        
        # Regular params (minimize loss)
        j = 1
        for param in self.param_groups[0]['params']:
            if param.grad is None:
                continue
            grad = param.grad
            #print(f"Gradient: {torch.norm(grad)} for regular param {j}")
            j += 1
            #print(f'Param before: {param.data}', end='')
            param.data = param.data - self.param_groups[0]['lr']*grad
            #print(f'Param after: {param.data}')
        
        # Lagrange params (maximize loss)
        i = 1
        for param in self.param_groups[1]['params']:
            if param.grad is None:
                continue
            grad = param.grad
            #print(f'Param before: {param.data}', end='')
            #print(f"Gradient: {grad} for lagrange{i}")  
            i += 1
            param.data = param.data + self.param_groups[1]['lr_lagrange']*grad
            #print(f'Param after: {param.data}')

        return loss

    """ def zero_grad(self):
        #Clears the gradients of all optimized parameters.
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_() """