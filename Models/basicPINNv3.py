import torch 
import torch.nn as nn
import torch.optim as optim

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
    # This will be a sum of relu functions applied to the learnable parameters * -1000
    gama = 100
    neg_loss = torch.relu(-gama*model.M1) + torch.relu(-gama*model.M2) + torch.relu(-gama*model.M3) + torch.relu(-gama*model.D1) + torch.relu(-gama*model.D2) + torch.relu(-gama*model.D3) + torch.relu(-gama*model.K1) + torch.relu(-gama*model.K2) + torch.relu(-gama*model.E1)

    # Define the total loss as the sum of the data loss and the physics-informed loss
    print(alpha, beta, data_loss.detach(), physics_loss.detach(), neg_loss.detach()/100, end='\r')
    #print(alpha, beta, data_loss.detach(), physics_loss.detach(), end='\r')
    total_loss = alpha*data_loss + beta*physics_loss + beta*neg_loss
    #total_loss = torch.abs(torch.mean(residualMass1 + residualMass2))
    #total_loss = data_loss
    return total_loss, data_loss, physics_loss, neg_loss