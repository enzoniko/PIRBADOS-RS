import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        
        self.hidden1 = nn.Linear(10, 64) # 10 inputs (x2, x2_dot, x3, x3_dot, y2, y2_dot, y3, y3_dot, omega, t)

        self.hidden2 = nn.Linear(64, 64) # Any number of hidden layers can be added

        self.output = nn.Linear(64, 4) # 4 outputs (the accelerations) (x2_ddot, x3_ddot, y2_ddot, y3_ddot)

        # Define learnable parameters (masses, damping coefficients, spring constants, x1 and y1 and their derivatives)
        self.M1 = nn.Parameter(torch.tensor(1.0))
        self.M2 = nn.Parameter(torch.tensor(1.0))
        self.M3 = nn.Parameter(torch.tensor(1.0))
        self.D1 = nn.Parameter(torch.tensor(1.0))
        self.D2 = nn.Parameter(torch.tensor(1.0))
        self.D3 = nn.Parameter(torch.tensor(1.0))
        self.K1 = nn.Parameter(torch.tensor(1.0))
        self.K2 = nn.Parameter(torch.tensor(1.0))
        self.E1 = nn.Parameter(torch.tensor(1.0))


        self.x1 = nn.Parameter(torch.tensor(1.0))
        self.y1 = nn.Parameter(torch.tensor(1.0))
        self.x1_dot = nn.Parameter(torch.tensor(1.0))
        self.y1_dot = nn.Parameter(torch.tensor(1.0))
        self.x1_ddot = nn.Parameter(torch.tensor(1.0))
        self.y1_ddot = nn.Parameter(torch.tensor(1.0))

        self.g = torch.tensor(9.81)  # Gravity constant (can be set as a constant)

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = self.output(x)
        return x

    def compute_residuals(self, x, pred):
        x2_ddot, x3_ddot, y2_ddot, y3_ddot = torch.split(pred, 1, dim=1) # Get predicted accelerations
        x2, x2_dot, x3, x3_dot, y2, y2_dot, y3, y3_dot, omega, t = torch.split(x, 1, dim=1) # Get the input features

        # Compute residuals based on system of equations
        residual1 = self.M1 * self.x1_ddot + self.D1 * self.x1_dot + self.K1 * (self.x1 - x2) + self.K2 * (self.x1 - x3) - self.M1 * omega**2 * self.E1 * torch.cos(omega * t) # Without Fxs term
        residual2 = self.M1 * self.y1_ddot + self.D1 * self.y1_dot + self.K1 * (self.y1 - y2) + self.K2 * (self.y1 - y3) - self.M1 * omega**2 * self.E1 * torch.sin(omega * t) + self.M1 * self.g  # Without Fys term
        residual3 = self.M2 * y2_ddot + self.D2 * y2_dot + self.K1 * (y2 - self.y1) - (self.M2 * x2_ddot + self.D2 * x2_dot + self.K1 * (x2 - self.x1)) + self.M2 * self.g
        residual4 = self.M3 * y3_ddot + self.D3 * y3_dot + self.K2 * (y3 - self.y1) - (self.M3 * x3_ddot + self.D3 * x3_dot + self.K2 * (x3 - self.x1)) + self.M3 * self.g

        return residual1, residual2, residual3, residual4

model = PINN()


def custom_loss(model, x, y_true):
    y_pred = model(x)
    data_loss = nn.MSELoss()(y_pred, y_true)

    residuals = model.compute_residuals(x, y_pred)
    physics_loss = torch.mean(torch.stack([torch.mean(residual**2) for residual in residuals]))

    total_loss = data_loss + physics_loss
    return total_loss


# Example data (replace with actual data)
X_train = torch.rand(1000, 10)  # 1000 samples, 9 features
y_train = torch.rand(1000, 4)  # 1000 samples, 4 targets

optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 2000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    loss = custom_loss(model, X_train, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Print the learned parameters
for name, param in model.named_parameters():
    print(name, param.data)