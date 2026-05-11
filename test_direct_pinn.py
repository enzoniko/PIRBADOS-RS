from Models.basicPINNv4 import PINN, custom_loss
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import optim

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available GPU devices
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    # Print the name of each GPU device
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    device = 'cuda:0'
else:
    print("No GPU available, using CPU instead.")



class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class ReduceLROnPlateau:
    def __init__(self, optimizer, factor=0.1, patience=10, min_lr=1e-6, verbose=True):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0

    def step(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._reduce_lr()

    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            new_lr = max(param_group['lr'] * self.factor, self.min_lr)
            if param_group['lr'] > self.min_lr:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f"Reducing learning rate to {new_lr}")
            self.counter = 0


if __name__ == "__main__":
    # Instantiate the model
    model = PINN().to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # Define callbacks
    early_stopping = EarlyStopping(patience=2000, min_delta=1e-4)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1000, min_lr=1e-4)

    # Training loop parameters
    epochs = int(1e6)
    best_loss = float('inf')
    model_save_path = 'best_model.pth'

    # Load the Data/X.pth and Data/Y.pth files
    X = torch.load('Data/X_normal.pth')
    y = torch.load('Data/Y_normal.pth')

    X = X[:, :249998, :].to(device)
    y = y[:, :249998, :].to(device)

    
    # Add a new feature to the data: the time `t` in seconds.
    # The time should start at 0 for each sample and increase by 2e-5 seconds for each timestamp.
    t = torch.arange(0, 2e-5 * (X.size(1)), 2e-5).view(1, -1, 1).expand(X.size(0), -1, -1).to(device)
    
    X = torch.cat((X, t), dim=2)

    # Reshape the tensors to 2D
    X = X.view(-1, X.size(-1))
    y = y.view(-1, y.size(-1))

    # Get the max and min of omega (X[:, -1])
    max_omega = X[:, -1].max()
    min_omega = X[:, -1].min()

    # Take 10000 random samples
    indices = torch.randperm(X.size(0))[:4000]
    X = X[indices]
    y = y[indices]

    # Split the data into training (80%), validation (10%), and test (10%) sets
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Extract the training data
    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])

    # Calculate min and max values from the training data
    max_x_values = X_train.max(dim=0)[0]
    max_x_values[-1] = 4.99996
    max_x_values[-2] = max_omega
    min_x_values = X_train.min(dim=0)[0]
    min_x_values[-1] = 0.0
    min_x_values[-2] = min_omega
    max_y_values = y_train.max(dim=0)[0]
    min_y_values = y_train.min(dim=0)[0]

    """ max_x_values = torch.load(f'max_x_values.pth')
    min_x_values = torch.load(f'min_x_values.pth')
    max_y_values = torch.load(f'max_y_values.pth')
    min_y_values = torch.load(f'min_y_values.pth') """

    # Save the min and max values
    torch.save(max_x_values, 'max_x_values.pth')
    torch.save(min_x_values, 'min_x_values.pth')
    torch.save(max_y_values, 'max_y_values.pth')
    torch.save(min_y_values, 'min_y_values.pth')

    # Define a normalization function
    def normalize(data, min_values, max_values):
        return (data - min_values) / (max_values - min_values)

    # Normalize the training data
    X_train_normalized = normalize(X_train, min_x_values, max_x_values)
    y_train_normalized = normalize(y_train, min_y_values, max_y_values)

    # Normalize the validation data
    X_val = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
    y_val = torch.stack([val_dataset[i][1] for i in range(len(val_dataset))])

    X_val_normalized = normalize(X_val, min_x_values, max_x_values)
    y_val_normalized = normalize(y_val, min_y_values, max_y_values)

    # Normalize the test data
    X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])

    X_test_normalized = normalize(X_test, min_x_values, max_x_values)
    y_test_normalized = normalize(y_test, min_y_values, max_y_values)

    # UNCOMMENT THIS BLOCK TO USE THE ORIGINAL DATA WITHOUT NORMALIZATION
    """ X_train_normalized = X_train
    y_train_normalized = y_train

    X_val_normalized = X_val
    y_val_normalized = y_val

    X_test_normalized = X_test
    y_test_normalized = y_test """

    # Now you can create DataLoaders for the normalized datasets
    train_loader = DataLoader(TensorDataset(X_train_normalized, y_train_normalized), batch_size=1000, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_normalized, y_val_normalized), batch_size=1000, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_normalized, y_test_normalized), batch_size=1000, shuffle=False)

    min_x_values = min_x_values.to(device)
    max_x_values = max_x_values.to(device)
    min_y_values = min_y_values.to(device)
    max_y_values = max_y_values.to(device)

    alpha = torch.tensor(1000.0, requires_grad=False).to(device)
    beta = torch.tensor(1.0, requires_grad=False).to(device)
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)
            # Perform backpropagation
            optimizer.zero_grad()

            # Compute the loss
            loss = custom_loss(model, X_train, y_train, max_x_values, min_x_values, max_y_values, min_y_values, alpha, beta)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                loss = custom_loss(model, X_val, y_val, max_x_values, min_x_values, max_y_values, min_y_values, alpha, beta)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Print the losses
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss}, Validation Loss = {val_loss}")

        # Early stopping and learning rate reduction based on validation loss
        lr_scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved with validation loss {best_loss}")
            alpha = alpha * 0.99
            alpha = torch.clamp(alpha, 800.0)
            beta = beta * 1.2
            beta = torch.clamp(beta, 1.0, 1000.0)

        early_stopping(val_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break

    # Testing
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            loss = custom_loss(model, X_test, y_test, max_x_values, min_x_values, max_y_values, min_y_values, 1.0, 1.0)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss = {test_loss}")

    # Print the learned parameters
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")