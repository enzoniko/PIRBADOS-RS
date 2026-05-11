from Models.basicPINNv3 import PINN, custom_loss
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import optim

from Data.LoadData import data_paths
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

from tqdm import tqdm

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

import numpy as np
from scipy.stats import skew, kurtosis

def plot_residuals_by_variable_and_type(residuals_dict, output_dir):
    """
    Generate a summary plot with boxplots and violin plots of residuals
    for each variable across different data types.

    Parameters:
    - residuals_dict: A dictionary where keys are data types and values are residual tensors of shape (samples, variables).
    - output_dir: Directory to save the generated plot.
    """
    #variables = ['x2_ddot', 'y2_ddot', 'x3_ddot', 'y3_ddot', 'r1', 'r2', 'r3', 'r4']
    variables = ['x2_ddot', 'y2_ddot', 'x3_ddot', 'y3_ddot']
    num_variables = len(variables)
    data_types = list(residuals_dict.keys())
    num_data_types = len(data_types)

    fig, axes = plt.subplots(num_variables, 1, figsize=(45, 6 * num_variables), sharex=True)

    if num_variables == 1:
        axes = [axes]

    for var_idx, ax in enumerate(axes):
        boxplot_data = [residuals_dict[key][:, var_idx].detach().cpu().numpy() for key in data_types]
        ax.violinplot(boxplot_data, positions=np.arange(1, num_data_types + 1), showextrema=True, showmedians=True)
        ax.boxplot(boxplot_data, positions=np.arange(1, num_data_types + 1), widths=0.3, patch_artist=True, boxprops=dict(facecolor="lightblue"))

        # Annotate statistics for each data type
        for type_idx, data_type in enumerate(data_types):
            residuals = residuals_dict[data_type][:, var_idx].detach().cpu().numpy()
            mean_val = np.mean(residuals)
            median_val = np.median(residuals)
            std_val = np.std(residuals)
            skew_val = skew(residuals)
            kurt_val = kurtosis(residuals)

            ax.text(
                x=type_idx + 1, 
                y=ax.get_ylim()[1] * 0.9, 
                s=(f"Mean: {mean_val:.2f}\n"
                   f"Median: {median_val:.2f}\n"
                   f"Std: {std_val:.2f}\n"
                   f"Skew: {skew_val:.2f}\n"
                   f"Kurt: {kurt_val:.2f}"),
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
            )

        #ax.set_yscale("symlog", linthresh=1e-3)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        #ax.set_title(f"Residuals for {variables[var_idx]}")
        ax.set_ylabel(f"Residuals for {variables[var_idx]}")

    axes[-1].set_xlabel("Data Types")
    axes[-1].set_xticks(np.arange(1, num_data_types + 1))
    axes[-1].set_xticklabels(data_types, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/residuals_summary_plot.png")
    plt.close()


def main():

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

    X = X[:, :249998, :]
    y = y[:, :249998, :]

    
    # Add a new feature to the data: the time `t` in seconds.
    # The time should start at 0 for each sample and increase by 2e-5 seconds for each timestamp.
    t = torch.arange(0, 2e-5 * (X.size(1)), 2e-5).view(1, -1, 1).expand(X.size(0), -1, -1)
    
    X = torch.cat((X, t), dim=2)

    # Reshape the tensors to 2D
    X = X.reshape(-1, X.size(-1))
    y = y.reshape(-1, y.size(-1))

    # Get the max and min of omega (X[:, -1])
    max_omega = X[:, -1].max()
    min_omega = X[:, -1].min()

    # Take 10000 random samples
    indices = torch.randperm(X.size(0))[:10000]
    # Take the first 10000 samples
    #indices = torch.arange(10000)
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
    for epoch in tqdm(range(epochs)):
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
        for X_test, y_test in tqdm(test_loader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            loss = custom_loss(model, X_test, y_test, max_x_values, min_x_values, max_y_values, min_y_values, 1.0, 1.0)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss = {test_loss}")

    # Print the learned parameters
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")

    
    # Main block to compute residuals and plot
    model.eval()
    residuals_dict = {}

    batch_size = 1024  # You can adjust this value based on your GPU memory

    with torch.no_grad():
        for key in tqdm(data_paths):
            X_temp, Y_temp = torch.load(f'Data/X_{key}.pth'), torch.load(f'Data/Y_{key}.pth')
            X_temp = X_temp[:, :249998, :]
            Y_temp = Y_temp[:, :249998, :]

            t = torch.arange(0, 2e-5 * (X_temp.size(1)), 2e-5).view(1, -1, 1).expand(X_temp.size(0), -1, -1)
            X_temp = torch.cat((X_temp, t), dim=2)

            X_temp_flattened = X_temp.reshape(-1, X_temp.size(-1))
            Y_temp_flattened = Y_temp.reshape(-1, Y_temp.size(-1))

            X_temp_flattened = normalize(X_temp_flattened, min_x_values.cpu(), max_x_values.cpu())
            Y_temp_flattened = normalize(Y_temp_flattened, min_y_values.cpu(), max_y_values.cpu())


            all_residuals = [] # To store residuals from all batches
            all_r1 = []
            all_r2 = []
            all_r3 = []
            all_r4 = []
            all_rm1 = []
            all_rm2 = []


            for batch_start in range(0, X_temp_flattened.size(0), batch_size):
                batch_end = min(batch_start + batch_size, X_temp_flattened.size(0))

                X_batch = X_temp_flattened[batch_start:batch_end].to(device)
                Y_batch = Y_temp_flattened[batch_start:batch_end].to(device)

                Y_pred_batch = model(X_batch)

                Y_pred_batch = Y_pred_batch * (max_y_values - min_y_values) + min_y_values
                Y_batch = Y_batch * (max_y_values - min_y_values) + min_y_values
                X_batch = X_batch * (max_x_values - min_x_values) + min_x_values

                residuals_batch = Y_pred_batch - Y_batch

                # Get the physical residuals from the models:
                r1_batch, r2_batch, r3_batch, r4_batch, rm1_batch, rm2_batch = model.compute_residuals(X_batch, Y_pred_batch)

                all_residuals.append(residuals_batch.cpu()) # Move back to CPU to avoid GPU memory build up
                all_r1.append(r1_batch.cpu())
                all_r2.append(r2_batch.cpu())
                all_r3.append(r3_batch.cpu())
                all_r4.append(r4_batch.cpu())
                # rm1 and rm2 are not used in the final concatenation, so no need to store them for now if memory is critical, can be added back if needed.


            # Concatenate the residuals from all batches
            residuals = torch.cat(all_residuals, dim=0).to(device) #Move to device for concatenation if needed, or do everything on CPU if possible
            r1 = torch.cat(all_r1, dim=0).to(device)
            r2 = torch.cat(all_r2, dim=0).to(device)
            r3 = torch.cat(all_r3, dim=0).to(device)
            r4 = torch.cat(all_r4, dim=0).to(device)


            residuals = torch.cat((residuals, r1, r2, r3, r4), dim=1)
            residuals_dict[key] = residuals.cpu() # Store on CPU to free up GPU memory for next iteration

    # Call the plotting function
    
    # Save the residuals dictionary
    torch.save(residuals_dict, 'residuals_dict.pth')


    


if __name__ == "__main__":
    
    #main()

    residuals_dict = torch.load('residuals_dict.pth')
    output_dir = "Images"  # Replace with your actual output directory
    plot_residuals_by_variable_and_type(residuals_dict, output_dir)
