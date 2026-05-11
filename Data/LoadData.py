# Sript to load data from the csv files of the MAFAULDA dataset

# The dataset is composed of the following columns:
""" 
    column 1
    tachometer signal that allows to estimate rotation frequency;

    columns 2 to 4
    underhang bearing accelerometer (axial, radiale tangential direction);

    columns 5 to 7
    overhang bearing accelerometer (axial, radiale tangential direction);

    column 8
    microphone. 

"""

import pandas as pd
import numpy as np
import os
import torch

data_paths = {
    'normal': 'Data/normal/',
    'overhang_ball_fault_0g': 'Data/overhang/ball_fault/0g/',
    'overhang_ball_fault_6g': 'Data/overhang/ball_fault/6g/',
    'overhang_ball_fault_20g': 'Data/overhang/ball_fault/20g/',
    'overhang_ball_fault_35g': 'Data/overhang/ball_fault/35g/',
    'overhang_cage_fault_0g': 'Data/overhang/cage_fault/0g/',
    'overhang_cage_fault_6g': 'Data/overhang/cage_fault/6g/',
    'overhang_cage_fault_20g': 'Data/overhang/cage_fault/20g/',
    'overhang_cage_fault_35g': 'Data/overhang/cage_fault/35g/',
    'overhang_outer_race_fault_0g': 'Data/overhang/outer_race/0g/',
    'overhang_outer_race_fault_6g': 'Data/overhang/outer_race/6g/',
    'overhang_outer_race_fault_20g': 'Data/overhang/outer_race/20g/',
    'overhang_outer_race_fault_35g': 'Data/overhang/outer_race/35g/',
    'underhang_ball_fault_0g': 'Data/underhang/ball_fault/0g/',
    'underhang_ball_fault_6g': 'Data/underhang/ball_fault/6g/',
    'underhang_ball_fault_20g': 'Data/underhang/ball_fault/20g/',
    'underhang_ball_fault_35g': 'Data/underhang/ball_fault/35g/',
    'underhang_cage_fault_0g': 'Data/underhang/cage_fault/0g/',
    'underhang_cage_fault_6g': 'Data/underhang/cage_fault/6g/',
    'underhang_cage_fault_20g': 'Data/underhang/cage_fault/20g/',
    'underhang_cage_fault_35g': 'Data/underhang/cage_fault/35g/',
    'underhang_outer_race_fault_0g': 'Data/underhang/outer_race/0g/',
    'underhang_outer_race_fault_6g': 'Data/underhang/outer_race/6g/',
    'underhang_outer_race_fault_20g': 'Data/underhang/outer_race/20g/',
    'underhang_outer_race_fault_35g': 'Data/underhang/outer_race/35g/',
    'horizontal_misalignment_fault_0.5mm': 'Data/horizontal-misalignment/0.5mm/',
    'horizontal_misalignment_fault_1.0mm': 'Data/horizontal-misalignment/1.0mm/',
    'horizontal_misalignment_fault_1.5mm': 'Data/horizontal-misalignment/1.5mm/',
    'horizontal_misalignment_fault_2.0mm': 'Data/horizontal-misalignment/2.0mm/',
    'vertical_misalignment_fault_0.51mm': 'Data/vertical-misalignment/0.51mm/',
    'vertical_misalignment_fault_0.63mm': 'Data/vertical-misalignment/0.63mm/',
    'vertical_misalignment_fault_1.27mm': 'Data/vertical-misalignment/1.27mm/',
    'vertical_misalignment_fault_1.40mm': 'Data/vertical-misalignment/1.40mm/',
    'vertical_misalignment_fault_1.78mm': 'Data/vertical-misalignment/1.78mm/',
    'vertical_misalignment_fault_1.90mm': 'Data/vertical-misalignment/1.90mm/',
    'imbalance_fault_6g': 'Data/imbalance/6g/',
    'imbalance_fault_10g': 'Data/imbalance/10g/',
    'imbalance_fault_15g': 'Data/imbalance/15g/',
    'imbalance_fault_20g': 'Data/imbalance/20g/',
    'imbalance_fault_25g': 'Data/imbalance/25g/',
    'imbalance_fault_30g': 'Data/imbalance/30g/',
    'imbalance_fault_35g': 'Data/imbalance/35g/',
}

def load_csvs(data_path):

    multivariate_time_series = []
    # For each csv file in the data path
    for csv_file in os.listdir(data_path):
        # Load the csv file
        df = pd.read_csv(data_path + csv_file, usecols=[2, 3, 5, 6])

        # Get the name of the csv file
        name = csv_file.split('.')[0]

        # Get the data from the csv file
        data = df.to_numpy()

        # Get the number of columns
        n_cols = data.shape[1]

        # Get the number of rows
        n_rows = data.shape[0]

        print(f'Reading {name} with {n_rows} rows and {n_cols} columns')

        # Append the data to the multivariate time series
        multivariate_time_series.append(data)

    multivariate_time_series = np.array(multivariate_time_series)
    multivariate_time_series = torch.tensor(multivariate_time_series, dtype=torch.float32)
    print(f'Shape of the multivariate time series: {multivariate_time_series.shape}')
    return multivariate_time_series

def expand_data(data):

    """
    Transforms acceleration data into displacement and velocity data using FFT with PyTorch.
    
    Parameters:
    - data: 3D torch tensor of shape (samples, timesteps, 4)
      The 4 features are assumed to be:
        1. Underhang radial acceleration (x_2 ddot)
        2. Underhang tangential acceleration (y_2 ddot)
        3. Overhang radial acceleration (x_3 ddot)
        4. Overhang tangential acceleration (y_3 ddot)
    
    Returns:
    - expanded_data: 3D torch tensor of shape (samples, timesteps, 12)
      The expanded features include displacements and velocities for each of the original accelerations.
    """

    samples, timesteps, features = data.shape
    
    # Fourier transform of the data (along the time axis)
    fft_data = torch.fft.fft(data, dim=1)
    
    # Frequency axis
    freqs = torch.fft.fftfreq(timesteps, d=1.0).to(data.device)  # Ensure the frequency tensor is on the same device
    
    # Create complex constants
    j2pi_f = 2j * torch.pi * freqs
    
    # Avoid division by zero or small values for the DC component
    j2pi_f_safe = j2pi_f.clone()
    j2pi_f_safe[j2pi_f == 0] = torch.inf  # Avoid division by zero by setting to inf (which will result in zero velocity/displacement)
    
    # Calculate displacements and velocities using the FFT method
    displacement_fft = fft_data / (-j2pi_f_safe**2).unsqueeze(0).unsqueeze(-1)
    velocity_fft = fft_data / j2pi_f_safe.unsqueeze(0).unsqueeze(-1)
    
    # Set the DC component (zero frequency) to zero
    displacement_fft[:, 0, :] = 0
    velocity_fft[:, 0, :] = 0
    
    # Inverse FFT to get time-domain signals
    displacement = torch.fft.ifft(displacement_fft, dim=1).real
    velocity = torch.fft.ifft(velocity_fft, dim=1).real
    
    # Concatenate the original accelerations with calculated displacements and velocities
    expanded_data = torch.cat((data, velocity, displacement), dim=2)
    
    print(f'Shape of the expanded data: {expanded_data.shape}')
    return expanded_data

def get_omegas(folder_path):
    omegas = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):

            # Get the angular velocity from the file name
            omega = file[:-4]
            omega = float(omega)

            # Convert the angular velocity from Hz to rad/s
            omega = omega * 2 * np.pi

            omegas.append(omega)

    omegas = torch.tensor(omegas, dtype=torch.float32)
    return omegas

# def get_omegas(file_path=None):
#             omegas = torch.tensor([127.3928, 262.5065, 338.4274,  90.0757, 136.4004, 140.2608,  95.2229,
#                 374.4577, 369.3105, 199.4534, 329.4199, 229.0497, 253.4989, 147.9816,
#                 222.6158, 325.5595, 299.8235,  77.2078, 266.3669, 379.6049, 123.5324,
#                 108.0909, 350.0086, 207.1742, 290.8160, 185.2987, 101.6569, 386.0389,
#                 307.5443, 154.4156, 240.6309, 211.0346, 236.7705, 280.5216, 274.0876,
#                 285.6688, 181.4383, 193.0195, 311.4047,  82.3550, 247.0649, 356.4426,
#                 364.1634, 343.5746, 115.8117, 172.4307, 160.8495, 167.2835, 214.8950
#             ])

#             print("OMEGAS SHAPE", omegas.shape)
#             return omegas

if __name__ == '__main__':

    for key in data_paths:             
        data = load_csvs(data_paths[key])

        omegas = get_omegas(data_paths[key])

        print(omegas)
        # Extend the dataset with displacement and velocity calculated from the accelerometer data
        expanded_data = expand_data(data)

        # Create X and Y data from the expanded data
        # The X is the displacements and velocities and the Y is the accelerations
        X = expanded_data[:, :, 4:]
        Y = expanded_data[:, :, :4]

        # Add the angular velocities to the X data
        X = torch.cat((X, omegas.unsqueeze(1).unsqueeze(-1).repeat(1, X.shape[1], 1)), dim=2)
        print(f'Shape of the X data: {X.shape}')

        # Save the X and Y data
        torch.save(X, f'Data/X_{key}.pth')
        torch.save(Y, f'Data/Y_{key}.pth')

        # Check for NaN values
        print(torch.isnan(expanded_data).any())

        print(expanded_data[0, 0, :])
        print(expanded_data[0, 1, :])