import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import sys

def load_and_parse_data(acc_path, gyro_path):
    # Load the data from .npy files
    acc_data = np.load(acc_path)
    gyro_data = np.load(gyro_path)
    
    # Assuming there are 6 participants and each has 5 IMUs
    num_participants = 6
    num_imus = 5
    
    # Each dataset has X, Y, Z data per IMU per participant
    # Assuming data is stacked in the order: P1I1, P1I2, ..., P1I5, P2I1, ..., P6I5
    # And each sensor's data is stored as [X, Y, Z]
    
    # Data structures to store the parsed data
    parsed_acc = {f'Participant_{i + 1}': {} for i in range(num_participants)}
    parsed_gyro = {f'Participant_{i + 1}': {} for i in range(num_participants)}
    
    # Iterate through each participant and IMU to slice the data accordingly
    for p in range(num_participants):
        for i in range(num_imus):
            start_index = p * num_imus + i
            parsed_acc[f'Participant_{p + 1}'][f'IMU_{i + 1}'] = {
                'X': acc_data[start_index, :, 0],
                'Y': acc_data[start_index, :, 1],
                'Z': acc_data[start_index, :, 2]
            }
            parsed_gyro[f'Participant_{p + 1}'][f'IMU_{i + 1}'] = {
                'X': gyro_data[start_index, :, 0],
                'Y': gyro_data[start_index, :, 1],
                'Z': gyro_data[start_index, :, 2]
            }
            
    return parsed_acc, parsed_gyro

if __name__ == '__main__':
    # Example file paths, replace these with actual paths when deploying
    acc_path = 'Data_tests/ACC_signal.npy'
    gyro_path = 'Data_tests/Gyro_signal.npy'
    acc_data, gyro_data = load_and_parse_data(acc_path, gyro_path)
    print('Acceleration Data:', acc_data)
    print('Gyroscope Data:', gyro_data)

