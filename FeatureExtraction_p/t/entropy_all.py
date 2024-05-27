import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

def calculate_entropy(series, window_size):
    entropy_values = []
    for i in range(len(series)):
        window = series[max(i - window_size + 1, 0):i + 1]
        _, counts = np.unique(window, return_counts=True)
        probs = counts / counts.sum()
        entropy_values.append(entropy(probs))
    return pd.Series(entropy_values)

def Entropy_all_subjects(sampling_window):
    entropy_data_all_patients = {}

    for subject in subjects:
        entropy_data_patient = {}
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]

        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]

            dataset_acc = pd.DataFrame(acc_data_imu)
            dataset_rot = pd.DataFrame(rot_data_imu)

            entropy_acc = dataset_acc.apply(lambda x: calculate_entropy(x, sampling_window))
            entropy_rot = dataset_rot.apply(lambda x: calculate_entropy(x, sampling_window))

            entropy_data_patient[imu_location] = {'acc_entropy': entropy_acc, 'rot_entropy': entropy_rot}
        
        entropy_data_all_patients[subject] = entropy_data_patient

    return entropy_data_all_patients

sampling_window = 3
overall_entropy_data = Entropy_all_subjects(sampling_window)
print(overall_entropy_data)
