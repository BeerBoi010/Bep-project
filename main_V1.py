import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import pandas as pd

#### Importing of necessary functions for algorithm  ###############################################
import RMS_V2
import Mean_V2
import labels_interpolation

##### VARIABLES ##########################################################################################
# later toevoegen dat random wordt gekozen wie train en test is
train_amount = 5
sampling_window = 3
min_periods = 1
test_amount = train_amount

### Setting up the test and training sets and labels ############################################################

X_train_RMS = RMS_V2.RMS_train(train_amount, sampling_window, min_periods)
X_test_RMS = RMS_V2.RMS_train(test_amount, sampling_window, min_periods)

X_train_Mean = Mean_V2.Mean_train(train_amount, sampling_window, min_periods)
X_test_Mean = Mean_V2.Mean_test(test_amount, sampling_window, min_periods)

Y_train_labels = labels_interpolation.expanded_matrices[:train_amount]
Y_test_labels = labels_interpolation.expanded_matrices[test_amount:]


# Create lists to store data and labels for each patient
X_data_patients_train = []
labels_patients_train = []

# Iterate over each patient
for subject in X_train_RMS:

    # Initialize combined_data_patient for each patient
    combined_data_patient = []

    # Combine accelerometer and gyroscope data horizontally
    for imu_location in X_train_RMS[subject]:

        acc_rms_imu = X_train_RMS[subject][imu_location]["acc_rms"]
        rot_rms_imu = X_train_RMS[subject][imu_location]["rot_rms"]
        acc_mean_imu = X_train_Mean[subject][imu_location]["acc_mean"]
        rot_mean_imu = X_train_Mean[subject][imu_location]["rot_mean"]

        combined_data_imu = np.hstack((acc_rms_imu, rot_rms_imu, acc_mean_imu, rot_mean_imu))
        combined_data_patient.append(combined_data_imu)  # Append each sensor's data

    # Stack the data from all sensors for this patient
    X_data_patients_train.append(np.hstack(combined_data_patient))

# Combine data from all patients
combined_X_data_train = np.concatenate(X_data_patients_train)

print(combined_X_data_train.shape)


# Create lists to store data and labels for each patient
X_data_patients_test = []
labels_patients_test = []

# Iterate over each patient
for subject in X_test_RMS:

    # Initialize combined_data_patient for each patient
    combined_data_patient = []

    # Combine accelerometer and gyroscope data horizontally
    for imu_location in X_test_RMS[subject]:

        acc_rms_imu = X_test_RMS[subject][imu_location]["acc_rms"]
        rot_rms_imu = X_test_RMS[subject][imu_location]["rot_rms"]
        acc_mean_imu = X_test_Mean[subject][imu_location]["acc_mean"]
        rot_mean_imu = X_test_Mean[subject][imu_location]["rot_mean"]

        combined_data_imu = np.hstack((acc_rms_imu, rot_rms_imu, acc_mean_imu, rot_mean_imu))
        combined_data_patient.append(combined_data_imu)  # Append each sensor's data

    # Stack the data from all sensors for this patient
    X_data_patients_test.append(np.hstack(combined_data_patient))

# Combine data from all patients
combined_X_data_test = np.concatenate(X_data_patients_test)

print(combined_X_data_test.shape)