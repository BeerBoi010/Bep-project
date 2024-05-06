import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import pandas as pd

####    Importing of necessary functions for algorithm  ###############################################
import RMS_V2
import Mean_V2
import labels_interpolation

##### VARIABLES ##########################################################################################
#later toevoegen dat random wordt gekozen wie train en test is 
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
for subject in X_test_RMS:
    acc_data_patient = acc[subject]
    rot_data_patient = rot[subject]
    labels_patient = [] 


    for row in annotation[subject]:
        label = int(row[2])
        start_time = float(row[0])
        end_time = float(row[1])
        duration = end_time - start_time
        num_measurements = round(duration * Hz)
        #print("variables",start_time,end_time,label,duration,num_measurements)
        labels_patient.extend([label] * num_measurements)
    
    if subject == 'drinking_HealthySubject6_Test':
        labels_patient = labels_patient[:-5]  # Delete the last 5 labels

    # Combine accelerometer and gyroscope data horizontally
    combined_data_patient = []
    for imu_location in imu_locations:
        acc_data_imu = acc_data_patient[imu_location]
        rot_data_imu = rot_data_patient[imu_location]
        combined_data_imu = np.hstack((acc_data_imu, rot_data_imu))
        combined_data_patient.extend(combined_data_imu.T)
    
    # Add data and labels to the lists
    X_data_patients_train.append(np.vstack(combined_data_patient).T)
    labels_patients_train.append(labels_patient)

# Combine data and labels from all patients
combined_X_data = np.concatenate(X_data_patients_train)
combined_labels = np.concatenate(labels_patients_train)

print(combined_labels)
print(combined_X_data.shape,combined_labels.shape)

#Scale numerical features to a similar range to prevent some features from dominating others.
#Common scaling techniques include StandardScaler (mean=0, std=1) and MinMaxScaler (scaling features to a range).

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Split the combined dataset and label array
#X_train, X_test, y_train, y_test = train_test_split(combined_X_data, combined_labels, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(combined_X_data)
y_train = combined_labels