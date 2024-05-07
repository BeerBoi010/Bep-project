import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

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
X_test_RMS = RMS_V2.RMS_test(test_amount, sampling_window, min_periods)

X_train_Mean = Mean_V2.Mean_train(train_amount, sampling_window, min_periods)
X_test_Mean = Mean_V2.Mean_test(test_amount, sampling_window, min_periods)

Y_train_labels = labels_interpolation.expanded_matrices[:train_amount]
Y_test_labels = labels_interpolation.expanded_matrices[test_amount:]


labels_train = []

for item in Y_train_labels:
    for i in item:
        labels_train.append(i[1])

labels_test = []

for item in Y_test_labels:
    for i in item:
        labels_test.append(i[1])

# Dictionary to map labels to numerical values
label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}

# Convert labels to numerical values
y_train = [label_mapping[label] for label in labels_train]
y_test = [label_mapping[label] for label in labels_test]

print("y_test",len(y_test))

# Create lists to store data and labels for each patient
X_data_patients_train = []

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
X_train = combined_X_data_train
print(combined_X_data_train.shape)


# Create lists to store data and labels for each patient
X_data_patients_test = []

# Iterate over each patient
for subject in X_test_RMS:
    print("test subject", subject)
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
X_test = combined_X_data_test

print(combined_X_data_test.shape)

# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions 
y_test_pred = clf.predict(X_test)
print("y_test_pred",len(y_test_pred))
y_train_pred = clf.predict(X_train)



# Display classification report
print("Classification Report of train data:")
print(classification_report(y_train, y_train_pred))

# Display classification report
print("Classification Report of test data:")
print(classification_report(y_test, y_test_pred))

 # Create an empty list of size equal to the length of predictions or true labels
element_numbers = list(range(len(y_test_pred)))

# Plot for y_pred
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)  # 1 row, 2 columns, plot number 1
plt.plot(element_numbers, y_test_pred, label='Predictions', color='blue')
plt.xlabel('Element Numbers')
plt.ylabel('Predicted Labels')
plt.title(f'Predicted Labels - {subject}')
plt.legend()

# Plot for y_test
plt.subplot(1, 2, 2)  # 1 row, 2 columns, plot number 2
plt.plot(element_numbers, y_test, label='True Labels', color='green')
plt.xlabel('Element Numbers')
plt.ylabel('True Labels')
plt.title(f'True Labels - {subject}')
plt.legend()

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# Get feature importances
importances = clf.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.show()


# Visualize one of the decision trees in the Random Forest
plt.figure(figsize=(150, 10))
plot_tree(clf.estimators_[0], feature_names=[f'feature {i}' for i in range(X_train.shape[1])], filled=True)
plt.show()

