import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import sys


# Define IMU locations
#imu_locations = ['hand_IMU']
#imu_locations = ['hand_IMU', 'lowerarm_IMU']
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']


# Iterate over each subject and IMU location
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()
#pre = np.load("data_Preprocessed.npy", allow_pickle=True).item()


annotation2 = np.load("Data_tests/Annotated times/time_ranges_subject_2.npy", allow_pickle=True)
annotation3 = np.load("Data_tests/Annotated times/time_ranges_subject_3.npy", allow_pickle=True)
annotation4 = np.load("Data_tests/Annotated times/time_ranges_subject_4.npy", allow_pickle=True)
annotation5 = np.load("Data_tests/Annotated times/time_ranges_subject_5.npy", allow_pickle=True)
annotation6 = np.load("Data_tests/Annotated times/time_ranges_subject_6.npy", allow_pickle=True)
annotation7 = np.load("Data_tests/Annotated times/time_ranges_subject_7.npy", allow_pickle=True)

print(annotation2)

# Define the label mapping dictionary
label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}

# Map the letters to numbers in the loaded array
mapped_labels2 = [[item[0], item[1], label_mapping[item[2]]] for item in annotation2]
mapped_labels3 = [[item[0], item[1], label_mapping[item[2]]] for item in annotation3]
mapped_labels4 = [[item[0], item[1], label_mapping[item[2]]] for item in annotation4]
mapped_labels5 = [[item[0], item[1], label_mapping[item[2]]] for item in annotation5]
mapped_labels6 = [[item[0], item[1], label_mapping[item[2]]] for item in annotation6]
mapped_labels7 = [[item[0], item[1], label_mapping[item[2]]] for item in annotation7]

# Convert the mapped labels list to a NumPy array
annotation2_numbers = np.array(mapped_labels2)
annotation3_numbers = np.array(mapped_labels3)
annotation4_numbers = np.array(mapped_labels4)
annotation5_numbers = np.array(mapped_labels5)
annotation6_numbers = np.array(mapped_labels6)
annotation7_numbers = np.array(mapped_labels7)

annotation = {'drinking_HealthySubject2_Test':annotation2_numbers,'drinking_HealthySubject3_Test':annotation3_numbers,
              'drinking_HealthySubject4_Test':annotation4_numbers,'drinking_HealthySubject5_Test':annotation5_numbers,
              'drinking_HealthySubject6_Test':annotation6_numbers,'drinking_HealthySubject7_Test':annotation7_numbers
              }

x_acceleration = acc['drinking_HealthySubject2_Test']['hand_IMU']
Hz = len(x_acceleration)/38.1

# Create lists to store data and labels for each patient
X_data_patients_train = []
labels_patients_train = []

# Iterate over each patient
for subject in subjects[:4]:
    acc_data_patient = acc[subject]
    rot_data_patient = rot[subject]
    labels_patient = [] 

    measurement_list = [] 

    for row in annotation[subject]:
        label = int(row[2])
        start_time = float(row[0])
        end_time = float(row[1])
        duration = end_time - start_time
        num_measurements = round(duration * Hz)
        measurement_list.append(num_measurements)
        labels_patient.append(label)

    # if subject == 'drinking_HealthySubject6_Test':
    #     labels_patient = labels_patient[:-5]  # Delete the last 5 labels

    # Initialize a list to store data for each movement
X_data_movements = []

# Iterate over each annotation and extract the data
start_idx = 0
for num_meas in measurement_list:
    acc_data_movement = {imu_location: [] for imu_location in imu_locations}
    rot_data_movement = {imu_location: [] for imu_location in imu_locations}

    # Iterate over each measurement within the movement
    for i in range(start_idx, min(start_idx + num_meas, 1905)):
        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]
            
            # Extract the data for this measurement
            acc_measurement = acc_data_imu[i]
            rot_measurement = rot_data_imu[i]

            acc_data_movement[imu_location].append(acc_measurement)
            rot_data_movement[imu_location].append(rot_measurement)

    # Calculate mean for each IMU sensor
    mean_acc_movement = np.concatenate([np.mean(acc_data_movement[imu_loc], axis=0) for imu_loc in imu_locations])
    mean_rot_movement = np.concatenate([np.mean(rot_data_movement[imu_loc], axis=0) for imu_loc in imu_locations])

    # Flatten and append the data
    combined_data_movement = np.concatenate([mean_acc_movement, mean_rot_movement])
    X_data_movements.append(combined_data_movement)

    # Update the start index for the next movement
    start_idx += num_meas

# Add the data for this patient to the overall list
X_data_patients_train.append(X_data_movements)
labels_patients_train.append(labels_patient)


# Combine data and labels from all patients
X_train = np.concatenate(X_data_patients_train)
y_train = np.concatenate(labels_patients_train)

print(X_train.shape, y_train.shape)

# Create lists to store data and labels for each patient
X_data_patients_test = []
labels_patients_test = []

# Iterate over each patient
for subject in subjects[4:]:
    acc_data_patient = acc[subject]
    rot_data_patient = rot[subject]
    labels_patient = [] 

    measurement_list = [] 

    for row in annotation[subject]:
        label = int(row[2])
        start_time = float(row[0])
        end_time = float(row[1])
        duration = end_time - start_time
        num_measurements = round(duration * Hz)
        measurement_list.append(num_measurements)
        labels_patient.append(label)

    # if subject == 'drinking_HealthySubject6_Test':
    #     labels_patient = labels_patient[:-5]  # Delete the last 5 labels

    # Initialize a list to store data for each movement
X_data_movements = []

# Iterate over each annotation and extract the data
start_idx = 0
for num_meas in measurement_list:
    acc_data_movement = {imu_location: [] for imu_location in imu_locations}
    rot_data_movement = {imu_location: [] for imu_location in imu_locations}

    # Iterate over each measurement within the movement
    for i in range(start_idx, min(start_idx + num_meas, 1905)):
        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]
            
            # Extract the data for this measurement
            acc_measurement = acc_data_imu[i]
            rot_measurement = rot_data_imu[i]

            acc_data_movement[imu_location].append(acc_measurement)
            rot_data_movement[imu_location].append(rot_measurement)

    # Calculate mean for each IMU sensor
    mean_acc_movement = np.concatenate([np.mean(acc_data_movement[imu_loc], axis=0) for imu_loc in imu_locations])
    mean_rot_movement = np.concatenate([np.mean(rot_data_movement[imu_loc], axis=0) for imu_loc in imu_locations])

    # Flatten and append the data
    combined_data_movement = np.concatenate([mean_acc_movement, mean_rot_movement])
    X_data_movements.append(combined_data_movement)

    # Update the start index for the next movement
    start_idx += num_meas

# Add the data for this patient to the overall list
X_data_patients_test.append(X_data_movements)
labels_patients_test.append(labels_patient)


# Combine data and labels from all patients
X_test = np.concatenate(X_data_patients_test)
y_test = np.concatenate(labels_patients_test)

print(X_test.shape, y_test.shape)


# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions 
y_test_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)

# Display classification report
print("Classification Report of test data:")
print(classification_report(y_test, y_test_pred))

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


# # Visualize one of the decision trees in the Random Forest
# plt.figure(figsize=(150, 10))
# plot_tree(clf.estimators_[0], feature_names=[f'feature {i}' for i in range(X_train.shape[1])], filled=True)
