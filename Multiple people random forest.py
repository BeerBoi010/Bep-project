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

# Define the label mapping dictionary
label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3, 0: "N", 1: "A", 2:"B", 3: "C"}

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
X_data_patients = []
labels_patients = []

# Iterate over each patient
for subject in subjects[:5]:
    acc_data_patient = acc[subject]
    rot_data_patient = rot[subject]
    labels_patient = [] 

    for row in annotation[subject]:
        label = row[2]
        start_time = float(row[0])
        end_time = float(row[1])
        duration = end_time - start_time
        num_measurements = round(duration * Hz)
        labels_patient.extend([label] * num_measurements)
    
    # Combine accelerometer and gyroscope data horizontally
    combined_data_patient = []
    for imu_location in imu_locations:
        acc_data_imu = acc_data_patient[imu_location]
        rot_data_imu = rot_data_patient[imu_location]
        combined_data_imu = np.hstack((acc_data_imu, rot_data_imu))
        combined_data_patient.extend(combined_data_imu.T)
    
    # Add data and labels to the lists
    X_data_patients.append(np.vstack(combined_data_patient).T)
    labels_patients.append(labels_patient)

# Combine data and labels from all patients
combined_X_data = np.concatenate(X_data_patients)
combined_labels = np.concatenate(labels_patients)

print(combined_labels)
print(combined_X_data.shape,combined_labels.shape)

# Split the combined dataset and label array
X_train, X_test, y_train, y_test = train_test_split(combined_X_data, combined_labels, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

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
# plt.show()

# Initialize an empty list to store combined data for testing
combined_data_patient_test = []

# Iterate over each IMU location
for imu_location in imu_locations:
    # Extract accelerometer and gyroscope data for the specified IMU location
    acc_data_imu_test = acc['drinking_HealthySubject7_Test'][imu_location]
    rot_data_imu_test = rot['drinking_HealthySubject7_Test'][imu_location]
    
    # Combine accelerometer and gyroscope data horizontally
    combined_data_imu_test = np.hstack((acc_data_imu_test, rot_data_imu_test))
    
    # Append the combined data for the current IMU location to the list
    combined_data_patient_test.extend(combined_data_imu_test.T)

X_data = np.vstack(combined_data_patient_test).T

# Make predictions using the trained Random Forest classifier
predicted_labels = clf.predict(X_data)


np.set_printoptions(threshold=sys.maxsize)

# Print the predicted labels
print(predicted_labels)

