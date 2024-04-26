import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Define IMU locations
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

# Iterate over each subject and IMU location
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

# Define the label mapping dictionary
label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}

# Load data for all patients
acc_data = {}
rot_data = {}
for subject in subjects:
    acc_data[subject] = acc[subject]
    rot_data[subject] = rot[subject]

# Combine accelerometer and gyroscope data for all patients and IMU sensors
X_data_per_sensor = []
for imu_location in imu_locations:
    combined_data_imu = np.hstack(
        (np.vstack([acc_data[subject][imu_location] for subject in subjects]),
         np.vstack([rot_data[subject][imu_location] for subject in subjects])))
    X_data_per_sensor.append(combined_data_imu)

# Combine data from all IMU sensors
X_data = np.vstack(X_data_per_sensor)

# Get the sampling frequency (Hz) from the accelerometer data of the first subject
x_acceleration = acc['drinking_HealthySubject2_Test']['hand_IMU']
Hz = len(x_acceleration) / 38.1

# Combine labels for all patients
labels_per_measurement = []
for subject in subjects:
    i = 2
    annotations = np.load(f"Data_tests/Annotated times/time_ranges_subject_" + str(i) + ".npy", allow_pickle=True)
    for row in annotations:
        label = label_mapping[row[2]]
        start_time = float(row[0])
        end_time = float(row[1])
        duration = end_time - start_time
        num_measurements = round(duration * Hz)
        labels_per_measurement.extend([label] * num_measurements)
    i += 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_data, labels_per_measurement, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
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
