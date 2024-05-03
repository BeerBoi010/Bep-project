import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import sys

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
print(len(acc['drinking_HealthySubject6_Test']['hand_IMU']))


#print(annotation2_numbers)

#Plot annotated time ranges for subject 3
plt.figure(figsize=(10, 4))
plt.title("Annotated Time Ranges for Subject 3")
plt.xlabel("Time")
plt.ylabel("Activity Label")
for start, end, label in annotation3_numbers:
    plt.plot([start, end], [label, label], color='blue', linewidth=3)
plt.yticks([0, 1, 2, 3], ['N', 'A', 'B', 'C'])  # Mapping numerical labels to annotated letters
plt.grid(True)
plt.show()
