import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Feature_Extraction import RMS_V2
from Feature_Extraction import Mean_V2
from Feature_Extraction import  Slope_V2
from Feature_Extraction import Max_V2
from Feature_Extraction import Min_V2
from Feature_Extraction import Standard_Deviation
from Random_forest import labels_interpolation


###Description: Try to split movements into their respective movement lists


# #variables
# sampling_window = 3
# min_periods = 1

# Define IMU locations
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

# Iterate over each subject and IMU location
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()
all_labels = labels_interpolation.expanded_matrices

acc_2 = acc['drinking_HealthySubject2_Test']['hand_IMU']
# print(acc_2)
# print(all_labels)

labels = []
###### for-loops to make annotation list for random forest method ###########################################################################
for item in all_labels[:1]:
    for i in item:
        labels.append(i[1])

# Dictionary to map labels to numerical values
label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}

# Convert labels to numerical values
text2 = [label_mapping[label] for label in labels]

labels_data = {'annotations', 'x', 'y', 'z'}

data = pd.DataFrame(acc_2, text2)

# plt.figure()
# plt.plot(text2)
# plt.show()

# N_acc2 = text2[value == '1']
# print(N_acc2)