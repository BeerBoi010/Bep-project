import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys

#important variables:
window_size = 3



# Define IMU locations
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

# Iterate over each subject and IMU location
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

annotation2 = np.load("Data_tests/Annotated times/time_ranges_subject_2.npy", allow_pickle=True)
annotation3 = np.load("Data_tests/Annotated times/time_ranges_subject_3.npy", allow_pickle=True)
annotation4 = np.load("Data_tests/Annotated times/time_ranges_subject_4.npy", allow_pickle=True)
annotation5 = np.load("Data_tests/Annotated times/time_ranges_subject_5.npy", allow_pickle=True)
annotation6 = np.load("Data_tests/Annotated times/time_ranges_subject_6.npy", allow_pickle=True)
annotation7 = np.load("Data_tests/Annotated times/time_ranges_subject_7.npy", allow_pickle=True)

data = [[1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]]
# Convert the array to a DataFrame
df = pd.DataFrame(data)
#acc2 = df['drinking_HealthySubject2_Test']
print(df.head)

# acc_dat = pd.DataFrame(acc)
# acc_2 = acc_dat['drinking_HealthySubject2_Test']

# print(acc_2)


# Calculate rolling mean for each row with window size 3
rolling_means = df.rolling(window=2,   min_periods=1).mean()

#print(rolling_means)
x_acceleration = acc['drinking_HealthySubject2_Test']['hand_IMU']
x_accT = x_acceleration.T

dataset = pd.DataFrame(x_accT)
roller = dataset.rolling(window=3,   min_periods=1).mean()

print(roller[0])
print(x_accT[0])

# plt.figure()
# plt.plot(x_accT[0])
# plt.show()