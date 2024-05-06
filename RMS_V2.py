import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



###Description: Calculates RMS-value for given data over a sampling window, working down and giving values for everor of the dataset.


#variables
sampling_window = 3
min_periods = 1


# Define IMU locations
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

# Iterate over each subject and IMU location
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

x_acceleration2 = acc['drinking_HealthySubject2_Test']['hand_IMU']



def RMS(data):
    Squared = np.square(data)

    #open up a pandas to add a rolling mean for calculations
    dataset_sub2= pd.DataFrame(Squared)

    #The rolling mean calculates the rolling mean for the entire row
    roller= dataset_sub2.rolling(sampling_window, min_periods).mean()
    test2 = dataset_sub2.rolling(sampling_window, min_periods, step = 2).mean()

    test = np.sqrt(test2)
    RMS= np.sqrt(roller)
    return RMS, test


print(RMS(x_acceleration2)[1])