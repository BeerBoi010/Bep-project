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

print(acc)

rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

x_acceleration2 = acc['drinking_HealthySubject2_Test']['hand_IMU']

def RMS():
    ###function: calculate RMS-values for all patients, with acc and gyr data.
    # Iterate over each patient
    for subject in subjects[:5]:
        #calcluation of values for every imu sensor
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]

        # Combine accelerometer and gyroscope data horizontally
        
        combined_data_patient = []
        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]
            
            #calcluation of Squared matrices for one sensor, for one patient
            Squared_acc = np.square(acc_data_imu)
            Squared_rot = np.square(rot_data_imu)

            #open up a pandas to add a rolling mean for calculations
            dataset_acc = pd.DataFrame(Squared_acc)
            dataset_rot = pd.DataFrame(Squared_rot)

            print("acc dataframe", dataset_acc)
            print("rot dataframe", dataset_rot)

            #The rolling mean calculates the rolling mean for the entire row
            Squaredmean_acc= dataset_acc.rolling(sampling_window, min_periods).mean()
            Squaredmean_rot = dataset_rot.rolling(sampling_window, min_periods).mean()

            RMS_acc = np.sqrt(Squaredmean_acc)
            RMS_rot = np.sqrt(Squaredmean_rot)

            
            
            # combined_data_imu = np.hstack((acc_data_imu, rot_data_imu))
            # combined_data_patient.extend(combined_data_imu.T)
    
    # Add data and labels to the lists
    # X_data_patients_train.append(np.vstack(combined_data_patient).T)
    # labels_patients_train.append(labels_patient)
    
    # Combine data and labels from all patients
    # combined_X_data = np.concatenate(X_data_patients_train)

    # print(combined_X_data.shape,combined_labels.shape)

    # return RMS, test

print(RMS())


###Tjalles interpolatiecode

# annotation2 = np.load("Data_tests/Annotated times/time_ranges_subject_2.npy", allow_pickle=True)
# annotation3 = np.load("Data_tests/Annotated times/time_ranges_subject_3.npy", allow_pickle=True)
# annotation4 = np.load("Data_tests/Annotated times/time_ranges_subject_4.npy", allow_pickle=True)
# annotation5 = np.load("Data_tests/Annotated times/time_ranges_subject_5.npy", allow_pickle=True)
# annotation6 = np.load("Data_tests/Annotated times/time_ranges_subject_6.npy", allow_pickle=True)
# annotation7 = np.load("Data_tests/Annotated times/time_ranges_subject_7.npy", allow_pickle=True)
# annotation_matrices = [annotation2, annotation3, annotation4, annotation5, annotation6, annotation7]

# # Function to perform interpolation for a single row
# def interpolate_row(row, cumulative_count):
#     # Convert start and end time to floats
#     start_time = float(row[0])
#     end_time = float(row[1])
#     # Original label
#     label = row[2]
#     # Calculate the number of samples
#     num_samples = round((end_time - start_time) * sampling_frequency)
#     # Create expanded rows with data points and label
#     expanded_rows = [[cumulative_count + i + 1, label] for i in range(num_samples)]
#     # Update cumulative count
#     cumulative_count += num_samples
#     return expanded_rows, cumulative_count

# # Initialize list to store expanded rows for all participants
# expanded_matrices = []

# # Iterate over each participant's annotation matrix
# for annotation_matrix in annotation_matrices:
#     # Initialize variables to keep track of the cumulative count
#     cumulative_count = 0
#     # Initialize list to store expanded rows for the current participant
#     expanded_matrix = []
#     # Iterate over each row in the annotation matrix and perform interpolation
#     for row in annotation_matrix:
#         expanded_rows, cumulative_count = interpolate_row(row, cumulative_count)
#         expanded_matrix.extend(expanded_rows)
#     # Append the expanded matrix for the current participant to the list
#     expanded_matrices.append(expanded_matrix)

# exp_annotations2 = expanded_matrices[0]
# exp_annotations3 = expanded_matrices[1]
# exp_annotations4 = expanded_matrices[2]
# exp_annotations5 = expanded_matrices[3]
# exp_annotations6 = expanded_matrices[4]
# exp_annotations7 = expanded_matrices[5]

