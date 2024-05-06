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


#######################################################

############train####################

def Mean_train():
    ###function: calculate mean-values for all patients, with acc and gyr data.
    mean_data_all_patients = {}

    # Iterate over each patient
    for subject in subjects[:5]:

        #calcluation of values for every imu sensor
        mean_data_patient = {}
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]
        

        # Combine accelerometer and gyroscope data horizontally
        
        combined_data_patient = []
        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]

            #open up a pandas to add a rolling mean for calculations
            dataset_acc = pd.DataFrame(acc_data_imu)
            dataset_rot = pd.DataFrame(rot_data_imu)


            #The rolling mean calculates the rolling mean for the entire row
            mean_acc= dataset_acc.rolling(sampling_window, min_periods).mean()
            mean_rot= dataset_rot.rolling(sampling_window, min_periods).mean()


            # Store RMS data for the current sensor location in the dictionary
            mean_data_patient[imu_location] = {'acc_rms': mean_acc, 'rot_rms': mean_rot}
        
        # Store RMS data for the current patient in the dictionary
        mean_data_all_patients[subject] = mean_data_patient
    
    # Return the dictionary containing RMS data for all patients
    return mean_data_all_patients

#############################################################################

##########test############
def Mean_train():
    ###function: calculate mean-values for all patients, with acc and gyr data.
    mean_data_all_patients = {}

    # Iterate over each patient
    for subject in subjects[:5]:

        #calcluation of values for every imu sensor
        mean_data_patient = {}
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]
        

        # Combine accelerometer and gyroscope data horizontally
        
        combined_data_patient = []
        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]

            #open up a pandas to add a rolling mean for calculations
            dataset_acc = pd.DataFrame(acc_data_imu)
            dataset_rot = pd.DataFrame(rot_data_imu)


            #The rolling mean calculates the rolling mean for the entire row
            mean_acc= dataset_acc.rolling(sampling_window, min_periods).mean()
            mean_rot= dataset_rot.rolling(sampling_window, min_periods).mean()


            # Store RMS data for the current sensor location in the dictionary
            mean_data_patient[imu_location] = {'acc_rms': mean_acc, 'rot_rms': mean_rot}
        
        # Store RMS data for the current patient in the dictionary
        mean_data_all_patients[subject] = mean_data_patient
    
    # Return the dictionary containing RMS data for all patients
    return mean_data_all_patients

print(Mean())


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

