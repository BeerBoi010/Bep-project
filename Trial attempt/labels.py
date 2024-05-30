import numpy as np
import matplotlib.pyplot as plt

from Feature_Extraction import RMS_V2, Mean_V2, Slope_V2, Max_V2, Min_V2, Standard_Deviation

train_amount = 5
sampling_window = 3
min_periods = 1
test_amount = train_amount

sampling_window_RMS = 3
sampling_window_min_max = 3
sampling_window_mean = 3
sampling_window_STD = 3
sampling_window_slope = 3
test_person = 7


# Load data
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

X_train_RMS = RMS_V2.RMS_train(subjects, sampling_window_RMS, min_periods)

X_train_Mean = Mean_V2.Mean_train(subjects, sampling_window_mean, min_periods)

X_train_Slope = Slope_V2.Slope_train(subjects, sampling_window_slope, min_periods)

X_train_Max = Max_V2.Max_train(subjects, sampling_window_min_max, min_periods)

X_train_Min = Min_V2.Min_train(subjects, sampling_window_min_max, min_periods)

X_train_STD = Standard_Deviation.STD_train(subjects, sampling_window_STD, min_periods)

# Load annotations
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

annotation = {'drinking_HealthySubject2_Test': annotation2_numbers, 'drinking_HealthySubject3_Test': annotation3_numbers,
              'drinking_HealthySubject4_Test': annotation4_numbers, 'drinking_HealthySubject5_Test': annotation5_numbers,
              'drinking_HealthySubject6_Test': annotation6_numbers, 'drinking_HealthySubject7_Test': annotation7_numbers}

# Calculate the sampling rate
x_acceleration = acc['drinking_HealthySubject2_Test']['hand_IMU']
Hz = len(x_acceleration) / 38.1



X_data_patients_train = []

for subject in subjects:

    # Initialize combined_data_patient for each patient
    combined_data_patient = []

    # Combine accelerometer and gyroscope data horizontally
    for imu_location in acc[subject]:

        acc_rms_imu = X_train_RMS[subject][imu_location]["acc_rms"]
        rot_rms_imu = X_train_RMS[subject][imu_location]["rot_rms"]
        acc_mean_imu = X_train_Mean[subject][imu_location]["acc_mean"]
        rot_mean_imu = X_train_Mean[subject][imu_location]["rot_mean"]
        acc_slope_imu = X_train_Slope[subject][imu_location]["acc_slope"]
        rot_slope_imu = X_train_Slope[subject][imu_location]["rot_slope"]
        acc_max_imu = X_train_Max[subject][imu_location]["acc_max"]
        rot_max_imu = X_train_Max[subject][imu_location]["rot_max"]
        acc_min_imu = X_train_Min[subject][imu_location]["acc_min"]
        rot_min_imu = X_train_Min[subject][imu_location]["rot_min"]
        acc_STD_imu = X_train_STD[subject][imu_location]["acc_STD"]
        rot_STD_imu = X_train_STD[subject][imu_location]["rot_STD"]

        combined_data_imu = np.hstack((acc_rms_imu, rot_rms_imu, acc_mean_imu, rot_mean_imu,acc_slope_imu,rot_slope_imu,
                                       acc_max_imu,rot_max_imu,acc_min_imu,rot_min_imu,acc_STD_imu,rot_STD_imu))
        combined_data_patient.append(combined_data_imu)  # Append each sensor's data

    # Stack the data from all sensors for this patient
    X_data_patients_train.append(np.hstack(combined_data_patient))

combined_X_data_train = np.concatenate(X_data_patients_train)
X_train = combined_X_data_train

print(X_data_patients_train)


# Iterate over each patient
for subject in subjects:
    acc_data_patient = acc[subject]
    rot_data_patient = rot[subject]
    labels_patient = []

    for row in annotation[subject]:
        label = int(row[2])
        start_time = float(row[0])
        end_time = float(row[1])
        duration = end_time - start_time
        num_measurements = round(duration * Hz)
        labels_patient.extend([label] * num_measurements)

    labels_patient = np.array(labels_patient).reshape(-1, 1)



def split_on_last_column_change(array):
    result = []
    current_segment = [array[0]]

    for row in array[1:]:
        if row[-1] == current_segment[-1][-1]:
            current_segment.append(row)
        else:
            result.append(np.array(current_segment))
            current_segment = [row]
    
    result.append(np.array(current_segment))
    return result

split_data_all_acc = {}
split_data_all_rot = {}

for subject in subjects:
    split_data_all_acc[subject] = {}
    split_data_all_rot[subject] = {}
    for imu_location in acc[subject]:
        split_data_all_acc[subject][imu_location] = split_on_last_column_change(acc[subject][imu_location])
        split_data_all_rot[subject][imu_location] = split_on_last_column_change(rot[subject][imu_location])



