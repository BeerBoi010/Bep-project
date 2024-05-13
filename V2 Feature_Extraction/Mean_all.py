import numpy as np
import pandas as pd

imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

annotation2 = np.load("Data_tests/Annotated times/time_ranges_subject_2.npy", allow_pickle=True)
annotation3 = np.load("Data_tests/Annotated times/time_ranges_subject_3.npy", allow_pickle=True)
annotation4 = np.load("Data_tests/Annotated times/time_ranges_subject_4.npy", allow_pickle=True)
annotation5 = np.load("Data_tests/Annotated times/time_ranges_subject_5.npy", allow_pickle=True)
annotation7 = np.load("Data_tests/Annotated times/time_ranges_subject_7.npy", allow_pickle=True)
annotation6 = np.array([
    ['0', '1.4', 'N'],
    ['1.4', '2.6', 'A'],
    ['2.6', '4.8', 'B'],
    ['4.8', '6', 'C'],
    ['6', '7.5', 'B'],
    ['7.5', '8.8', 'A'],
    ['8.8', '11.3', 'N'],
    ['11.3', '12.6', 'A'],
    ['12.6', '13.6', 'B'],
    ['13.6', '14.3', 'C'],
    ['14.3', '15.7', 'B'],
    ['15.7', '17.6', 'A'],
    ['17.6', '18.4', 'N'],
    ['18.4', '19.2', 'A'],
    ['19.2', '20.3', 'B'],
    ['20.3', '21.3', 'C'],
    ['21.3', '22.5', 'B'],
    ['22.5', '23.5', 'A'],
    ['23.5', '24.5', 'N'],
    ['24.5', '25.4', 'A'],
    ['25.4', '26.7', 'B'],
    ['26.7', '27.7', 'C'],
    ['27.7', '29', 'B'],
    ['29', '30', 'A'],
    ['30', '31.4', 'N'],
    ['31.4', '32.1', 'A'],
    ['32.1', '33.3', 'B'],
    ['33.3', '34.4', 'C'],
    ['34.4', '35.8', 'B'],
    ['35.8', '37', 'A'],
    ['37', '38.1', 'N']
])

annotations_matrices = [annotation2, annotation3, annotation4, annotation5, annotation6, annotation7]

def Mean_per_movement(annotation, acc_data, rot_data, imu_locations, sampling_window, min_periods):
    mean_data = {}

    for imu_location in imu_locations:
        acc_data_imu = acc_data[imu_location]
        rot_data_imu = rot_data[imu_location]

        for movement_idx, movement in enumerate(annotation):
            start_time, end_time, movement_label = movement

            start_index = int(float(start_time) * 50)  # Convert time to index (assuming 50 Hz sampling rate)
            end_index = int(float(end_time) * 50)

            acc_data_movement = acc_data_imu[start_index:end_index]
            rot_data_movement = rot_data_imu[start_index:end_index]

            # Compute rolling mean for each movement instance
            rolling_mean_acc = []
            rolling_mean_rot = []
            for i in range(len(acc_data_movement) - sampling_window + 1):
                acc_window = acc_data_movement[i:i+sampling_window]
                rot_window = rot_data_movement[i:i+sampling_window]
                rolling_mean_acc.append(pd.DataFrame(acc_window).mean().values)
                rolling_mean_rot.append(pd.DataFrame(rot_window).mean().values)

            movement_key = f"{movement_label}_{movement_idx}"  # Create movement key
            if imu_location not in mean_data:
                mean_data[imu_location] = {}
            if movement_key not in mean_data[imu_location]:
                mean_data[imu_location][movement_key] = {'acc_mean': rolling_mean_acc, 'rot_mean': rolling_mean_rot}

    return mean_data

def Mean_all_subjects(sampling_window, min_periods):
    mean_data_all_patients = {}

    for i, subject in enumerate(subjects):
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]
        annotation = annotations_matrices[i]

        mean_data_patient = Mean_per_movement(annotation, acc_data_patient, rot_data_patient, imu_locations, sampling_window, min_periods)

        mean_data_all_patients[subject] = mean_data_patient

    return mean_data_all_patients

sampling_window = 3
min_periods = 1
overall_mean_data = Mean_all_subjects(sampling_window, min_periods)
print(overall_mean_data['drinking_HealthySubject2_Test']['hand_IMU']['N_0'])
