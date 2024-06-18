#description: implemented features for the model to train on 


### Importing of necessary libraries ###############################################################################################
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import seaborn as sns
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.decomposition import PCA

#### Importing of necessary functions for algorithm  #############################################################################
from Feature_Extraction import RMS_V2
from Feature_Extraction import Mean_V2
from Feature_Extraction import  Slope_V2
from Feature_Extraction import Max_V2
from Feature_Extraction import Min_V2
from Feature_Extraction import Standard_Deviation
from Feature_Extraction import entropy_V2
import labels_interpolation

''' Full datasets'''
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()
all_labels = labels_interpolation.expanded_matrices

subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
         'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

Labels = ['X-acceleration', 'Y-acceleration', 'Z-acceleration']


##### VARIABLES ######################################################################################################
'''later toevoegen dat random wordt gekozen wie train en test is'''

sampling_window = 3
min_periods = 1
'''' sampling windows with respective values'''
sampling_window_RMS = 3
sampling_window_min_max = 3
sampling_window_mean = 3
sampling_window_STD = 3
sampling_window_slope = 3
sampling_window_entropy = 3

test_person = 7
subjects.remove(f'drinking_HealthySubject{test_person}_Test')
subjects_train = subjects
subjects_test = [f'drinking_HealthySubject{test_person}_Test']

test_labels = all_labels[test_person - 2]
all_labels.pop(test_person - 2)
train_labels = all_labels
#test_person = int(input('Which subject woudl you like to test on (2-7) ? '))
Y_train_labels = train_labels
Y_test_labels = test_labels
X_train_RMS = RMS_V2.RMS_train(subjects_train, sampling_window_RMS, min_periods)
X_test_RMS = RMS_V2.RMS_test(subjects_test, sampling_window_RMS, min_periods)

X_train_Mean = Mean_V2.Mean_train(subjects_train, sampling_window_mean, min_periods)
X_test_Mean = Mean_V2.Mean_test(subjects_test, sampling_window_mean, min_periods)

X_train_Slope = Slope_V2.Slope_train(subjects_train, sampling_window_slope, min_periods)
X_test_Slope = Slope_V2.Slope_test(subjects_test, sampling_window_slope, min_periods)

X_train_Max = Max_V2.Max_train(subjects_train, sampling_window_min_max, min_periods)
X_test_Max = Max_V2.Max_test(subjects_test, sampling_window_min_max, min_periods)

X_train_Min = Min_V2.Min_train(subjects_train, sampling_window_min_max, min_periods)
X_test_Min = Min_V2.Min_test(subjects_test, sampling_window_min_max, min_periods)

X_train_STD = Standard_Deviation.STD_train(subjects_train, sampling_window_STD, min_periods)
X_test_STD = Standard_Deviation.STD_test(subjects_test, sampling_window_STD, min_periods)

labels_train = []
for item in Y_train_labels:
    for i in item:
        labels_train.append(i[1])

labels_test = []
for item in Y_test_labels:
    labels_test.append(item[1])

label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}

y_train = [label_mapping[label] for label in labels_train]
y_test = [label_mapping[label] for label in labels_test]

X_data_patients_train = []

for subject in X_train_RMS:
    combined_data_patient = []

    for imu_location in X_train_RMS[subject]:
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

        combined_data_imu = np.hstack((acc_rms_imu, rot_rms_imu, acc_mean_imu, rot_mean_imu, acc_slope_imu, rot_slope_imu,
                                       acc_max_imu, rot_max_imu, acc_min_imu, rot_min_imu, acc_STD_imu, rot_STD_imu))
        combined_data_patient.append(combined_data_imu)

    X_data_patients_train.append(np.hstack(combined_data_patient))

combined_X_data_train = np.concatenate(X_data_patients_train)
X_train = combined_X_data_train

X_data_patients_test = []

for subject in X_test_RMS:
    combined_data_patient = []

    for imu_location in X_test_RMS[subject]:
        acc_rms_imu = X_test_RMS[subject][imu_location]["acc_rms"]
        rot_rms_imu = X_test_RMS[subject][imu_location]["rot_rms"]
        acc_mean_imu = X_test_Mean[subject][imu_location]["acc_mean"]
        rot_mean_imu = X_test_Mean[subject][imu_location]["rot_mean"]
        acc_slope_imu = X_test_Slope[subject][imu_location]["acc_slope"]
        rot_slope_imu = X_test_Slope[subject][imu_location]["rot_slope"]
        acc_max_imu = X_test_Max[subject][imu_location]["acc_max"]
        rot_max_imu = X_test_Max[subject][imu_location]["rot_max"]
        acc_min_imu = X_test_Min[subject][imu_location]["acc_min"]
        rot_min_imu = X_test_Min[subject][imu_location]["rot_min"]
        acc_STD_imu = X_test_STD[subject][imu_location]["acc_STD"]
        rot_STD_imu = X_test_STD[subject][imu_location]["rot_STD"]

        combined_data_imu = np.hstack((acc_rms_imu, rot_rms_imu, acc_mean_imu, rot_mean_imu, acc_slope_imu, rot_slope_imu,
                                       acc_max_imu, rot_max_imu, acc_min_imu, rot_min_imu, acc_STD_imu, rot_STD_imu))
        combined_data_patient.append(combined_data_imu)

    X_data_patients_test.append(np.hstack(combined_data_patient))

combined_X_data_test = np.concatenate(X_data_patients_test)
X_test = combined_X_data_test

clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

y_test_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)





#######################################################################################################################
### Importing and naming of the datasets ##############################################################################

#Figure for the acceleration
plt.figure(figsize= (12,4))
plt.grid()
plt.title("Acceleration data Participant 6")
plt.plot(acc['drinking_HealthySubject6_Test']['hand_IMU'], label = Labels)
plt.legend()
plt.ylabel("Acceleration [m/s²]")
plt.xlabel("Time [s]")

# Scale the x-axis down by 50
current_ticks = plt.gca().get_xticks()
plt.gca().set_xticks(current_ticks)  # Ensure the current ticks are used
plt.gca().set_xticklabels([str(tick / 50) for tick in current_ticks])  # Scale down the tick labels by 50
plt.xlim([0, 38.1*50])
plt.show()

# Generate incorrect indices
incorrect_indices = [i for i in range(len(y_test)) if y_test[i] != y_test_pred[i]]

def plot_with_highlight(ax, data, incorrect_indices, label):
    x_data = data[:, 0]  # Assuming the x-axis acceleration data is the first column
    correct_plotted = False
    incorrect_plotted = False
    
    for i in range(len(x_data) - 1):
        if i in incorrect_indices:
            if not incorrect_plotted:
                ax.plot([i, i+1], [x_data[i], x_data[i+1]], color='red', label='Incorrect classification')
                incorrect_plotted = True
            else:
                ax.plot([i, i+1], [x_data[i], x_data[i+1]], color='red')
        else:
            if not correct_plotted:
                ax.plot([i, i+1], [x_data[i], x_data[i+1]], color='green', label='Correct classification')
                correct_plotted = True
            else:
                ax.plot([i, i+1], [x_data[i], x_data[i+1]], color='green')
                
    ax.set_xlabel('Element number')
    ax.set_ylabel('X Acceleration value')
    ax.set_title('X_acceleration for participant 6')
    ax.legend()

# Example usage
plt.figure()
plot_with_highlight(plt.gca(), acc[f'drinking_HealthySubject{test_person}_Test']['hand_IMU'], incorrect_indices, 'hand_IMU')
plt.grid()
plt.tight_layout()
plt.show()