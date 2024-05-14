### Importing of necessary libraries ###############################################################################################
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt


#### Importing of necessary functions for algorithm  #############################################################################
from Feature_Extraction import RMS_V2
from Feature_Extraction import Mean_V2
from Feature_Extraction import  Slope_V2
from Feature_Extraction import Max_V2
from Feature_Extraction import Min_V2
from Feature_Extraction import Standard_Deviation
from Random_forest import labels_interpolation
import Feature_importance


##### VARIABLES ######################################################################################################
'''later toevoegen dat random wordt gekozen wie train en test is'''

train_amount = 5
sampling_window = 3
min_periods = 1
test_amount = train_amount
'''' sampling windows with respective values'''
sampling_window_RMS = 3
sampling_window_min_max = 3
sampling_window_mean = 3
sampling_window_STD = 3
sampling_window_slope = 3
test_person = 3
#test_person = int(input('Which subject woudl you like to test on (2-7) ? '))

#######################################################################################################################
### Importing and naming of the datasets ##############################################################################

#### Importing of necessary functions for algorithm  #############################################################################
from Feature_Extraction import RMS_V2, Mean_V2, Slope_V2, Max_V2, Min_V2, Standard_Deviation
from Random_forest import labels_interpolation
from scipy.stats import pearsonr



##### VARIABLES ######################################################################################################
# '''later toevoegen dat random wordt gekozen wie train en test is'''

train_amount = 5
sampling_window = 3
min_periods = 1
test_amount = train_amount
'''' sampling windows with respective values'''
sampling_window_RMS = 3
sampling_window_min_max = 3
sampling_window_mean = 3
sampling_window_STD = 3
sampling_window_slope = 3
test_person = 2
#test_person = int(input('Which subject woudl you like to test on (2-7) ? '))

#######################################################################################################################
### Importing and naming of the datasets ##############################################################################

''' Full datasets'''
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

# print(acc)
all_labels = labels_interpolation.expanded_matrices



subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
        'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# ########must be  into movements(not per measurement!)######################

# subjects.remove(f'drinking_HealthySubject{test_person}_Test')
subjects_train = subjects
subjects_test = [f'drinking_HealthySubject{test_person}_Test']
#print(subjects_test)

# test_labels = all_labels[test_person - 2]
# #print("test labels:",test_labels)

# all_labels.pop(test_person - 2)
train_labels = all_labels
#print("train labels:",train_labels)

#################################################################################################################
### Setting up the test and training sets with labels ###########################################################

# X_train_RMS = RMS_V2.RMS_train(subjects_train, sampling_window_RMS, min_periods)
# X_test_RMS = RMS_V2.RMS_test(subjects_test, sampling_window_RMS, min_periods)

# X_train_Mean = Mean_V2.Mean_train(subjects_train, sampling_window_mean, min_periods)
# X_test_Mean = Mean_V2.Mean_test(subjects_test, sampling_window_mean, min_periods)

# X_train_Slope = Slope_V2.Slope_train(subjects_train, sampling_window_slope, min_periods)
# X_test_Slope = Slope_V2.Slope_test(subjects_test, sampling_window_slope, min_periods)

X_train_Max = Max_V2.Max_train(subjects_train, sampling_window_min_max, min_periods)
X_test_Max = Max_V2.Max_test(subjects_test, sampling_window_min_max, min_periods)

X_train_Min = Min_V2.Min_train(subjects_train, sampling_window_min_max, min_periods)
X_test_Min = Min_V2.Min_test(subjects_test, sampling_window_min_max, min_periods)

# X_train_STD = Standard_Deviation.STD_train(subjects_train, sampling_window_STD, min_periods)
# X_test_STD = Standard_Deviation.STD_test(subjects_test, sampling_window_STD, min_periods)


###################### Recovered from chatGPT ############################

# Dictionary to hold correlation lists for each subject
correlations = {}

# Iterate over each subject
for subject in subjects:
        # Initialize the correlation list for the current subject
        corr = []
        
        # Get the number of rows (assuming all subjects have the same number of rows)
        # num_rows = len(X_test_Min[subject]['hand_IMU']["acc_min"])

        # Iterate over the rows for the current subject
        for number in range(3):
            x = X_train_Max[subject]['hand_IMU']["acc_max"][number]
            y = X_train_Min[subject]['hand_IMU']["acc_min"][number]
            correlation, _ = pearsonr(x, y)
            corr.append(correlation)  # Append the correlation value to the list

        # Store the correlation list in the dictionary
        correlations[subject] = corr

# Print the correlation lists for each subject
for subject, corr in correlations.items():
    print(f'The correlations for {subject} are: {corr}')