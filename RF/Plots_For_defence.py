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
# from Feature_Extraction import RMS_V2
# from Feature_Extraction import Mean_V2
# from Feature_Extraction import  Slope_V2
# from Feature_Extraction import Max_V2
# from Feature_Extraction import Min_V2
# from Feature_Extraction import Standard_Deviation
# from Feature_Extraction import entropy_V2
import labels_interpolation


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
#test_person = int(input('Which subject woudl you like to test on (2-7) ? '))

#######################################################################################################################
### Importing and naming of the datasets ##############################################################################

''' Full datasets'''
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()
all_labels = labels_interpolation.expanded_matrices



subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
         'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

Labels = ['X-acceleration', 'Y-acceleration', 'Z-acceleration']


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