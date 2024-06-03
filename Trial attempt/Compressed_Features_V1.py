import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

from Feature_Extraction import RMS_V2, Mean_V2, Slope_V2, Max_V2, Min_V2, Standard_Deviation
import labels_interpolation

train_amount = 6
sampling_window = 3
min_periods = 1
test_amount = train_amount

sampling_window_RMS = 3
sampling_window_min_max = 3
sampling_window_mean = 3
sampling_window_STD = 3
sampling_window_slope = 3
test_person = 7
print(f'drinking_HealthySubject{test_person}_Test')
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

all_labels = labels_interpolation.expanded_matrices

subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

subjects.remove(f'drinking_HealthySubject{test_person}_Test')
subjects_train = subjects
subjects_test = [f'drinking_HealthySubject{test_person}_Test']

##################################################################################################################################
Y_test_labels = all_labels[test_person - 2]
all_labels.pop(test_person - 2)
Y_train_labels = all_labels

labels_train = []
###### for-loops to make annotation list for random forest method ###########################################################################
# for item in Y_train_labels:
#     filtered_array = []
#     for i in range(0, len(Y_train_labels), 5):
#     # Append every 5th element to the filtered_array
#         filtered_array.append(Y_train_labels[i])
#         labels_train.append(filtered_array)
# print("labels train", labels_train, labels_train.shape)

# labels_test1 = []
# for item in Y_test_labels:
#     labels_test1.append(item[1])
# labels_test = labels_test1
# # print("labels test", labels_test)

# # Dictionary to map labels to numerical values
# label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}

# # Convert labels to numerical values
# y_train = [label_mapping[label] for label in labels_train]
# y_test = [label_mapping[label] for label in labels_test]
#############################################################################################################################
#%% Extracting features: 
window_length_sec = 0.5 # 0.5 second
overlap = 0.5
windows_AllSubject = []
Labels_AllSubject = []
window_counts_AllSubject = [] 

#stacking of acceleration and rotation matrices next to each other. This way we can prime the data before the feature extraction.
X_data_patients_dict = {}

for subject in subjects:
    combined_data_patient = []
    for imu in acc[subject]:
        combined_data_imu = np.hstack((acc[subject][imu], rot[subject][imu]))
        combined_data_patient.append(combined_data_imu)
    #Dictionary with the combined acc and rot data per subject
    X_data_patients_dict[subject] = np.hstack((combined_data_patient))

# Combine data for all subjects into a single array
combined_X_data_train = np.concatenate(list(X_data_patients_dict.values()))
FullCombinedData = combined_X_data_train

#print(X_data_patients_dict['drinking_HealthySubject5_Test'],X_data_patients_dict['drinking_HealthySubject5_Test'].shape)


################## Setting up the feature matrix ###################
feature_matrix = {'drinking_HealthySubject2_Test': [],'drinking_HealthySubject3_Test': [], 'drinking_HealthySubject4_Test': [],
                        'drinking_HealthySubject5_Test': [],'drinking_HealthySubject6_Test': [],'drinking_HealthySubject7_Test': []}
for patient in X_data_patients_dict:
    X_data_patients_dict[patient] = np.array_split(X_data_patients_dict[patient], 381)
    for split in X_data_patients_dict[patient]:
        #Setting up features that loop through the columns: mean_x_acc,mean_y_acc....,Mean_x_rot. For all features, so
        #a row of 5*5*6 = 150 features 
        Mean = np.mean(split, axis=0)
        STD = np.std(split, axis=0)
        RMS = np.sqrt(np.mean(split**2, axis=0))  # RMS value of each column
        MIN = np.min(split, axis=0)
        MAX = np.max(split, axis=0)
        feature_matrix[patient].append(np.hstack((Mean,STD,RMS,MIN,MAX)))

feature_matrix['drinking_HealthySubject2_Test'] = np.array(feature_matrix['drinking_HealthySubject2_Test'])
print(feature_matrix['drinking_HealthySubject2_Test'],feature_matrix['drinking_HealthySubject2_Test'].shape)
#print(X_data_patients_dict['drinking_HealthySubject2_Test'],X_data_patients_dict['drinking_HealthySubject2_Test'].shape)