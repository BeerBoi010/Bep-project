import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import sys
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

test_labels = all_labels[test_person - 2]
all_labels.pop(test_person - 2)
train_labels = all_labels

num_features = 30
for subject in subjects:
    subject_feature_matrix = np.zeros((0, num_features))

    for imu_location in acc[subject]:
        data_imu = [] 
        #Stacking of both acc and rot data to put through features
        acc_data_imu_split = np.array_split(acc[subject][imu_location],381)
        rot_data_imu_split = np.array_split(rot[subject][imu_location],381)
        full_data = np.hstack((acc_data_imu_split, rot_data_imu_split))

        for split in full_data:
            for step in split:      #calculate the features for each channel (column)
                Mean = np.mean(step, axis=0)
                STD = np.std(step, axis=0)
                RMS = np.sqrt(np.mean(step**2, axis=0))  #RMS value of each column.
                MIN = np.min(step, axis=0)
                MAX = np.max(step, axis=0)
                window_features = np.hstack((Mean, STD, RMS, MIN, MAX))

        # Append the features for the current window to subject_feature_matrix
        subject_feature_matrix = np.vstack((subject_feature_matrix, window_features))

    # Append the features for the current subject to Feature_Matrix    
    Feature_Matrix = np.vstack((Feature_Matrix, subject_feature_matrix))

print(Feature_Matrix)
Y_train_labels = train_labels
Y_test_labels = test_labels

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


windowsize = 1905/5

y_train= np.array_split(y_train, windowsize)
y_train = np.array(y_train)

print(y_train)
print(y_train.shape)

np.set_printoptions(threshold=sys.maxsize)

y_test= np.array_split(y_test, windowsize)
y_test = np.array(y_test)

print(y_test)

