#########################################

#uses best parameters found in gridsearch,added filter,added print for most important lda features, removed mistakes

########################

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

train_amount = 5
sampling_window = 50
min_periods = 1
test_amount = train_amount
sampling_window_RMS = 50
sampling_window_min_max = 50
sampling_window_mean = 50
sampling_window_STD = 50
sampling_window_slope = 50
test_person = 5


# train_amount = 5
# sampling_window = 3
# min_periods = 1
# test_amount = train_amount

# sampling_window_RMS = 3
# sampling_window_min_max = 3
# sampling_window_mean = 3
# sampling_window_STD = 3
# sampling_window_slope = 3
# test_person = 7
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

clf = RandomForestClassifier(n_estimators=100,min_samples_leaf=1,max_depth=10, random_state=42)
clf.fit(X_train, y_train)

y_test_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)

print("Classification Report of train data:")
print(classification_report(y_train, y_train_pred))

print("Classification Report of test data:")
print(classification_report(y_test, y_test_pred))

element_numbers = list(range(len(y_test_pred)))

### Setting up plots to illustrate code
plt.figure(figsize=(12, 6))
plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['hand_IMU'])
plt.xlabel('Element number')
plt.ylabel('Acceleration value')
plt.title(f'hand_IMU - {subjects_test[0]}')
plt.show

plt.figure(figsize=(12, 6))

plt.subplot(2, 4, 1)
plt.plot(element_numbers, y_test_pred, label='Predictions', color='blue')
plt.xlabel('Element Numbers')
plt.ylabel('Predicted Labels')
plt.title(f'Predicted Labels - {subjects_test[0]}')
plt.legend()

plt.subplot(2, 4, 2)
plt.plot(element_numbers, y_test, label='True Labels', color='green')
plt.xlabel('Element Numbers')
plt.ylabel('True Labels')
plt.title(f'True Labels - {subjects_test[0]}')
plt.legend()

plt.subplot(2, 4, 3)
plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['hand_IMU'])
plt.xlabel('Element number')
plt.ylabel('Acceleration value')
plt.title(f'hand_IMU - {subjects_test[0]}')

plt.subplot(2, 4, 5)
plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['lowerarm_IMU'])
plt.xlabel('Element number')
plt.ylabel('Acceleration value')
plt.title(f'lowerarm_IMU - {subjects_test[0]}')

plt.subplot(2, 4, 6)
plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['upperarm_IMU'])
plt.xlabel('Element number')
plt.ylabel('Acceleration value')
plt.title(f'upperarm_IMU - {subjects_test[0]}')

plt.subplot(2, 4, 7)
plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['shoulder_IMU'])
plt.xlabel('Element number')
plt.ylabel('Acceleration value')
plt.title(f'shoulder_IMU - {subjects_test[0]}')

plt.subplot(2, 4, 8)
plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['sternum_IMU'])
plt.xlabel('Element number')
plt.ylabel('Acceleration value')
plt.title(f'sternum_IMU - {subjects_test[0]}')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

plt.plot(element_numbers, y_test_pred, label='Predictions', color='black')
plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['hand_IMU'])
plt.xlabel('Element Numbers')
plt.ylabel('Predicted Labels')
plt.title(f'Predicted Labels vs Acceleration Data - {subjects_test[0]}')
plt.legend()
plt.show()

# Compute confusion matrix for test data
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Label maps for confusion matrix
label_mapping = {0: 'N', 1: 'A', 2: 'B', 3: 'C'}

# Plot confusion matrix
print("Confusion Matrix:\n", conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16},
            xticklabels=[label_mapping[key] for key in label_mapping.keys()],
            yticklabels=[label_mapping[key] for key in label_mapping.keys()])
plt.xlabel('Predicted Labels', fontsize = 12)
plt.ylabel('True Labels', fontsize = 12)
plt.title(f'Confusion Matrix of subject 7', fontsize = 12)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tick_params(axis='both', which='minor', labelsize=10)
plt.show()

# plt.plot(element_numbers, y_test_pred, label='Predictions', color='black')
# plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['hand_IMU'])
# plt.xlabel('Element Numbers')
# plt.ylabel('Predicted Labels')
# plt.title(f'Predicted Labels vs Acceleration Data - {subjects_test[0]}')
# plt.legend()
# plt.show()
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 4, 1)
# plt.plot(element_numbers, y_test_pred, label='Predictions', color='blue')
# plt.plot(element_numbers, y_test, label='True Labels', color='black')
# plt.xlabel('Element Numbers')
# plt.ylabel('Predicted Labels')
# plt.title(f'Predicted Labels - {subjects_test[0]}')
# plt.legend()

# plt.subplot(2, 4, 2)
# plt.plot(element_numbers, y_test, label='True Labels', color='green')
# plt.xlabel('Element Numbers')
# plt.ylabel('True Labels')
# plt.title(f'True Labels - {subjects_test[0]}')
# plt.legend()

# plt.subplot(2, 4, 3)
# plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['hand_IMU'])
# plt.xlabel('Element number')
# plt.ylabel('Acceleration value')
# plt.title(f'hand_IMU - {subjects_test[0]}')

# plt.subplot(2, 4, 5)
# plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['lowerarm_IMU'])
# plt.xlabel('Element number')
# plt.ylabel('Acceleration value')
# plt.title(f'lowerarm_IMU - {subjects_test[0]}')

# plt.subplot(2, 4, 6)
# plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['upperarm_IMU'])
# plt.xlabel('Element number')
# plt.ylabel('Acceleration value')
# plt.title(f'upperarm_IMU - {subjects_test[0]}')

# plt.subplot(2, 4, 7)
# plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['shoulder_IMU'])
# plt.xlabel('Element number')
# plt.ylabel('Acceleration value')
# plt.title(f'shoulder_IMU - {subjects_test[0]}')

# plt.subplot(2, 4, 8)
# plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['sternum_IMU'])
# plt.xlabel('Element number')
# plt.ylabel('Acceleration value')
# plt.title(f'sternum_IMU - {subjects_test[0]}')

# plt.tight_layout()
# plt.show()





importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]

# plt.figure(figsize=(10, 6))
# plt.title("Feature Importances")
# plt.bar(range(X_train.shape[1]), importances[indices], align="center")
# plt.xticks(range(X_train.shape[1]), indices)
# plt.xlabel("Feature Index")
# plt.ylabel("Feature Importance")
# plt.show()

num_classes = len(np.unique(y_train))
n_components_lda = min(num_classes - 1, X_train.shape[1])


lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
X_train_lda = lda.fit_transform(X_train, y_train)
print(X_train_lda.shape)
X_test_lda = lda.transform(X_test)

pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

clf_lda = RandomForestClassifier(n_estimators=100,min_samples_leaf=1,max_depth=10, random_state=42)
clf_lda.fit(X_train_lda, y_train)
y_test_pred_lda = clf_lda.predict(X_test_lda)

clf_pca = RandomForestClassifier(n_estimators=100,min_samples_leaf=1,max_depth=10, random_state=42)
clf_pca.fit(X_train_pca, y_train)
y_test_pred_pca = clf_pca.predict(X_test_pca)

print("Classification Report of test data for LDA:")
print(classification_report(y_test, y_test_pred_lda))

print("Classification Report of test data for PCA:")
print(classification_report(y_test, y_test_pred_pca, zero_division=1))

lda_feature_importance = np.abs(lda.coef_[0])

n_features_lda = lda.n_features_in_

lda_feature_importance /= np.sum(lda_feature_importance)
#Get the indices of the most important features
lda_important_features_indices = np.argsort(lda_feature_importance)[::-1]

# Print the most important features
top_n = 30  # Number of top features to print
print(f"Top {top_n} most important features from LDA:")
for i in range(top_n):
    print(f"LDA Feature {lda_important_features_indices[i]}: Importance {lda_feature_importance[lda_important_features_indices[i]]:.4f}")
    print(f'MDI Feature {indices[i]}: Importance {importances[i]}')


print("Feature Importances from LDA:")
print(lda_feature_importance[:30])

pca_explained_variance_ratio = pca.explained_variance_ratio_

# print("Explained Variance Ratios from PCA:")
# print(pca_explained_variance_ratio)

pca_feature_importance = np.cumsum(pca_explained_variance_ratio)

pca_feature_importance /= np.sum(pca_feature_importance)

print("Feature Importances from PCA:")
print(pca_feature_importance[:30])

#Get feature importances
importances = clf.feature_importances_



#Plot all feature importances

plt.figure(figsize=(10, 6))
plt.title("Feature Importances MDI")
plt.bar(range(X_train.shape[1])[:top_n], importances[indices][:top_n], align="center")
plt.xticks(range(X_train.shape[1])[:top_n], indices[:top_n])
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.show()

# # plt.figure(figsize=(12, 6))
# plt.figure(figsize=(10, 6))
# plt.bar(range(n_features_lda)[:top_n], important_features_indices[:top_n], align="center", color='orange', label='LDA')
# plt.xlabel("Feature Index")
# plt.ylabel("Feature Importance (LDA)")
# plt.legend()

plt.figure(figsize=(10, 6))
plt.bar(range(X_train_pca.shape[1])[:], pca_feature_importance[:], align="center", color='green', label='PCA')
plt.xlabel("PCA Component Index")
plt.ylabel("Feature Importance (PCA)")
plt.legend()
plt.show()
