import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm  # Import tqdm library for progress bars
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import seaborn as sns

from Feature_Extraction import RMS_V2, Mean_V2, Slope_V2, Max_V2, Min_V2, Standard_Deviation
import labels_interpolation

# Define parameters
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
all_labels = labels_interpolation.expanded_matrices

# Prepare subjects and labels
subjects = [f'drinking_HealthySubject{i+2}_Test' for i in range(6)]
subjects.remove(f'drinking_HealthySubject{test_person}_Test')
subjects_train = subjects
subjects_test = [f'drinking_HealthySubject{test_person}_Test']

test_labels = all_labels[test_person - 2]
all_labels.pop(test_person - 2)
train_labels = all_labels

labels_train = [i[1] for item in train_labels for i in item]
labels_test = [item[1] for item in test_labels]
label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}

y_train = [label_mapping[label] for label in labels_train]
y_test = [label_mapping[label] for label in labels_test]

# Feature extraction
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

# Combine features
def combine_features(subjects, rms, mean, slope, max_val, min_val, std_dev):
    combined_data_patients = []
    for subject in subjects:
        combined_data_patient = []
        for imu_location in rms[subject]:
            combined_data_imu = np.hstack((
                rms[subject][imu_location]["acc_rms"], rms[subject][imu_location]["rot_rms"],
                mean[subject][imu_location]["acc_mean"], mean[subject][imu_location]["rot_mean"],
                slope[subject][imu_location]["acc_slope"], slope[subject][imu_location]["rot_slope"],
                max_val[subject][imu_location]["acc_max"], max_val[subject][imu_location]["rot_max"],
                min_val[subject][imu_location]["acc_min"], min_val[subject][imu_location]["rot_min"],
                std_dev[subject][imu_location]["acc_STD"], std_dev[subject][imu_location]["rot_STD"]
            ))
            combined_data_patient.append(combined_data_imu)
        combined_data_patients.append(np.hstack(combined_data_patient))
    return np.concatenate(combined_data_patients)

X_train = combine_features(subjects_train, X_train_RMS, X_train_Mean, X_train_Slope, X_train_Max, X_train_Min, X_train_STD)
X_test = combine_features(subjects_test, X_test_RMS, X_test_Mean, X_test_Slope, X_test_Max, X_test_Min, X_test_STD)

print(X_train.shape)

ovr_clf = OneVsRestClassifier(SVC(C=0.1, gamma=0.01, kernel="rbf",random_state=42))
ovr_clf.fit(X_train, y_train)

# Predictions
y_test_pred = ovr_clf.predict(X_test)
y_train_pred = ovr_clf.predict(X_train)

print("Classification Report of train data:")
print(classification_report(y_train, y_train_pred))

print("Classification Report of test data:")
print(classification_report(y_test, y_test_pred))

# Define minimum duration threshold
min_duration_threshold = 5  # Minimum number of measurements for a label to be considered valid

# Post-processing function
def filter_short_labels(predictions, min_duration):
    filtered_predictions = predictions.copy()
    current_label = predictions[0]
    current_start = 0

    for i in range(1, len(predictions)):
        if predictions[i] != current_label:
            if i - current_start < min_duration:
                filtered_predictions[current_start:i] = [filtered_predictions[current_start - 1]] * (i - current_start) if current_start > 0 else [0] * (i - current_start)
            current_label = predictions[i]
            current_start = i

    # Check the last segment
    if len(predictions) - current_start < min_duration:
        filtered_predictions[current_start:] = [filtered_predictions[current_start - 1]] * (len(predictions) - current_start) if current_start > 0 else [0] * (len(predictions) - current_start)
    
    return filtered_predictions

# Apply the filter to test predictions
y_test_pred_filtered = filter_short_labels(y_test_pred, min_duration_threshold)

# Apply the filter to train predictions if needed
y_train_pred_filtered = filter_short_labels(y_train_pred, min_duration_threshold)

# Generate Classification Report for filtered predictions
print("Classification Report of test data (filtered):")
print(classification_report(y_test, y_test_pred_filtered))

print("Classification Report of train data (filtered):")
print(classification_report(y_train, y_train_pred_filtered))

# Plot filtered predictions
element_numbers = list(range(len(y_test_pred_filtered)))

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(element_numbers, y_test_pred_filtered, label='Filtered Predictions', color='blue')
plt.xlabel('Element Numbers')
plt.ylabel('Predicted Labels')
plt.title(f'Filtered Predicted Labels - {subjects_test[0]}')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(element_numbers, y_test, label='True Labels', color='green')
plt.xlabel('Element Numbers')
plt.ylabel('True Labels')
plt.title(f'True Labels - {subjects_test[0]}')
plt.legend()

plt.tight_layout()
plt.show()

# Compute confusion matrix for test data
conf_matrix = confusion_matrix(y_test, y_test_pred_filtered)

# Label maps for confusion matrix
label_mapping = {0: 'N', 1: 'A', 2: 'B', 3: 'C'}

# Plot confusion matrix
print("Confusion Matrix:\n", conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[label_mapping[key] for key in label_mapping.keys()],
            yticklabels=[label_mapping[key] for key in label_mapping.keys()])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix for {test_person}')
plt.show()

num_classes = len(np.unique(y_train))
n_components_lda = min(num_classes - 1, X_train.shape[1])

lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Using the determined parameters for OvA classification with SVC
ova_clf_lda = OneVsRestClassifier(SVC(C=0.1, gamma=0.01, kernel="rbf", random_state=42))
ova_clf_lda.fit(X_train_lda, y_train)
y_test_pred_lda = ova_clf_lda.predict(X_test_lda)

ova_clf_pca = OneVsRestClassifier(SVC(C=0.1, gamma=0.01, kernel="rbf", random_state=42))
ova_clf_pca.fit(X_train_pca, y_train)
y_test_pred_pca = ova_clf_pca.predict(X_test_pca)

print("Classification Report of test data for LDA:")
print(classification_report(y_test, y_test_pred_lda))

print("Classification Report of test data for PCA:")
print(classification_report(y_test, y_test_pred_pca, zero_division=1))

lda_feature_importance = np.abs(lda.coef_[0])
n_features_lda = lda.n_features_in_
lda_feature_importance /= np.sum(lda_feature_importance)

print("Feature Importances from LDA:")
print(lda_feature_importance)

pca_explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratios from PCA:")
print(pca_explained_variance_ratio)

pca_feature_importance = np.cumsum(pca_explained_variance_ratio)
pca_feature_importance /= np.sum(pca_feature_importance)

print("Feature Importances from PCA:")
print(pca_feature_importance)

plt.figure(figsize=(10, 6))
plt.bar(range(n_features_lda), lda_feature_importance, align="center", color='orange', label='LDA')
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance (LDA)")
plt.legend()

plt.figure(figsize=(10, 6))
plt.bar(range(X_train_pca.shape[1]), pca_feature_importance, align="center", color='green', label='PCA')
plt.xlabel("PCA Component Index")
plt.ylabel("Feature Importance (PCA)")
plt.legend()
plt.show()
