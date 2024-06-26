import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import seaborn as sns
from collections import Counter

from Feature_Extraction import RMS_V2, Mean_V2, Slope_V2, Max_V2, Min_V2, Standard_Deviation
import labels_interpolation

#test_person = 5
bin_size = 20
bin_val = int(1905/bin_size)

#print(f'drinking_HealthySubject{test_person}_Test')
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

all_labels = labels_interpolation.expanded_matrices

subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# subjects.remove(f'drinking_HealthySubject{test_person}_Test')
subjects_train = subjects
#subjects_test = [f'drinking_HealthySubject{test_person}_Test']

##################################################################################################################################
Y_train_labels = all_labels


def majority_vote(labels):
    """Returns the label with the highest count in the labels list."""
    counter = Counter(labels)
    return counter.most_common(1)[0][0]

def process_labels(labels, bin_size, overlap):
    """Processes the labels with majority voting in each window with the specified overlap."""
    step = bin_size - overlap
    processed_labels = []
    
    for start in range(0, len(labels) - bin_size + 1, step):
        window = labels[start:start + bin_size]
        majority_label = majority_vote([label[1] for label in window])
        processed_labels.append(majority_label)
    
    return processed_labels

# Parameters
overlap = 0

labels_train = []
for item in Y_train_labels:
    labels_train.extend(process_labels(item, bin_size, overlap))
print("labels train", labels_train, len(labels_train))

# Dictionary to map labels to numerical values
label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}

# Convert labels to numerical values
y_train = [label_mapping[label] for label in labels_train]

print(len(y_train))

#############################################################################################################################

#stacking of acceleration and rotation matrices for all imu sensors next to each other. This way we can prime the data before the feature extraction.
X_data_patients_dict = {}

for subject in subjects_train:
    combined_data_patient = []
    for imu in acc[subject]:
        combined_data_imu = np.hstack((acc[subject][imu], rot[subject][imu]))
        combined_data_patient.append(combined_data_imu)
    #Dictionary with the combined acc and rot data per subject
    X_data_patients_dict[subject] = np.hstack((combined_data_patient))

print('Full shape of one patient', X_data_patients_dict['drinking_HealthySubject2_Test'].shape)
################## Setting up the feature matrix ###################
feature_dict = {'drinking_HealthySubject2_Test': [],'drinking_HealthySubject3_Test': [], 'drinking_HealthySubject4_Test': [],'drinking_HealthySubject5_Test': [],
                  
                        'drinking_HealthySubject6_Test': [],'drinking_HealthySubject7_Test': []}
for patient in X_data_patients_dict:
    #Calls the array for one subject and splits it in equal parts of five
    X_data_patients_dict[patient] = np.array_split(X_data_patients_dict[patient], bin_val)
    for split in X_data_patients_dict[patient]:
        #print(split,split.shape)
        #Setting up features that loop through the columns: mean_x_acc,mean_y_acc....,Mean_x_rot. For all featured and 5 imu sensors so
        #a row of 5*6*6 = 180 features 
        Mean = np.mean(split, axis=0)
        STD = np.std(split, axis=0)
        RMS = np.sqrt(np.mean(split**2, axis=0))  # RMS value of each column
        MIN = np.min(split, axis=0)
        MAX = np.max(split, axis=0)
        Slope = (np.take(split, -1, axis =0) - np.take(split, 0, axis =0))/bin_size
        #appends all features in a dictionary for each patient 
        feature_dict[patient].append(np.hstack((Mean,STD,RMS,MIN,MAX, Slope)))

# Combine all feature arrays into a single array
compressed_array_train = np.concatenate(list(feature_dict.values()), axis=0)

print(compressed_array_train.shape,(np.array(y_train)).shape)


#Set up a random train-test split for a subjective model
X_train, X_test, y_train, y_test = train_test_split(compressed_array_train, y_train, test_size=0.2, random_state=42)


ovr_clf = OneVsOneClassifier(SVC(C=10, gamma=0.01, kernel="rbf",random_state=42))
ovr_clf.fit(X_train, y_train)

# Predictions
y_test_pred = ovr_clf.predict(X_test)
y_train_pred = ovr_clf.predict(X_train)

print("Classification Report of train data:")
print(classification_report(y_train, y_train_pred))

print("Classification Report of test data:")
print(classification_report(y_test, y_test_pred))

# Compute confusion matrix for test data
conf_matrix = confusion_matrix(y_test, y_test_pred)

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
plt.title("Confusion Matrix Random Split")
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
ova_clf_lda = OneVsOneClassifier(SVC(C=10, gamma=0.01, kernel="rbf", random_state=42))
ova_clf_lda.fit(X_train_lda, y_train)
y_test_pred_lda = ova_clf_lda.predict(X_test_lda)

ova_clf_pca = OneVsOneClassifier(SVC(C=10, gamma=0.01, kernel="rbf", random_state=42))
ova_clf_pca.fit(X_train_pca, y_train)
y_test_pred_pca = ova_clf_pca.predict(X_test_pca)

print("Classification Report of test data for LDA:")
print(classification_report(y_test, y_test_pred_lda))

# Compute confusion matrix for test data
conf_matrix = confusion_matrix(y_test, y_test_pred_lda)

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
plt.title("Confusion Matrix Random Split LDA")
plt.show()

print("Classification Report of test data for PCA:")
print(classification_report(y_test, y_test_pred_pca, zero_division=1))

lda_feature_importance = np.abs(lda.coef_[0])
n_features_lda = lda.n_features_in_
lda_feature_importance /= np.sum(lda_feature_importance)

# Get the indices of the most important features
important_features_indices = np.argsort(lda_feature_importance)[::-1]

# Print the most important features
top_n = 30  # Number of top features to print
print(f"Top {top_n} most important features from LDA:")
for i in range(top_n):
    print(f"Feature {important_features_indices[i]}: Importance {lda_feature_importance[important_features_indices[i]]:.4f}")


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
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(range(X_train_pca.shape[1]), pca_feature_importance, align="center", color='green', label='PCA')
plt.xlabel("PCA Component Index")
plt.ylabel("Feature Importance (PCA)")
plt.legend()
plt.show()
