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
from collections import Counter

from Feature_Extraction import RMS_V2, Mean_V2, Slope_V2, Max_V2, Min_V2, Standard_Deviation
import labels_interpolation

test_person = 5
bin_size = 1
bin_val = int(1905/bin_size)

print(f'drinking_HealthySubject{test_person}_Test')
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

all_labels = labels_interpolation.expanded_matrices

subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# subjects.remove(f'drinking_HealthySubject{test_person}_Test')
subjects_train = subjects
subjects_test = [f'drinking_HealthySubject{test_person}_Test']

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

# labels_train = []
# ###### for-loops to make annotation list for random forest method ###########################################################################
# # Iterate over each item in Y_train_labels
# for item in Y_train_labels:
#     # Iterate over the item, taking every 5th element
#     for i in range(0, len(item), bin_size):
#         labels_train.append(item[i][1])  # Append the 2nd element of every 5th sublist
# print("labels train", labels_train, len(labels_train))

# # Dictionary to map labels to numerical values
# label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}

# # Convert labels to numerical values
# y_train = [label_mapping[label] for label in labels_train]
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

# Combine data for all subjects into a single array
combined_X_data_train = np.concatenate(list(X_data_patients_dict.values()))
FullCombinedData = combined_X_data_train


################## Setting up the feature matrix ###################
feature_dict = {'drinking_HealthySubject2_Test': [],'drinking_HealthySubject3_Test': [], 'drinking_HealthySubject4_Test': [],'drinking_HealthySubject5_Test': [],
                  
                        'drinking_HealthySubject6_Test': [],'drinking_HealthySubject7_Test': []}
for patient in X_data_patients_dict:
    #Calls the array for one subject and splits it in equal parts of five
    X_data_patients_dict[patient] = np.array_split(X_data_patients_dict[patient], bin_val)

    for split in X_data_patients_dict[patient]:

        #Setting up features that loop through the columns: mean_x_acc,mean_y_acc....,Mean_x_rot. For all featured and 5 imu sensors so
        #a row of 5*5*6 = 150 features 
        Mean = np.mean(split, axis=0)
        STD = np.std(split, axis=0)
        RMS = np.sqrt(np.mean(split**2, axis=0))  # RMS value of each column
        MIN = np.min(split, axis=0)
        MAX = np.max(split, axis=0)
        #appends all features in a dictionary for each patient 
        feature_dict[patient].append(np.hstack((Mean,STD,RMS,MIN,MAX)))

# feature_matrix['drinking_HealthySubject2_Test'] = np.array(feature_matrix['drinking_HealthySubject2_Test'])
# print(feature_matrix['drinking_HealthySubject2_Test'],feature_matrix['drinking_HealthySubject2_Test'].shape)
# #print(X_data_patients_dict['drinking_HealthySubject2_Test'],X_data_patients_dict['drinking_HealthySubject2_Test'].shape

# Combine all feature arrays into a single array
compressed_array_train = np.concatenate(list(feature_dict.values()), axis=0)

print(compressed_array_train.shape,(np.array(y_train)).shape)

X_train, X_val, y_train, y_val = train_test_split(compressed_array_train, y_train, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = rf_model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(accuracy)
report = classification_report(y_val, y_pred)
print(report)

# Plot visualizations

# Confusion Matrix Plot
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance Plot
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()

# Classification Report Metrics Plot
report_dict = classification_report(y_val, y_pred, output_dict=True)

metrics_df = pd.DataFrame(report_dict).transpose()
metrics_df = metrics_df.iloc[:-3, :3]  # Exclude the last 3 summary rows

metrics_df.plot(kind='bar', figsize=(12, 6))
plt.title('Classification Report Metrics')
plt.xlabel('Classes')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.show()

# Decision Trees Plot
plt.figure(figsize=(20, 10))
plot_tree(rf_model.estimators_[0], feature_names=[f'Feature {i}' for i in range(X_train.shape[1])], filled=True, max_depth=3)
plt.title("Decision Tree from Random Forest")
plt.show()