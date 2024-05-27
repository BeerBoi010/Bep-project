import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import labels_interpolation

# Define parameters
train_amount = 5
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

# One-hot encode labels
y_train_oh = to_categorical(y_train)
y_test_oh = to_categorical(y_test)

# Reshape the labels to match the desired shape
y_train_oh = y_train_oh.reshape((train_amount, -1, y_train_oh.shape[-1]))
y_test_oh = y_test_oh.reshape((1, -1, y_test_oh.shape[-1]))

# Function to combine raw data
def prepare_raw_data(subjects, acc, rot):
    data = []
    for subject in subjects:
        subject_data = []
        for imu_location in acc[subject]:
            acc_data = acc[subject][imu_location]
            rot_data = rot[subject][imu_location]
            imu_data = np.hstack((acc_data, rot_data))
            subject_data.append(imu_data)
        # Stack all IMU sensor data horizontally
        subject_data = np.hstack(subject_data)
        data.append(subject_data)
    return np.array(data)

# Prepare the training and test data
X_train_raw = prepare_raw_data(subjects_train, acc, rot)
print(X_train_raw.shape)
X_test_raw = prepare_raw_data(subjects_test, acc, rot)

# Concatenate X_train_raw for multiple subjects
X_train_raw = np.concatenate(X_train_raw, axis=0)

# Reshape data for 1D CNN input (samples, timesteps, features)
X_train_raw = X_train_raw.reshape((-1, X_train_raw.shape[1], X_train_raw.shape[2]))
X_test_raw = X_test_raw.reshape((-1, X_test_raw.shape[1], X_test_raw.shape[2]))

# Define the CNN model with 1D convolutions
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

# Input shape
input_shape = (X_train_raw.shape[1], X_train_raw.shape[2])
num_classes = y_train_oh.shape[-1]

# Create and compile the model
model = create_cnn_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train_raw, y_train_oh, epochs=20, batch_size=32)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test_raw, y_test_oh)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Predict on test data
y_test_pred = model.predict(X_test_raw)
y_test_pred_classes = np.argmax(y_test_pred, axis=2)
y_test_true_classes = np.argmax(y_test_oh, axis=2)

# Classification report
print("Classification Report of test data:")
print(classification_report(y_test_true_classes.ravel(), y_test_pred_classes.ravel(), zero_division=1))

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test_true_classes.ravel(), y_test_pred_classes.ravel())
label_mapping_inv = {v: k for k, v in label_mapping.items()}

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[label_mapping_inv[key] for key in range(num_classes)],
            yticklabels=[label_mapping_inv[key] for key in range(num_classes)])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix for {test_person}')
plt.show()

# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
