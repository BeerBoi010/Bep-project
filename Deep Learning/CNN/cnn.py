import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Reshape
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
import math
import time
import labels_interpolation

# Load data
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()
all_labels = labels_interpolation.expanded_matrices

# Define parameters
train_amount = 5
test_person = 7
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

# Function to normalize data
def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

# Function to combine raw data
def prepare_raw_data(subjects, acc, rot):
    data = []
    for subject in subjects:
        subject_data = []
        for imu_location in acc[subject]:
            acc_data = acc[subject][imu_location]
            rot_data = rot[subject][imu_location]
            imu_data = np.hstack((acc_data, rot_data))
            imu_data = normalize_data(imu_data)
            subject_data.append(imu_data)
        subject_data = np.hstack(subject_data)
        data.append(subject_data)
    return np.array(data)

# Prepare the training and test data
X_train_raw = prepare_raw_data(subjects_train, acc, rot)
X_test_raw = prepare_raw_data(subjects_test, acc, rot)

# Define the CNN model with 1D convolutions
def create_cnn_model(input_shape, output_shape):
    model = Sequential()
    
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(output_shape[0] * output_shape[1], activation='softmax'))
    model.add(Reshape(output_shape))
    
    return model

# Input and output shapes
input_shape = (1905, 30)
output_shape = (1905, 4)

# Create and compile the model
model = create_cnn_model(input_shape, output_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Learning rate scheduler
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)
tensorboard = TensorBoard(log_dir=f"logs/{time.time()}")

# Train the model
history = model.fit(X_train_raw, y_train_oh, epochs=20, callbacks=[lrate, tensorboard])

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
            xticklabels=[label_mapping_inv[key] for key in range(output_shape[1])],
            yticklabels=[label_mapping_inv[key] for key in range(output_shape[1])])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix for {test_person}')
plt.show()

# Plot training accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()