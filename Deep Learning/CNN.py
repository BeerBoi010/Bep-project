import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import labels_interpolation

# Load IMU data
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

# Function to get IMU data and labels for a subject
def get_subject_data(subject, annotations):
    acc_data = acc[subject]
    rot_data = rot[subject]
    combined_data = np.concatenate((acc_data, rot_data), axis=2)
    labels = np.array([label_mapping[label] for _, label in annotations])
    return combined_data, labels

# Define label mapping
label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}

# Set the test person index
test_person = 3

# Prepare subjects and labels
subjects = [f'drinking_HealthySubject{i+2}_Test' for i in range(6)]
subjects.remove(f'drinking_HealthySubject{test_person}_Test')
subjects_train = subjects
subjects_test = [f'drinking_HealthySubject{test_person}_Test']

# Get training and testing data
X_train_list, y_train_list = [], []
for i, subject in enumerate(subjects_train):
    data, labels = get_subject_data(subject, exp_annotations[i])
    X_train_list.append(data)
    y_train_list.append(labels)

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)

# Get testing data
X_test, y_test = get_subject_data(subjects_test[0], exp_annotations[test_person - 2])

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=len(label_mapping))
y_test = to_categorical(y_test, num_classes=len(label_mapping))

# Define the CNN model
model = Sequential()

# Convolutional layers
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# Flatten layer
model.add(Flatten())

# Fully connected layers
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_mapping), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# Save the model
model.save('cnn_arm_submovements_recognition_model.h5')