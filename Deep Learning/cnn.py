import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load annotation data
annotation2 = np.load("Data_tests/Annotated times/time_ranges_subject_2.npy", allow_pickle=True)
annotation3 = np.load("Data_tests/Annotated times/time_ranges_subject_3.npy", allow_pickle=True)
annotation4 = np.load("Data_tests/Annotated times/time_ranges_subject_4.npy", allow_pickle=True)
annotation5 = np.load("Data_tests/Annotated times/time_ranges_subject_5.npy", allow_pickle=True)
annotation7 = np.load("Data_tests/Annotated times/time_ranges_subject_7.npy", allow_pickle=True)
annotation6 = np.array([
    ['0', '1.4', 'N'],
    ['1.4', '2.6', 'A'],
    ['2.6', '4.8', 'B'],
    ['4.8', '6', 'C'],
    ['6', '7.5', 'B'],
    ['7.5', '8.8', 'A'],
    ['8.8', '11.3', 'N'],
    ['11.3', '12.6', 'A'],
    ['12.6', '13.6', 'B'],
    ['13.6', '14.3', 'C'],
    ['14.3', '15.7', 'B'],
    ['15.7', '17.6', 'A'],
    ['17.6', '18.4', 'N'],
    ['18.4', '19.2', 'A'],
    ['19.2', '20.3', 'B'],
    ['20.3', '21.3', 'C'],
    ['21.3', '22.5', 'B'],
    ['22.5', '23.5', 'A'],
    ['23.5', '24.5', 'N'],
    ['24.5', '25.4', 'A'],
    ['25.4', '26.7', 'B'],
    ['26.7', '27.7', 'C'],
    ['27.7', '29', 'B'],
    ['29', '30', 'A'],
    ['30', '31.4', 'N'],
    ['31.4', '32.1', 'A'],
    ['32.1', '33.3', 'B'],
    ['33.3', '34.4', 'C'],
    ['34.4', '35.8', 'B'],
    ['35.8', '37', 'A'],
    ['37', '38.1', 'N']
])


annotation_matrices = [annotation2, annotation3, annotation4, annotation5, annotation6, annotation7]

# Define the sampling frequency
sampling_frequency = 50  # Hz

# Function to perform interpolation for a single row
def interpolate_row(row, cumulative_count):
    start_time = float(row[0])
    end_time = float(row[1])
    label = row[2]
    num_samples = round((end_time - start_time) * sampling_frequency)
    expanded_rows = [[cumulative_count + i, label] for i in range(num_samples)]
    cumulative_count += num_samples
    return expanded_rows, cumulative_count

# Initialize list to store expanded rows for all participants
expanded_matrices = []

# Iterate over each participant's annotation matrix
for annotation_matrix in annotation_matrices:
    cumulative_count = 0
    expanded_matrix = []
    for row in annotation_matrix:
        expanded_rows, cumulative_count = interpolate_row(row, cumulative_count)
        expanded_matrix.extend(expanded_rows)
    expanded_matrices.append(expanded_matrix)

exp_annotations = expanded_matrices

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