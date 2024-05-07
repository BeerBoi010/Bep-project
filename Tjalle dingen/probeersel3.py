import numpy as np
import matplotlib.pyplot as plt

# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()


x_acceleration = acc['drinking_HealthySubject2_Test']
# print(x_acceleration)

annotation2 = np.load("Data_tests/Annotated times/time_ranges_subject_2.npy", allow_pickle=True)
annotation3 = np.load("Data_tests/Annotated times/time_ranges_subject_3.npy", allow_pickle=True)
annotation4 = np.load("Data_tests/Annotated times/time_ranges_subject_4.npy", allow_pickle=True)
annotation5 = np.load("Data_tests/Annotated times/time_ranges_subject_5.npy", allow_pickle=True)
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
annotation7 = np.load("Data_tests/Annotated times/time_ranges_subject_7.npy", allow_pickle=True)

annotation_matrices = [annotation2, annotation3, annotation4, annotation5, annotation6, annotation7]

# Define the sampling frequency
sampling_frequency = 50  # Hz

# Function to perform interpolation for a single row
def interpolate_row(row, cumulative_count):
    # Convert start and end time to floats
    start_time = float(row[0])
    end_time = float(row[1])
    # Original label
    label = row[2]
    # Calculate the number of samples
    num_samples = round((end_time - start_time) * sampling_frequency)
    # Create expanded rows with data points and label
    expanded_rows = [[cumulative_count + i + 1, label] for i in range(num_samples)]
    # Update cumulative count
    cumulative_count += num_samples
    return expanded_rows, cumulative_count

# Initialize list to store expanded rows for all participants
expanded_matrices = []

# Iterate over each participant's annotation matrix
for annotation_matrix in annotation_matrices:
    # Initialize variables to keep track of the cumulative count
    cumulative_count = 0
    # Initialize list to store expanded rows for the current participant
    expanded_matrix = []
    # Iterate over each row in the annotation matrix and perform interpolation
    for row in annotation_matrix:
        expanded_rows, cumulative_count = interpolate_row(row, cumulative_count)
        expanded_matrix.extend(expanded_rows)
    # Append the expanded matrix for the current participant to the list
    expanded_matrices.append(expanded_matrix)

# # Plot the data for each participant
for participant_index, expanded_matrix in enumerate(expanded_matrices):
    plt.figure(figsize=(10, 5))
    # Extract x-values (time) and y-values (labels) from expanded_matrix
    x_values = [row[0] for row in expanded_matrix]
    y_values = [row[1] for row in expanded_matrix]
    # Plot the data with smaller marker size
    plt.plot(x_values, y_values, marker='o', linestyle='-', markersize=2, label=f'Participant {participant_index + 1}')
    plt.xlabel('Time')
    plt.ylabel('Label')
    plt.title('Interpolated Data')
    plt.grid(True)
    plt.legend()
    plt.show()

# labels2 = expanded_matrices[0]
# # print(labels2)
# for row in labels2:
#     x_values = [row[0] for row in expanded_matrix]
#     y_values = [row[1] for row in expanded_matrix]

# plt.plot(x_values, y_values, marker='o', linestyle='-', markersize=2, label=f'Participant 1')
# plt.xlabel('Time')
# plt.ylabel('Label')
# plt.title('Interpolated Data')
# plt.grid(True)
# plt.legend()
# plt.show()

# Plotting
for sensor, data in x_acceleration.items():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(data[:, 0], label='X Acceleration', color='r')
    ax.plot(data[:, 1], label='Y Acceleration', color='g')
    ax.plot(data[:, 2], label='Z Acceleration', color='b')
    ax.set_title(f'Acceleration Data for {sensor}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Acceleration')
    ax.legend()
    ax.grid(True)  # Adding grid
    plt.show()