import numpy as np
import matplotlib.pyplot as plt


annotation2 = np.load("Data_tests/Annotated times/time_ranges_subject_2.npy", allow_pickle=True)

# Define the sampling frequency
sampling_frequency = 50  # Hz

# Initialize variables to keep track of the cumulative count
cumulative_count = 0

# Function to perform interpolation for a single row
def interpolate_row(row):
    global cumulative_count  # Use global variable for cumulative count
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
    return expanded_rows

# Initialize list to store expanded rows
expanded_matrix = []

# Iterate over each row and perform interpolation
for row in annotation2:
    expanded_rows = interpolate_row(row)
    expanded_matrix.extend(expanded_rows)

#print("Expanded rows:")
# for row in expanded_matrix:
#     print(row)

#print(expanded_matrix)

# Extract x-values (time) and y-values (labels) from expanded_matrix
x_values = [row[0] for row in expanded_matrix]
y_values = [row[1] for row in expanded_matrix]

# Plot the data
plt.plot(x_values, y_values, marker='o', linestyle='-')
plt.xlabel('Time')
plt.ylabel('Label')
plt.title('Interpolated Data')
plt.grid(True)
plt.show()