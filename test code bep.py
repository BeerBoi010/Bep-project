import numpy as np
import matplotlib.pyplot as plt

# Load the .npy files
<<<<<<< HEAD
# acc = np.load(r"C:\Users\Mieke\Downloads\ACC_signal.npy", allow_pickle=True).item()
# rot = np.load(r"C:\Users\Mieke\Downloads\Gyro_signal.npy", allow_pickle=True).item()

acc = np.load(r"C:\Users\Eigenaar\Downloads\ACC_signal.npy", allow_pickle=True).item()
rot = np.load(r"C:\Users\Eigenaar\Downloads\Gyro_signal.npy", allow_pickle=True).item()
=======
acc = np.load(r"C:\Users\ninad\OneDrive\Uni\Jaar3 Q1,2,3,4\BEP\Data\ACC_signal.npy", allow_pickle=True).item()
rot = np.load(r"C:\Users\ninad\OneDrive\Uni\Jaar3 Q1,2,3,4\BEP\Data\Gyro_signal.npy", allow_pickle=True).item()
>>>>>>> a01f48dfdc9c3efcad8a3f6d22c8aae701c5bed2

# Define IMU locations
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

# Iterate over each subject and IMU location
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

for subject in subjects:
    for imu_location in imu_locations:
        # Extract acceleration data for the current subject and IMU location
        acc_data = acc[subject][imu_location]
        # Extract rotation data for the current subject and IMU location
        rot_data = rot[subject][imu_location]

        # Extract X, Y, and Z acceleration
        x_acceleration = acc_data[:, 0]
        y_acceleration = acc_data[:, 1]
        z_acceleration = acc_data[:, 2]

        # Extract X, Y, and Z rotation
        x_rotation = rot_data[:, 0]
        y_rotation = rot_data[:, 1]
        z_rotation = rot_data[:, 2]

        # Plot acceleration data
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(x_acceleration, label='X Acceleration')
        plt.plot(y_acceleration, label='Y Acceleration')
        plt.plot(z_acceleration, label='Z Acceleration')
        plt.title(f'Acceleration Data for {subject} - {imu_location}')
        plt.xlabel('Time')
        plt.ylabel('Acceleration')
        plt.legend()
        plt.grid(True)

        # Plot rotation data
        plt.subplot(1, 2, 2)
        plt.plot(x_rotation, label='X Rotation')
        plt.plot(y_rotation, label='Y Rotation')
        plt.plot(z_rotation, label='Z Rotation')
        plt.title(f'Rotation Data for {subject} - {imu_location}')
        plt.xlabel('Time')
        plt.ylabel('Rotation')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
