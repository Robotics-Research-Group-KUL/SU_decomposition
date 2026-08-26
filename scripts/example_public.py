import numpy as np
import scipy

from sulib import su_decomp

# Generate trajectory data of a helical translation
N = 30  # number of samples
time_total = 2  # seconds
time_axis = np.linspace(0, time_total, N)
dt = time_total / (N - 1)  # [s]

r = 0.2  # radius of the circular trajectory
p_x = np.array([r * np.cos(np.pi * time_axis / time_total)])
p_y = np.array([r * np.sin(np.pi * time_axis / time_total)])
p_z = np.array([r * time_axis / time_total])
p = np.vstack([p_x, p_y, p_z])

T = np.zeros((4, 4, N))
for k in range(N):
    T[0:3, 3, k] = p[:, k]
    T[0:3, 0:3, k] = np.eye(3)
    T[3, 3, k] = 1

# Calculate the DUTIR of this pose trajectory
dutir, _ = su_decomp.pose_trajectory_to_dutir(T, dt, L=0.3, twist_type="body")

print(" ")
print("Single DUTIR sample for a helical translation")
print(np.round(dutir[:, :, 10], 5))


# Generate trajectory data of a rotation about a fixed axis

N = 60
time_total = 4  # seconds
time_axis = np.linspace(0, time_total, N)
dt = time_total / (N - 1)  # [s]

T = np.zeros((4, 4, N))
for k in range(N):
    ROT = scipy.spatial.transform.Rotation.from_euler("z", 270 * time_axis[k] / time_total, degrees=True).as_matrix()
    T[0:3, 0:3, k] = ROT
    T[3, 3, k] = 1

    # displace body frame origin away from the zero vector
    T_disp = np.eye(4)
    T_disp[0, 3] = 0.2
    T[:, :, k] = T[:, :, k] @ T_disp


# Calculate the DUTIR of this pose trajectory
dutir, _ = su_decomp.pose_trajectory_to_dutir(T, dt, L=0.3, twist_type="body")

print(" ")
print("Single DUTIR sample for fixed axis rotation")
print(np.round(dutir[:, :, 10], 5))


# Generate trajectory data of a precession motion

T = np.zeros((4, 4, N))
for k in range(N):
    # First rotation
    ROT1 = scipy.spatial.transform.Rotation.from_euler("z", 270 * time_axis[k] / time_total, degrees=True).as_matrix()
    T[0:3, 0:3, k] = ROT1
    T[3, 3, k] = 1

    # Displace body frame origin away from the zero vector
    T_disp = np.eye(4)
    T_disp[0, 3] = 0.2
    T[:, :, k] = T[:, :, k] @ T_disp

    # Second rotation -> precession
    ROT2 = scipy.spatial.transform.Rotation.from_euler("x", 270 * time_axis[k] / time_total, degrees=True).as_matrix()
    T[0:3, 0:3, k] = T[0:3, 0:3, k] @ ROT2


# Calculate the DUTIR of this pose trajectory
dutir, _ = su_decomp.pose_trajectory_to_dutir(T, dt, L=0.3, twist_type="body")

print(" ")
print("Single DUTIR sample for a precession")
print(np.round(dutir[:, :, 10], 5))
