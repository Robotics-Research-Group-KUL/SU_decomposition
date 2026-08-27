import numpy as np

import sulib

# Generate trajectory data of a helical translation
T, dt = sulib.su_decomp.generate_synthetic_pose_trajectory(trajectory_type="translation_3D")

# Calculate the DUTIR of this pose trajectory
dutir, _ = sulib.su_decomp.pose_trajectory_to_dutir(T, dt, L=0.3, twist_type="body")

print(" ")
print("Single DUTIR sample for a helical translation")
print(np.round(dutir[:, :, 10], 5))


# Generate trajectory data of a rotation about a fixed axis
T, dt = sulib.su_decomp.generate_synthetic_pose_trajectory(trajectory_type="rotation_1D")

# Calculate the DUTIR of this pose trajectory
dutir, _ = sulib.su_decomp.pose_trajectory_to_dutir(T, dt, L=0.3, twist_type="body")

print(" ")
print("Single DUTIR sample for fixed axis rotation")
print(np.round(dutir[:, :, 10], 5))


# Generate trajectory data of a precession motion
T, dt = sulib.su_decomp.generate_synthetic_pose_trajectory(trajectory_type="rotation_3D")

# Calculate the DUTIR of this pose trajectory
dutir, _ = sulib.su_decomp.pose_trajectory_to_dutir(T, dt, L=0.3, twist_type="body")

print(" ")
print("Single DUTIR sample for a precession")
print(np.round(dutir[:, :, 10], 5))
