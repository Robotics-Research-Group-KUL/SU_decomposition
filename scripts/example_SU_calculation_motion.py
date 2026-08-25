# Import necessary libraries
import matplotlib.pyplot as plt
import numpy
import scipy

import sulib._data_handling as dh
import sulib._plotting as plotting
import sulib._preprocessing as pp
import sulib.su_decomp as su_decomp

############ Input ##########
input_trajectory = "pouring"
# options: 'helical_translation', 'axis_rotation', 'precession', 'pouring', 'contour_following',
#          'peg_on_hole_alignment'
progress_domain = "geometric"
# options: 'time', 'geometric'

############ Load and preprocess the trajectory and object data ##########
path_to_data = "data"
path_to_figures = "figures"

# Load the trajectory data
T_raw, N, dt = dh.load_demo_trajectory_motion(input_trajectory, path_to_data)

if progress_domain == "time":
    # Subsample raw trajectory data
    T, ds = T_raw[:, :, 0:N:3], 3 * dt
    N = T.shape[2]
elif progress_domain == "geometric":
    # Interpolate pose data to equidistant geometric progress steps
    s = pp.calculate_geom_progress_axis(T_raw, dt, L=0.3)
    ds = 0.02  # -> 2 cm
    T = pp.preprocess_pose_data(T_raw, s, ds)
    N = T.shape[2]

progress_total = (N - 1) * ds

# Load the data of the rigid body
if input_trajectory == "pouring":
    object_data = dh.load_data_kettle(path_to_data)
    T_kettle_wrt_tracker = dh.load_tracker_kettle_calibration_data()
    nb_vertices = object_data["vertices"].shape[0]
    hom_vertices = numpy.column_stack([object_data["vertices"], numpy.ones(nb_vertices)])
    calibrated_vertices = T_kettle_wrt_tracker @ hom_vertices.T
    object_data["vertices"] = calibrated_vertices[:3, :].T
else:
    object_data = dh.create_cube_data()

# Plot the original rigid-body trajectory
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection="3d")
key_values_body_frame, key_values_rigid_object = [0, -1], [0, -1]
ax = plotting.plot_trajectory_origin(ax, T, color="b", linewidth=3.0)
ax = plotting.plot_frames(ax, T, key_values_body_frame, color="b", linewidth=3.0, arrow_len=0.08)
ax = plotting.plot_rigid_bodies(ax, T, key_values_rigid_object, object_data)
ax = plotting.ax_settings_general(ax)
if input_trajectory == "pouring":
    ax = plotting.ax_settings_pouring_trajectory(ax)
fig.savefig(rf"{path_to_figures}/input_trajectory.svg")

############ Introduce variations in coordinate frame ##########
nb_body_frame_transformations = 2

# Initialize the transformation matrices of the body frame
body_frame_transformations = [numpy.eye(4) for j in range(nb_body_frame_transformations)]

# Define body frame 1
body_frame_transformations[0][:3, 3] = numpy.array([0.1, -0.13, 0.04])

# Define body frame 2
body_frame_transformations[1][:3, 3] = numpy.array([0.1, 0.08, -0.04])
rot2 = scipy.spatial.transform.Rotation.from_euler("xzx", [120, 70, 0], degrees=True).as_matrix()
body_frame_transformations[1][:3, :3] = rot2

# Initialise the resulting trajectories
T_var = [numpy.zeros(T.shape) for j in range(nb_body_frame_transformations)]

# Apply the body frame transformations
for j in range(nb_body_frame_transformations):
    for k in range(N):
        T_var[j][:, :, k] = T[:, :, k] @ body_frame_transformations[j]

# Plot the rigid-body trajectories with new body frames
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection="3d")
ax = plotting.plot_rigid_bodies(ax, T, key_values_rigid_object, object_data)
key_values_body_frame, key_values_rigid_object = [0, -1], [0, -1]
colors = ["r", "b"]
for j in range(nb_body_frame_transformations):
    ax = plotting.plot_trajectory_origin(ax, T_var[j], color=colors[j], linewidth=3.0)
    ax = plotting.plot_frames(
        ax,
        T_var[j],
        key_values_body_frame,
        color=colors[j],
        linewidth=3.0,
        arrow_len=0.08,
    )
ax = plotting.ax_settings_general(ax)
if input_trajectory == "pouring":
    ax = plotting.ax_settings_pouring_trajectory(ax)
fig.savefig(rf"{path_to_figures}/trajectories_with_different_body_frames.svg")

############ Calculate and plot the twist trajectory and dutir representation ##########

fig_twist, axes_twist = plotting.initialize_plot_twist_trajectory(progress_domain, input_trajectory)
fig_dutir, axes_dutir = plotting.initialize_plot_dutir(progress_domain, input_trajectory)
fig_dutir_reg, axes_dutir_reg = plotting.initialize_plot_dutir(progress_domain, input_trajectory)

linewidths = [3.0, 1.5]
for j in range(nb_body_frame_transformations):
    dutir, twist = su_decomp.pose_trajectory_to_dutir(T_var[j], ds, twist_type="body")
    plotting.plot_twist_trajectory(axes_twist, twist, progress_total, color=colors[j])
    plotting.plot_dutir(axes_dutir, dutir, progress_total, color=colors[j], linewidth=linewidths[j])

    dutir_reg, _ = su_decomp.pose_trajectory_to_dutir(T_var[j], ds, L=0.3, twist_type="body")
    plotting.plot_dutir(axes_dutir_reg, dutir_reg, progress_total, color=colors[j], linewidth=linewidths[j])

fig_twist.savefig(rf"{path_to_figures}/twists.svg")
fig_dutir.savefig(rf"{path_to_figures}/dutir.svg")
fig_dutir_reg.savefig(rf"{path_to_figures}/dutir_reg.svg")
