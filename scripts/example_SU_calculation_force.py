# Import necessary libraries
import numpy
import scipy

import sulib._data_handling as dh
import sulib._plotting as plotting
import sulib._preprocessing as pp
import sulib._robotics as rob
import sulib.su_decomp as su_decomp

############ Input ##########
input_trajectory = "peg_on_hole_alignment"
# options: 'contour_following', peg_on_hole_alignment'
progress_domain = "geometric"
# options: 'time', 'geometric'

############ Load and preprocess the trajectory and object data ##########
path_to_data = "data"
path_to_figures = "figures"

# Load the trajectory data
T_raw, wrench_raw, N, dt = dh.load_demo_trajectory_force(input_trajectory, path_to_data)

if progress_domain == "time":
    # Subsample raw trajectory data
    T, ds = T_raw[:, :, 0:N:3], 3 * dt
    wrench = wrench_raw[:, 0:N:3]
    N = T.shape[2]
elif progress_domain == "geometric":
    # Interpolate pose data to equidistant geometric progress steps
    s = pp.calculate_geom_progress_axis(T_raw, dt, L=0.3)
    ds = 0.02  # -> 2 cm
    T = pp.preprocess_pose_data(T_raw, s, ds)
    wrench = pp.preprocess_wrench_data(wrench_raw, s, ds)
    N = T.shape[2]

progress_total = (N - 1) * ds

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
wrench_var = [numpy.zeros(wrench.shape) for j in range(nb_body_frame_transformations)]

# Apply the body frame transformations
for j in range(nb_body_frame_transformations):
    for k in range(N):
        T_inv = rob.inverse_T(body_frame_transformations[j])
        wrench_var[j][:3, k] = T_inv[:3, :3] @ wrench[:3, k]
        wrench_var[j][3:6, k] = T_inv[:3, :3] @ wrench[3:6, k] + numpy.cross(T_inv[:3, 3], wrench_var[j][:3, k])

############ Plot the wrench trajectory ##########
colors = ["r", "b"]
fig_wrench, axes_wrench = plotting.initialize_plot_wrench_trajectory(progress_domain, input_trajectory)
for j in range(nb_body_frame_transformations):
    plotting.plot_twist_trajectory(axes_wrench, wrench_var[j], progress_total, color=colors[j])
fig_wrench.savefig(rf"{path_to_figures}/wrenches.svg")

############ Calculate and plot the dutir representation ##########
fig_dutir, axes_dutir = plotting.initialize_plot_dutir(progress_domain, input_trajectory)
fig_dutir_reg, axes_dutir_reg = plotting.initialize_plot_dutir(progress_domain, input_trajectory)

linewidths = [3.0, 1.5]
for j in range(nb_body_frame_transformations):
    dutir = su_decomp.compute_dutir_from_screw_traj(wrench_var[j])
    plotting.plot_dutir(axes_dutir, dutir, progress_total, color=colors[j], linewidth=linewidths[j])

    dutir_reg = su_decomp.compute_dutir_from_screw_traj(wrench_var[j], L=0.3)
    plotting.plot_dutir(axes_dutir_reg, dutir_reg, progress_total, color=colors[j], linewidth=linewidths[j])

fig_dutir.savefig(rf"{path_to_figures}/dutir.svg")
fig_dutir_reg.savefig(rf"{path_to_figures}/dutir_reg.svg")
