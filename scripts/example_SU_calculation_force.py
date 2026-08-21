# Import necessary libraries
import numpy
import scipy

import sulib.data_handling as dh
import sulib.plotting as plotting
import sulib.robotics as rob
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
    s = rob.calculate_geom_progress_axis(T_raw, dt, L=0.3)
    ds = 0.02  # -> 2 cm
    N = dh.calculate_number_of_equidistant_steps_in_array(s, stepsize=ds)
    s_equidistant = dh.make_array_equidistant(s, N)
    T = rob.interpT(s, T_raw, s_equidistant)
    wrench = numpy.vstack([numpy.interp(s_equidistant, s, wrench_raw[i, :]) for i in range(wrench_raw.shape[0])])

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

############ Calculate the SU decomposition ##########

# Initialise the results
Xi = [numpy.zeros((6, 3, N - 3)) for j in range(nb_body_frame_transformations)]
U = [numpy.zeros((6, 3, N - 3)) for j in range(nb_body_frame_transformations)]
U_reg = [numpy.zeros((6, 3, N - 3)) for j in range(nb_body_frame_transformations)]

for j in range(nb_body_frame_transformations):
    # Smooth the wrench trajectory
    wrench_smooth = scipy.ndimage.gaussian_filter1d(wrench_var[j], sigma=1.0, axis=1, mode="nearest")

    # Perform the successive SU decompositions along the trajectory
    for k in range(N - 3):
        # Restructure twist data into successive overlapping windows of size (6,3)
        Xi_ = numpy.column_stack([wrench_smooth[:, k], wrench_smooth[:, k + 1], wrench_smooth[:, k + 2]])

        # Compute U matrix without regularization
        U_, _, _ = su_decomp.SU(Xi_)

        # Compute U matrix with regularization
        U_reg_, _, _ = su_decomp.SU(Xi_, L=0.3)

        # Store the results
        Xi[j][:, :, k] = Xi_
        U[j][:, :, k] = U_
        U_reg[j][:, :, k] = U_reg_


############ Plot the results ##########
colors = ["r", "b"]
fig, axes = plotting.initialize_plot_wrench_trajectory(progress_domain, input_trajectory)
for j in range(nb_body_frame_transformations):
    axes = plotting.plot_twist_trajectory(axes, Xi[j][:, 0, :], progress_total, color=colors[j])
fig.savefig(rf"{path_to_figures}/wrenches.svg")

fig, axes = plotting.initialize_plot_U_wrench(progress_domain, input_trajectory)
linewidths = [3.0, 1.5]
for j in range(nb_body_frame_transformations):
    axes = plotting.plot_U(axes, U[j], progress_total, color=colors[j], linewidth=linewidths[j])
fig.savefig(rf"{path_to_figures}/U.svg")

fig, axes = plotting.initialize_plot_U(progress_domain, input_trajectory)
for j in range(nb_body_frame_transformations):
    axes = plotting.plot_U(axes, U_reg[j], progress_total, color=colors[j], linewidth=linewidths[j])
fig.savefig(rf"{path_to_figures}/U_reg.svg")
