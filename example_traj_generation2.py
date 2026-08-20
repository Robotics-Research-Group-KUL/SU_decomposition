# Import necessary libraries
import sys, scipy, numpy, src.data_handling, src.robotics, src.SU_decomp, src.plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
from scipy.spatial.transform import Rotation as R__

path_to_data = 'Data'
path_to_figures = 'figures'

############ Load and preprocess the trajectory data ##########

############ Input ##########
input_trajectory = 'pouring' 
# options: 'helical_translation', 'axis_rotation', 'precession', 'pouring', 'contour_following',
#          'peg_on_hole_alignment'
progress_domain = 'geometric'
# options: 'time', 'geometric'
L = 0.3

############ Load and preprocess the trajectory and object data ##########
path_to_data = 'Data'
path_to_figures = 'figures'

# Load the trajectory data
T_raw, N, dt, time_total = src.data_handling.load_demo_trajectory_motion(input_trajectory,path_to_data)

if progress_domain == 'time':
    # Subsample raw trajectory data
    T, ds = T_raw[:,:,0:N:3], 3*dt
    N = T.shape[2]
elif progress_domain == 'geometric':
    # Interpolate pose data to equidistant geometric progress steps
    s = src.robotics.calculate_geom_progress_axis(T_raw, dt, L=L)
    ds = 0.02 # -> 2 cm
    N = src.data_handling.calculate_number_of_equidistant_steps_in_array(s, stepsize = ds)
    s_equidistant = src.data_handling.make_array_equidistant(s, N)
    T = src.robotics.interpT(s, T_raw, s_equidistant)

# Load the data of the rigid body
if input_trajectory == 'pouring':
    object_data = src.data_handling.load_data_kettle(path_to_data)
    T_kettle_wrt_tracker = src.data_handling.load_tracker_kettle_calibration_data()
    nb_vertices = object_data['vertices'].shape[0]
    hom_vertices = numpy.column_stack([object_data['vertices'],numpy.ones(nb_vertices)])
    calibrated_vertices = T_kettle_wrt_tracker @ hom_vertices.T
    object_data['vertices'] = calibrated_vertices[:3,:].T
else:
    object_data = src.data_handling.create_cube_data()

# Plot the original rigid-body trajectory
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
key_values_body_frame, key_values_rigid_object = [0,-1], [0,-1]
ax = src.plotting.plot_trajectory_origin(ax, T, color = 'b', linewidth = 3.)
ax = src.plotting.plot_frames(ax, T, key_values_body_frame , color = 'b', linewidth = 3., arrow_len = 0.08)
ax = src.plotting.plot_rigid_bodies(ax, T, key_values_rigid_object, object_data)
ax = src.plotting.ax_settings_general(ax)
if input_trajectory == 'pouring':
    ax = src.plotting.ax_settings_pouring_trajectory(ax)
fig.savefig(rf"{path_to_figures}/input_trajectory.svg")

############ Calculate the SU decomposition ########## 

# Initialise the results
Xi = numpy.zeros((6,3,N-3))
U = numpy.zeros((6,3,N-3))

# Calculate body twist trajectory
bodytwist = src.robotics.calculate_bodytwist_from_poses(T,ds)
dtwist_body = src.robotics.calculate_dtwist_from_poses(T)

dtwist_mf = numpy.zeros((6,N-3))

# Perform the successive SU decompositions along the trajectory
for k in range(N-3): 

    # Restructure twist data into successive overlapping windows of size (6,3)
    Xi_ = numpy.column_stack([bodytwist[:,k], bodytwist[:,k+1], bodytwist[:,k+2]])

    # Compute U matrix with regularization
    U_, R, p = src.SU_decomp.SU(Xi_, L = L)

    # Express dtwist in invariant frame
    R_T = R.T
    dtwist_mf[:3,k] = R_T @ dtwist_body[:3,k+2]
    dtwist_mf[3:6,k] = R_T @ dtwist_body[3:6,k+2] + R_T @ (numpy.cross(dtwist_body[:3,k+2],p))

    # Store the results
    Xi[:,:,k] = Xi_
    U[:,:,k] = U_


############ Plot the results ########## 
fig, axes = src.plotting.initialize_plot_U(progress_domain, input_trajectory)
axes = src.plotting.plot_U(axes, U, time_total, color = 'b', linewidth = 2.0)
fig.savefig(rf"{path_to_figures}/U_reg.svg")

##################### Trajectory generalization ####################################################
T_target = numpy.array([[0., 0., -1., -0.5],[1., 0., 0., 2.2],[0., -1., 0., -2.], [0.,  0.,  0.,  1.]])

nb_targets = 1
generated_trajectories = numpy.zeros((nb_targets,4,4,N))
T_gen_all = numpy.zeros(())
for Q in range(nb_targets):
    
    T_target[0,3] += 0.25

    # Reconstruction by integration
    weights = numpy.ones(N-3)
    weights_min = numpy.ones(N-3)
    stepsize = 1.
    shape_diff_min = numpy.ones(N-3)*10*10
    average_shape_diff_min = 10**10
    average_shape_diff_prev = 10**10

    for test in range(100):

        # Initialization
        T_rec = numpy.zeros((4,4,N))
        T_rec[:,:,0:3] = T[:,:,0:3]

        twist_rec = numpy.zeros((6,N-1))
        twist_rec[:,0] = numpy.squeeze(src.robotics.calculate_bodytwist_from_poses(T_rec[:,:,0:2],ds))
        twist_rec[:,1] = numpy.squeeze(src.robotics.calculate_bodytwist_from_poses(T_rec[:,:,1:3],ds))

        Xi_rec = numpy.zeros((6,3,N-3))
        Xi_rec[:,0,0] = twist_rec[:,0]
        Xi_rec[:,1,0] = twist_rec[:,1]

        twist_matrix = numpy.zeros((4,4))
        twist_ = numpy.zeros(6)
        dtwist_ = numpy.zeros(6)

        shape_diff = numpy.zeros(N-3)
        for k in range(N-3):

            # Reconstruct moving frame
            _, R, p = src.SU_decomp.SU(Xi_rec[:,:,k], L = L)

            # Express dtwist in body frame
            dtwist_[0:3] = R @ dtwist_mf[:3,k]
            dtwist_[3:6] = R @ dtwist_mf[3:6,k] - numpy.cross(dtwist_[0:3], p)

            # Calculate endpose (assuming open loop reconstruction)
            twist_matrix[0:3,0:3] = src.robotics.skew(dtwist_[0:3])
            twist_matrix[0:3,3] = dtwist_[3:6]
            T_end_= T_rec[:,:,2+k] @ scipy.linalg.expm(twist_matrix)

            # Calculate error on target pose expressed in world frame
            T_error_world = T_target @ src.robotics.inverse_T(T_end_)

            # Transform the pose error to a finite twist
            dtwist_matrix_error_world = src.robotics.logm_pose(T_error_world)

            # Compute correction to achieve target
            dtwist_matrix_correction_world = weights[k]*dtwist_matrix_error_world/(N-3-k)

            # Reconstruct local twist from DUTIR
            twist_[0:3] = R @ U[0:3,2,k]
            twist_[3:6] = R @ U[3:6,2,k] - numpy.cross(twist_[0:3], p)
            twist_rec[:,2+k] = twist_
            Xi_rec[:,2,k] = twist_
            twist_matrix[0:3,0:3] = src.robotics.skew(twist_[0:3])
            twist_matrix[0:3,3] = twist_[3:6]
            
            # Integrate twists to pose
            T_rec[:,:,3+k] = scipy.linalg.expm(dtwist_matrix_correction_world) @ T_rec[:,:,3+k-1] @ scipy.linalg.expm(twist_matrix*ds)

            # Calculate trajectory shape after correction
            corrected_bodytwist = src.robotics.calculate_bodytwist_from_poses(T_rec[:,:,2+k:4+k],ds)
            Xi_corrected = numpy.copy(Xi_rec[:,:,k])
            Xi_corrected[:,2] = corrected_bodytwist[:,0]
            U_corrected, _, _ = src.SU_decomp.SU(Xi_corrected, L = L)

            # Xi_rec[:,:,k] = Xi_corrected*1.

            shape_diff[k] = numpy.sqrt(L*2*numpy.sum((U_corrected[:3,:]-U[:3,:,k])**2) + numpy.sum((U_corrected[3:6,:]-U[3:6,:,k])**2))/3.

            if k < N-4:
                # Update twist window for reconstruction from DUTIR
                Xi_rec[:,0,k+1] = Xi_rec[:,1,k]
                Xi_rec[:,1,k+1] = Xi_rec[:,2,k]
                
        generated_trajectories[Q,:,:,:] = T_rec
        error_target = numpy.sum((T_rec[:,:,-1]-T_target)**2)
        # print(error_target)

        average_shape_diff = numpy.mean(shape_diff)
        print(average_shape_diff)
        # print(average_shape_diff < average_shape_diff_prev)

        if (average_shape_diff < average_shape_diff_min):
            # Update weights
            average_shape_diff_min = average_shape_diff*1.
            shape_diff_min = shape_diff*1.
            weights_min = weights*1.
            for k in range(N-3):
                direction = ((average_shape_diff - shape_diff[k])/average_shape_diff)
                weights[k] += 2*direction*stepsize*1.
            weights = weights/numpy.sum(weights)*(N-3)
        else:
            stepsize = stepsize/2.
            for k in range(N-3):
                direction = ((average_shape_diff_min - shape_diff_min[k])/average_shape_diff_min)
                weights[k] = weights_min[k] + direction*stepsize*1.
            weights = weights/numpy.sum(weights)*(N-3)

        if stepsize < 10**(-5):
            break


# Plot the reconstructed rigid-body trajectory
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
key_values_body_frame, key_values_rigid_object = [0,-1], [0,-1]
for Q in range(nb_targets):
    T_gen = generated_trajectories[Q,:,:,:]
    ax = src.plotting.plot_trajectory_origin(ax, T_gen, color = 'r', linewidth = 3.)
    ax = src.plotting.plot_frames(ax, T_gen, key_values_body_frame , color = 'r', linewidth = 3., arrow_len = 0.08)
    ax = src.plotting.plot_rigid_bodies(ax, T_gen, key_values_rigid_object, object_data)
    ax = src.plotting.ax_settings_general(ax)
if input_trajectory == 'pouring':
    ax = src.plotting.ax_settings_pouring_trajectory(ax)
fig.savefig(rf"{path_to_figures}/generated_trajectories.svg")
