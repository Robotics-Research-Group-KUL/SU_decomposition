import matplotlib.pyplot as plt
import numpy
import scipy

import src.data_handling
import src.plotting
import src.robotics
import src.su_decomp


def load_trajectory(input_trajectory,path_to_data):

    # Load the trajectory data
    T_raw, N, dt = src.data_handling.load_demo_trajectory_motion(input_trajectory,path_to_data)
    total_time = (N-1)*dt

    # Subsample raw trajectory data
    T_sub = T_raw[:,:,0:N:3]
    dt = 3*dt
    
    return T_sub, dt, total_time

def compute_SU(T,dt,L):

    # Initialise the results
    N = T.shape[2]
    U = [numpy.zeros((6,3,N-3)), numpy.zeros((6,3,N-3))]

    # Calculate body twist trajectory
    twist = src.robotics.calculate_bodytwist_from_poses(T,dt)

    # Smooth the body twist trajectory
    twist_smooth = scipy.ndimage.gaussian_filter1d(twist, sigma = 1, axis=1, mode='nearest')

    # Perform the successive SU decompositions along the trajectory
    for k in range(N-3): 

        # Restructure twist data into successive overlapping windows of size (6,3)
        Xi_ = numpy.column_stack([twist_smooth[:,k], twist_smooth[:,k+1], twist_smooth[:,k+2]])

        # Compute U matrix without regularization
        U_, _, _ = src.su_decomp.SU(Xi_)

        # Compute U matrix with regularization
        U_reg_, _, _ = src.su_decomp.SU(Xi_, L = L)

        # Store the results
        U[0][:,:,k] = U_reg_
        U[1][:,:,k] = U_

    return U

def plot_rigid_body_trajectory(T,input_trajectory,path_to_data,path_to_figures):

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

def plot_U(U,total_time,input_trajectory,path_to_figures):

    fig, axes = src.plotting.initialize_plot_U('time', input_trajectory)
    linewidths = [3.0, 1.5]
    colors = ['b','r']
    for j in range(2):
        axes = src.plotting.plot_U(axes, U[j], total_time, color = colors[j], linewidth = linewidths[j])
    fig.savefig(rf"{path_to_figures}/U.svg")



