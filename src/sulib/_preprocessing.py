import numpy as np
from scipy.ndimage import gaussian_filter1d

import sulib._robotics as rob


def regulate_matrix(A):
    """
    Add regularization for numerical stability
    """
    tol = 10.0 ** (-12)
    inc = 10.0 ** (-12)
    if np.linalg.norm(A[:, 0]) < tol:
        A[0, 0] = add_increment(A[0, 0], inc)
    if np.linalg.norm(np.cross(A[:, 0], A[:, 1])) / np.linalg.norm(A[:, 0]) < tol:
        A[1, 1] = add_increment(A[1, 1], inc)
    if np.linalg.norm(np.cross(A[:, 0], A[:, 1])) / np.linalg.norm(A[:, 0]) < tol:
        A[0, 1] = add_increment(A[0, 1], inc)

    return A


def add_increment(a, inc):
    """
    add a signed increment to the element
    """
    if np.abs(a + inc) > np.abs(a - inc):
        a += inc
    else:
        a -= inc

    return a


def remove_offset_array(array):
    array -= array[0]
    return array


def calculate_number_of_equidistant_steps_in_array(array, stepsize=0.02):
    array_without_offset = remove_offset_array(array)
    N = int(1 + np.floor(array_without_offset[-1] / stepsize))
    return N


def make_array_equidistant(array, N):
    array_equidistant = np.linspace(array[0], array[-1], N)
    return array_equidistant


def preprocess_time_axis(t, stepsize):
    t0 = remove_offset_array(t)
    N = calculate_number_of_equidistant_steps_in_array(t0, stepsize=stepsize)
    t0_equidistant = make_array_equidistant(t0, N)
    return t0_equidistant, t0


def preprocess_pose_data(T, t, dt):

    t0_equi, t0 = preprocess_time_axis(t, stepsize=dt)

    # Interpolate pose data to equidistant stepsize
    T = rob.interpT(t0, T, t0_equi)

    return T


def preprocess_wrench_data(wrench, t, dt):

    t0_equi, t0 = preprocess_time_axis(t, stepsize=dt)

    # Interpolate wrench data to equidistant timesteps
    wrench = np.vstack([np.interp(t0_equi, t0, wrench[i, :]) for i in range(wrench.shape[0])])

    return wrench


def filter_pose_data(T, sigma=10):

    for j in range(4):
        for k in range(3):
            T[k, j, :] = gaussian_filter1d(T[k, j, :], sigma=sigma, mode="nearest")

    N = T.shape[2]
    for j in range(N):
        R = T[:3, :3, j]
        U, _, VT = np.linalg.svd(R)
        R = U @ VT
        T[:3, :3, j] = np.copy(R)

    return T


def calculate_geom_progress_axis(T, dt, L):
    """
    This function calculates the screwbased geometric progress axis for rigid
    body trajectories from input pose data. The twist components are
    estimated from the pose data using a finite differences scheme.
    INPUT : T (4x4xN)      -> Input pose trajectory
          : dt             -> Corresponding time step [s]
          : L              -> value for the characteristic length
    OUTPUT: s (1xN)   -    -> calculated geometric progress axis
    """

    N = T.shape[2]
    s = np.zeros(N)
    bodytwists = rob.poses_to_bodytwists(T, dt)

    for k in range(N - 1):
        omega = bodytwists[0:3, k]
        vel = bodytwists[3:6, k]
        if np.linalg.norm(omega) == 0:
            v1 = np.linalg.norm(vel)
        else:
            p_perp = np.cross(omega, vel) / np.dot(omega, omega)
            if np.linalg.norm(p_perp) > L:
                p_regularized = L * p_perp / np.linalg.norm(p_perp)
                vel_regularized = vel + np.cross(omega, p_regularized)
                v1 = np.linalg.norm(vel_regularized)
            else:
                v1 = np.dot(vel, omega) / np.linalg.norm(omega)

        s_dot = np.sqrt(L**2 * np.dot(omega, omega) + v1**2)
        s[k + 1] = s[k] + s_dot * dt

    return s
