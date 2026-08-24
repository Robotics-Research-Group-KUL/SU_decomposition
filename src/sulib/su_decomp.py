import numpy as np

import sulib.robotics as rob


def RU(A):
    """
    Compute an SO(3)-constrained QR-like decomposition of a 3x3 matrix.
    The first two columns of A are orthonormalized using the Gram-Schmidt procedure.
    The third basis vector is constructed as the cross product of the first two,
    ensuring that the resulting matrix R has determinant +1.

    Input:
    A: np.ndarray of shape (3, 3)

    Output:
    R: np.ndarray of shape (3, 3)
    U: np.ndarray of shape (3, 3)
    """

    # Add regularization for numerical stability
    if np.linalg.norm(A[:, 0]) == 0:
        A[0, 0] += 10 ** (-15)
    if np.linalg.norm(np.cross(A[:, 0], A[:, 1])) == 0:
        A[1, 1] += 10 ** (-15)
    if np.linalg.norm(np.cross(A[:, 0], A[:, 1])) == 0:
        A[0, 1] += 10 ** (-15)

    # Gramm-Shmidt orthogonalisation
    ex = A[:, 0] / np.linalg.norm(A[:, 0])

    proj = np.dot(A[:, 1], ex) * ex
    ey = A[:, 1] - proj
    ey = ey / np.linalg.norm(ey)

    ez = np.cross(ex, ey)
    ez /= np.linalg.norm(ez)
    R = np.column_stack((ex, ey, ez))

    U = R.T @ A

    return R, U


def SU(X, L=10**10):
    """
    Compute the SU-decomposition of a 6x3 matrix.

    Input:
    X: np.ndarray of shape (6, 3)
    L: float, optional
       Regularization parameter

    Output:
    S: np.ndarray of shape (6, 6)
    U: np.ndarray of shape (6, 3)

    """

    # Calculate R
    X1 = X[0:3, :]
    X2 = X[3:6, :]
    R, U1 = RU(X1)
    RTX2 = R.T @ X2

    # Calculate p_star
    z = RTX2[1, 0] / U1[0, 0]
    y = -RTX2[2, 0] / U1[0, 0]
    x = (RTX2[2, 1] + U1[0, 1] * y) / U1[1, 1]
    p_star = np.array([x, y, z])

    # Perform regularization
    if x**2 + y**2 + z**2 > L**2:  # regularization active
        # Regularize p_star
        if y**2 + z**2 > L**2:  # Case 2
            x = 0
            y = L * y / np.sqrt(y**2 + z**2)
            z = L * z / np.sqrt(y**2 + z**2)
        else:  # Case 1
            x = np.sign(x) * np.sqrt(L**2 - y**2 - z**2)
        p_star = np.array([x, y, z])

        p = R @ p_star
        U2 = RTX2 - rob.skew(p_star) @ U1

        # Regularize R
        _, U2_tri = RU(U2)

        # input matrix = [L*U1 U2]
        # target matrix = [L*U1 U2_tri]
        C = L**2 * U1[:, 0:2] @ U1[:, 0:2].T + U2[:, 0:2] @ U2_tri[:, 0:2].T

        # Compute SVD
        U, _, Vt = np.linalg.svd(C)
        V = Vt.T
        Rc = V @ U.T

        # Ensure Rc is a proper rotation matrix (det = +1)
        if np.linalg.det(Rc) < 0:
            U[:, -1] *= -1
            Rc = V @ U.T
        U1 = Rc @ U1
        U2 = Rc @ U2
        R = R @ Rc.T
    else:
        p = R @ p_star
        U2 = RTX2 - rob.skew(p_star) @ U1

    U = np.vstack((U1, U2))
    S = np.vstack((np.hstack((R, np.zeros((3, 3)))), np.hstack((rob.skew(p) @ R, R))))

    assert np.isclose(np.sum((S @ U - X)**2), 0.)

    return S, U


def compute_dutir_from_pose_traj(T, ds, L=10**10, twist_type="body"):
    """
    Compute the DUTIR from a rigid-body pose trajectory.
    The pose trajectory is first converted into a screw trajectory
    (currently a body-twist trajectory), after which the SU-decomposition
    is applied successively to overlapping windows of three twist samples.

    Input:
    T  : numpy.ndarray, shape (4, 4, N)
    ds : float
    L  : float, optional
    twist_type : str, optional. Currently only "body" is supported.

    Output:
    dutir : numpy.ndarray, shape (6, 3, N-3)
    twist : numpy.ndarray, shape (6, N-1)

    """

    T_shape = T.shape
    assert len(T_shape) == 3, "Input pose trajectory has wrong dimensions."
    assert T_shape[0] == 4, "Input pose trajectory has wrong dimensions."
    assert T_shape[1] == 4, "Input pose trajectory has wrong dimensions."
    assert T_shape[2] >= 4, (
        "Input pose trajectory has insufficent number of trajectory samples, "
        "a minimum of four pose samples is required."
    )

    # Calculate body twist trajectory
    if twist_type == "body":
        twist = rob.calculate_bodytwist_from_poses(T, ds)
    else:
        raise TypeError("Wrong twist type, supported type(s) are 'body'.")

    # Calculate the dutir from the twist trajectory
    dutir = compute_dutir_from_screw_traj(twist, L)

    return dutir, twist


def compute_dutir_from_screw_traj(screw, L=10**10):
    """
    Compute the DUTIR from a screw trajectory. The DUTIR is computed by
    applying the SU-decomposition successively to overlapping windows
    of three screw samples.

    Input:
    screw  : numpy.ndarray, shape (6, N)
    L  : float, optional

    Output:
    dutir : numpy.ndarray, shape (6, 3, N-2)

    """

    screw_shape = screw.shape
    assert len(screw_shape) == 2, "Input screw trajectory has wrong dimensions."
    assert screw_shape[0] == 6, "Input screw trajectory has wrong dimensions."
    assert screw_shape[1] >= 3, (
        "Input screw trajectory has insufficent number of trajectory samples, a minimum of three samples is required."
    )

    # Perform the successive SU decompositions along the trajectory
    N = screw_shape[1]
    dutir = np.zeros((6, 3, N - 2))
    for k in range(N - 2):
        # Restructure twist data into successive overlapping windows of size (6,3)
        Xi = np.column_stack([screw[:, k], screw[:, k + 1], screw[:, k + 2]])

        # Compute U matrix
        _, U = SU(Xi, L=L)

        # Store the results
        dutir[:, :, k] = U

    return dutir
