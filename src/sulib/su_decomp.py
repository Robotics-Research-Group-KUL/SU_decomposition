import numpy as np

import sulib._data_handling as _dh
import sulib._robotics as _rob


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

    # Check whether input is of correct type and shape
    if not isinstance(A, np.ndarray):
        raise TypeError("A must be a NumPy array")
    if A.shape != (3, 3):
        raise ValueError(f"Expected shape (3, 3), got {A.shape}")

    # Ensure A is pseudo-regular to avoid divide by zero
    A = _dh.regulate_matrix(A)

    # Compute QR-decomposition
    R, U = np.linalg.qr(A)

    # Apply sign convention
    if U[0, 0] < 0:
        R[:, 0] *= -1
        U[0, :] *= -1
    if U[1, 1] < 0:
        R[:, 1] *= -1
        U[1, :] *= -1
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
        U[2, :] *= -1

    return R, U


def SU(X, L=10.0**10):
    """
    Compute the SU-decomposition of a 6x3 matrix.

    Input:
    X : np.ndarray of shape (6, 3)
    L : float, optional
        Regularization parameter controlling the trade-off between a
        coordinate-system-invariant representation and a representation
        with a coordinate-system-dependent translational component.

        The parameter L can be interpreted as a characteristic length scale.
        When the estimated screw axis of X[:,0] is farther than approximately L from the
        origin, the screw axis of X[:,0] is constrained to a maximum distance of L
        and a pure translational component is introduced. This translational component is
        anchored to the origin of the coordinate system.

        Hence, larger values of L favor a purely coordinate-system-invariant representation,
        but, for motions that are nearly translational, this may lead to an unintuitive representation
        with very distant screw axes.

    Output:
    S: np.ndarray of shape (6, 6)
    U: np.ndarray of shape (6, 3)

    """

    # Check whether input is of correct type and shape
    if not isinstance(X, np.ndarray):
        raise TypeError("A must be a NumPy array")
    if X.shape != (6, 3):
        raise ValueError(f"Expected shape (6, 3), got {X.shape}")
    if not isinstance(L, float) or L < 0.0:
        raise ValueError("L must be a positive float")

    # Divide X into two square block matrices X1 and X2
    X1 = X[0:3, :]
    X2 = X[3:6, :]

    # Ensure X1 is pseudo-regular to avoid divide by zero
    X1 = _dh.regulate_matrix(X1)

    # Compute the RU-decomposition of X1
    R, U1 = RU(X1)

    # Express the columns of X2 in a different basis defined by the columns of R
    RTX2 = R.T @ X2

    # Calculate p_star = [x y z]. p_star represents the intersection point of
    # the screw axis of X[:,0] and the common normal of the screw axes of X[:,0] and X[:,1].
    # The coordinates of p_star are expressed in the basis defined by the columns of R.
    z = RTX2[1, 0] / U1[0, 0]
    y = -RTX2[2, 0] / U1[0, 0]
    x = (RTX2[2, 1] + U1[0, 1] * y) / U1[1, 1]

    p_star = np.array([x, y, z])

    # Regularization:
    # Ensure the norm of p_star does not exceed a predefined value: || p_star || < L
    # This regularization procedure is designed such that the lower diagonal terms
    # of U[3:6,0] are minimized.
    if x**2 + y**2 + z**2 > L**2:  # Check whether || p_star || > L
        # Check which regularization strategy has to be applied
        if y**2 + z**2 >= L**2:
            # Case 1: there does not exists a point on the screw axis of X[:,0]
            #         with a distance to the origin smaller than L

            # Choose the point on the screw axis of X[:,0] with smallest distance to the origin
            x = 0

            # Uniformly normalize the y- and z-coordinates of p_star
            y = L * y / np.sqrt(y**2 + z**2)
            z = L * z / np.sqrt(y**2 + z**2)

        else:
            # Case 2: there exists at least one point on the screw axis of X[:,0]
            #         with a distance to the origin smaller than L

            # p_star is an intersection point of the screw axis of X[:,0] and a spherical manifold
            # with radius L around the origin.
            # In theory, there are always two intersection points with the spherical manifold,
            # but the 'sign(x)' ensures that the intersection point closest to the original p_star is chosen.
            x = np.sign(x) * np.sqrt(L**2 - y**2 - z**2)

        # Construct the complete p_star
        p_star = np.array([x, y, z])

        # Express p_star in the original base: p_star -> p
        p = R @ p_star

        # Compute U2
        U2 = RTX2 - _rob.skew(p_star) @ U1

        # Regularize R such that U1 and U2 are both as close as possible to
        # upper-triangular matrices. This regularization of R is achieved as R_reg = R @ Rc.T ,
        # with Rc a corrective rotation matrix. The matrix Rc is found by solving
        # an orthogonal procrustes problem.

        # Calculate exact upper-triangular form of U2 as a reference
        _, U2_tri = RU(U2)

        # Compute covariance matrix C
        C = L**2 * U1[:, 0:2] @ U1[:, 0:2].T + U2[:, 0:2] @ U2_tri[:, 0:2].T

        # Compute the SVD of C
        U, _, Vt = np.linalg.svd(C)
        V = Vt.T

        # Compute the optimal corrective rotation matrix Rc
        Rc = V @ U.T

        # Ensure Rc is a proper rotation matrix (det = +1)
        if np.linalg.det(Rc) < 0:
            U[:, -1] *= -1
            Rc = V @ U.T

        # Compute the regularized version of R
        R = R @ Rc.T

        # Compute the values of U1 and U2 based on the regularized version of R
        U1 = Rc @ U1
        U2 = Rc @ U2

    else:
        # No regularization required

        # Express p_star in the original base: p_star -> p
        p = R @ p_star

        # Compute U2
        U2 = RTX2 - _rob.skew(p_star) @ U1

    # Construct the complete U and S matrices
    U = np.vstack((U1, U2))
    S = np.vstack((np.hstack((R, np.zeros((3, 3)))), np.hstack((_rob.skew(p) @ R, R))))

    return S, U


def pose_trajectory_to_dutir(pose_trajectory, ds, L=10.0**10, twist_type="body"):
    """
    Compute the DUTIR from a rigid-body pose trajectory.
    The pose trajectory is first converted into a screw trajectory
    (currently a body-twist trajectory), after which the SU-decomposition
    is applied successively to overlapping windows of three twist samples.

    Input:
    pose_trajectory  : numpy.ndarray, shape (4, 4, N)
    ds : float
    L  : float, optional
        Regularization parameter controlling the trade-off between a
        coordinate-system-invariant representation and a representation
        with a coordinate-system-dependent translational component.
        -> see SU()

    twist_type : str, optional. Currently "body" and "spatial" are supported.

    Output:
    dutir : numpy.ndarray, shape (6, 3, N-3)
    twist_trajectory : numpy.ndarray, shape (6, N-1)

    """

    # Check whether input is of correct type and shape
    if not isinstance(pose_trajectory, np.ndarray):
        raise TypeError("pose_trajectory must be a NumPy array")
    if pose_trajectory.ndim != 3 or pose_trajectory.shape[:2] != (4, 4) or pose_trajectory.shape[2] < 4:
        raise ValueError(f"Expected shape (4, 4, N) with N >= 4, got {pose_trajectory.shape}")
    if not isinstance(L, float) or L < 0.0:
        raise ValueError("L must be a positive float")
    if twist_type not in ["body", "spatial"]:
        raise TypeError(
            "twist_type must be one of the supported types. Currently supported types: 'body' and 'spatial'."
        )

    # Calculate body twist trajectory
    if twist_type == "body":
        twist_trajectory = _rob.poses_to_bodytwists(pose_trajectory, ds)
    elif twist_type == "spatial":
        twist_trajectory = _rob.poses_to_spatialtwists(pose_trajectory, ds)

    # Calculate the dutir from the twist trajectory
    dutir = screw_trajectory_to_dutir(twist_trajectory, L)

    return dutir, twist_trajectory


def screw_trajectory_to_dutir(screw_trajectory, L=10.0**10):
    """
    Compute the DUTIR from a screw trajectory. The DUTIR is computed by
    applying the SU-decomposition successively to overlapping windows
    of three screw samples.

    Input:
    screw_trajectory  : numpy.ndarray, shape (6, N), N is the number of trajectory samples
    L  : float, optional
        Regularization parameter controlling the trade-off between a
        coordinate-system-invariant representation and a representation
        with a coordinate-system-dependent translational component.
        -> see SU()

    Output:
    dutir : numpy.ndarray, shape (6, 3, N-2)

    """

    # Check whether input is of correct type and shape
    if not isinstance(screw_trajectory, np.ndarray):
        raise TypeError("screw_trajectory must be a NumPy array")
    if screw_trajectory.ndim != 2 or screw_trajectory.shape[0] != 6 or screw_trajectory.shape[1] < 3:
        raise ValueError(f"Expected shape (6, N) with N >= 3, got {screw_trajectory.shape}")
    if not isinstance(L, float) or L < 0.0:
        raise ValueError("L must be a positive float")

    # Perform the successive SU decompositions along the trajectory
    N = screw_trajectory.shape[1]
    dutir = np.zeros((6, 3, N - 2))
    for k in range(N - 2):
        # Restructure twist data into successive overlapping windows of size (6,3)
        Xi = np.column_stack([screw_trajectory[:, k], screw_trajectory[:, k + 1], screw_trajectory[:, k + 2]])

        # Compute U matrix
        _, U = SU(Xi, L=L)

        # Store the results
        dutir[:, :, k] = U

    return dutir


## Generation of synthetic pose trajectories for numerical examples


def generate_synthetic_pose_trajectory(trajectory_type="rotation_3D"):
    """
    Generate trajectory data of a synthetic pose trajectory.

    Input:
    trajectory_type : str. Currently 'rotation_1D', 'rotation_3D' and 'translation_3D' are supported.

    Output:
    T: numpy.ndarray, shape (4, 4, 20), pose trajectory
    dt: float, timestep
    """

    if trajectory_type not in ["rotation_1D", "rotation_3D", "translation_3D"]:
        raise TypeError(
            "trajectory_type must be one of the supported types. "
            "Currently supported types: 'rotation_1D', 'rotation_3D' and 'translation_3D'."
        )

    N = 100  # number of trajectory samples
    time_total = 2  # seconds
    time_axis = np.linspace(0, time_total, N)
    dt = time_total / (N - 1)  # time resolution
    T_init = np.zeros((4, 4, N))  # initialisation of pose trajectory

    match trajectory_type:
        case "rotation_1D":
            T = _dh.generate_precession(T_init, N, time_axis, time_total)
        case "rotation_3D":
            T = _dh.generate_precession(T_init, N, time_axis, time_total)
            T = _dh.generate_spin(T, N, time_axis, time_total)
        case "translation_3D":
            T = _dh.generate_helical_translation(T_init, N, time_axis, time_total)

    return T, dt
