# tests/test_example.py
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import sulib.su_decomp as su_decomp
from sulib._robotics import skew


def test_SU_trivial():
    Xi = np.zeros((6, 3))
    S, U = su_decomp.SU(Xi)
    assert np.isclose(np.sum((S @ U - Xi) ** 2), 0.0)


def test_SU_trivial_reg():
    Xi = np.zeros((6, 3))
    S, U = su_decomp.SU(Xi, L=0.3)
    assert np.isclose(np.sum((S @ U - Xi) ** 2), 0.0)


def test_SU_1D_rotation():
    Xi = np.zeros((6, 3))
    Xi[0, :] = np.ones(3)
    S, U = su_decomp.SU(Xi)
    assert np.isclose(np.sum((S @ U - Xi) ** 2), 0.0)


def test_SU_1D_rotation_reg():
    Xi = np.zeros((6, 3))
    Xi[0, :] = np.ones(3)
    S, U = su_decomp.SU(Xi, L=0.3)
    assert np.isclose(np.sum((S @ U - Xi) ** 2), 0.0)


def test_SU_trivial_reg_max():
    Xi = np.zeros((6, 3))
    S, U = su_decomp.SU(Xi, L=0.0)
    assert np.isclose(np.sum((S @ U - Xi) ** 2), 0.0)


def test_SU_noise():
    Xi = np.random.rand(6, 3)
    S, U = su_decomp.SU(Xi)
    assert np.isclose(np.sum((S @ U - Xi) ** 2), 0.0)


def test_SU_noise_reg():
    Xi = np.random.rand(6, 3)
    S, U = su_decomp.SU(Xi, L=0.3)
    assert np.isclose(np.sum((S @ U - Xi) ** 2), 0.0)


def test_SU_noise_reg_max():
    Xi = np.random.rand(6, 3)
    S, U = su_decomp.SU(Xi, L=0.0)
    assert np.isclose(np.sum((S @ U - Xi) ** 2), 0.0)


def test_SU_invariance():
    # Test exact invariance in the unregularized case
    Xi1 = np.random.rand(6, 3)
    S1, U1 = su_decomp.SU(Xi1)

    p = np.random.rand(3)
    euler_angles = 20.0 * np.random.rand(3)
    R_ = Rot.from_euler("XYZ", euler_angles, degrees=True)
    R = R_.as_matrix()
    print(R)
    S_ = np.vstack((np.hstack((R, np.zeros((3, 3)))), np.hstack((skew(p) @ R, R))))

    Xi2 = S_ @ Xi1
    S2, U2 = su_decomp.SU(Xi2)

    assert np.isclose(np.sum((U1 - U2) ** 2), 0.0)
    assert np.isclose(np.sum((S_ @ S1 - S2) ** 2), 0.0)


def test_dutir_trivial():
    N = 100
    ds = 0.01
    T = np.zeros((4, 4, N))
    for k in range(N):
        T[:, :, k] = np.eye(4)
    su_decomp.pose_trajectory_to_dutir(T, ds, L=0.3, twist_type="body")
