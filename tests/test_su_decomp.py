# tests/test_example.py
import numpy as np

import sulib.su_decomp as su_decomp


def test_SU_trivial():
    Xi = np.zeros((6, 3))
    su_decomp.SU(Xi)


def test_dutir_trivial():
    N = 100
    ds = 0.01
    T = np.zeros((4, 4, N))
    for k in range(N):
        T[:, :, k] = np.eye(4)
    su_decomp.compute_dutir_from_pose_traj(T, ds, L=0.3, twist_type="body")
