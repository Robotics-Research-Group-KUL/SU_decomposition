# tests/test_example.py
import numpy as np

import sulib.su_decomp as su_decomp


def test_SU_trivial():
    Xi = np.zeros((6,3))
    U, R, p = su_decomp.SU(Xi)
