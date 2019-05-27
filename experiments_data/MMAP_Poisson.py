import numpy as np


class MmapPoisson:
    def __init__(self):
        # BMMAP
        self.test_matrD_0 = np.array([[-32.]])
        self.test_matrD = np.array([[32.]])

        self.test_q = 0.8
        self.test_n = 1

        self.priority_part = 0.7

        self.test_N = 1
        self.p_hp = 0.4

        # Server PH
        self.test_vect_beta = np.array([[1.]])
        self.test_matrS = np.array([[-10.]])

        # Timer PH
        self.test_vect_gamma = np.array([[1.]])
        self.test_matrGamma = np.array([[-5.]])
