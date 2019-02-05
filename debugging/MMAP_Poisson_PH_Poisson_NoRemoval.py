import numpy as np


class MmapPoissonPhPoissonNoRemoval:
    def __init__(self):
        # BMMAP
        self.test_matrD_0 = np.array([[-10.]])
        self.test_matrD = np.array([[10.]])

        self.test_q = 0.8
        self.test_n = 1

        self.priority_part = 0.7

        self.test_N = 5
        self.p_hp = 10. ** (-5)

        # Server PH
        self.test_vect_beta = np.array([[1., 0.]])
        self.test_matrS = np.array([[-20., 20.],
                                    [0., -20.]])

        # Timer PH
        self.test_vect_gamma = np.array([[1.]])
        self.test_matrGamma = np.array([[-10. ** (-5)]])
