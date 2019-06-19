import numpy as np


class Bmmap502PhPh:
    def __init__(self, matrS_elem=20):
        # BMMAP
        self.test_matrD_0 = np.array([[-8.28142513, 0.],
                                      [0., -0.26874977]])

        self.test_matrD = np.array([[8.22628993, 0.0551352],
                                    [0.14964989, 0.11909988]])

        self.test_qs = (0.8, 0.2)
        self.test_ns = (5, 2)

        self.priority_part = 0.1

        self.test_N = 10
        self.p_hp = 0.4

        # Server PH
        self.test_vect_beta = np.array([[1., 0.]])
        self.test_matrS = np.array([[-matrS_elem, matrS_elem],
                                    [0., -matrS_elem]])

        # Timer PH
        self.test_vect_gamma = np.array([[1., 0.]])
        self.test_matrGamma = np.array([[-10., 10.],
                                        [0., -10.]])
