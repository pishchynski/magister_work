import numpy as np


class Bmmap302PhPh:
    def __init__(self, matrS_elem=20):
        # BMMAP
        self.test_matrD_0 = np.array([[-5.408, 0.],
                                      [0., -0.1755]]) * 4.319328938619707

        self.test_matrD = np.array([[5.372, 0.036],
                                    [0.09772, 0.07778]]) * 4.319328938619707

        self.test_q = 0.8
        self.test_n = 3

        self.priority_part = 0.7

        self.test_N = 5
        self.p_hp = 0.4

        # Server PH
        self.test_vect_beta = np.array([[1., 0.]])
        self.test_matrS = np.array([[-matrS_elem, matrS_elem],
                                    [0., -matrS_elem]])

        # Timer PH
        self.test_vect_gamma = np.array([[1., 0.]])
        self.test_matrGamma = np.array([[-10., 10.],
                                        [0., -10.]])
