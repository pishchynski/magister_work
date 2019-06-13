import numpy as np


class Bmmap502PhPh:
    def __init__(self, matrS_elem=20):
        # BMMAP
        self.test_matrD_0 = np.array([[-5.408, 0.],
                                      [0., -0.1755]]) * 3.121801121221731

        self.test_matrD = np.array([[5.372, 0.036],
                                    [0.09772, 0.07778]]) * 3.121801121221731

        self.test_q = 0.8
        self.test_n = 5

        self.priority_part = 0.9

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
