import numpy as np


class Bmmap3PoissonPhPh:
    def __init__(self):
        # BMMAP
        self.test_matrD_0 = np.array([[-17.2743362832]])
        self.test_matrD = np.array([[17.2743362832]])

        self.test_q = 0.8
        self.test_n = 3

        self.test_N = 5
        self.p_hp = 0.4

        # Server PH
        self.test_vect_beta = np.array([[1., 0.]])
        self.test_matrS = np.array([[-20., 20.],
                                    [0., -20.]])

        # Timer PH
        self.test_vect_gamma = np.array([[1., 0.]])
        self.test_matrGamma = np.array([[-10., 10.],
                                        [0., -10.]])
