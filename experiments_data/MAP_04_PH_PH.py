import numpy as np


class Map04PhPh:
    def __init__(self):
        # BMMAP
        self.test_matrD_0 = np.array([[-86, 0.01],
                                      [0.02, -2.76]]) * 1.80258382661

        self.test_matrD = np.array([[85, 0.99],
                                    [0.2, 2.54]]) * 1.80258382661

        self.test_qs = (0.8, 0.8)
        self.test_ns = (1, 1)

        self.priority_part = 0.99999999999

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
